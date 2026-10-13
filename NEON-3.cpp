#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rawllm_common.hpp"
#include "rawllm_util.hpp"
#include "rawllm_json.hpp"
#include "rawllm_loader.hpp"
#include "rawllm_nctr_loader.hpp"
#include "rawllm_rocm.hpp"
#include "rawllm_vulkan.hpp"
#include "rawllm_forward.hpp"
#if defined(NANITY_ENABLE_IDLE_LOOP)
#include "nectar_diskmem.hpp"
#include "nectar_vision.hpp"
#include "nectar_splice.hpp"
#include "nectar_cot.hpp"
#include "nectar_zram.hpp"
#endif

// =============================================================================
// FIX (bug 2): VERBOSE_LOG timing macros.
// Previously the Python flag passed -DVERBOSE_LOG=1 but no C++ code read it,
// so the "per-stage timing" advertised in the UI was a no-op.  These macros
// write microsecond-stamped lines to stderr whenever VERBOSE_LOG is defined.
// =============================================================================
#ifdef VERBOSE_LOG
#  define VLOG(msg) \
    do { \
        auto _now = std::chrono::steady_clock::now(); \
        auto _us  = std::chrono::duration_cast<std::chrono::microseconds>( \
                        _now.time_since_epoch()).count(); \
        std::cerr << "[VERBOSE " << _us << "us] " << (msg) << "\n"; \
    } while(0)
#  define VLOG_ELAPSED(label, t0) \
    do { \
        auto _dt = std::chrono::steady_clock::now() - (t0); \
        std::cerr << "[VERBOSE] " << (label) << " " \
                  << std::chrono::duration_cast<std::chrono::milliseconds>(_dt).count() \
                  << "ms\n"; \
    } while(0)
#else
#  define VLOG(msg)             do {} while(0)
#  define VLOG_ELAPSED(lbl, t0) do {} while(0)
#endif


// =============================================================================
// FIX (bug 9 — tokenizer family mismatch): Phi-4-mini-instruct's config.json
// declares "AutoTokenizer": "Xenova/gpt-4o" — a byte-level BPE tokenizer
// (the GPT-2 / GPT-4o / tiktoken family), NOT SentencePiece. There is no ▁
// (U+2581) word-start marker anywhere in that vocab; instead every raw byte
// is remapped through a fixed byte<->unicode table before BPE ever sees it
// (so byte 0x20 "space" becomes the vocab character "Ġ", and a long tail of
// other bytes map to obscure Latin-1/printable Unicode codepoints that look
// like ordinary text but each stand in for exactly one raw byte).
//
// The previous Tokenizer searched for ▁-prefixed substrings (which never
// match) and greedily matched literal text against the vocab (which mostly
// finds nothing, because the vocab strings are byte-remapped, not literal
// text) — and on decode, it stripped a ▁ sequence that doesn't exist while
// leaving every byte-remapped vocab string un-translated, which is exactly
// what produces output like "_l<5256155535353535...": real vocab strings
// emitted as literal characters instead of being run through the inverse
// byte<->unicode mapping.
//
// This replacement is a real byte-level BPE codec:
//   1. ByteCodec: the canonical GPT-2-style byte<->unicode table.
//   2. pretokenize(): splits input into word/number/punctuation/whitespace
//      chunks. This is a hand-rolled approximation of the regex GPT-2/
//      tiktoken-family tokenizers pretokenize with ('s|'t|'re|...| ?\p{L}+|
//      ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+) — std::regex has no Unicode
//      property classes, so non-ASCII bytes are simply treated as "letter"
//      class. This is NOT byte-exact for every script, but it is the right
//      shape of pretokenizer for this vocab family, vs. none at all.
//   3. bpe_merge(): real merge-rule application over tok_meta.merges
//      (rank = list position, lowest rank wins, repeat to fixed point) —
//      the part the old comment admitted was missing.
//   4. decode_one(): inverse byte<->unicode mapping per vocab string, fed
//      through a small streaming buffer that withholds any trailing bytes
//      that don't yet form a complete UTF-8 codepoint — necessary because
//      a single printed character can be split across two adjacent BPE
//      tokens, and the call sites JSON-escape/print each piece immediately
//      as it's produced. flush() drains anything still buffered once
//      generation stops (EOS or max_tokens).
// =============================================================================
struct ByteCodec {
    std::array<std::string, 256> byte_to_str;          // raw byte -> vocab-alphabet UTF-8 string
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;   // vocab-alphabet codepoint -> raw byte

    static std::string encode_cp(uint32_t cp) {
        // All codepoints this table produces are < 0x800 (verified by
        // construction below), so 1- or 2-byte UTF-8 is always sufficient.
        std::string s;
        if (cp < 0x80) {
            s += (char)cp;
        } else {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        }
        return s;
    }

    ByteCodec() {
        std::vector<int> bs;
        for (int b = (int)'!'; b <= (int)'~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs;

        std::array<bool, 256> present{};
        for (int b : bs) present[b] = true;

        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!present[b]) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }

        for (size_t i = 0; i < bs.size(); ++i) {
            uint8_t  byte = (uint8_t)bs[i];
            uint32_t cp   = (uint32_t)cs[i];
            byte_to_str[byte] = encode_cp(cp);
            cp_to_byte[cp]    = byte;
        }
    }
};

// Decode one UTF-8 codepoint starting at s[i]. Returns false (and advances
// by exactly 1 byte) on malformed input so callers can pass the raw byte
// through rather than getting stuck.
static bool utf8_decode_one(const std::string& s, size_t i, uint32_t& cp, size_t& len) {
    uint8_t c = (uint8_t)s[i];
    if ((c & 0x80) == 0x00) { cp = c; len = 1; return true; }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        cp = (uint32_t)(c & 0x1F) << 6 | (uint8_t(s[i+1]) & 0x3F);
        len = 2; return true;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        cp = (uint32_t)(c & 0x0F) << 12 | (uint32_t)(uint8_t(s[i+1]) & 0x3F) << 6
                                         | (uint8_t(s[i+2]) & 0x3F);
        len = 3; return true;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        cp = (uint32_t)(c & 0x07) << 18 | (uint32_t)(uint8_t(s[i+1]) & 0x3F) << 12
                                         | (uint32_t)(uint8_t(s[i+2]) & 0x3F) << 6
                                         | (uint8_t(s[i+3]) & 0x3F);
        len = 4; return true;
    }
    len = 1; return false;
}

// Length of the prefix of buf that consists only of complete UTF-8
// codepoints, i.e. where to cut so we never emit a torn multi-byte char.
static size_t complete_utf8_prefix_len(const std::string& buf) {
    size_t n = buf.size();
    if (n == 0) return 0;
    size_t back_limit = std::min<size_t>(4, n);
    for (size_t back = 1; back <= back_limit; ++back) {
        uint8_t c = (uint8_t)buf[n - back];
        if ((c & 0xC0) == 0x80) continue;   // continuation byte, keep walking back
        int need;
        if      ((c & 0x80) == 0x00) need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else need = 1;                       // invalid lead byte: don't buffer forever
        return (back < (size_t)need) ? (n - back) : n;
    }
    return n;   // more than 4 trailing continuation bytes with no lead found: malformed, flush all
}

struct Tokenizer {
    const loader::TokenizerMeta&             meta;
    std::unordered_map<std::string, int32_t> vocab_map;   // byte-level-BPE string -> id
    std::unordered_map<std::string, int32_t> bpe_ranks;   // "left right" -> merge priority (lower = earlier)
    int32_t bos_id, eos_id;
    ByteCodec codec;
    std::string pending_bytes;   // streaming-decode buffer: bytes not yet a complete UTF-8 char

    // FIX (bug 10 — root cause of garbled output even after the byte-BPE
    // codec fix): control tokens like <|system|>, <|user|>, <|assistant|>,
    // <|end|> are single atomic vocab entries the model was trained to
    // treat as structural markers — NOT literal text. The old encode() had
    // no concept of this: literal "<|system|>" in a prompt got pretokenized
    // and BPE-merged character-by-character like ordinary text (<, |, s, y,
    // s, t, e, m, |, >...), so the model never actually saw its own control
    // tokens, just a punctuation-heavy approximation of their spelling. No
    // amount of correct RoPE/byte-codec/EOS-id fixing helps if the prompt
    // itself never contains the tokens the model was fine-tuned to expect.
    //
    // special_tokens here is auto-populated from the vocab itself (any
    // entry of the exact shape "<|...|>", with no '|' in the middle) rather
    // than a hardcoded list, so it stays correct if the vocab changes —
    // sorted longest-first so e.g. "<|end|>" can't accidentally be matched
    // as a prefix of something longer at the same position.
    std::vector<std::string> special_tokens;

    static bool looks_like_special_token(const std::string& s) {
        if (s.size() < 4) return false;
        if (s[0] != '<' || s[1] != '|' || s[s.size()-2] != '|' || s.back() != '>') return false;
        for (size_t i = 2; i + 2 < s.size(); ++i)
            if (s[i] == '|' || s[i] == '<' || s[i] == '>') return false;   // keep it a single clean tag
        return true;
    }

    explicit Tokenizer(const loader::TokenizerMeta& m)
        : meta(m), bos_id(m.bos_id), eos_id(m.eos_id)
    {
        vocab_map.reserve(m.tokens.size());
        for (size_t i = 0; i < m.tokens.size(); ++i)
            vocab_map[m.tokens[i]] = static_cast<int32_t>(i);

        // tok_meta.merges entries are "left right" (single-space separated,
        // standard GGUF tokenizer.ggml.merges convention) in priority order.
        bpe_ranks.reserve(m.merges.size());
        for (size_t i = 0; i < m.merges.size(); ++i)
            bpe_ranks[m.merges[i]] = static_cast<int32_t>(i);

        for (const auto& t : m.tokens)
            if (looks_like_special_token(t)) special_tokens.push_back(t);
        std::sort(special_tokens.begin(), special_tokens.end(),
                  [](const std::string& a, const std::string& b){ return a.size() > b.size(); });
    }

    // Returns the matching special token string at text[pos], or empty if
    // none matches there. Longest-match since special_tokens is presorted.
    const std::string* match_special_token(const std::string& text, size_t pos) const {
        for (const auto& s : special_tokens)
            if (text.compare(pos, s.size(), s) == 0) return &s;
        return nullptr;
    }

    // FIX (bug 11 — SPM/SentencePiece tokenizer support): the code below
    // previously assumed every model uses byte-level BPE (GPT-2 style
    // alphabet + merge list), unconditionally. Llama/TinyLlama-family
    // models declare tokenizer.ggml.model == "llama" and use SentencePiece
    // vocabularies instead — raw UTF-8 vocab strings (word-boundary marked
    // with U+2581 '▁', not a byte-remap alphabet) segmented by unigram-LM
    // score, not a merge-rank list. Feeding an SPM vocab through the GPT-2
    // path means almost every merged symbol misses vocab_map, and encode()
    // falls all the way back to one <0xXX> byte-fallback token per input
    // byte — a segmentation the model never saw in training, hence
    // coherent-looking-but-meaningless output regardless of how correct
    // the transformer math is.
    bool is_spm = (meta.model == "llama");

    // "<0xXX>" -> the single raw byte it represents, or -1 if `s` doesn't
    // match that exact shape. Used on both tokenizer paths: GPT-2 byte-BPE
    // models can still emit a literal byte-fallback token in rare cases,
    // and SPM models use it as their primary out-of-vocab escape hatch.
    static int parse_byte_fallback(const std::string& s) {
        if (s.size() != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return -1;
        auto hex = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            return -1;
        };
        int hi = hex(s[3]), lo = hex(s[4]);
        if (hi < 0 || lo < 0) return -1;
        return (hi << 4) | lo;
    }

    // SPM greedy score-based merge: repeatedly merge the adjacent symbol
    // pair whose concatenation exists in vocab_map with the highest score,
    // until no mergeable pair remains. This is the standard emulation of
    // SentencePiece's unigram segmentation via BPE-shaped merging — scores
    // (from tokenizer.ggml.scores) stand in for the merge-rank list that
    // GPT-2-style vocabs ship instead.
    std::vector<std::string> spm_merge(std::vector<std::string> symbols) const {
        while (symbols.size() > 1) {
            float  best_score = -std::numeric_limits<float>::infinity();
            size_t best_i     = SIZE_MAX;
            for (size_t k = 0; k + 1 < symbols.size(); ++k) {
                auto it = vocab_map.find(symbols[k] + symbols[k + 1]);
                if (it == vocab_map.end()) continue;
                float sc = (it->second >= 0 && (size_t)it->second < meta.scores.size())
                           ? meta.scores[it->second] : 0.0f;
                if (sc > best_score) { best_score = sc; best_i = k; }
            }
            if (best_i == SIZE_MAX) break;
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }
        return symbols;
    }

    // SPM encode for one literal chunk: ' ' -> '▁' (U+2581, SentencePiece's
    // word-boundary marker), leading '▁' added so the first word gets one
    // too (matches SentencePiece's add_dummy_prefix default), start from
    // individual UTF-8 characters (not bytes — SPM vocab strings are plain
    // UTF-8, unlike the GPT-2 byte-remap alphabet), then greedily merge.
    // Characters SPM has no single-codepoint vocab entry for fall back to
    // one <0xXX> token per UTF-8 byte, same escape hatch as the BPE path.
    void encode_literal_chunk_spm(const std::string& literal, std::vector<int32_t>& ids) const {
        std::string prefixed;
        prefixed.reserve(literal.size() + 3);
        prefixed += "\xE2\x96\x81";   // ▁ (U+2581) UTF-8 bytes
        for (char c : literal) {
            if (c == ' ') prefixed += "\xE2\x96\x81";
            else          prefixed += c;
        }

        std::vector<std::string> symbols;
        size_t i = 0;
        while (i < prefixed.size()) {
            uint32_t cp; size_t len;
            if (!utf8_decode_one(prefixed, i, cp, len)) { len = 1; }
            symbols.push_back(prefixed.substr(i, len));
            i += len;
        }

        for (const std::string& sym : spm_merge(std::move(symbols))) {
            auto it = vocab_map.find(sym);
            if (it != vocab_map.end()) { ids.push_back(it->second); continue; }
            for (unsigned char b : sym) {
                char hex[8];
                std::snprintf(hex, sizeof(hex), "<0x%02X>", b);
                auto hit = vocab_map.find(hex);
                if (hit != vocab_map.end()) ids.push_back(hit->second);
            }
        }
    }

    // Split text into pretokenization chunks (word / number / punctuation /
    // whitespace runs), each independently fed through BPE merges below.
    static std::vector<std::string> pretokenize(const std::string& text) {
        static const char* kContractions[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
        auto is_space       = [](unsigned char c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\v'||c=='\f'; };
        auto is_ascii_letter = [](unsigned char c){ return std::isalpha(c) != 0; };
        auto is_ascii_digit  = [](unsigned char c){ return std::isdigit(c) != 0; };
        auto utf8_len        = [](unsigned char c)->size_t {
            if ((c & 0x80) == 0x00) return 1;
            if ((c & 0xE0) == 0xC0) return 2;
            if ((c & 0xF0) == 0xE0) return 3;
            if ((c & 0xF8) == 0xF0) return 4;
            return 1;
        };

        std::vector<std::string> chunks;
        size_t i = 0, n = text.size();
        while (i < n) {
            unsigned char c = (unsigned char)text[i];

            // Contractions bind to the preceding word in real GPT-2 BPE, but
            // splitting them off as their own chunk still segments correctly
            // against this vocab (each contraction piece is itself a vocab
            // entry), so treat them as their own chunk for simplicity.
            if (c == '\'') {
                bool matched = false;
                for (const char* ctok : kContractions) {
                    size_t clen = std::strlen(ctok);
                    if (text.compare(i, clen, ctok) == 0) {
                        chunks.push_back(ctok);
                        i += clen; matched = true; break;
                    }
                }
                if (matched) continue;
            }

            if (is_space(c)) {
                size_t j = i;
                while (j < n && is_space((unsigned char)text[j])) ++j;
                size_t wslen = j - i;
                if (j == n) {
                    // Trailing whitespace at end of input: its own chunk.
                    chunks.push_back(text.substr(i, wslen));
                    i = j;
                    continue;
                } else if (wslen > 1) {
                    // All but the last space is its own chunk; the final
                    // space is left for the next chunk to pick up as a
                    // leading space (mirrors " ?\p{L}+" etc. in the regex).
                    chunks.push_back(text.substr(i, wslen - 1));
                    i = j - 1;
                    continue;
                }
                // wslen == 1 and not at end: fall through below so the
                // word/number/punct branch picks this up as a leading
                // space (its own `c == ' '` check handles it). Looping
                // back here with i unchanged would spin forever.
            }

            size_t start = i;
            bool leading_space = false;
            if (c == ' ') {
                leading_space = true;
                ++i;
                if (i >= n) { chunks.push_back(" "); break; }
                c = (unsigned char)text[i];
            }
            (void)leading_space;

            if (is_ascii_letter(c) || c >= 0x80) {
                size_t j = i;
                while (j < n) {
                    unsigned char cj = (unsigned char)text[j];
                    if (is_ascii_letter(cj))      { ++j; }
                    else if (cj >= 0x80)          { j += utf8_len(cj); }
                    else break;
                }
                chunks.push_back(text.substr(start, j - start));
                i = j;
            } else if (is_ascii_digit(c)) {
                size_t j = i;
                while (j < n && is_ascii_digit((unsigned char)text[j])) ++j;
                chunks.push_back(text.substr(start, j - start));
                i = j;
            } else {
                size_t j = i;
                while (j < n) {
                    unsigned char cj = (unsigned char)text[j];
                    if (is_space(cj) || is_ascii_letter(cj) || is_ascii_digit(cj) || cj >= 0x80) break;
                    ++j;
                }
                if (j == i) j = i + 1;   // guarantee forward progress
                chunks.push_back(text.substr(start, j - start));
                i = j;
            }
        }
        return chunks;
    }

    // Repeatedly merge the lowest-rank adjacent pair until none apply.
    std::vector<std::string> bpe_merge(std::vector<std::string> symbols) const {
        while (symbols.size() > 1) {
            int    best_rank = std::numeric_limits<int>::max();
            size_t best_i    = SIZE_MAX;
            for (size_t k = 0; k + 1 < symbols.size(); ++k) {
                auto it = bpe_ranks.find(symbols[k] + " " + symbols[k + 1]);
                if (it != bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i    = k;
                }
            }
            if (best_i == SIZE_MAX) break;
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }
        return symbols;
    }

    // Encode text -> token-id sequence via real byte-level BPE, with
    // special control tokens (<|system|>, <|end|>, etc.) recognized as
    // atomic ids rather than being shredded through BPE as literal text.
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const {
        std::vector<int32_t> ids;
        if (add_bos && bos_id >= 0) ids.push_back(bos_id);

        auto encode_literal_chunk = [&](const std::string& literal) {
            if (is_spm) { encode_literal_chunk_spm(literal, ids); return; }
            for (const std::string& chunk : pretokenize(literal)) {
                std::vector<std::string> symbols;
                symbols.reserve(chunk.size());
                for (unsigned char b : chunk) symbols.push_back(codec.byte_to_str[b]);

                for (const std::string& sym : bpe_merge(std::move(symbols))) {
                    auto it = vocab_map.find(sym);
                    if (it != vocab_map.end()) {
                        ids.push_back(it->second);
                        continue;
                    }
                    // Vocab/merges mismatch (shouldn't happen with a consistent
                    // GGUF, but don't silently drop input): fall back to
                    // per-byte lookup of this merged symbol's original bytes.
                    size_t bi = 0;
                    while (bi < sym.size()) {
                        uint32_t cp; size_t len;
                        if (!utf8_decode_one(sym, bi, cp, len)) { ++bi; continue; }
                        auto bit = codec.cp_to_byte.find(cp);
                        if (bit != codec.cp_to_byte.end()) {
                            char hex[8];
                            std::snprintf(hex, sizeof(hex), "<0x%02X>", bit->second);
                            auto hit = vocab_map.find(hex);
                            if (hit != vocab_map.end()) ids.push_back(hit->second);
                        }
                        bi += len;
                    }
                }
            }
        };

        size_t i = 0, n = text.size(), literal_start = 0;
        while (i < n) {
            const std::string* tag = special_tokens.empty() ? nullptr : match_special_token(text, i);
            if (tag) {
                if (i > literal_start) encode_literal_chunk(text.substr(literal_start, i - literal_start));
                ids.push_back(vocab_map.at(*tag));   // present by construction: tag came from the vocab itself
                i += tag->size();
                literal_start = i;
            } else {
                ++i;
            }
        }
        if (literal_start < n) encode_literal_chunk(text.substr(literal_start, n - literal_start));

        return ids;
    }

    // Decode a single token id -> UTF-8 bytes, run through the inverse
    // byte<->unicode mapping, withholding any trailing incomplete UTF-8
    // sequence (it'll complete once the next token's bytes are appended).
    std::string decode_one(int32_t id) {
        if (id < 0 || id >= (int32_t)meta.tokens.size()) return "";
        const std::string& t = meta.tokens[id];

        // FIX (bug 11, decode half): "<0xXX>" is a byte-fallback token
        // representing exactly one raw byte — not six literal characters.
        // Previously this fell through the loop below, where each ASCII
        // char of "<0xE4>" mostly self-maps through cp_to_byte (GPT-2's
        // byte-remap table sends most printable ASCII to itself), so the
        // token decoded as the literal text "<0xE4>" instead of the single
        // byte 0xE4 it stands for. That's also why multi-byte UTF-8
        // characters, split across several <0xXX> tokens by the encoder,
        // came out as literal "<0x..>" text or "�" instead of reassembling
        // into the intended character — this check needs to run before the
        // per-character loop, on both tokenizer paths (GPT-2 vocabs can
        // also emit a literal byte-fallback token, not just SPM ones).
        int bf = parse_byte_fallback(t);
        if (bf >= 0) {
            pending_bytes += (char)(uint8_t)bf;
            size_t emit_len = complete_utf8_prefix_len(pending_bytes);
            std::string out = pending_bytes.substr(0, emit_len);
            pending_bytes.erase(0, emit_len);
            return out;
        }

        std::string raw;
        raw.reserve(t.size());
        if (is_spm) {
            // SPM vocab strings are plain UTF-8, not byte-remapped — emit
            // as-is except for the '▁' (U+2581) word-boundary marker,
            // which SentencePiece uses in place of a literal space.
            size_t i = 0;
            while (i < t.size()) {
                uint32_t cp; size_t len;
                if (!utf8_decode_one(t, i, cp, len)) { raw += t[i]; ++i; continue; }
                if (cp == 0x2581) raw += ' ';
                else              raw += t.substr(i, len);
                i += len;
            }
        } else {
            size_t i = 0;
            while (i < t.size()) {
                uint32_t cp; size_t len;
                if (!utf8_decode_one(t, i, cp, len)) { raw += t[i]; ++i; continue; }
                auto it = codec.cp_to_byte.find(cp);
                if (it != codec.cp_to_byte.end()) {
                    raw += (char)it->second;
                } else {
                    // Not part of the byte-remap alphabet — this is a literal
                    // special-token character (e.g. from "<|im_start|>"), not a
                    // remapped byte. Emit it verbatim.
                    raw += t.substr(i, len);
                }
                i += len;
            }
        }

        pending_bytes += raw;
        size_t emit_len = complete_utf8_prefix_len(pending_bytes);
        std::string out = pending_bytes.substr(0, emit_len);
        pending_bytes.erase(0, emit_len);
        return out;
    }

    // Drain any bytes still withheld by decode_one (call once generation
    // stops — EOS or max_tokens — so a trailing torn character isn't lost).
    std::string flush() {
        std::string out = std::move(pending_bytes);
        pending_bytes.clear();
        return out;
    }
};


// =============================================================================
// FIX (bug 8): Top-p (nucleus) sampler using fast_expf from rawllm_common.hpp.
// Previously the only sampling call was commented out; logits were never used.
// =============================================================================
int32_t sample_top_p(const float* logits, int n_vocab,
                     float temperature, float top_p,
                     std::mt19937& rng)
{
    // Greedy decode when temperature ≈ 0 or top_p ≈ 0.
    if (temperature < 1e-5f || top_p < 1e-5f) {
        return static_cast<int32_t>(
            std::max_element(logits, logits + n_vocab) - logits);
    }

    // Compute softmax with temperature via fast_expf.
    std::vector<float> probs(n_vocab);
    float max_l = *std::max_element(logits, logits + n_vocab);
    float sum   = 0.f;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = engine::fast_expf((logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < n_vocab; ++i) probs[i] /= sum;

    // Sort indices by probability descending.
    std::vector<int32_t> idx(n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(),
                      idx.begin() + std::min(n_vocab, 200), // cap sort for speed
                      idx.end(),
                      [&](int a, int b){ return probs[a] > probs[b]; });

    // Accumulate until nucleus covers top_p of probability mass.
    float cumsum = 0.f;
    int   cutoff = 1;
    for (int i = 0; i < std::min(n_vocab, 200); ++i) {
        cumsum += probs[idx[i]];
        cutoff  = i + 1;
        if (cumsum >= top_p) break;
    }

    // Sample uniformly within the nucleus.
    std::uniform_real_distribution<float> dist(0.f, cumsum);
    float r   = dist(rng);
    float acc = 0.f;
    for (int i = 0; i < cutoff; ++i) {
        acc += probs[idx[i]];
        if (r <= acc) return idx[i];
    }
    return idx[0];
}


// =============================================================================
// FIX (bug 8 — root cause of garbled output): forward() used to be a stub
// that filled out_logits with zeros, so sample_top_p() sampled uniformly at
// random over the whole vocab every step. The real transformer math lives
// in rawllm_forward.hpp — the fixed NANITY architecture (RMSNorm -> separate
// Q/K/V proj -> RoPE -> causal GQA attention -> out-proj -> SwiGLU FFN), per
// layer. See that file's header comment for scope/limitations before
// reporting follow-up bugs.
//
// FIX (perf): forward() used to recompute the whole context from scratch on
// every generated token (no KV cache) — O(seq²) total work per reply. Both
// ModelWeights AND a KVCache are now built once per process (function-local
// statics — safe because main() loads exactly one GGUF per process
// lifetime, so cfg/gguf never change between calls) and reused across every
// chat turn — including across turns now: cached_tokens records which token
// ids are currently sitting in the cache, and run_generation() below
// matches that against each new turn's freshly-tokenized prompt to find how
// much can be reused instead of re-prefilled. The underlying ~1GB of
// allocated K/V storage is only paid for once, not per turn, regardless.
// =============================================================================
struct EngineState {
    fwd::ModelWeights    mw;
    fwd::KVCache          cache;
    std::vector<int32_t>  cached_tokens;   // invariant: cached_tokens.size() == cache.length
};

template <typename Loader>
static EngineState& get_engine_state(const engine::Config& cfg,
                                      const Loader& gguf)
{
    // FIX (OOM crash on first chat turn — "Engine closed stdout unexpectedly"):
    // this used to size the cache off cfg.ctx_len (the model's full trained
    // context length — 131072 for Phi-4-mini). KVCache::assign() zero-fills
    // every element immediately, so that's not a lazy reservation; it forces
    // the OS to commit ~32GB on the spot the moment the first request comes
    // in, regardless of how short that request is, and the kernel OOM-kills
    // the process before it can write a single byte back. cfg.kv_window
    // (rawllm_loader.hpp's validate_config(), defaulted to min(ctx_len,4096))
    // exists specifically to bound this and was never wired in here.
    // ensure_cache_room()/kv_cache_shift() already handle sliding the window
    // forward once it fills, so capping the allocation at kv_window doesn't
    // lose correctness — it just means conversations beyond kv_window tokens
    // gracefully evict instead of the cache ever trying to be ctx_len long.
    static EngineState state{
        fwd::ModelWeights::build(gguf, cfg),
        fwd::KVCache(cfg.n_layer, (size_t)cfg.n_kv_head, (size_t)cfg.head_dim, cfg.kv_window),
        {}
    };
    return state;
}


// =============================================================================
// Graceful context-window overflow: called before every forward() call with
// the number of NEW tokens about to be processed. If there isn't room,
// evicts the oldest tokens (keeping a small fixed "sink" prefix, à la
// StreamingLLM/llama.cpp context-shift) and RoPE-rotates the survivors down
// by the discarded amount via fwd::kv_cache_shift() — see that function's
// comment for why the rotation makes this exact rather than approximate.
// cached_tokens is trimmed in lockstep so cross-turn prefix matching never
// compares against positions that no longer mean what they used to.
//
// This replaces the old behavior of just stopping generation once ctx_len
// was hit mid-reply. It's still graceful FORGETTING, not free context — the
// discarded token range is genuinely gone from the model's perspective —
// the difference is generation continues instead of halting.
// =============================================================================
// `keep` defaults to 4 (the original hardcoded StreamingLLM sink count) for
// every existing caller (chat/answer path). The idle loop (run_idle_loop(),
// below) passes its own keep count instead — the size of its pinned
// seed+sink region — since §1 of the idle-loop doc requires that region to
// survive eviction regardless of how many dedicated sink tokens it does or
// doesn't also carry.
//
// Returns the number of positions actually discarded (0 if no shift
// happened this call). run_idle_loop() uses this to pop exactly that many
// entries off its own token-aligned text buffer before handing them to
// nectar_diskmem — the two are guaranteed to line up because both are
// FIFO over the same sliding region, discard-count-for-discard-count.
static size_t ensure_cache_room(const engine::Config& cfg, EngineState& st, size_t seq,
                                 size_t keep = 4) {
    if (st.cache.length + seq <= st.cache.capacity) return 0;

    keep = std::min(keep, st.cache.length);

    if (keep + seq > st.cache.capacity) {
        throw std::runtime_error(
            "ensure_cache_room(): " + std::to_string(seq) + " new token(s) plus " +
            std::to_string(keep) + " sink token(s) exceed kv_window=" +
            std::to_string(st.cache.capacity) + " even after evicting everything "
            "else. Reduce the prompt/max_tokens, or raise the kv_window default "
            "in rawllm_loader.hpp's validate_config() (currently min(ctx_len, "
            "4096)) and rebuild -- note memory scales as n_layer * kv_window * "
            "kv_dim * 4 bytes * 2, so raise it deliberately.");
    }

    size_t need        = (st.cache.length + seq) - st.cache.capacity;
    size_t cushion     = st.cache.capacity / 4;   // avoid re-shifting on every next token
    size_t max_discard = st.cache.length - keep;
    size_t discard      = std::min(max_discard, need + cushion);
    if (discard == 0) return 0;

    VLOG("context shift: keep=" + std::to_string(keep) +
         " discard=" + std::to_string(discard) +
         " length_before=" + std::to_string(st.cache.length));

    const std::vector<float> freqs = fwd::rope_make_freqs(cfg.head_dim, cfg.rope_base);
    size_t rope_half = cfg.rope_dim_count ? (cfg.rope_dim_count / 2) : (cfg.head_dim / 2);
    fwd::kv_cache_shift(st.cache, keep, discard, cfg.n_kv_head, cfg.head_dim, rope_half,
                        cfg.rope_scale, freqs.data());

    st.cached_tokens.erase(st.cached_tokens.begin() + keep,
                            st.cached_tokens.begin() + keep + discard);

    VLOG("context shift done: length_after=" + std::to_string(st.cache.length));
    return discard;
}


// =============================================================================
// FIX (bug 10): this used to emit ChatML (<|im_start|>role\n...<|im_end|>\n),
// which doesn't exist anywhere in this model's vocab (confirmed via
// check_tokenizer.py — no <|im_start|>/<|im_end|>/<|im_sep|> entries at
// all). Phi-4-mini-instruct's actual published format, straight from
// Microsoft's model card, is simpler and has no newlines:
//   <|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>
// with the trailing <|assistant|> (no closing <|end|>) left as the
// generation prompt for the model to continue from.
// =============================================================================
static std::string format_chat(const json::Value& messages) {
    std::string prompt;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& m   = messages[i];
        const auto& role = m["role"].get_str();
        const auto& body = m["content"].get_str();
        if      (role == "system")    prompt += "<|system|>"    + body + "<|end|>";
        else if (role == "user")      prompt += "<|user|>"      + body + "<|end|>";
        else if (role == "assistant") prompt += "<|assistant|>" + body + "<|end|>";
    }
    prompt += "<|assistant|>";
    return prompt;
}


// =============================================================================
// Core autoregressive generation, shared by both single-shot generate() and
// the persistent run_interactive() loop. Decodes one token at a time,
// calling on_token(id) for each piece — so the caller decides whether to
// print plain text (generate()) or NDJSON (run_interactive()) without
// duplicating the prefill/decode logic itself.
//
// Cross-turn cache reuse: NEON.py resends the full conversation as text
// every turn, so prompt_ctx is re-tokenized from scratch each call. Causal
// attention guarantees position i's K/V depends only on tokens[0..i], so
// wherever the new prompt's tokens match what's already in the cache
// (token for token — even a single divergent id, e.g. from tokenizer
// re-segmentation rather than new content, stops the match), those
// positions don't need to be reprocessed at all. Only the new suffix after
// the match point gets prefilled. This is exact, never approximate: a
// mismatch just means a shorter reused prefix, never a wrong one.
// =============================================================================
template <typename Loader, typename OnToken>
static void run_generation(const engine::Config& cfg,
                            const Loader& gguf,
                            const Tokenizer& tok,   // only eos_id is read here; decode_one (stateful) is called by the on_token callback in our callers, not here
                            const std::vector<int32_t>& prompt_ctx,
                            int max_new_tokens,
                            float temperature,
                            float top_p,
                            util::ThreadPool& pool,
                            std::mt19937& rng,
                            OnToken&& on_token)
{
    auto& st = get_engine_state(cfg, gguf);

    size_t match_len = 0;
    size_t max_check = std::min({ st.cached_tokens.size(), prompt_ctx.size(), st.cache.length });
    while (match_len < max_check && st.cached_tokens[match_len] == prompt_ctx[match_len])
        ++match_len;

    // K/V being cached doesn't cache logits — always leave at least one
    // "new" token so we get a fresh forward call to sample from, even if
    // the entire new prompt was already cached verbatim (e.g. regenerate).
    if (!prompt_ctx.empty() && match_len >= prompt_ctx.size())
        match_len = prompt_ctx.size() - 1;

    st.cache.length = match_len;
    st.cached_tokens.resize(match_len);
    std::vector<int32_t> new_prompt_tokens(prompt_ctx.begin() + match_len, prompt_ctx.end());
    VLOG("cache reuse: matched=" + std::to_string(match_len) + "/" +
         std::to_string(prompt_ctx.size()) + " new_prefill=" +
         std::to_string(new_prompt_tokens.size()));

    std::vector<float> logits(cfg.n_vocab, 0.f);

    ensure_cache_room(cfg, st, new_prompt_tokens.size());
    VLOG("prefill len=" + std::to_string(new_prompt_tokens.size()));
    auto t_pre = std::chrono::steady_clock::now();
    fwd::transformer_forward(cfg, st.mw, new_prompt_tokens, st.cache, logits.data(), pool);
    VLOG_ELAPSED("prefill", t_pre);
    st.cached_tokens.insert(st.cached_tokens.end(), new_prompt_tokens.begin(), new_prompt_tokens.end());

    int32_t next = sample_top_p(logits.data(), (int)cfg.n_vocab, temperature, top_p, rng);

    // FIX (diagnostics): NANITY_DEBUG_LOGITS=1 dumps the top-10 (logit,
    // token) candidates for the FIRST generated token only -- the one
    // produced straight from the prefill forward pass, before any
    // per-step decode/cache-reuse machinery runs. This is the single
    // cheapest way to tell "the forward pass itself is producing bad
    // logits" apart from "something about incremental decode/caching goes
    // wrong a few tokens in" -- run with --prompt, temperature=0, top_p=1
    // for a fully deterministic single sample.
    if (std::getenv("NANITY_DEBUG_LOGITS")) {
        std::vector<int32_t> idx(cfg.n_vocab);
        std::iota(idx.begin(), idx.end(), 0);
        size_t topn = std::min((size_t)10, idx.size());
        std::partial_sort(idx.begin(), idx.begin() + topn, idx.end(),
            [&](int32_t a, int32_t b){ return logits[a] > logits[b]; });
        std::cerr << "[DEBUG top-" << topn << " first-token candidates]\n";
        for (size_t i = 0; i < topn; ++i) {
            int32_t id = idx[i];
            const std::string& s = (id >= 0 && (size_t)id < gguf.tok_meta.tokens.size())
                                    ? gguf.tok_meta.tokens[id] : std::string("<oob>");
            std::cerr << "  #" << i << "  id=" << id << "  logit=" << logits[id]
                      << "  vocab_str=" << s << "\n";
        }
        std::cerr << std::flush;
    }

    for (int step = 0; step < max_new_tokens; ++step) {
        if (next == tok.eos_id) break;
        on_token(next);

        ensure_cache_room(cfg, st, 1);

        VLOG("decode step=" + std::to_string(step));
        auto t_dec = std::chrono::steady_clock::now();
        std::vector<int32_t> one_token{ next };
        fwd::transformer_forward(cfg, st.mw, one_token, st.cache, logits.data(), pool);
        VLOG_ELAPSED("decode", t_dec);
        st.cached_tokens.push_back(next);

        next = sample_top_p(logits.data(), (int)cfg.n_vocab, temperature, top_p, rng);
    }
}


// =============================================================================
// Single-shot mode: streams each decoded piece to stdout immediately.
// =============================================================================
template <typename Loader>
static void generate(const engine::Config& cfg,
                     const Loader& gguf,
                     Tokenizer& tok,
                     const std::string& prompt,
                     int   max_new_tokens,
                     float temperature,
                     float top_p,
                     util::ThreadPool& pool,
                     std::mt19937& rng)
{
    VLOG("encode prompt len=" + std::to_string(prompt.size()));
    auto t_enc = std::chrono::steady_clock::now();
    std::vector<int32_t> ctx = tok.encode(prompt, /*add_bos=*/true);
    VLOG_ELAPSED("tokenize", t_enc);

    auto t_gen = std::chrono::steady_clock::now();
    run_generation(cfg, gguf, tok, ctx, max_new_tokens, temperature, top_p, pool, rng,
        [&](int32_t id) {
            std::cout << tok.decode_one(id) << std::flush;
        });
    std::cout << tok.flush() << '\n' << std::flush;
    VLOG_ELAPSED("generate total", t_gen);
}


// =============================================================================
// FIX (bug 7, C++ side): Persistent stdin/stdout interactive loop.
//
// Protocol (one JSON object per line each way):
//   IN  → {"messages":[…], "max_tokens":256, "temperature":0.7, "top_p":0.9}
//   OUT ← {"token":"…"}   (one per generated piece, then…)
//   OUT ← {"done":true}
//
// NEON.py spawns the binary once (--interactive) and reuses the process for
// every chat turn, eliminating per-turn model-load latency and giving the
// Python layer a place to accumulate conversation history.
// =============================================================================
template <typename Loader>
static void run_interactive(const engine::Config& cfg,
                             const Loader& gguf,
                             Tokenizer& tok,
                             util::ThreadPool& pool)
{
    std::mt19937 rng(std::random_device{}());
    std::string  line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        auto req = json::try_parse(line);
        if (req.is_null()) {
            std::cout << "{\"error\":\"bad JSON\"}\n" << std::flush;
            continue;
        }

        int   max_tokens  = req["max_tokens"].get_int(256);
        float temperature = req["temperature"].get_float(0.7f);
        float top_p       = req["top_p"].get_float(0.9f);

        std::string prompt;
        const auto& msgs = req["messages"];
        if (msgs.is_array() && msgs.size() > 0) {
            prompt = format_chat(msgs);
        } else {
            prompt = req["prompt"].get_str();
        }

        // Generate and emit each token as a JSON line.
        VLOG("interactive: encode prompt");
        auto t_enc = std::chrono::steady_clock::now();
        std::vector<int32_t> ctx = tok.encode(prompt, /*add_bos=*/true);
        VLOG_ELAPSED("interactive encode", t_enc);

        run_generation(cfg, gguf, tok, ctx, max_tokens, temperature, top_p, pool, rng,
            [&](int32_t id) {
                std::string piece = tok.decode_one(id);
                // Emit as JSON so the Python side can parse it without fragile
                // substring matching.
                std::cout << "{\"token\":\"" << json::escape(piece) << "\"}\n"
                          << std::flush;
            });
        // FIX: decode_one() withholds any trailing bytes that don't yet form
        // a complete UTF-8 codepoint (it can be split across the last two
        // tokens). Drain that buffer now, or the final character of a reply
        // can silently vanish whenever generation stops mid-codepoint.
        std::string tail = tok.flush();
        if (!tail.empty()) {
            std::cout << "{\"token\":\"" << json::escape(tail) << "\"}\n"
                      << std::flush;
        }
        std::cout << "{\"done\":true}\n" << std::flush;
    }
}


// =============================================================================
// Anti-repetition for long idle generation (no retraining).
//
// Why the idle loop collapses: every token the model emits is fed back as
// context, and a small LM's own recent text is the strongest signal it sees.
// Once a phrase repeats, attention to the previous copy makes the next copy
// more likely (self-reinforcing). Plain temperature/top-p can't break that.
// Three cheap, retrain-free countermeasures live here:
//   1. repetition/frequency penalty over a recent window (logit-level)
//   2. no-repeat n-gram ban (hard: never complete an n-gram already seen)
//   3. loop detector, used by the idle loop to inject a steering turn
// =============================================================================
struct AntiRepeat {
    // v2: v1's 1.15 penalty over ALL tokens (incl. "the", ",", ".") starved the
    // model of punctuation/function words and produced run-on word salad.
    // Penalties now touch only content-word tokens, and are much gentler.
    float  rep_penalty      = 1.08f;  // >1 discourages content words seen in window (1.0 = off)
    float  freq_penalty     = 0.0f;   // subtracted per prior occurrence in window (content words only)
    size_t window           = 128;    // how far back penalties look
    int    no_repeat_ngram  = 8;      // 0 = off; only blocks verbatim copy-loops at this length
    float  min_p            = 0.05f;  // drop tokens < min_p * p(top); cuts the weird-word tail
    size_t turn_tokens      = 200;    // soft cap: end the assistant turn at the next sentence end
    size_t max_runon        = 60;     // tokens without . ! ? or newline => degraded, rewind
};

// Tokens eligible for repetition penalty: plain alphabetic words, 4+ letters
// (after the byte-BPE space marker). Punctuation, digits, newlines, short
// function words and special tokens are never penalized.
static std::vector<char> build_penalty_mask(const std::vector<std::string>& toks, size_t n_vocab)
{
    std::vector<char> m(n_vocab, 0);
    for (size_t i = 0; i < toks.size() && i < n_vocab; ++i) {
        const std::string& t = toks[i];
        size_t b = 0;
        if (t.size() >= 2 && (unsigned char)t[0] == 0xC4 && (unsigned char)t[1] == 0xA0) b = 2;  // 'Ġ'
        size_t len = t.size() - b;
        if (len < 4) continue;
        { const std::string w = t.substr(b); if (w == "think" || w == "answer") continue; }  // tag words must stay cheap to emit
        bool ok = true;
        for (size_t j = b; j < t.size(); ++j)
            if (!std::isalpha((unsigned char)t[j])) { ok = false; break; }
        m[i] = ok ? 1 : 0;
    }
    return m;
}

static void apply_anti_repeat(std::vector<float>& logits,
                              const std::vector<int32_t>& hist,
                              size_t pinned, const AntiRepeat& ar,
                              const std::vector<char>& mask)
{
    const size_t V = logits.size();
    const size_t end = hist.size();

    if (end > pinned && (ar.rep_penalty != 1.0f || ar.freq_penalty != 0.0f)) {
        const size_t start = std::max(pinned, end > ar.window ? end - ar.window : (size_t)0);
        std::unordered_map<int32_t, int> counts;
        for (size_t i = start; i < end; ++i) ++counts[hist[i]];
        for (const auto& kv : counts) {
            if (kv.first < 0 || (size_t)kv.first >= V || !mask[kv.first]) continue;
            float& l = logits[kv.first];
            if (ar.rep_penalty != 1.0f) l = (l > 0.f) ? l / ar.rep_penalty : l * ar.rep_penalty;
            l -= ar.freq_penalty * (float)kv.second;
        }
    }

    const int n = ar.no_repeat_ngram;
    if (n >= 2 && end > pinned && (end - pinned) >= (size_t)n) {
        const size_t m = (size_t)n - 1;
        const size_t scan_from = std::max(pinned, end > 2048 ? end - 2048 : (size_t)0);
        for (size_t i = scan_from; i + m < end; ++i) {
            bool match = true;
            for (size_t j = 0; j < m; ++j)
                if (hist[i + j] != hist[end - m + j]) { match = false; break; }
            if (match) {
                int32_t banned = hist[i + m];
                if (banned >= 0 && (size_t)banned < V) logits[banned] = -1e30f;
            }
        }
    }

    if (ar.min_p > 0.f) {
        float mx = *std::max_element(logits.begin(), logits.end());
        float thr = mx + std::log(ar.min_p);
        for (auto& l : logits) if (l < thr) l = -1e30f;
    }
}

// True if the recent tail looks like a loop: very low token diversity, or the
// last 16 tokens already occurred earlier in the last 512.
static bool looks_like_loop(const std::vector<int32_t>& hist, size_t pinned)
{
    if (hist.size() < pinned + 96) return false;
    const size_t end = hist.size();
    {
        const size_t w = 96;
        std::unordered_set<int32_t> uniq(hist.end() - w, hist.end());
        if ((double)uniq.size() / (double)w < 0.40) return true;
    }
    const size_t k = 16;
    const size_t scan_from = std::max(pinned, end > 512 ? end - 512 : (size_t)0);
    for (size_t i = scan_from; i + k < end - k; ++i) {
        bool match = true;
        for (size_t j = 0; j < k; ++j)
            if (hist[i + j] != hist[end - k + j]) { match = false; break; }
        if (match) return true;
    }
    return false;
}

// Steering prompts are anchored to the topic and ask for ONE SHORT paragraph.
// Short, well-formed turns are what a small instruct model is good at, and
// they keep every turn inside its trained length distribution (the model
// degrades when a single assistant turn runs for 600+ tokens).
static std::string make_steer(size_t idx, const std::string& topic)
{
    static const char* aspects[] = {
        "its history and origins",
        "a specific real-world example",
        "its main criticisms and limitations",
        "how it works in practice day to day",
        "the biggest challenges it faces today",
        "a common misconception about it",
        "how it compares with a similar alternative",
        "what might change about it in the future",
    };
    const std::string subj = topic.empty() ? "the topic" : topic;
    return "Continue explaining " + subj + ", now focusing on " +
           aspects[idx % (sizeof(aspects) / sizeof(aspects[0]))] +
           ". Write one short paragraph of 3 to 5 complete sentences and do not repeat earlier points.";
}

// =============================================================================
// Standby CoT cycle (--idle-cot)
//
//   <think> ...private reasoning... </think> <answer> ...result... </answer>
//   -> runtime injects the next <think> with a NEW stimulus -> repeat forever
//
// The runtime owns the structure; the model only fills in the text:
//   * the format is taught in-context (pinned system prompt + one worked example)
//   * the closing/opening tags are INJECTED by the runtime, not sampled, so the
//     structure cannot be lost even if the 1.7B model forgets it
//   * a fresh stimulus (lore line / vision delta / built-in curiosity prompt)
//     opens every think block, which is what moves it off the old topic
//   * when nothing is queued it sometimes spends the cycle on a question for
//     the user; typed lines on stdin interrupt at any time and get a real reply
// =============================================================================
struct CotConfig {
    bool   enabled              = false;
    bool   default_seed         = false;   // build the built-in CoT system prompt as the pinned seed
    bool   hide_think           = false;   // keep <think>/<next> text off stdout (still in context + disk log)
    bool   user_input           = true;    // read stdin lines as user events
    size_t think_tokens         = 100;     // soft cap for the whole think block (2 short fields)
    size_t answer_tokens        = 120;     // soft cap per idle answer (x2 when answering the user)
    size_t field_tokens         = 40;      // soft cap per think field
    size_t next_tokens          = 30;      // soft cap for the model's own "what next" line
    size_t min_think_tokens     = 4;       // tokens each think field must contain before it may end
    size_t min_answer_tokens    = 12;
    long   ask_timeout_s        = 120;     // stop waiting for a user reply after this long
    int    lore_percent         = 15;      // chance a cycle opens with a lore line instead of the model's own next step
    std::string task;                      // a prompt to FOLLOW first
    std::string system_text;               // replaces the built-in system prompt (loaded from --idle-system-file)
    std::string input_file;                // second input route: every line appended to this file is a user message
    std::vector<std::string> stimuli;      // [0, n_lore) = lore lines, then built-in fallbacks
    size_t n_lore               = 0;
    float  speak_bias           = -1.0f;   // logit added to "Reply: yes" on self-directed cycles (negative = lean toward staying silent)
    float  user_speak_bias      = 0.5f;    // same, when the cycle answers a user message (never forced either way)
    bool   input_line           = false;   // pinned "> " input line on the bottom terminal row; model output scrolls above it
    bool   no_echo              = false;   // turn the terminal's input echo off (typed text no longer lands inside the output)
    size_t ask_gap              = 10;      // minimum cycles between questions to the user (no "want more?" after every answer)
    int    ask_after_user_percent = 25;    // chance that finishing a reply to the user leads it to consider checking in at all
    size_t speak_gap            = 5;       // cycles it stays quiet after speaking: a person says a thing once, then lets it land
    size_t quiet_cycles         = 8;       // after the model stays silent on a user message, no new questions for this many cycles
};

static const char* kCotBuild = "cot-build 11 (pacing + visible input line + rare check-ins)";   // printed at startup so a stale binary is obvious

// Events from outside the generation loop (typed lines, vision deltas, future tool results).
static mstream::SpliceQueue g_splice;

static std::vector<std::string> default_stimuli()
{
    return {
        "Explain how an everyday object works, such as a refrigerator or a bicycle.",
        "Think of an interesting fact about the ocean and explain why it matters.",
        "Describe a simple idea from mathematics in plain words.",
        "Pick a historical invention and explain how it changed daily life.",
        "Imagine a small problem a person might have today and suggest a practical fix.",
        "Explain how plants turn sunlight into food.",
        "Think about how a computer stores and retrieves a file.",
        "Describe how weather forecasts are made.",
        "Pick an animal and explain one surprising thing about how it lives.",
        "Explain the difference between weather and climate.",
        "Think about why bridges stay up and what can make them fail.",
        "Describe how a language changes over hundreds of years.",
        "Explain how vaccines train the immune system.",
        "Think about what makes a good habit stick.",
        "Describe how electricity gets from a power plant to a home.",
        "Explain how sound travels and why echoes happen.",
        "Pick a city and describe what makes its layout unusual.",
        "Think about how maps are made and why every flat map distorts something.",
        "Explain how a simple machine like a lever gives you an advantage.",
        "Describe how the moon affects the tides.",
        "Think about how to learn a new skill efficiently.",
        "Explain how money moves when you pay by card.",
        "Describe what happens inside a volcano before it erupts.",
        "Think about why stories are easier to remember than lists of facts.",
    };
}

// Pinned seed that teaches the format in-context. `<|im_start|>` is omitted
// when the tokenizer's BOS id already IS <|im_start|> (add_bos=true prepends
// it), so the sequence never starts with two.
// The built-in system prompt (the standby format). Replaceable with --idle-system-file.
static std::string default_cot_system_text()
{
    return std::string(
         "You are NANITY, a small autonomous assistant. When no user is typing you are in standby: you think on your own and decide for yourself what to do next.\n"
         "Every reply has this format:\n"
         "<think>\nTask: one line saying what is being asked or noticed.\nPlan: one line saying what to do.\nReply: yes or no\n</think>\n"
         "<answer>\nThe answer, in first person. Only write this part when Reply is yes.\n</answer>\n"
         "<next>\nOne line saying what you will do next.\n</next>\n"
         "Rules: keep the think block to those three short lines and never repeat them in the answer. "
         "Reply yes only when the user needs or asked to hear something now. "
         "Reply no when there is nothing to say, when the user is finished (for example \"thanks, that's all\"), or when a note is only something to think about quietly; "
         "then skip the answer and go straight to <next>. "
         "Keep answers short, 3 to 5 sentences. "
         "Explain a thing once, like a person would: never repeat yourself or go back over a topic you already covered, unless you realize you left out something important. "
         "After you ask the user something, wait quietly for their reply (Reply: no) and keep thinking. "
         "Messages starting with [Self-note] are your own private plans, not questions from the user: carry them out in first person and never praise or comment on them. "
         "If a self-note says to ask the user something, reply yes and make the answer that question, addressed directly to the user. "
         "Never start an answer with praise such as \"Great question\". "
         "What you do next can be anything that fits: look deeper at a detail, double-check something you said, move to a related topic, "
         "ask the user something, check in with the user, or wait for their reply.\n"
         "Example 1:\n"
         "<think>\nTask: Explain how rainbows form.\nPlan: Mention sunlight entering raindrops, bending, reflecting inside, and spreading into colors.\nReply: yes\n</think>\n"
         "<answer>\nA rainbow forms when sunlight enters raindrops, reflects off the inside, and exits, with each color bending at a slightly "
         "different angle. That spreads white light into a band of colors.\n</answer>\n"
         "<next>\nI could explain why rainbows are curved.\n</next>\n"
         "Example 2:\n"
         "<think>\nTask: The user said thanks and is finished.\nPlan: Nothing to add, so stay quiet.\nReply: no\n</think>\n"
         "<next>\nI could think about something I am curious about on my own.\n</next>\n");
}

static std::string build_cot_seed(const Tokenizer& tok, const std::string& task, const std::string& system_override)
{
    auto it = tok.vocab_map.find("<|im_start|>");
    const bool bos_is_im_start = (it != tok.vocab_map.end() && it->second == tok.bos_id);
    return cotx::compose_seed(bos_is_im_start,
                              system_override.empty() ? default_cot_system_text() : system_override,
                              task.empty() ? std::string("Standby mode. Pick something interesting to explain.") : task);
}

// =============================================================================
// Idle loop (NECTAR idle-loop architecture v3, §8 build-order step 1):
// continuous streaming generation with StreamingLLM-style sink/window
// eviction, so idle CoT can run indefinitely on a fixed-size context
// instead of halting once the cache fills. Deliberately excludes the
// pinned implant proper, vision deltas, and eviction-to-disk (§§3-5 of the
// doc) — those are build-order steps 2-4. What this step DOES let you run
// is the open comparison §1 calls out before committing to a sink design:
//
//   --idle-seed "<text>" --idle-sink-tokens 0   → seed-block-as-sink-only
//   --idle-seed "<text>" --idle-sink-tokens N   → seed block + N dedicated
//                                                   neutral sink tokens
//
// `keep` = seed_tokens.size() + sink_tokens; both are pinned via the same
// ensure_cache_room()/kv_cache_shift() machinery the answer path already
// uses for its 4-token default, just with a caller-supplied keep count.
// Watch cache.length in the periodic [Idle] log across a long run: if
// generation degrades (repetition, incoherence) after a shift with
// sink_tokens=0 but stays stable with sink_tokens>0, that's the doc's
// signal to keep dedicated sink tokens; if seed-only holds up, they're
// redundant and step 2 (the real implant) can skip them.
//
// Own EngineState: get_engine_state()'s function-local static means this
// must run as its own process (`--idle-loop`), never in the same process
// as `--interactive`/single-shot — that's what keeps this loop's KV cache
// and cached_tokens from ever touching the answer path's, per §2's "don't
// let them share state." The separation is "two processes," not "two
// objects in one process."
//
// No natural stop: idle CoT isn't supposed to terminate, so EOS is treated
// as a soft paragraph break (emit a newline, keep generating) rather than
// end-of-reply. --idle-max-tokens caps a run for benchmarking; 0 runs until
// killed.
// =============================================================================
// Which zRAM compression backend (nectar_zram.hpp) the idle loop uses when
// a chunk ages out of the live window. Declared unconditionally (not
// inside the NANITY_ENABLE_IDLE_LOOP guard below) because IdleLoopArgs
// references it regardless of build config — CLI parsing always needs
// somewhere to put --idle-compression's value, even in a build that can't
// act on it. SoftSlots is intentionally not wired into run_idle_loop()
// yet — see kv_cache_compact()'s doc comment in rawllm_forward.hpp for
// exactly what it needs (a forward pass over reserved slot tokens BEFORE
// the discard, and kv_cache_compact() instead of kv_cache_shift() at that
// eviction) and why it needs a trained compression objective to produce
// anything but noise. Selecting it is a hard error at run_idle_loop()
// entry until that wiring exists, rather than silently falling back to
// None.
enum class IdleCompressionKind { None, TextSummary, SoftSlots };

#if defined(NANITY_ENABLE_IDLE_LOOP)

// Standalone forward pass used ONLY by the TextSummary zRAM compressor to
// summarize a chunk of idle-loop text. Deliberately does NOT touch
// get_engine_state()'s singleton EngineState/KVCache — that singleton IS
// the idle loop's own live context; running a second generation through
// it here (even with a save/restore) would mean this function's prompt
// tokens and the idle loop's real content fighting over the exact same
// cache.length/cached_tokens bookkeeping the idle loop is mid-decode on.
// Builds a small throwaway KVCache instead, sized only for this call, so
// it structurally cannot collide with or be observed by the idle loop's
// state. Uses its OWN Tokenizer instance (fresh from gguf.tok_meta) for
// the same reason: decode_one()/flush() carry partial-UTF8 state that the
// idle loop's own `tok` is mid-way through — sharing it here would
// corrupt whichever one decodes next.
template <typename Loader>
static std::string summarize_chunk_standalone(const engine::Config& cfg,
                                                const fwd::ModelWeights& mw,
                                                const Loader& gguf,
                                                util::ThreadPool& pool,
                                                const std::string& chunk_text,
                                                size_t max_summary_tokens,
                                                std::mt19937& rng)
{
    Tokenizer local_tok(gguf.tok_meta);
    std::string prompt = "Summarize the following in one short sentence:\n" + chunk_text + "\nSummary:";
    std::vector<int32_t> ctx = local_tok.encode(prompt, /*add_bos=*/true);
    if (ctx.empty()) return {};

    size_t cap = ctx.size() + max_summary_tokens + 8;   // exactly this call's needs, nothing shared
    fwd::KVCache local_cache(cfg.n_layer, (size_t)cfg.n_kv_head, (size_t)cfg.head_dim, cap);

    std::vector<float> logits(cfg.n_vocab, 0.f);
    fwd::transformer_forward(cfg, mw, ctx, local_cache, logits.data(), pool);
    int32_t next = sample_top_p(logits.data(), (int)cfg.n_vocab, /*temperature=*/0.3f, /*top_p=*/0.9f, rng);

    std::string out;
    for (size_t i = 0; i < max_summary_tokens && next != local_tok.eos_id; ++i) {
        out += local_tok.decode_one(next);
        std::vector<int32_t> one{ next };
        fwd::transformer_forward(cfg, mw, one, local_cache, logits.data(), pool);
        next = sample_top_p(logits.data(), (int)cfg.n_vocab, 0.3f, 0.9f, rng);
    }
    out += local_tok.flush();
    return out;
}

template <typename Loader>
static void run_idle_loop(const engine::Config& cfg,
                           const Loader& gguf,
                           Tokenizer& tok,
                           util::ThreadPool& pool,
                           const std::string& seed_text,
                           size_t sink_tokens,
                           long   max_tokens,   // 0 = infinite
                           size_t log_every,
                           float  temperature,
                           float  top_p,
                           const std::string& disk_path,
                           size_t disk_flush_every_tokens,
                           long   disk_flush_interval_s,
                           bool   vision_enabled,
                           long   vision_poll_ms,
                           // ── zRAM tier (nectar_zram.hpp) ────────────────
                           IdleCompressionKind compression_kind,
                           size_t chunk_tokens,           // 0 disables zRAM entirely — behavior identical to pre-zRAM
                           size_t zram_ring_capacity,
                           const std::string& zram_index_path,
                           size_t summary_max_tokens,
                           double zram_max_keep_fraction, // cap on keep/kv_window before a summary stops being pinned
                           const AntiRepeat& anti_repeat,
                           bool   steer_enabled,
                           const std::string& steer_topic,
                           const CotConfig& cot)
{
    auto& st = get_engine_state(cfg, gguf);
    std::mt19937 rng(std::random_device{}());

    if (compression_kind == IdleCompressionKind::SoftSlots) {
        throw std::runtime_error(
            "run_idle_loop(): --idle-compression=slots is not wired yet — "
            "kv_cache_compact() (rawllm_forward.hpp) exists and is ready, but "
            "this loop still needs the forward-pass-before-discard sequencing "
            "and a chunk-aligned discard trigger to call it correctly. See "
            "kv_cache_compact()'s doc comment for exactly what's missing. Use "
            "--idle-compression=text or omit the flag for now.");
    }

    zram::ChunkRing zram_ring(zram_ring_capacity, zram_index_path);
    std::deque<std::string> pending_chunk;
    size_t chunk_range_start = 0;   // set once pinned region is known, below
    size_t next_chunk_id     = 0;

    // ---- Build the pinned region: implant block (§3/§4 — seed_text is now
    // the actual compressed personality distillation, not a stand-in;
    // §8 step 2) followed by optional dedicated neutral sink tokens (§1).
    std::string first_stim;                                   // used as the first user turn when no --idle-task is given
    if (cot.enabled && cot.task.empty() && !cot.stimuli.empty())
        first_stim = cot.stimuli[(size_t)std::chrono::steady_clock::now().time_since_epoch().count() % cot.stimuli.size()];
    const std::string seed_task = cot.task.empty() ? (first_stim.empty() ? std::string() : "[Standby] " + first_stim) : cot.task;
    const std::string seed_eff = (cot.enabled && cot.default_seed) ? build_cot_seed(tok, seed_task, cot.system_text) : seed_text;
    std::vector<int32_t> pinned = tok.encode(seed_eff, /*add_bos=*/true);
    const size_t seed_len = pinned.size();

    if (sink_tokens > 0) {
        // StreamingLLM's finding is that early *position*, not specific
        // content, is what stabilizes attention — so any bland, repeatable
        // text is fine here. Encode once, then repeat/truncate to exactly
        // sink_tokens ids.
        static const std::string kNeutralFiller =
            "The quick brown fox jumps over the lazy dog.";
        std::vector<int32_t> filler = tok.encode(kNeutralFiller, /*add_bos=*/false);
        if (filler.empty())
            throw std::runtime_error("run_idle_loop(): neutral filler encoded to zero tokens");
        pinned.reserve(pinned.size() + sink_tokens);
        for (size_t i = 0; i < sink_tokens; ++i)
            pinned.push_back(filler[i % filler.size()]);
    }

    // Mutable, not const: a TextSummary compression cycle grows this by
    // permanently pinning newly-produced summary tokens (see the splice
    // block inside the main loop below), the same way the original
    // implant+sink region is pinned here at startup. zram_max_keep_fraction
    // caps how far that growth is allowed to go before a summary stops
    // being pinned (still recorded in the ring/index either way — only
    // whether it re-enters the live cache is capped).
    size_t keep = pinned.size();
    chunk_range_start = keep;
    if (keep >= cfg.kv_window) {
        throw std::runtime_error(
            "run_idle_loop(): pinned region (" + std::to_string(keep) +
            " tokens: " + std::to_string(seed_len) + " seed + " +
            std::to_string(sink_tokens) + " sink) already fills or exceeds "
            "kv_window=" + std::to_string(cfg.kv_window) + " -- shorten "
            "--idle-seed, lower --idle-sink-tokens, or raise kv_window.");
    }

    // §4 "Implant Sizing Constraint": the implant is a permanent tax on
    // every cycle's budget, forever, so it should stay a small minority of
    // kv_window, not compete with rolling vision/CoT content for room.
    // 20% is a soft line, not from the doc itself -- just a sanity check
    // worth having before a long unattended run rather than after.
    if (seed_len * 5 > cfg.kv_window) {
        std::cerr << "[Idle] WARNING: implant is " << seed_len << " tokens, "
                  << (100 * seed_len / cfg.kv_window) << "% of kv_window="
                  << cfg.kv_window << " -- doc §4 wants this a small minority "
                  << "so it doesn't crowd out rolling content. Consider "
                  << "compressing the implant text.\n";
    }

    std::cerr << "[Idle] pinned region: " << keep << " tokens ("
              << seed_len << " implant + " << sink_tokens << " dedicated sink), "
              << "kv_window=" << cfg.kv_window << "\n";

    // ---- Disk write path (§5.1): batched, source-tagged. `aging` mirrors
    // exactly the sliding region's contents in FIFO order — one entry per
    // generated token, pushed the instant it's generated and popped the
    // instant ensure_cache_room() reports it as discarded. That one-to-one
    // correspondence (rather than e.g. "just buffer whatever was recently
    // generated") is what makes this a real eviction-triggered write, not
    // an approximation of one.
    diskmem::WriteBuffer idle_disk(disk_path, diskmem::Source::Idle,
                                    disk_flush_every_tokens,
                                    std::chrono::seconds(disk_flush_interval_s));
    std::deque<std::string> aging;

    // ---- Mid-stream injection (§7, §8 step 5): vision deltas spliced into
    // the live context between tokens. Off by default — vision_enabled
    // requires a real Hyprland+grim(+tesseract) desktop to produce
    // anything; on any other machine (including this build/test sandbox)
    // vis.poll() just returns nullopt forever, so leaving it on is safe,
    // but it's opt-in so a headless/CI run doesn't pay the per-interval
    // hyprctl/grim shellout cost for nothing.
    //
    // Deltas are grounding context for the CoT, not spoken output: they go
    // into the KV cache the same way generated tokens do, but are never
    // written to stdout (no TTS for "user switched to kitty").
    vision::VisionModule vis;
    mstream::IntervalGate vision_gate{std::chrono::milliseconds(vision_poll_ms)};
    // §9 "Vision token budget" risk: short deltas only. A caption longer
    // than this is truncated at the character level before encoding —
    // crude, but keeps a single injection from ever competing seriously
    // with rolling CoT content for kv_window room.
    constexpr size_t kMaxAnnotationChars = 300;

    std::vector<float> logits(cfg.n_vocab, 0.f);

    ensure_cache_room(cfg, st, pinned.size(), keep);
    fwd::transformer_forward(cfg, st.mw, pinned, st.cache, logits.data(), pool);
    st.cached_tokens = pinned;

    int32_t next = sample_top_p(logits.data(), (int)cfg.n_vocab, temperature, top_p, rng);

    // Every discarded piece still goes to idle_disk (the "Disk" tier,
    // unchanged) — this lambda additionally accumulates it toward a zRAM
    // chunk when chunk_tokens > 0, so enabling zRAM is strictly additive,
    // never a substitute for the existing raw disk log.
    auto evict_piece = [&](const std::string& piece) {
        idle_disk.add(piece);
        if (chunk_tokens == 0) return;   // zRAM disabled — behavior identical to before this feature existed

        pending_chunk.push_back(piece);
        if (pending_chunk.size() < chunk_tokens) return;

        std::string chunk_text;
        for (const auto& s : pending_chunk) chunk_text += s;
        size_t range_end = chunk_range_start + pending_chunk.size();

        zram::CompressResult result;
        if (compression_kind == IdleCompressionKind::TextSummary) {
            result.splice_text = summarize_chunk_standalone(cfg, st.mw, gguf, pool,
                                                              chunk_text, summary_max_tokens, rng);
        }
        // SoftSlots is rejected before this loop ever starts (see the
        // guard at the top of run_idle_loop), so no branch for it here.

        zram::ChunkPointer ptr;
        ptr.chunk_id          = next_chunk_id++;
        ptr.token_range_start = chunk_range_start;
        ptr.token_range_end   = range_end;
        ptr.summary           = result.splice_text;
        ptr.num_soft_slots    = result.num_soft_slots;   // always 0 on the TextSummary path
        ptr.raw_text          = chunk_text;
        zram_ring.push(ptr);

        // Fold the summary into the pinned region permanently — same
        // ensure_cache_room()/forward() primitives the startup implant and
        // the vision splice below already use, just growing `keep`
        // afterward so this survives every future shift too, not only the
        // next one. zram_max_keep_fraction stops that growth from
        // eventually crowding out the whole rolling window; a summary that
        // exceeds it is still recorded in the ring/index, just not pinned.
        if (!result.splice_text.empty()) {
            std::vector<int32_t> summary_tokens = tok.encode(" " + result.splice_text, /*add_bos=*/false);
            double prospective_fraction = (double)(keep + summary_tokens.size()) / (double)cfg.kv_window;
            if (!summary_tokens.empty() && prospective_fraction <= zram_max_keep_fraction) {
                ensure_cache_room(cfg, st, summary_tokens.size(), keep);
                fwd::transformer_forward(cfg, st.mw, summary_tokens, st.cache, logits.data(), pool);
                st.cached_tokens.insert(st.cached_tokens.end(), summary_tokens.begin(), summary_tokens.end());
                keep += summary_tokens.size();
                std::cerr << "[Idle][zRAM] chunk " << ptr.chunk_id << " (" << pending_chunk.size()
                          << " tok) summarized + pinned (" << summary_tokens.size()
                          << " tok); keep=" << keep << "/" << cfg.kv_window << "\n";
            } else {
                std::cerr << "[Idle][zRAM] chunk " << ptr.chunk_id << " summarized but NOT pinned "
                          << "(would push keep to " << (100.0 * prospective_fraction) << "% of kv_window, "
                          << "over the " << (100.0 * zram_max_keep_fraction) << "% cap) — still recorded in ring/index.\n";
            }
        }

        pending_chunk.clear();
        chunk_range_start = range_end;
    };

    const std::vector<char> penal_mask = build_penalty_mask(tok.meta.tokens, cfg.n_vocab);
    const bool managed = steer_enabled || cot.enabled;   // output is held until a sentence completes

    long   produced = 0;
    size_t steer_idx = 0;
    std::string pending_out;
    size_t appended_since_cp = 0;      // cache positions appended since the last-sentence-end checkpoint
    bool   cp_valid          = true;
    size_t since_sent_end    = 0;
    size_t turn_tokens       = 0;

    // ---- CoT state. One cycle = one ChatML exchange:
    //   user turn  : a user message, an [Event], or a [Self-note] (the model's own previous <next> line)
    //   assistant  : <think>Task/Plan</think> <answer>..</answer> <next>the model's own next step</next>
    enum class Phase { THINK, ANSWER, NEXT };
    enum class Kind  { NORMAL, ASK, USER };
    Phase  phase = Phase::THINK;
    Kind   kind  = Kind::NORMAL;
    size_t phase_tokens = 0, think_field = 0, field_tokens = 0;
    long   cycle_no = 0;
    const std::vector<std::string> think_labels = { "Task:", "Plan:" };
    std::string tail, answer_text, next_text, next_opener, last_note;
    std::deque<std::string> recent_notes;
    std::deque<size_t>      recent_stim;
    // Conversation state: after the model asks the user something it keeps thinking
    // (cycles go on) until a reply is spliced in or the wait times out.
    bool        awaiting = false;
    std::string pending_question;
    auto        asked_at = std::chrono::steady_clock::now();

    const cotx::TagVocab tv = cotx::build_tag_vocab(tok.meta.tokens, (size_t)cfg.n_vocab);
    std::vector<int32_t> praise_ids;                     // opening-praise tokens ("Great", "Thanks", ...)
    const cotx::ChoiceVocab cv = cotx::build_choice_vocab(tok.meta.tokens, (size_t)cfg.n_vocab);
    bool   deciding = false;      // the think block is at its "Reply:" field: the model picks yes (speak) or no (stay in thought)
    bool   spoke = false;         // this cycle produced an <answer>
    bool   event_cycle = false;   // this cycle was opened by an outside event (vision / tool)
    size_t ask_cooldown = 0;      // cycles during which new questions are suspended
    size_t cycles_since_ask = 1000;   // cycles since the last question to the user
    bool   ask_suppressed = false;    // the model wanted to check in, but it is too soon: keep that as a private thought
    size_t speak_cooldown = 0;    // cycles it stays quiet after speaking
    bool   override_repeat = false;   // a self-declared correction may revisit a topic it already covered
    bool   force_quiet = false;       // a pacing rule decided this cycle's Reply
    std::string quiet_why, task_text;
    std::vector<std::string> covered; // Task lines of what has already been said
    for (const char* w : {"Great", "Good", "Excellent", "Thank", "Thanks", "Sure", "Certainly",
                          "Absolutely", "Indeed", "Wow", "Interesting", "Your", "You"})
        for (const std::string& key : std::vector<std::string>{ std::string(w), std::string("\xC4\xA0") + w }) {
            auto it = tok.vocab_map.find(key);
            if (it != tok.vocab_map.end()) praise_ids.push_back(it->second);
        }

    auto emit = [&](const std::string& t) {
        if (cot.enabled && cot.hide_think && (phase == Phase::THINK || phase == Phase::NEXT)) return;
        if (managed) pending_out += t; else std::cout << t << std::flush;
    };
    auto flush_out = [&]() {
        if (!pending_out.empty()) { std::cout << pending_out << std::flush; pending_out.clear(); }
    };
    // Splice text into the live context (cache + token history + aging buffer).
    auto inject = [&](const std::string& text) {
        std::vector<int32_t> toks = tok.encode(text, /*add_bos=*/false);
        if (toks.empty()) return;
        mstream::push_injected_text(aging, toks.size(), text);
        size_t d = ensure_cache_room(cfg, st, toks.size(), keep);
        for (size_t i = 0; i < d && !aging.empty(); ++i) { evict_piece(aging.front()); aging.pop_front(); }
        fwd::transformer_forward(cfg, st.mw, toks, st.cache, logits.data(), pool);
        st.cached_tokens.insert(st.cached_tokens.end(), toks.begin(), toks.end());
        appended_since_cp = 0; cp_valid = true; since_sent_end = 0; tail.clear();   // injection point is a consistent checkpoint
    };
    // Truncate the KV cache back to the last sentence end: the degraded tokens
    // never happened as far as the model is concerned.
    auto try_rewind = [&]() -> bool {
        if (!(cp_valid && appended_since_cp > 0 &&
              appended_since_cp <= st.cached_tokens.size() &&
              appended_since_cp <= st.cache.length &&
              (st.cache.length - appended_since_cp) > keep)) return false;
        const size_t n = appended_since_cp;
        st.cache.length -= n;
        st.cached_tokens.resize(st.cached_tokens.size() - n);
        for (size_t i = 0; i < n && !aging.empty(); ++i) aging.pop_back();
        pending_out.clear();
        (void)tok.flush();
        appended_since_cp = 0; since_sent_end = 0; tail.clear();
        return true;
    };
    auto pick_from = [&](size_t lo, size_t hi) -> std::string {   // random stimulus in [lo,hi), avoiding recent picks
        if (hi <= lo || hi > cot.stimuli.size()) return "Pick any interesting topic and think about it.";
        size_t idx = lo;
        for (int tries = 0; tries < 8; ++tries) {
            idx = lo + (size_t)(rng() % (hi - lo));
            if (std::find(recent_stim.begin(), recent_stim.end(), idx) == recent_stim.end()) break;
        }
        recent_stim.push_back(idx);
        if (recent_stim.size() > std::min<size_t>(8, (hi - lo) / 2)) recent_stim.pop_front();
        return cot.stimuli[idx];
    };
    auto shorten = [](std::string t, size_t n) { if (t.size() > n) { t.resize(n); t += "..."; } return t; };

    // Open the next exchange. What it is about comes from, in order:
    //   1. a queued outside event (vision / tool)            -> [Event]
    //   2. occasionally a lore line                          -> [Self-note]
    //   3. the model's OWN <next> line from the last cycle   -> [Self-note]   (asks the user if the model said so)
    //   4. a built-in stimulus, only if the model's line was empty / too short / a near-copy
    auto begin_next_cycle = [&](bool first) {
        std::string pre, utext;
        event_cycle = false;
        if (first) {
            kind = cot.task.empty() ? Kind::NORMAL : Kind::USER;
            if (!cot.default_seed && !cot.task.empty())
                pre = "<|im_end|>\n<|im_start|>user\n" + cot.task + "<|im_end|>\n<|im_start|>assistant\n";
            if (!cot.task.empty()) pending_out += "[TASK] " + cot.task + "\n";
        } else {
            std::string label = "[Self-note] ", text;
            kind = Kind::NORMAL;
            if (ask_cooldown > 0) --ask_cooldown;
            if (cycles_since_ask < 100000) ++cycles_since_ask;
            ask_suppressed = false;
            mstream::SpliceEvent ev;
            const bool from_event = g_splice.try_pop_other(ev);
            event_cycle = from_event;
            // Questions are only allowed when not already waiting for a reply and not in the quiet period
            // that follows the model choosing silence on a user message (e.g. "I am satisfied").
            const cotx::NextKind nk = cotx::classify_next(last_note);
            const bool note_asks = nk == cotx::NextKind::Ask;
            const bool note_corrects = nk == cotx::NextKind::Correct;
            const bool may_ask = !awaiting && ask_cooldown == 0 && cycles_since_ask >= cot.ask_gap;
            if (speak_cooldown > 0) --speak_cooldown;
            override_repeat = false;
            if (from_event) {
                label = "[Event] ";
                text = ev.text;
            } else if (cot.n_lore > 0 && (int)(rng() % 100) < cot.lore_percent) {
                text = "Something from my background: " + pick_from(0, cot.n_lore);
            } else if (cotx::note_is_usable(last_note, recent_notes) &&
                       (note_corrects || !cotx::repeats_topic(last_note, covered))) {   // no going back over covered ground...
                kind = (note_asks && may_ask) ? Kind::ASK : Kind::NORMAL;
                ask_suppressed = note_asks && !may_ask;       // too soon to check in again: it stays a private thought
                if (kind == Kind::ASK) cycles_since_ask = 0;
                override_repeat = note_corrects;                                       // ...unless it says it left something out
                text = last_note;
                recent_notes.push_back(last_note);
                if (recent_notes.size() > 4) recent_notes.pop_front();
            } else {
                text = "Something to think about: " + pick_from(0, cot.stimuli.size());
            }
            if (awaiting && !from_event) {
                const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - asked_at).count();
                if (waited > (double)cot.ask_timeout_s) {
                    text = "The user has not replied to \"" + shorten(pending_question, 100) + "\". " + text;
                    awaiting = false;
                } else {
                    text = "I am still waiting for the user to answer \"" + shorten(pending_question, 100) + "\". Meanwhile: " + text;
                }
            }
            utext = label + text;
            pre = "<|im_end|>\n<|im_start|>user\n" + utext + "<|im_end|>\n<|im_start|>assistant\n";
        }
        const std::string first_label = (kind == Kind::ASK) ? std::string("Task: Ask the user one short question about") : think_labels[0];
        inject(pre + "<think>\n" + first_label);
        phase = Phase::THINK; phase_tokens = 0; think_field = 0; field_tokens = 0; ++cycle_no;
        deciding = false; spoke = false; force_quiet = false;
        answer_text.clear(); next_text.clear(); task_text.clear();
        emit("\n<think>\n" + first_label);
        std::cerr << "[Idle][cot] cycle " << cycle_no
                  << (kind == Kind::ASK ? " -> asking the user" : (kind == Kind::USER ? " -> following the prompt" : ""))
                  << (awaiting ? " (waiting for user reply)" : "")
                  << (utext.empty() ? std::string() : ": " + utext.substr(0, 110)) << "\n";
    };

    bool input_line_on = false;
    if (cot.enabled && cot.user_input) {
        if (cot.input_line) input_line_on = mstream::start_input_line(g_splice);          // needs stdin AND stdout on a terminal
        if (!input_line_on) mstream::start_stdin_pump(g_splice);
        std::cerr << (input_line_on
            ? "[Idle][cot] input line active on the bottom row of the terminal (Ctrl-C quits and restores the terminal)\n"
            : "[Idle][cot] type a line + Enter to talk to it; it is spliced in at the next sentence end, answered, then standby thinking resumes\n");
        if (cot.input_line && !input_line_on) std::cerr << "[Idle][cot] --idle-input-line needs a terminal on stdin and stdout; using plain stdin instead\n";
    }
    if (cot.enabled && cot.user_input && cot.no_echo && !input_line_on) {
        if (mstream::disable_tty_echo()) std::cerr << "[Idle][cot] terminal echo off (restored on exit and on Ctrl-C)\n";
        else std::cerr << "[Idle][cot] --idle-no-echo ignored: stdin is not a terminal\n";
    }
    if (cot.enabled && !cot.input_file.empty()) {
        mstream::start_file_pump(g_splice, cot.input_file);
        std::cerr << "[Idle][cot] also reading user lines appended to " << cot.input_file << "\n";
    }
    if (cot.enabled) {   // on stdout on purpose: stderr is often redirected to a log
        std::cout << "[" << kCotBuild << "]\n[input] " << (!cot.user_input ? "stdin input is off" : (input_line_on ? "type in the bottom line + Enter" : "type a line + Enter"))
                  << (cot.input_file.empty() ? std::string() : std::string(" or append a line to ") + cot.input_file)
                  << " - it is spliced in at the end of the current sentence and shown as [USER]\n" << std::flush;
    }

    auto suppress_special = [&](std::vector<float>& l) {
        if (tok.eos_id >= 0 && (size_t)tok.eos_id < l.size()) l[tok.eos_id] = -1e30f;   // <|im_end|>: the runtime closes turns, not the model
        if (tok.bos_id >= 0 && (size_t)tok.bos_id < l.size()) l[tok.bos_id] = -1e30f;   // <|im_start|>
    };
    auto sample_next = [&]() -> int32_t {
        std::vector<float> adj(logits.begin(), logits.end());
        bool in_tag = false;
        if (cot.enabled) {
            suppress_special(adj);
            // Structure guard: the runtime injects every OPENING tag, so the model may only ever
            // produce closing ones. Opening-tag tokens are banned; after a bare "<" only "/" may
            // follow; after "</" only this block's own word (think / answer / next).
            for (int32_t id : tv.lt_open) adj[(size_t)id] = -1e30f;
            auto restrict_to = [&](const std::vector<int32_t>& ids) {
                if (ids.empty()) return;
                std::vector<std::pair<int32_t, float>> saved;
                saved.reserve(ids.size());
                for (int32_t id : ids) saved.emplace_back(id, adj[(size_t)id]);
                std::fill(adj.begin(), adj.end(), -1e30f);
                for (const auto& pr : saved) adj[(size_t)pr.first] = pr.second;
            };
            if (!tail.empty() && tail.back() == '<') restrict_to(tv.slash);
            else if (tail.size() >= 2 && tail.compare(tail.size() - 2, 2, "</") == 0) restrict_to(tv.close_word[(int)phase]);

            const size_t lt = tail.rfind('<');
            in_tag = (lt != std::string::npos && tail.find('>', lt) == std::string::npos);   // mid-tag: leave it alone
            if (!in_tag) {
                size_t since, min_p;
                if (phase == Phase::THINK)     { since = field_tokens; min_p = cot.min_think_tokens; }
                else if (phase == Phase::NEXT) { since = phase_tokens; min_p = 3; }
                else                           { since = phase_tokens; min_p = cot.min_answer_tokens; }
                if (since < min_p) {                                     // force real content before a field/block can end
                    for (int32_t id : tv.lt) adj[(size_t)id] = -1e30f;
                    if (phase != Phase::ANSWER) for (int32_t id : tv.nl) adj[(size_t)id] = -1e30f;
                }
                if (phase == Phase::ANSWER && phase_tokens < 2)          // no "Great question!" openers
                    for (int32_t id : praise_ids) adj[(size_t)id] = -1e30f;
            }
        }
        if (cot.enabled && deciding) {                                   // the Reply field: a forced choice, free to go either way
            // Asking the user always means speaking; a user message leans neutral; idle notes lean quiet.
            float bias = (kind == Kind::ASK) ? 1000.f
                       : (kind == Kind::USER ? cot.user_speak_bias
                       : (event_cycle ? 0.f : (override_repeat ? 2.0f : cot.speak_bias)));
            if (force_quiet) bias = -1000.f;                             // pacing rule: this cycle stays in thought
            std::vector<std::pair<int32_t, float>> saved;
            for (int32_t id : cv.yes) saved.emplace_back(id, adj[(size_t)id] + bias);
            for (int32_t id : cv.no)  saved.emplace_back(id, adj[(size_t)id]);
            if (!saved.empty()) {
                std::fill(adj.begin(), adj.end(), -1e30f);
                for (const auto& pr : saved) adj[(size_t)pr.first] = pr.second;
            }
            in_tag = true;                                               // (flag reused) skip repetition penalties for this token
        }
        if (!in_tag) apply_anti_repeat(adj, st.cached_tokens, keep, anti_repeat, penal_mask);
        return sample_top_p(adj.data(), (int)cfg.n_vocab, temperature, top_p, rng);
    };

    if (cot.enabled) {
        begin_next_cycle(/*first=*/true);
        flush_out();
        next = sample_next();
    }

    while (max_tokens == 0 || produced < max_tokens) {
        std::string piece;
        const bool eos_this_step = (next == tok.eos_id);
        if (eos_this_step) {
            piece = "\n";   // soft break
        } else {
            piece = tok.decode_one(next);
            ++produced;
            ++turn_tokens;
        }
        const bool sent_end = eos_this_step || piece.find_first_of(".!?\n") != std::string::npos;
        emit(piece);
        aging.push_back(piece);

        size_t discard = ensure_cache_room(cfg, st, 1, keep);
        if (discard > 0) cp_valid = false;
        for (size_t i = 0; i < discard && !aging.empty(); ++i) {
            evict_piece(aging.front());
            aging.pop_front();
        }

        std::vector<int32_t> one_token{ next };
        fwd::transformer_forward(cfg, st.mw, one_token, st.cache, logits.data(), pool);
        st.cached_tokens.push_back(next);

        if (sent_end) {
            if (managed) flush_out();
            appended_since_cp = 0;
            cp_valid = true;
            since_sent_end = 0;
        } else {
            ++appended_since_cp;
            ++since_sent_end;
        }

        // ---- Vision: in CoT mode a delta is queued as an outside event (consumed at the next
        // cycle boundary, never spliced into the middle of a block); otherwise spliced in place.
        if (vision_enabled && vision_gate.ready(std::chrono::steady_clock::now())) {
            auto delta = vis.poll();
            if (delta && cot.enabled) {
                std::string d = *delta;
                if (d.size() > kMaxAnnotationChars) d.resize(kMaxAnnotationChars);
                g_splice.push(mstream::SpliceKind::Vision, "I just noticed on screen: " + d);
                std::cerr << "[Idle] queued vision event: " << d << "\n";
            } else if (delta) {
                std::string annotation = "\n[Vision: " + *delta + "]\n";
                if (annotation.size() > kMaxAnnotationChars) {
                    annotation.resize(kMaxAnnotationChars);
                    annotation += "...]\n";
                }
                std::vector<int32_t> ann_tokens = tok.encode(annotation, /*add_bos=*/false);
                if (!ann_tokens.empty()) {
                    mstream::push_injected_text(aging, ann_tokens.size(), annotation);
                    size_t ann_discard = ensure_cache_room(cfg, st, ann_tokens.size(), keep);
                    for (size_t i = 0; i < ann_discard && !aging.empty(); ++i) {
                        evict_piece(aging.front());
                        aging.pop_front();
                    }
                    fwd::transformer_forward(cfg, st.mw, ann_tokens, st.cache, logits.data(), pool);
                    st.cached_tokens.insert(st.cached_tokens.end(), ann_tokens.begin(), ann_tokens.end());
                    cp_valid = false;
                    std::cerr << "[Idle] injected vision delta (" << ann_tokens.size()
                              << " tokens): " << *delta << "\n";
                }
            }
        }

        if (log_every > 0 && produced > 0 && produced % (long)log_every == 0) {
            std::cerr << "[Idle] produced=" << produced
                      << " cache.length=" << st.cache.length
                      << "/" << st.cache.capacity
                      << " disk_pending=" << idle_disk.pending_tokens() << "\n";
        }

        if (cot.enabled) {
            // ================= CoT cycle management =================
            tail += piece;
            if (tail.size() > 32) tail.erase(0, tail.size() - 32);
            ++phase_tokens;
            if (phase == Phase::THINK)  ++field_tokens;
            if (phase == Phase::ANSWER) answer_text += piece;
            if (phase == Phase::THINK && think_field == 0) task_text += piece;
            if (phase == Phase::NEXT)   next_text += piece;

            // (a) A user event is spliced in at the next sentence end (or after 24 tokens without one):
            //     close whatever is open, answer the user, then standby thinking resumes after that answer.
            bool handled_user = false;
            if (g_splice.has_user() && (sent_end || since_sent_end >= 24 || phase_tokens < 2)) {
                mstream::SpliceEvent ev;
                if (g_splice.try_pop_user(ev)) {
                    handled_user = true;
                    flush_out();
                    const char* closer = (phase == Phase::THINK) ? "\n</think>\n<answer>\n(one moment)\n</answer>"
                                       : (phase == Phase::NEXT ? "\n</next>" : "\n</answer>");
                    phase = Phase::ANSWER;
                    emit("\n[USER] " + ev.text + "\n");
                    flush_out();
                    inject(std::string(closer) + "<|im_end|>\n<|im_start|>user\n" + ev.text +
                           "<|im_end|>\n<|im_start|>assistant\n<think>\n" + think_labels[0]);
                    kind = Kind::USER; awaiting = false; deciding = false; spoke = false; event_cycle = false;
                    override_repeat = false; force_quiet = false; ask_suppressed = false; task_text.clear();
                    phase = Phase::THINK; phase_tokens = 0; think_field = 0; field_tokens = 0;
                    answer_text.clear(); next_text.clear();
                    emit("<think>\n" + think_labels[0]);
                    std::cerr << "[Idle][cot] user event spliced in -> replying\n";
                }
            }

            if (!handled_user) {
                // (b) tag detection (model-closed), caps, and degradation (run-on / loop)
                bool degraded = false;
                const char* why = "";
                if (since_sent_end >= anti_repeat.max_runon) { degraded = true; why = "run-on"; }
                else if (phase != Phase::NEXT && phase_tokens >= 24 && looks_like_loop(st.cached_tokens, keep)) { degraded = true; why = "loop"; }

                const bool t_closed   = phase == Phase::THINK  && tail.find("</think>") != std::string::npos;
                const bool t_answered = phase == Phase::THINK  && tail.find("<answer>") != std::string::npos;
                const bool a_closed   = phase == Phase::ANSWER && tail.find("</answer>") != std::string::npos;
                const bool a_thinks   = phase == Phase::ANSWER && tail.find("<think>")  != std::string::npos;
                const bool n_closed   = phase == Phase::NEXT   && tail.find("</next>")  != std::string::npos;
                const bool next_end   = phase == Phase::NEXT   && piece.find('\n') != std::string::npos;
                const size_t cap = (phase == Phase::THINK) ? cot.think_tokens
                                 : (phase == Phase::NEXT ? cot.next_tokens
                                 : (kind == Kind::USER ? cot.answer_tokens * 2
                                 : (kind == Kind::ASK ? std::min<size_t>(cot.answer_tokens, 48) : cot.answer_tokens)));
                const bool field_step = phase == Phase::THINK &&
                    (piece.find('\n') != std::string::npos || (sent_end && field_tokens >= cot.field_tokens));
                const bool capped = (sent_end && phase_tokens >= cap) || phase_tokens >= cap * 2;
                const bool ask_done = phase == Phase::ANSWER && kind == Kind::ASK && phase_tokens >= 4 &&
                                      piece.find('?') != std::string::npos;   // a question asked: stop there
                if (capped && !degraded) why = "length cap";
                if (ask_done) why = "question complete";
                const bool force = capped || degraded;

                auto enter_next = [&]() {
                    // The model now writes ITS OWN next step. The opener only biases it toward variety.
                    // Right after serving the user the openers lean toward checking in with them.
                    next_opener = cotx::pick_opener((unsigned)rng(), awaiting ? cotx::OpenerMode::Waiting
                                                   : (kind == Kind::USER && spoke && cycles_since_ask >= cot.ask_gap && (int)(rng() % 100) < cot.ask_after_user_percent
                                                      ? cotx::OpenerMode::AfterUser : cotx::OpenerMode::General));
                    inject("\n<next>\n" + next_opener);
                    phase = Phase::NEXT; phase_tokens = 0; next_text.clear();
                    emit("\n<next>\n" + next_opener);
                };

                if (phase == Phase::THINK && deciding && field_tokens >= 1) {
                    // The model just chose. yes -> open <answer>.  no -> no answer at all: close the think block
                    // and go straight to its own <next> line, i.e. it stays in thought.
                    deciding = false;
                    const bool speak = cotx::says_yes(piece);
                    flush_out();
                    if (speak) {
                        spoke = true;
                        phase = Phase::ANSWER; phase_tokens = 0;
                        const std::string txt = "\n</think>\n<answer>\n";
                        inject(txt);
                        emit(txt);
                        if (kind == Kind::ASK)        emit("[ASK] ");
                        else if (kind == Kind::USER)  emit("[REPLY] ");
                    } else {
                        if (kind == Kind::USER) ask_cooldown = cot.quiet_cycles;   // closure: don't start asking again right away
                        inject("\n</think>");
                        emit("\n</think>");
                        std::cerr << "[Idle][cot] staying in thought (" << (force_quiet ? quiet_why : std::string("its own choice")) << ")\n";
                        enter_next();
                    }
                } else if (phase == Phase::THINK && !t_closed && !t_answered && !force && field_step) {
                    // A think field finished: inject the next label, or close the think block after the last one.
                    const bool had_nl = piece.find('\n') != std::string::npos;
                    flush_out();
                    if (think_field + 1 < think_labels.size()) {
                        ++think_field; field_tokens = 0;
                        const std::string lbl = (had_nl ? "" : "\n") + think_labels[think_field];
                        inject(lbl);
                        emit(lbl);
                    } else {
                        // Text fields done: the model now chooses between speaking and staying in thought.
                        deciding = true; field_tokens = 0;
                        // Pacing, like a person: don't talk over a question you are waiting on, don't talk again right
                        // after speaking, and don't re-explain what you already covered (unless correcting yourself).
                        force_quiet = false; quiet_why.clear();
                        if (kind == Kind::NORMAL && !override_repeat && !event_cycle) {
                            if (awaiting)                                         { force_quiet = true; quiet_why = "waiting for the user's reply"; }
                            else if (speak_cooldown > 0)                          { force_quiet = true; quiet_why = "it just spoke"; }
                            else if (ask_suppressed)                              { force_quiet = true; quiet_why = "not checking in again so soon"; }
                            else if (cotx::repeats_topic(task_text, covered))     { force_quiet = true; quiet_why = "already explained that"; }
                        }
                        const std::string lbl = std::string(had_nl ? "" : "\n") + "Reply:";
                        inject(lbl);
                        emit(lbl);
                    }
                } else if (phase == Phase::THINK && (t_closed || t_answered || force)) {
                    bool rewound = false;
                    if (!t_closed && !t_answered && degraded) rewound = try_rewind();
                    if (!rewound) flush_out();
                    deciding = false; spoke = true;
                    phase = Phase::ANSWER; phase_tokens = 0;
                    if (t_answered) {
                        tail.clear();                                   // model opened <answer> itself
                    } else {
                        const std::string txt = t_closed ? "\n<answer>\n" : "\n</think>\n<answer>\n";
                        inject(txt);
                        emit(txt);
                    }
                    if (kind == Kind::ASK)        emit("[ASK] ");
                    else if (kind == Kind::USER)  emit("[REPLY] ");
                    if (force && !t_closed && !t_answered)
                        std::cerr << "[Idle][cot] think closed by runtime (" << why << (rewound ? ", rewound" : "") << ")\n";
                } else if (phase == Phase::ANSWER && (a_closed || a_thinks || force || ask_done)) {
                    bool rewound = false;
                    if (!a_closed && !a_thinks && degraded) rewound = try_rewind();
                    if (!rewound) flush_out();
                    if (!a_closed && !a_thinks) {
                        inject("\n</answer>");
                        emit("\n</answer>");
                        std::cerr << "[Idle][cot] answer closed by runtime (" << why << (rewound ? ", rewound" : "") << ")\n";
                    }
                    flush_out();
                    speak_cooldown = cot.speak_gap;                     // having spoken, it now lets that land
                    if (kind != Kind::ASK) {                            // remember what was covered so it is not explained twice
                        covered.push_back(cotx::clean_note(task_text, 120));
                        if (covered.size() > 16) covered.erase(covered.begin());
                    }
                    if (kind == Kind::ASK) {                            // the answer WAS the question: now we wait for the reply
                        std::string q = answer_text;
                        const size_t e = q.find("</answer>");
                        if (e != std::string::npos) q.resize(e);
                        pending_question = cotx::clean_note(q);
                        awaiting = true;
                        asked_at = std::chrono::steady_clock::now();
                        std::cerr << "[Idle][cot] question sent; thinking continues while waiting for the user\n";
                    }
                    if (a_thinks) {
                        phase = Phase::THINK; phase_tokens = 0; think_field = 0; field_tokens = 0; deciding = false; kind = Kind::NORMAL; tail.clear();
                    } else {
                        enter_next();
                    }
                } else if (phase == Phase::NEXT && (n_closed || force || next_end)) {
                    flush_out();
                    const bool had_nl = piece.find('\n') != std::string::npos;
                    if (!n_closed) {
                        const std::string c = std::string(had_nl ? "" : "\n") + "</next>";
                        inject(c);
                        emit(c);
                    }
                    last_note = cotx::clean_note(next_opener + next_text);
                    begin_next_cycle(/*first=*/false);
                    flush_out();
                }
            }
        } else if (steer_enabled) {
            // ---- non-CoT steering (v2): new user turn on EOS / length cap, rewind on run-on / loop
            bool need_steer = false, degraded = false;
            const char* why = "";
            if (eos_this_step)                                         { need_steer = true; why = "EOS"; }
            else if (sent_end && turn_tokens >= anti_repeat.turn_tokens){ need_steer = true; why = "turn cap"; }
            else if (since_sent_end >= anti_repeat.max_runon)          { need_steer = degraded = true; why = "run-on"; }
            else if (looks_like_loop(st.cached_tokens, keep))          { need_steer = degraded = true; why = "loop"; }

            if (need_steer) {
                bool rewound = false;
                if (degraded) rewound = try_rewind();
                if (!rewound) flush_out();
                std::cout << "\n" << std::flush;

                std::string steer = std::string(eos_this_step ? "" : "<|im_end|>") +
                                    "\n<|im_start|>user\n" + make_steer(steer_idx++, steer_topic) +
                                    "<|im_end|>\n<|im_start|>assistant\n";
                inject(steer);
                turn_tokens = 0;
                std::cerr << "[Idle][steer] " << why << (rewound ? " (rewound to last sentence)" : "") << " -> new user turn\n";
            }
        }

        next = sample_next();
    }

    if (!pending_out.empty()) std::cout << pending_out;
    std::cout << tok.flush() << std::flush;
    idle_disk.flush();   // don't lose a partial buffer on clean shutdown
    std::cerr << "\n[Idle] stopped after " << produced << " tokens (max_tokens reached). "
              << "still-live (never evicted, not on disk): " << aging.size() << " tokens.\n";
}
#endif // NANITY_ENABLE_IDLE_LOOP


// =============================================================================
// Model-format dispatch
// =============================================================================
// Sniffed from the file's first 4 bytes, not its extension — an extension
// can be renamed; the magic can't lie about what the parser downstream is
// about to assume.
enum class ModelFormat { GGUF, NCTR };

static ModelFormat sniff_format(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open model: " + path);
    char magic[4] = {0};
    f.read(magic, 4);
    if (!std::memcmp(magic, "GGUF", 4)) return ModelFormat::GGUF;
    if (!std::memcmp(magic, "NCTR", 4)) return ModelFormat::NCTR;
    throw std::runtime_error(
        "Unrecognized model format (first 4 bytes are neither 'GGUF' nor 'NCTR'): " + path);
}

// Everything downstream of "which loader do we use" — load, validate,
// --probe, tokenizer setup, and generation — lives here once and gets
// instantiated per Loader (loader::GGUFLoader or loader::NCTRLoader) rather
// than duplicated per format. This is exactly the body that used to be the
// second half of main(); only `gguf` -> `model` (generic name, not
// format-specific) and the type changed.
struct IdleLoopArgs {
    bool        enabled     = false;
    std::string seed_text;                 // implant text (§3/§4)
    size_t      sink_tokens = 0;
    long        max_tokens  = 0;           // 0 = infinite
    size_t      log_every   = 50;
    float       temperature = 0.9f;        // idle CoT defaults a bit hotter than chat
    float       top_p       = 0.95f;
    std::string disk_path   = "nectar_idle_memory.jsonl";  // §5.1
    size_t      disk_flush_every_tokens = 300;
    long        disk_flush_interval_s   = 30;
    bool        vision_enabled  = false;   // §3/§7 — off by default: needs Hyprland+grim+tesseract
    long        vision_poll_ms  = 2000;
    // ── zRAM tier (nectar_zram.hpp) — all off by default (chunk_tokens=0
    //    disables it entirely, identical behavior to before this existed)
    IdleCompressionKind compression_kind    = IdleCompressionKind::None;
    size_t               chunk_tokens        = 0;
    size_t               zram_ring_capacity  = 32;
    std::string          zram_index_path     = "nectar_zram_index.jsonl";
    size_t               summary_max_tokens  = 40;
    double               zram_max_keep_fraction = 0.5;
    AntiRepeat           anti_repeat;
    bool                 steer = false;
    std::string          steer_topic;
    CotConfig            cot;
};

template <typename Loader>
static int run_model(const std::string& model_path, engine::Config& cfg,
                      util::ThreadPool& pool, bool probe_only, bool interactive,
                      const std::string& input_payload,
                      const IdleLoopArgs& idle,
                      std::chrono::steady_clock::time_point t_start)
{
    VLOG("open: " + model_path);
    std::cerr << "[Storage] Mapping weights via mmap: " << model_path << "\n";
    Loader model;
    model.open(model_path);
    std::cerr << "[Storage] Mapped " << model.tensors.size() << " tensors.\n";

    VLOG("validate_config");
    // FIX (carried over): validate_config() throws a specific, actionable
    // message (which exact metadata key or tensor shape didn't match the
    // NANITY spec) instead of returning a bare false — main()'s catch block
    // surfaces that message directly, so there's nothing to special-case here.
    model.validate_config(cfg);
    std::cerr << "[Config]"
              << " n_vocab="   << cfg.n_vocab
              << " n_embd="    << cfg.n_embd
              << " n_layer="   << cfg.n_layer
              << " n_head="    << cfg.n_head
              << " n_kv_head=" << cfg.n_kv_head
              << " head_dim="  << cfg.head_dim
              << " n_ff="      << cfg.n_ff
              << " ctx_len="   << cfg.ctx_len
              << " rope_base=" << cfg.rope_base << "\n";

    VLOG_ELAPSED("model load+detect", t_start);

    // ── --probe: boot validation used by NEON.py's /api/boot ──────────────
    if (probe_only) {
        std::cout << "PROBE_OK\n" << std::flush;
        return 0;
    }

    // ── Tokenizer ──────────────────────────────────────────────────────────
    Tokenizer tok(model.tok_meta);
    std::cerr << "[Tokenizer] vocab=" << model.tok_meta.tokens.size()
              << " bos=" << tok.bos_id << " eos=" << tok.eos_id << "\n";
    {   // matters for any caller that builds ChatML text: encode(add_bos=true) already prepends BOS
        auto im = tok.vocab_map.find("<|im_start|>");
        std::cerr << "[Tokenizer] BOS " << ((im != tok.vocab_map.end() && im->second == tok.bos_id) ? "IS" : "is NOT")
                  << " <|im_start|>  (so ChatML prompt text should " << ((im != tok.vocab_map.end() && im->second == tok.bos_id) ? "NOT" : "")
                  << " start with <|im_start|> itself)\n";
    }

    // ── Persistent interactive mode (spawned by NEON.py on boot) ──────────
    if (interactive) {
        std::cerr << "[Mode] Interactive stdin/stdout loop — ready.\n";
        run_interactive(cfg, model, tok, pool);
        return 0;
    }

    // ── Idle loop mode (own process — see run_idle_loop()'s comment on why
    //    this must never share a process with --interactive) ──────────────
    if (idle.enabled) {
#if defined(NANITY_ENABLE_IDLE_LOOP)
        std::cerr << "[Mode] Idle loop — ready.\n";
        run_idle_loop(cfg, model, tok, pool, idle.seed_text, idle.sink_tokens,
                      idle.max_tokens, idle.log_every, idle.temperature, idle.top_p,
                      idle.disk_path, idle.disk_flush_every_tokens, idle.disk_flush_interval_s,
                      idle.vision_enabled, idle.vision_poll_ms,
                      idle.compression_kind, idle.chunk_tokens, idle.zram_ring_capacity,
                      idle.zram_index_path, idle.summary_max_tokens, idle.zram_max_keep_fraction,
                      idle.anti_repeat, idle.steer, idle.steer_topic, idle.cot);
        return 0;
#else
        std::cerr << "[Mode] Idle loop requested, but this build was compiled without "
                     "NANITY_ENABLE_IDLE_LOOP (needs nectar_diskmem.hpp/nectar_vision.hpp/"
                     "nectar_splice.hpp). Rebuild with -DNANITY_ENABLE_IDLE_LOOP and those "
                     "headers present to use --idle.\n";
        return 1;
#endif
    }

    // ── Single-shot mode (legacy / direct CLI testing) ─────────────────────
    if (input_payload.empty()) {
        std::cerr << "[Error] --prompt required in single-shot mode.\n";
        return 1;
    }

    std::cerr << "[Parser] Parsing input payload...\n";
    std::string prompt;
    if (!input_payload.empty() && input_payload.front() == '{') {
        auto req  = json::parse(input_payload);
        const auto& msgs = req["messages"];
        if (msgs.is_array() && msgs.size() > 0) {
            prompt = format_chat(msgs);
        } else {
            prompt = req["prompt"].get_str(input_payload);
        }
    } else {
        prompt = input_payload;
    }

    float temperature = 0.7f;
    float top_p       = 0.9f;
    int   max_tokens  = 256;

    std::mt19937 rng(std::random_device{}());
    std::cerr << "\n--- Generation ---\n";
    generate(cfg, model, tok, prompt, max_tokens, temperature, top_p, pool, rng);
    std::cerr << "--- End ---\n";
    return 0;
}


// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    // FIX: was `argc < 5` which rejected valid two-flag invocations like
    // `--model path --probe`.
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " --model <path.gguf|path.nctr>"
                     " [--prompt <text_or_json> | --interactive | --probe | --idle-loop]"
                     " [--threads N]\n"
                     " Idle-loop flags (only with --idle-loop):\n"
                     "   --idle-seed <text>            implant text, inline (required unless --idle-seed-file)\n"
                     "   --idle-seed-file <path>        implant text, read from a file (overrides --idle-seed)\n"
                     "   --idle-sink-tokens <N>         dedicated neutral sink tokens appended after the implant (default 0)\n"
                     "   --idle-max-tokens <N>          stop after N generated tokens, 0 = run forever (default 0)\n"
                     "   --idle-log-every <N>           log cache/disk stats every N tokens (default 50)\n"
                     "   --idle-temperature <f>         sampling temperature (default 0.9)\n"
                     "   --idle-top-p <f>               nucleus sampling top-p (default 0.95)\n"
                     "   --idle-rep-penalty <f>         repetition penalty over recent window, 1.0=off (default 1.15)\n"
                     "   --idle-freq-penalty <f>        per-occurrence logit penalty (default 0.02)\n"
                     "   --idle-rep-window <N>          tokens the penalties look back over (default 256)\n"
                     "   --idle-no-repeat-ngram <N>     never complete an n-gram already seen, 0=off (default 4)\n"
                     "   --idle-steer                   manage turns: new ChatML user turn on EOS/length cap, rewind+steer on run-on/loop\n"
                     "   --idle-cot                     standby CoT: <think>..</think><answer>..</answer><next>..</next>; the model writes its own next step each cycle, can ask the user and keeps thinking while waiting (builds its own seed if none given)\n"
                     "   --idle-lore <file>             one stimulus per line (character lore, goals, notes) mixed into the CoT stimuli\n"
                     "   --idle-ask-timeout <S>         stop waiting for a user reply after S seconds (default 120)\n"
                     "   --idle-lore-percent <N>        chance (%) a cycle opens with a lore line instead of the model's own next step (default 15)\n"
                     "   --idle-next-tokens <N>         soft cap for the model's own what-next line (default 30)\n"
                     "   --idle-think-tokens <N>        soft cap for the whole think block (default 100)\n"
                     "   --idle-answer-tokens <N>       soft cap per idle answer, doubled when answering the user (default 120)\n"
                     "   --idle-task <text>             prompt to follow first (answered via think/answer), then developed before standby drift\n"
                     "   --idle-min-think <N>           tokens each think field needs before it may end (default 4)\n"
                     "   --idle-hide-think              don't print <think> text (still kept in context and disk log)\n"
                     "   --idle-input-file <path>       also read user messages from lines appended to this file (e.g. echo \"hi\" >> path from another shell)\n"
                     "   --idle-speak-bias <f>          logit added to Reply:yes on idle cycles; negative = lean toward staying in thought (default -1.0)\n"
                     "   --idle-user-speak-bias <f>     same when answering a user message (default 0.5)\n"
                     "   --idle-quiet-cycles <N>        after it stays silent on a user message, no new questions for N cycles (default 8)\n"
                     "   --idle-speak-gap <N>           cycles it stays quiet after speaking, and it never re-explains a covered topic (default 5)\n"
                     "   --idle-ask-gap <N>             minimum cycles between questions to the user (default 10)\n"
                     "   --idle-ask-after-user <N>      chance (%) that finishing a reply to the user leads it to consider checking in (default 25)\n"
                     "   --idle-system-file <path>      replace the built-in system prompt with this file (built-in seed only; not used with --idle-seed)\n"
                     "   --idle-input-line              pinned input line on the bottom terminal row: you always see what you type, model output scrolls above\n"
                     "   --idle-no-echo                 turn off terminal input echo so typed text does not land inside the output (restored on exit / Ctrl-C)\n"
                     "   --idle-no-user-input           don't read stdin lines as user interrupts\n"
                     "   --idle-topic <text>            subject the steering prompts keep returning to (e.g. \"the United Nations\")\n"
                     "   --idle-min-p <f>               min-p filter, 0=off (default 0.05)\n"
                     "   --idle-turn-tokens <N>         soft cap on tokens per assistant turn (default 200)\n"
                     "   --idle-max-runon <N>           tokens without sentence end before rewind (default 60)\n"
                     "   --idle-disk-path <path>        JSONL file evicted idle text is appended to (default nectar_idle_memory.jsonl)\n"
                     "   --idle-disk-flush-tokens <N>   flush to disk after N buffered tokens (default 300)\n"
                     "   --idle-disk-flush-seconds <N>  or after N seconds, whichever comes first (default 30)\n"
                     "   --idle-vision                  enable vision-delta polling+injection (needs Hyprland+grim; OCR needs tesseract)\n"
                     "   --idle-vision-poll-ms <N>      poll interval for vision deltas (default 2000)\n"
                     "   --idle-compression <none|text|slots>  zRAM tier backend (default none; slots is not wired yet, see nectar_zram.hpp)\n"
                     "   --idle-chunk-tokens <N>        tokens per zRAM chunk before compression fires (default 0 = zRAM off)\n"
                     "   --idle-zram-ring <N>           chunk pointers kept resident before indexing to disk (default 32)\n"
                     "   --idle-zram-index <path>       JSONL index file for chunks that age out of the ring (default nectar_zram_index.jsonl)\n"
                     "   --idle-summary-tokens <N>      max tokens in a TextSummary compressor's output per chunk (default 40)\n"
                     "   --idle-zram-max-keep <f>       cap on keep/kv_window before a summary stops being pinned, 0-1 (default 0.5)\n";
        return 1;
    }

    std::string model_path;
    std::string input_payload;
    bool interactive = false;
    bool probe_only  = false;
    int  threads_override = 0;   // 0 = auto-detect from hardware_concurrency()
    IdleLoopArgs idle;
    std::string idle_seed_file;
    std::string idle_lore_file;
    std::string idle_system_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--model"       && i + 1 < argc) model_path    = argv[++i];
        else if (arg == "--prompt"      && i + 1 < argc) input_payload = argv[++i];
        else if (arg == "--interactive")                  interactive   = true;
        else if (arg == "--probe")                        probe_only    = true;
        else if (arg == "--threads"     && i + 1 < argc) threads_override = std::max(0, std::atoi(argv[++i]));
        else if (arg == "--idle-loop")                    idle.enabled  = true;
        else if (arg == "--idle-seed"             && i + 1 < argc) idle.seed_text  = argv[++i];
        else if (arg == "--idle-seed-file"        && i + 1 < argc) idle_seed_file  = argv[++i];
        else if (arg == "--idle-sink-tokens"      && i + 1 < argc) idle.sink_tokens = (size_t)std::max(0, std::atoi(argv[++i]));
        else if (arg == "--idle-max-tokens"       && i + 1 < argc) idle.max_tokens  = std::atol(argv[++i]);
        else if (arg == "--idle-log-every"        && i + 1 < argc) idle.log_every   = (size_t)std::max(0, std::atoi(argv[++i]));
        else if (arg == "--idle-temperature"      && i + 1 < argc) idle.temperature = std::strtof(argv[++i], nullptr);
        else if (arg == "--idle-top-p"            && i + 1 < argc) idle.top_p       = std::strtof(argv[++i], nullptr);
        else if (arg == "--idle-rep-penalty"      && i + 1 < argc) idle.anti_repeat.rep_penalty     = std::strtof(argv[++i], nullptr);
        else if (arg == "--idle-freq-penalty"     && i + 1 < argc) idle.anti_repeat.freq_penalty    = std::strtof(argv[++i], nullptr);
        else if (arg == "--idle-rep-window"       && i + 1 < argc) idle.anti_repeat.window          = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-no-repeat-ngram"  && i + 1 < argc) idle.anti_repeat.no_repeat_ngram = (int)std::strtol(argv[++i], nullptr, 10);
        else if (arg == "--idle-steer")                            idle.steer = true;
        else if (arg == "--idle-topic"            && i + 1 < argc) idle.steer_topic = argv[++i];
        else if (arg == "--idle-cot")                              idle.cot.enabled = true;
        else if (arg == "--idle-lore"             && i + 1 < argc) idle_lore_file = argv[++i];
        else if (arg == "--idle-ask-timeout"      && i + 1 < argc) idle.cot.ask_timeout_s = std::atol(argv[++i]);
        else if (arg == "--idle-lore-percent"     && i + 1 < argc) idle.cot.lore_percent  = std::atoi(argv[++i]);
        else if (arg == "--idle-next-tokens"      && i + 1 < argc) idle.cot.next_tokens   = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-think-tokens"     && i + 1 < argc) idle.cot.think_tokens  = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-answer-tokens"    && i + 1 < argc) idle.cot.answer_tokens = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-hide-think")                       idle.cot.hide_think = true;
        else if (arg == "--idle-task"             && i + 1 < argc) idle.cot.task = argv[++i];
        else if (arg == "--idle-min-think"        && i + 1 < argc) idle.cot.min_think_tokens = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-no-user-input")                    idle.cot.user_input = false;
        else if (arg == "--idle-no-echo")                          idle.cot.no_echo = true;
        else if (arg == "--idle-input-line")                       idle.cot.input_line = true;
        else if (arg == "--idle-system-file"      && i + 1 < argc) idle_system_file = argv[++i];
        else if (arg == "--idle-ask-gap"          && i + 1 < argc) idle.cot.ask_gap = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-ask-after-user"   && i + 1 < argc) idle.cot.ask_after_user_percent = std::atoi(argv[++i]);
        else if (arg == "--idle-speak-gap"        && i + 1 < argc) idle.cot.speak_gap = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-speak-bias"       && i + 1 < argc) idle.cot.speak_bias = std::strtof(argv[++i], nullptr);
        else if (arg == "--idle-user-speak-bias"  && i + 1 < argc) idle.cot.user_speak_bias = std::strtof(argv[++i], nullptr);
        else if (arg == "--idle-quiet-cycles"     && i + 1 < argc) idle.cot.quiet_cycles = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-input-file"       && i + 1 < argc) idle.cot.input_file = argv[++i];
        else if (arg == "--idle-min-p"            && i + 1 < argc) idle.anti_repeat.min_p      = std::strtof(argv[++i], nullptr);
        else if (arg == "--idle-turn-tokens"      && i + 1 < argc) idle.anti_repeat.turn_tokens = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-max-runon"        && i + 1 < argc) idle.anti_repeat.max_runon   = (size_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--idle-disk-path"        && i + 1 < argc) idle.disk_path   = argv[++i];
        else if (arg == "--idle-disk-flush-tokens" && i + 1 < argc) idle.disk_flush_every_tokens = (size_t)std::max(0, std::atoi(argv[++i]));
        else if (arg == "--idle-disk-flush-seconds"&& i + 1 < argc) idle.disk_flush_interval_s   = std::atol(argv[++i]);
        else if (arg == "--idle-vision")                             idle.vision_enabled = true;
        else if (arg == "--idle-vision-poll-ms"    && i + 1 < argc)  idle.vision_poll_ms = std::atol(argv[++i]);
        else if (arg == "--idle-compression"       && i + 1 < argc) {
            std::string v = argv[++i];
            if      (v == "none")  idle.compression_kind = IdleCompressionKind::None;
            else if (v == "text")  idle.compression_kind = IdleCompressionKind::TextSummary;
            else if (v == "slots") idle.compression_kind = IdleCompressionKind::SoftSlots;
            else { std::cerr << "[Error] --idle-compression: expected none|text|slots, got '" << v << "'\n"; return 1; }
        }
        else if (arg == "--idle-chunk-tokens"      && i + 1 < argc) idle.chunk_tokens       = (size_t)std::max(0, std::atoi(argv[++i]));
        else if (arg == "--idle-zram-ring"         && i + 1 < argc) idle.zram_ring_capacity = (size_t)std::max(1, std::atoi(argv[++i]));
        else if (arg == "--idle-zram-index"        && i + 1 < argc) idle.zram_index_path    = argv[++i];
        else if (arg == "--idle-summary-tokens"    && i + 1 < argc) idle.summary_max_tokens  = (size_t)std::max(1, std::atoi(argv[++i]));
        else if (arg == "--idle-zram-max-keep"     && i + 1 < argc) idle.zram_max_keep_fraction = std::strtod(argv[++i], nullptr);
    }

    if (idle.enabled && idle.compression_kind != IdleCompressionKind::None && idle.chunk_tokens == 0) {
        std::cerr << "[Error] --idle-compression=" << (idle.compression_kind == IdleCompressionKind::TextSummary ? "text" : "slots")
                   << " requires --idle-chunk-tokens > 0 (0 means zRAM stays off regardless of backend selected).\n";
        return 1;
    }

    if (idle.enabled && !idle_seed_file.empty()) {
        std::ifstream f(idle_seed_file, std::ios::binary);
        if (!f) {
            std::cerr << "[Error] --idle-seed-file: cannot open " << idle_seed_file << "\n";
            return 1;
        }
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        idle.seed_text = content;
    }

    if (idle.enabled && idle.cot.enabled) {
        idle.cot.default_seed = idle.seed_text.empty();
        if (!idle_system_file.empty()) {                    // the system prompt as DATA: edit the file, no recompile
            std::ifstream sf(idle_system_file, std::ios::binary);
            if (!sf) { std::cerr << "[Error] --idle-system-file: cannot open " << idle_system_file << "\n"; return 1; }
            idle.cot.system_text.assign((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
            std::cerr << "[Idle][cot] system prompt loaded from " << idle_system_file << " (" << idle.cot.system_text.size() << " bytes)\n";
        }   // no seed given -> use the built-in CoT system prompt
        if (!idle_lore_file.empty()) {
            std::ifstream lf(idle_lore_file);
            if (!lf) { std::cerr << "[Error] --idle-lore: cannot open " << idle_lore_file << "\n"; return 1; }
            std::string ln;
            while (std::getline(lf, ln)) {
                while (!ln.empty() && (ln.back() == '\r' || ln.back() == ' ')) ln.pop_back();
                if (!ln.empty()) idle.cot.stimuli.push_back(ln);
            }
        }
        idle.cot.n_lore = idle.cot.stimuli.size();            // [0, n_lore) = lore; defaults follow as fallbacks
        for (auto& d : default_stimuli()) idle.cot.stimuli.push_back(d);
    }
    if (idle.enabled && idle.seed_text.empty() && !idle.cot.enabled) {
        std::cerr << "[Error] --idle-loop requires --idle-seed <text> or --idle-seed-file <path> (or --idle-cot for the built-in CoT seed).\n";
        return 1;
    }

    if (model_path.empty()) {
        std::cerr << "[Error] --model is required.\n";
        return 1;
    }

    try {
        auto t_start = std::chrono::steady_clock::now();

        // ── Hardware init (format-independent, runs once regardless of
        //    which loader ends up being picked below) ────────────────────
        engine::Config cfg;

        if (threads_override > 0) {
            cfg.n_threads = threads_override;
        } else {
            unsigned hw = std::thread::hardware_concurrency();
            cfg.n_threads = hw > 0 ? (int)hw : 4;
        }
        std::cerr << "[Hardware] using " << cfg.n_threads << " CPU thread(s)"
                  << (threads_override > 0 ? " (--threads override)\n" : " (auto-detected)\n");

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
        VLOG("probing ROCm devices");
        auto& gpu = rocm::GpuContext::get();
        if (gpu.available()) {
            std::cerr << "[Hardware] ROCm GPU: " << gpu.n_devices << " device(s)\n";
        } else {
            std::cerr << "[Hardware] ROCm: no devices found, running CPU-only.\n";
        }
#endif

#if defined(USE_VULKAN)
        // EXPERIMENTAL, informational only (matches rawllm_vulkan.hpp's own
        // documented scope): this constructs the backend so a --vulkan build
        // reports real device/shader availability instead of staying
        // untested dead code, but proj_all_positions() below still always
        // runs on the CPU thread pool — nothing here changes which path
        // actual token generation takes. F32/Q4_0/Q8_0 fused GPU kernels
        // (with weight residency and batched multi-op submission) all
        // exist now (rawllm_vulkan.hpp), plus an optional fp16
        // cooperative-matrix GEMM path for prefill on hardware that
        // supports it, but routing real matvecs through any of this —
        // deciding when GPU beats CPU for a given tensor size, prefill-
        // vs-decode dispatch, the upload/readback cost model at that
        // point — is exactly the follow-up work rawllm_vulkan.hpp's
        // header comment describes, not done here.
        VLOG("probing Vulkan devices");
        try {
            static vk_backend::VulkanMatvecBackend vulkan_probe("shaders");
            std::cerr << "[Hardware] Vulkan: compute device + F32/Q4_0/Q8_0 matvec shaders "
                          "initialized (experimental — not yet used for generation)"
                       << (vulkan_probe.coop_matrix_supported() ? ", fp16 cooperative-matrix GEMM available.\n" : ".\n");
        } catch (const std::exception& e) {
            std::cerr << "[Hardware] Vulkan: unavailable (" << e.what() << "), running CPU-only.\n";
        }
#endif

#ifdef USE_NUMA
        util::ThreadPool pool(cfg.n_threads, /*numa_bind=*/true);
        std::cerr << "[Topology] NUMA-aware thread pool (" << cfg.n_threads << " threads)\n";
#else
        util::ThreadPool pool(cfg.n_threads, false);
#endif

        // ── Format dispatch ─────────────────────────────────────────────
        ModelFormat fmt = sniff_format(model_path);
        switch (fmt) {
            case ModelFormat::GGUF:
                std::cerr << "[Format] GGUF\n";
                return run_model<loader::GGUFLoader>(
                    model_path, cfg, pool, probe_only, interactive, input_payload, idle, t_start);
            case ModelFormat::NCTR:
                std::cerr << "[Format] NCTR\n";
                return run_model<loader::NCTRLoader>(
                    model_path, cfg, pool, probe_only, interactive, input_payload, idle, t_start);
        }
        return 1;   // unreachable — silences -Wreturn-type on some compilers

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n";
        return 1;
    }
}
