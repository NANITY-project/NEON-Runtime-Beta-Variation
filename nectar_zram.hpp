#pragma once
// =============================================================================
// nectar_zram.hpp — Compressed-context ring buffer for the idle loop
// ("zRAM" tier, sitting between the live sliding-window KV cache — the
// "RAM" tier already implemented via ensure_cache_room()/kv_cache_shift()
// in NEON.cpp/rawllm_forward.hpp — and the raw append-only write-only log
// in nectar_diskmem.hpp — the "Disk" tier).
//
// Scope: this header is deliberately backend-agnostic. It does NOT know
// about engine::Config, ModelWeights, KVCache, or the tokenizer — same
// reasoning as nectar_diskmem.hpp / nectar_splice.hpp staying
// model-agnostic. What actually compresses a chunk is supplied by the
// caller (NEON.cpp) as a CompressFn, so either backend below can plug into
// the exact same ring/pointer machinery:
//
//   TextSummary — the model (or a smaller one) is asked to summarize the
//   chunk in plain text; the summary is real tokens, re-spliceable into
//   the pinned implant region with zero architecture changes. Works today,
//   no training required. Lossy in the ordinary "summarization loses
//   detail" sense, not in a "garbage in the cache" sense.
//
//   SoftSlots — a fixed number of reserved token ids are appended to the
//   chunk and forwarded through the model; their resulting K/V rows are
//   kept (via rawllm_forward.hpp's kv_cache_compact()) in place of the raw
//   chunk's rows. Mechanically this works with zero forward()/KVCache
//   interface changes. It is NOT real compression until the model has
//   been fine-tuned with an objective that makes those slots' K/V actually
//   encode the chunk (à la Gist Tokens / AutoCompressors / ICAE) — without
//   that training, the slots' K/V is attention output over untrained
//   tokens, i.e. noise the rest of generation would be attending to. See
//   make_soft_slot_compressor()'s comment below for the exact sequencing
//   this backend needs from its caller, which TextSummary does not.
// =============================================================================
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <string>

#include "nectar_diskmem.hpp"   // reuses diskmem::json_escape — one escaper, not two

namespace zram {

// What a compressor hands back for one chunk. Exactly one of the two
// "payload" fields is meaningful per backend; the other is left at its
// default. Nothing downstream branches on backend identity — it just
// checks which field is non-empty/non-zero, so adding a third backend
// later doesn't require touching ChunkRing or the call site's control
// flow, only CompressFn's implementation.
struct CompressResult {
    // TextSummary: plain-text summary of the chunk, ready to be
    // re-tokenized and folded into the pinned implant. Empty for
    // SoftSlots (there is nothing textual to splice — the compression
    // already lives in the KV cache once compress() returns).
    std::string splice_text;

    // SoftSlots: how many trailing KV-cache rows the caller should
    // preserve (via kv_cache_compact()) instead of discarding, because
    // this compressor already forwarded that many reserved slot tokens
    // and wrote their K/V. 0 for TextSummary (plain discard — nothing in
    // the cache survives this chunk's eviction, same as before zRAM
    // existed at all).
    size_t num_soft_slots = 0;
};

// Supplied by the engine (NEON.cpp), not by this header. Given the plain
// text of a chunk that's about to leave the live window, produce a
// CompressResult. A std::function rather than a virtual interface so this
// header never needs to know what a "model" is — same boundary
// nectar_diskmem.hpp/nectar_splice.hpp already draw.
//
// IMPORTANT SEQUENCING NOTE for SoftSlots implementations specifically:
// producing num_soft_slots > 0 requires having already run a forward()
// pass over the reserved slot tokens, appended to the LIVE cache, BEFORE
// the chunk's raw KV rows are discarded — the slots' causal attention has
// to see the chunk to have any chance of summarizing it. That means a
// SoftSlots CompressFn cannot be a pure "text in, metadata out" function
// the way TextSummary's is: it must capture references to the live
// engine state (cache, model, pool) and do its forward pass as a side
// effect of being called, before returning how many rows it just wrote.
// The call site (NEON.cpp) must then use kv_cache_compact(), not
// kv_cache_shift(), for that eviction. TextSummary's CompressFn has no
// such requirement — it can run any time after the chunk's text is known,
// including after the plain kv_cache_shift() discard already happened.
using CompressFn = std::function<CompressResult(const std::string& chunk_text)>;

// One compressed chunk's metadata. `raw_text` is kept only long enough to
// survive from push() to whenever this pointer ages out of the ring and
// gets indexed to disk (see ChunkRing::write_index_record) — it is NOT
// meant to accumulate in memory for the ring's full lifetime beyond that,
// which is why capacity_ below matters: a bounded ring bounds how much
// raw_text is resident at once, same motivation as the KV window itself
// being bounded.
struct ChunkPointer {
    size_t      chunk_id          = 0;
    // Positions here are counted in the idle loop's own eviction order —
    // "the Nth through Mth token this loop has ever evicted" — NOT a live
    // KV-cache slot index. Cache slots get reused after every
    // kv_cache_shift()/kv_cache_compact(), so a raw slot index would stop
    // meaning anything the moment the next shift happens; an
    // ever-incrementing eviction counter stays meaningful for as long as
    // the process runs, which is what a pointer meant to outlive the ring
    // (via the index file) actually needs.
    size_t      token_range_start = 0;   // inclusive
    size_t      token_range_end   = 0;   // exclusive
    std::string summary;                 // TextSummary output; empty for SoftSlots/none
    size_t      num_soft_slots    = 0;   // SoftSlots output; 0 otherwise
    std::string raw_text;                // full chunk text — dropped once indexed to disk
};

// Fixed-capacity ring of live ChunkPointers. "Live" means resident in this
// process's RAM — hence "zRAM" as a name, in the sense of "compressed
// context kept in RAM," not literally Linux zram/swap. When the ring is
// full, the oldest pointer is written to a small JSONL index file and
// dropped from RAM; its full raw_text survives only in that index record
// (and, redundantly, in nectar_diskmem's own append log if the caller
// also writes there — see NEON.cpp's wiring, which keeps both so the
// existing disk tier's behavior is unchanged by zRAM's existence).
//
// No read-back function here either, same as nectar_diskmem.hpp and for
// the same reason: "the idle loop never reads back" is a property this
// header enforces structurally by not exposing a query API, not a
// convention callers have to remember. A future answer-path retrieval
// module (out of scope here) reads the index file directly.
class ChunkRing {
public:
    ChunkRing(size_t capacity, std::string index_path)
        : capacity_(capacity), index_path_(std::move(index_path)) {}

    void push(ChunkPointer ptr) {
        ring_.push_back(std::move(ptr));
        while (ring_.size() > capacity_) {
            write_index_record(ring_.front());
            ring_.pop_front();
        }
    }

    // Splice material for the pinned implant: every summary currently
    // resident in the ring, oldest to newest, space-joined. Empty pieces
    // (SoftSlots pointers have no summary text) are skipped. Callers
    // decide how/when to re-encode and pin this — this just assembles the
    // text, it doesn't touch any cache or tokenizer.
    std::string joined_summaries() const {
        std::string out;
        for (const auto& p : ring_) {
            if (p.summary.empty()) continue;
            if (!out.empty()) out += ' ';
            out += p.summary;
        }
        return out;
    }

    size_t size() const { return ring_.size(); }
    size_t capacity() const { return capacity_; }

private:
    void write_index_record(const ChunkPointer& p) {
        std::ofstream f(index_path_, std::ios::app);
        if (!f) return;   // best-effort, same posture as diskmem::WriteBuffer
        f << "{\"chunk_id\":" << p.chunk_id
          << ",\"start\":" << p.token_range_start
          << ",\"end\":" << p.token_range_end
          << ",\"num_soft_slots\":" << p.num_soft_slots
          << ",\"summary\":\"" << diskmem::json_escape(p.summary) << "\""
          << ",\"raw_text\":\"" << diskmem::json_escape(p.raw_text) << "\"}\n";
    }

    size_t                   capacity_;
    std::string              index_path_;
    std::deque<ChunkPointer> ring_;
};

} // namespace zram
