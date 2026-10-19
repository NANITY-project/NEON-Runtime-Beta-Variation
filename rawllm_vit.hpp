#pragma once
// =============================================================================
// rawllm_vit.hpp — ViT = Vision Transformer tower: image -> LLM-embedding tokens
//
// Pipeline (all CPU, F32 activations, weights F32/F16/BF16/Q8_0 straight out of
// the mmapped GGUF):
//
//   RGB image (PPM/P6 bytes — exactly what vision::capture_frame()/`grim -t ppm`
//   already produce)
//     -> PIL-compatible resize (bilinear/bicubic, antialiased on downscale) so
//        HF-preprocessed reference outputs can be matched, + crop/pad policy
//     -> patchify + normalise (im2col in (c,py,px) order == flattened conv w)
//     -> patch embedding (GEMM), [CLS], + learned position embedding, [pre-LN]
//     -> N x pre-LN transformer block (bidirectional MHA + GELU MLP)
//     -> [post-LN] -> drop CLS -> optional pixel-shuffle token reduction
//     -> projector (none | linear | 2-layer MLP) to the LLM's n_embd
//
// This covers CLIP-ViT / SigLIP-ViT towers with LLaVA-style (linear/mlp2) or
// Idefics3/SmolVLM-style (pixel-shuffle + linear) connectors. It does NOT cover
// dynamic-resolution towers (Qwen2-VL / NaViT: no fixed position table) or
// cross-attention connectors (Flamingo/Perceiver resamplers).
//
// Optimisation notes
//   * all heavy ops are nn::linear (cache-blocked 6x16 AVX2 FMA GEMM; weights
//     are converted F16/Q8_0 -> F32 into an L1/L2-resident 16-column panel, so
//     a big F16 tower never needs an F32 copy in RAM).
//   * attention is blocked: per (head, 144-query block) the score tile
//     [144 x T] lives in a thread-local buffer (<= ~420 KB for T=729), QK^T and
//     PV both run through the same GEMM core (V is transposed once per head so
//     PV is also a "weights-row-contiguous" product). No TxT matrix is ever
//     materialised, and (head, block) tasks are spread across the ThreadPool.
//   * all scratch is allocated once in load(); encode() does not allocate.
// =============================================================================
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rawllm_nn.hpp"
#include "rawllm_mvae.hpp"    // TensorFile, from_ggml, row_view helpers
#include "rawllm_loader.hpp"

namespace vit {

using nn::Act;
using nn::Linear;

// -----------------------------------------------------------------------------
// Images
// -----------------------------------------------------------------------------
struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;     // h * w * 3, row-major
};

// Parses binary PPM (P6, maxval 255). Returns false on anything else.
inline bool parse_ppm(const std::string& b, Image& out) {
    size_t p = 0;
    auto skip = [&] {
        for (;;) {
            while (p < b.size() && std::isspace((unsigned char)b[p])) ++p;
            if (p < b.size() && b[p] == '#') { while (p < b.size() && b[p] != '\n') ++p; } else break;
        }
    };
    if (b.size() < 10 || b[0] != 'P' || b[1] != '6') return false;
    p = 2;
    long v[3];
    for (int i = 0; i < 3; ++i) {
        skip();
        if (p >= b.size() || !std::isdigit((unsigned char)b[p])) return false;
        long x = 0; while (p < b.size() && std::isdigit((unsigned char)b[p])) x = x * 10 + (b[p++] - '0');
        v[i] = x;
    }
    if (v[2] != 255 || v[0] <= 0 || v[1] <= 0) return false;
    ++p;   // single whitespace after maxval
    size_t need = (size_t)v[0] * (size_t)v[1] * 3;
    if (b.size() - p < need) return false;
    out.w = (int)v[0]; out.h = (int)v[1];
    out.rgb.assign((const uint8_t*)b.data() + p, (const uint8_t*)b.data() + p + need);
    return true;
}
inline bool load_ppm_file(const std::string& path, Image& out) {
    std::ifstream f(path, std::ios::binary); if (!f) return false;
    std::stringstream ss; ss << f.rdbuf(); return parse_ppm(ss.str(), out);
}

enum class Resample { Bilinear, Bicubic };

// Pillow-compatible separable resample (Resample.c: precompute_coeffs +
// horizontal pass then vertical pass, 8-bit intermediate). Antialiased when
// downscaling (filter support scales with the reduction factor), which is what
// HF image processors get from PIL. Integer fixed-point like Pillow's 8bpc path,
// so the result is meant to be bit-identical (verified in test_vit.py).
inline void pil_resize(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, Resample r) {
    auto filt = [&](double x) -> double {
        x = std::fabs(x);
        if (r == Resample::Bilinear) return x < 1.0 ? 1.0 - x : 0.0;
        const double a = -0.5;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
        return 0.0;
    };
    const double fsup = (r == Resample::Bilinear) ? 1.0 : 2.0;
    // Pillow's 8-bit path: weights are normalised in double, then quantised to 22-bit
    // fixed point, accumulated in int32 with a rounding bias, shifted back and clipped
    // — after EACH pass (horizontal first). Reproducing that exactly gives bit-identical
    // output to PIL.Image.resize (what HF image processors call).
    constexpr int PB = 22;
    struct Coef { int xmin, n; std::vector<int32_t> w; };
    auto make = [&](int in, int out) {
        std::vector<Coef> cs(out);
        double scale = (double)in / out, fscale = std::max(scale, 1.0), support = fsup * fscale, ss = 1.0 / fscale;
        for (int xx = 0; xx < out; ++xx) {
            double center = (xx + 0.5) * scale;
            int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
            int xmax = (int)(center + support + 0.5); if (xmax > in) xmax = in;
            xmax -= xmin;
            Coef c; c.xmin = xmin; c.n = xmax; c.w.assign(std::max(xmax, 1), 0);
            std::vector<double> k(std::max(xmax, 1), 0.0); double sum = 0;
            for (int x = 0; x < xmax; ++x) { k[x] = filt((x + xmin - center + 0.5) * ss); sum += k[x]; }
            for (int x = 0; x < xmax; ++x) {
                double kn = sum != 0 ? k[x] / sum : k[x];
                c.w[x] = kn < 0 ? (int32_t)(-0.5 + kn * (1 << PB)) : (int32_t)(0.5 + kn * (1 << PB));
            }
            cs[xx] = std::move(c);
        }
        return cs;
    };
    auto cx = make(sw, dw), cy = make(sh, dh);
    std::vector<uint8_t> tmp((size_t)dw * sh * 3);
    auto q = [](int32_t v) { v >>= PB; return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); };
    const int32_t bias = 1 << (PB - 1);
    for (int y = 0; y < sh; ++y)                                 // horizontal
        for (int x = 0; x < dw; ++x) {
            const Coef& c = cx[x]; int32_t s0 = bias, s1 = bias, s2 = bias;
            const uint8_t* row = src + (size_t)y * sw * 3;
            for (int k = 0; k < c.n; ++k) { const uint8_t* px = row + (size_t)(c.xmin + k) * 3; s0 += c.w[k] * px[0]; s1 += c.w[k] * px[1]; s2 += c.w[k] * px[2]; }
            uint8_t* o = tmp.data() + ((size_t)y * dw + x) * 3; o[0] = q(s0); o[1] = q(s1); o[2] = q(s2);
        }
    for (int y = 0; y < dh; ++y)                                 // vertical
        for (int x = 0; x < dw; ++x) {
            const Coef& c = cy[y]; int32_t s0 = bias, s1 = bias, s2 = bias;
            for (int k = 0; k < c.n; ++k) { const uint8_t* px = tmp.data() + ((size_t)(c.xmin + k) * dw + x) * 3; s0 += c.w[k] * px[0]; s1 += c.w[k] * px[1]; s2 += c.w[k] * px[2]; }
            uint8_t* o = dst + ((size_t)y * dw + x) * 3; o[0] = q(s0); o[1] = q(s1); o[2] = q(s2);
        }
}

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------
enum class Preprocess { Squash, ShortestCrop, PadSquare };
enum class ProjType { None, Linear, MLP2 };

struct Config {
    size_t image_size = 224, patch = 16, n_embd = 768, n_layer = 12, n_head = 12, n_ff = 3072, channels = 3;
    bool   use_cls = false, drop_cls = true, post_ln = true, pre_ln = false;
    float  ln_eps = 1e-6f;
    Act    act = Act::GeluTanh;
    int    feature_layer = -1;          // -1: all layers; -2: stop one layer early (LLaVA)
    size_t pixel_shuffle = 1;
    ProjType proj = ProjType::None;
    Act    proj_act = Act::GeluErf;
    Resample resample = Resample::Bicubic;
    Preprocess pre = Preprocess::Squash;
    float  mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
    size_t grid() const { return image_size / patch; }
    size_t n_patches() const { return grid() * grid(); }
    size_t n_tokens() const { return n_patches() + (use_cls ? 1 : 0); }
    size_t run_layers() const { long r = (long)n_layer + feature_layer + 1; return (size_t)std::max(0L, std::min((long)n_layer, r)); }
};

struct Options {
    bool  convert = false; nn::WType weights = nn::WType::F16;   // opt-in lossy re-encode at load
    util::ThreadPool* pool = nullptr;
};

// -----------------------------------------------------------------------------
// Encoder
// -----------------------------------------------------------------------------
class VisionEncoder {
public:
    Config cfg;

    void load(const loader::GGUFLoader& g, const Options& opt = {}) {
        opt_ = opt;
        if (g.meta_string("general.architecture") != "nanity-vit")
            throw std::runtime_error("vit: general.architecture != 'nanity-vit'");
        auto num = [&](const char* k, double d) { return g.meta_number(k, d); };
        auto str = [&](const char* k, const char* d) { return g.meta_string(k, d); };
        cfg.image_size = (size_t)num("nanity.vit.image_size", 0);
        cfg.patch      = (size_t)num("nanity.vit.patch_size", 0);
        cfg.n_embd     = (size_t)num("nanity.vit.n_embd", 0);
        cfg.n_layer    = (size_t)num("nanity.vit.n_layer", 0);
        cfg.n_head     = (size_t)num("nanity.vit.n_head", 0);
        cfg.n_ff       = (size_t)num("nanity.vit.n_ff", 0);
        cfg.channels   = (size_t)num("nanity.vit.n_channels", 3);
        cfg.use_cls    = num("nanity.vit.use_cls", 0) != 0;
        cfg.drop_cls   = num("nanity.vit.drop_cls", 1) != 0;
        cfg.pre_ln     = num("nanity.vit.pre_ln", 0) != 0;
        cfg.post_ln    = num("nanity.vit.post_ln", 1) != 0;
        cfg.ln_eps     = (float)num("nanity.vit.ln_eps", 1e-6);
        cfg.act        = nn::act_from_string(str("nanity.vit.act", "gelu_tanh"));
        cfg.feature_layer = (int)num("nanity.vit.feature_layer", -1);
        cfg.pixel_shuffle = (size_t)num("nanity.vit.pixel_shuffle", 1);
        cfg.proj_act   = nn::act_from_string(str("nanity.vit.proj_act", "gelu"));
        { std::string p = str("nanity.vit.projector", "none");
          cfg.proj = p == "none" ? ProjType::None : p == "linear" ? ProjType::Linear : p == "mlp2" ? ProjType::MLP2
                   : throw std::runtime_error("vit: unknown projector '" + p + "'"); }
        { std::string r = str("nanity.vit.resample", "bicubic");
          cfg.resample = r == "bilinear" ? Resample::Bilinear : Resample::Bicubic; }
        { std::string r = str("nanity.vit.preprocess", "squash");
          cfg.pre = r == "shortest_crop" ? Preprocess::ShortestCrop : r == "pad_square" ? Preprocess::PadSquare : Preprocess::Squash; }

        if (!cfg.image_size || !cfg.patch || cfg.image_size < cfg.patch) throw std::runtime_error("vit: bad image_size/patch_size");
        if (!cfg.n_embd || cfg.n_embd % cfg.n_head) throw std::runtime_error("vit: n_embd must be divisible by n_head");
        if (cfg.pixel_shuffle < 1 || cfg.grid() % cfg.pixel_shuffle) throw std::runtime_error("vit: pixel_shuffle must divide the patch grid");
        if (cfg.channels != 3) throw std::runtime_error("vit: only 3-channel RGB towers are supported");

        mvae::TensorFile tf(g);
        const size_t d = cfg.n_embd, T = cfg.n_tokens(), K = cfg.channels * cfg.patch * cfg.patch;
        patch_ = tf.linear("v.patch", K, d);
        if (cfg.use_cls) cls_ = tf.floats("v.cls");
        pos_ = tf.floats("v.pos");
        if (pos_.size() != T * d) throw std::runtime_error("vit: v.pos has " + std::to_string(pos_.size()) + " floats, expected " + std::to_string(T * d) + " (n_tokens*n_embd)");
        if (cfg.pre_ln) { pre_g_ = tf.floats("v.pre_ln.weight"); pre_b_ = tf.floats("v.pre_ln.bias"); }
        if (cfg.post_ln) { post_g_ = tf.floats("v.post_ln.weight"); post_b_ = tf.floats("v.post_ln.bias"); }
        if (tf.find("v.image_mean")) { auto m = tf.floats("v.image_mean"); auto s = tf.floats("v.image_std"); for (int i = 0; i < 3; ++i) { cfg.mean[i] = m[i]; cfg.stdv[i] = s[i]; } }
        blk_.resize(cfg.n_layer);
        for (size_t i = 0; i < cfg.n_layer; ++i) {
            std::string b = "v.blk." + std::to_string(i) + ".";
            auto& B = blk_[i];
            B.ln1_g = tf.floats(b + "ln1.weight"); B.ln1_b = tf.floats(b + "ln1.bias");
            B.ln2_g = tf.floats(b + "ln2.weight"); B.ln2_b = tf.floats(b + "ln2.bias");
            B.q = tf.linear(b + "attn_q", d, d); B.k = tf.linear(b + "attn_k", d, d);
            B.v = tf.linear(b + "attn_v", d, d); B.o = tf.linear(b + "attn_output", d, d);
            B.up = tf.linear(b + "ffn_up", d, cfg.n_ff); B.down = tf.linear(b + "ffn_down", cfg.n_ff, d);
        }
        size_t proj_in = d * cfg.pixel_shuffle * cfg.pixel_shuffle;
        out_dim_ = proj_in;
        if (cfg.proj != ProjType::None) { mm0_ = tf.linear("mm.0", proj_in); out_dim_ = mm0_.N; }
        if (cfg.proj == ProjType::MLP2) { mm2_ = tf.linear("mm.2", mm0_.N); out_dim_ = mm2_.N; }

        if (opt_.convert) {
            auto cv = [&](Linear& l) { if (!l.empty()) l.convert_to(opt_.weights); };
            cv(patch_); for (auto& B : blk_) { cv(B.q); cv(B.k); cv(B.v); cv(B.o); cv(B.up); cv(B.down); } cv(mm0_); cv(mm2_);
        }
        // scratch
        x_.assign(T * d, 0.f); h_.assign(T * d, 0.f); q_.assign(T * d, 0.f); k_.assign(T * d, 0.f);
        v_.assign(T * d, 0.f); vt_.assign(T * d, 0.f); a_.assign(T * d, 0.f); f_.assign(T * cfg.n_ff, 0.f);
        patches_.assign(cfg.n_patches() * K, 0.f);
        shuf_.assign(out_tokens() * proj_in, 0.f);
        mid_.assign(out_tokens() * (cfg.proj == ProjType::MLP2 ? mm0_.N : 0), 0.f);
        resized_.assign(cfg.image_size * cfg.image_size * 3, 0);
    }

    size_t out_tokens() const { size_t g = cfg.grid() / cfg.pixel_shuffle; return g * g + ((cfg.use_cls && !cfg.drop_cls) ? 1 : 0); }
    size_t out_dim() const { return out_dim_; }

    // ---- preprocessing: Image -> normalised patch matrix [n_patches, C*p*p] ----
    void preprocess(const Image& img, float* patches) {
        const size_t S = cfg.image_size;
        if (img.w <= 0 || img.h <= 0 || img.rgb.size() < (size_t)img.w * img.h * 3) throw std::runtime_error("vit: bad image");
        const uint8_t* src = img.rgb.data(); int sw = img.w, sh = img.h;
        std::vector<uint8_t> tmp;
        if (cfg.pre == Preprocess::PadSquare && sw != sh) {          // LLaVA-1.5 expand2square, bg = mean colour
            int m = std::max(sw, sh); tmp.resize((size_t)m * m * 3);
            uint8_t bg[3]; for (int c = 0; c < 3; ++c) bg[c] = (uint8_t)(int)(cfg.mean[c] * 255.f);
            for (size_t i = 0; i < (size_t)m * m; ++i) std::memcpy(&tmp[i * 3], bg, 3);
            int ox = (m - sw) / 2, oy = (m - sh) / 2;
            for (int y = 0; y < sh; ++y) std::memcpy(&tmp[((size_t)(y + oy) * m + ox) * 3], src + (size_t)y * sw * 3, (size_t)sw * 3);
            src = tmp.data(); sw = sh = m;
        }
        if (cfg.pre == Preprocess::ShortestCrop) {                   // CLIP: shortest side -> S, center crop SxS
            int nw, nh; if (sw <= sh) { nw = (int)S; nh = (int)((double)S * sh / sw); } else { nh = (int)S; nw = (int)((double)S * sw / sh); }
            std::vector<uint8_t> big((size_t)nw * nh * 3); pil_resize(src, sw, sh, big.data(), nw, nh, cfg.resample);
            int cx = (nw - (int)S) / 2, cy = (nh - (int)S) / 2;
            for (size_t y = 0; y < S; ++y) std::memcpy(&resized_[y * S * 3], &big[((size_t)(y + cy) * nw + cx) * 3], S * 3);
        } else {
            pil_resize(src, sw, sh, resized_.data(), (int)S, (int)S, cfg.resample);
        }
        const size_t g = cfg.grid(), p = cfg.patch, K = 3 * p * p;   // conv stride=patch, no padding: remainder rows/cols are ignored (e.g. 384/14 -> 27)
        float scale[3], bias[3]; for (int c = 0; c < 3; ++c) { scale[c] = 1.f / (255.f * cfg.stdv[c]); bias[c] = -cfg.mean[c] / cfg.stdv[c]; }
        for (size_t gy = 0; gy < g; ++gy) for (size_t gx = 0; gx < g; ++gx) {
            float* o = patches + (gy * g + gx) * K;
            for (size_t c = 0; c < 3; ++c) for (size_t py = 0; py < p; ++py) {
                const uint8_t* row = &resized_[((gy * p + py) * S + gx * p) * 3 + c];
                for (size_t px = 0; px < p; ++px) *o++ = row[px * 3] * scale[c] + bias[c];
            }
        }
    }

    // ---- full pipeline: Image -> out [out_tokens() x out_dim()] ----
    void encode(const Image& img, float* out) {
        preprocess(img, patches_.data());
        encode_patches(patches_.data(), out);
    }

    // ---- tower on an already-patchified, normalised matrix (used by tests) ----
    void encode_patches(const float* patches, float* out) {
        util::ThreadPool* pool = opt_.pool;
        const size_t d = cfg.n_embd, T = cfg.n_tokens(), np = cfg.n_patches(), K = cfg.channels * cfg.patch * cfg.patch;
        const size_t t0 = cfg.use_cls ? 1 : 0;
        nn::linear(patch_, patches, np, K, x_.data() + t0 * d, d, Act::None, pool);
        if (cfg.use_cls) std::memcpy(x_.data(), cls_.data(), d * sizeof(float));
        for (size_t t = 0; t < T; ++t) nn::add_inplace(x_.data() + t * d, pos_.data() + t * d, d);
        if (cfg.pre_ln) ln_rows(x_.data(), T, pre_g_.data(), pre_b_.data());

        for (size_t li = 0; li < cfg.run_layers(); ++li) {
            auto& B = blk_[li];
            std::memcpy(h_.data(), x_.data(), T * d * sizeof(float));
            ln_rows(h_.data(), T, B.ln1_g.data(), B.ln1_b.data());
            nn::linear(B.q, h_.data(), T, d, q_.data(), d, Act::None, pool);
            nn::linear(B.k, h_.data(), T, d, k_.data(), d, Act::None, pool);
            nn::linear(B.v, h_.data(), T, d, v_.data(), d, Act::None, pool);
            attention(T);
            nn::linear(B.o, a_.data(), T, d, h_.data(), d, Act::None, pool);
            for (size_t t = 0; t < T; ++t) nn::add_inplace(x_.data() + t * d, h_.data() + t * d, d);
            std::memcpy(h_.data(), x_.data(), T * d * sizeof(float));
            ln_rows(h_.data(), T, B.ln2_g.data(), B.ln2_b.data());
            nn::linear(B.up, h_.data(), T, d, f_.data(), cfg.n_ff, cfg.act, pool);
            nn::linear(B.down, f_.data(), T, cfg.n_ff, h_.data(), d, Act::None, pool);
            for (size_t t = 0; t < T; ++t) nn::add_inplace(x_.data() + t * d, h_.data() + t * d, d);
        }
        if (cfg.post_ln) ln_rows(x_.data(), T, post_g_.data(), post_b_.data());

        // drop CLS / pixel-shuffle into shuf_ (token-major, width d*s*s)
        const size_t s = cfg.pixel_shuffle, g = cfg.grid(), go = g / s, pin = d * s * s;
        const float* grid = x_.data() + ((cfg.use_cls && cfg.drop_cls) ? d : 0);
        const bool keep_cls = cfg.use_cls && !cfg.drop_cls;
        float* dst = shuf_.data(); const float* feat = nullptr;
        size_t n_out = go * go;
        if (keep_cls) { std::memcpy(dst, x_.data(), d * sizeof(float)); dst += pin; /* CLS only valid when s == 1 */ if (s != 1) throw std::runtime_error("vit: keep-CLS with pixel_shuffle is not supported"); grid = x_.data() + d; }
        for (size_t h2 = 0; h2 < go; ++h2) for (size_t w2 = 0; w2 < go; ++w2) {
            float* o = dst + (h2 * go + w2) * pin;
            for (size_t b = 0; b < s; ++b) for (size_t a = 0; a < s; ++a) {
                feat = grid + ((h2 * s + b) * g + (w2 * s + a)) * d;
                std::memcpy(o + (b * s + a) * d, feat, d * sizeof(float));
            }
        }
        const size_t n_tok = n_out + (keep_cls ? 1 : 0);
        if (cfg.proj == ProjType::None) { std::memcpy(out, shuf_.data(), n_tok * pin * sizeof(float)); return; }
        if (cfg.proj == ProjType::Linear) { nn::linear(mm0_, shuf_.data(), n_tok, pin, out, mm0_.N, Act::None, pool); return; }
        nn::linear(mm0_, shuf_.data(), n_tok, pin, mid_.data(), mm0_.N, cfg.proj_act, pool);
        nn::linear(mm2_, mid_.data(), n_tok, mm0_.N, out, mm2_.N, Act::None, pool);
    }

private:
    struct Block { std::vector<float> ln1_g, ln1_b, ln2_g, ln2_b; Linear q, k, v, o, up, down; };

    void ln_rows(float* x, size_t rows, const float* g, const float* b) {
        const size_t d = cfg.n_embd; const float eps = cfg.ln_eps;
        nn::parallel_for(opt_.pool, rows, 32, [&](size_t r0, size_t r1) { for (size_t r = r0; r < r1; ++r) nn::layernorm_row(x + r * d, d, g, b, eps); });
    }

    // bidirectional MHA over q_/k_/v_ ([T,d], head h = columns [h*hd,(h+1)*hd)) -> a_
    void attention(size_t T) {
        const size_t d = cfg.n_embd, H = cfg.n_head, hd = d / H;
        const float scale = 1.f / std::sqrt((float)hd);
        // transpose V per head: vt_[h][j][t] = v_[t][h*hd+j]
        nn::parallel_for(opt_.pool, H, 1, [&](size_t h0, size_t h1) {
            for (size_t h = h0; h < h1; ++h) {
                float* vt = vt_.data() + h * hd * T;
                for (size_t t = 0; t < T; ++t) { const float* src = v_.data() + t * d + h * hd; for (size_t j = 0; j < hd; ++j) vt[j * T + t] = src[j]; }
            }
        });
        constexpr size_t QB = 144;                             // multiple of the 6-row microkernel
        const size_t nqb = (T + QB - 1) / QB, tasks = H * nqb;
        nn::parallel_for(opt_.pool, tasks, 1, [&](size_t i0, size_t i1) {
            thread_local std::vector<float> S;
            if (S.size() < QB * T) S.resize(QB * T);
            for (size_t ti = i0; ti < i1; ++ti) {
                const size_t h = ti / nqb, qb = (ti % nqb) * QB, m = std::min(QB, T - qb);
                Linear Kh; Kh.type = nn::WType::F32; Kh.N = T; Kh.K = hd; Kh.w = (const uint8_t*)(k_.data() + h * hd); Kh.row_stride = d * sizeof(float);
                nn::gemm_range(Kh, q_.data() + qb * d + h * hd, m, d, S.data(), T, 0, T, Act::None);
                for (size_t r = 0; r < m; ++r) nn::softmax_scaled(S.data() + r * T, T, scale);
                Linear Vh; Vh.type = nn::WType::F32; Vh.N = hd; Vh.K = T; Vh.w = (const uint8_t*)(vt_.data() + h * hd * T); Vh.row_stride = T * sizeof(float);
                nn::gemm_range(Vh, S.data(), m, T, a_.data() + qb * d + h * hd, d, 0, hd, Act::None);
            }
        });
    }

    Options opt_;
    size_t out_dim_ = 0;
    Linear patch_, mm0_, mm2_;
    std::vector<float> cls_, pos_, pre_g_, pre_b_, post_g_, post_b_;
    std::vector<Block> blk_;
    std::vector<float> x_, h_, q_, k_, v_, vt_, a_, f_, patches_, shuf_, mid_;
    std::vector<uint8_t> resized_;
};

} // namespace vit
