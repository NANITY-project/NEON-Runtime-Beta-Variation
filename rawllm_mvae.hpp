#pragma once
// =============================================================================
// rawllm_mvae.hpp — MVAE = Motion Variational AutoEncoder inference (Ling et al. 2020)
//
// Implements the inference graphs of Ling et al. 2020 "Character Controllers
// Using Motion VAEs" as shipped in electronicarts/character-motion-vaes:
//
//   * PoseMixtureVAE  (MixedDecoder: 3 blended-expert layers + gating MLP)
//   * PoseVAE         (plain decoder; z and condition concatenated per layer)
//   * controller MLP  (PoseVAEController actor: ReLU stack -> tanh latent z)
//
// plus the encoder (for encoding a mocap frame into z — e.g. to imitate a
// reference clip, or to seed z from a recorded pose).
//
// ---- How it is optimised (all exact unless an opt-in flag says otherwise) ---
//
// 1. MoE layers are computed as ONE stacked GEMV, then blended.
//    PyTorch's MixedDecoder blends the *weights*: W_mix = sum_e a_e W_e, then
//    x @ W_mix. That writes and re-reads a (in x out) matrix per layer per
//    frame (for the 267-pose checkpoint: 3 layers x ~75K floats of scratch
//    traffic on top of the actual matmul). Since it is linear in W,
//        x @ (sum_e a_e W_e) + sum_e a_e b_e  ==  sum_e a_e (x @ W_e + b_e)
//    so we run x against the stacked [E*out, in] matrix once (a plain GEMV that
//    only *reads* weights, and that works unchanged with F16/Q8_0 storage), and
//    blend the E partial outputs. Same FLOPs, no mixed-weight materialisation.
//
// 2. Concatenations are never built: [z | h] lives in one preallocated buffer
//    whose z half is written once per frame and whose h half is overwritten
//    layer by layer.
//
// 3. Zero heap allocation per step; single-threaded by default (a pool barrier
//    costs more than a whole 267-pose frame), optional pool for big models.
//
// ---- Opt-in lossy knobs (off by default, per NEON's convention) ------------
//   Options::weights   : re-encode weights F32 -> F16 / BF16 / Q8_0 at load
//   Options::moe_top_k : evaluate only the k highest-gate experts per frame
//                        (renormalised). 0 = all experts = exact.
// =============================================================================
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rawllm_nn.hpp"
#include "rawllm_loader.hpp"

namespace mvae {

using nn::Act;
using nn::Linear;
using nn::WType;

inline WType from_ggml(loader::GGMLType t) {
    switch (t) {
        case loader::GGMLType::F32:  return WType::F32;
        case loader::GGMLType::F16:  return WType::F16;
        case loader::GGMLType::BF16: return WType::BF16;
        case loader::GGMLType::Q8_0: return WType::Q8_0;
        default: throw std::runtime_error("mvae: unsupported tensor type (use F32/F16/BF16/Q8_0)");
    }
}

// GGUF tensor lookup helpers (shape[0] = in/K, shape[1] = out/N, like PyTorch Linear)
struct TensorFile {
    const loader::GGUFLoader& g;
    explicit TensorFile(const loader::GGUFLoader& gg) : g(gg) {}
    const loader::TensorInfo* find(const std::string& n) const {
        for (auto& t : g.tensors) if (t.name == n) return &t;
        return nullptr;
    }
    const loader::TensorInfo& need(const std::string& n) const {
        auto* t = find(n); if (!t) throw std::runtime_error("mvae: missing tensor '" + n + "'"); return *t;
    }
    // Copy any float-typed tensor to F32 (for norm stats, biases)
    std::vector<float> floats(const std::string& n) const {
        const auto& t = need(n);
        size_t cnt = 1; for (auto d : t.shape) cnt *= (size_t)d;
        Linear tmp; tmp.borrow(from_ggml(t.type), t.data_ptr, 1, cnt);
        std::vector<float> out(cnt); tmp.dequant_row(0, out.data()); return out;
    }
    Linear linear(const std::string& stem, size_t expect_k = 0, size_t expect_n = 0) const {
        const auto& w = need(stem + ".weight");
        if (w.shape.size() != 2) throw std::runtime_error("mvae: " + stem + ".weight must be 2-D");
        Linear L; L.borrow(from_ggml(w.type), w.data_ptr, (size_t)w.shape[1], (size_t)w.shape[0]);
        if ((expect_k && L.K != expect_k) || (expect_n && L.N != expect_n))
            throw std::runtime_error("mvae: " + stem + ".weight has shape [K=" + std::to_string(L.K) + ",N=" +
                                     std::to_string(L.N) + "], expected K=" + std::to_string(expect_k) +
                                     " N=" + std::to_string(expect_n));
        if (find(stem + ".bias")) {
            auto bv = floats(stem + ".bias");
            if (bv.size() != L.N) throw std::runtime_error("mvae: " + stem + ".bias size mismatch");
            L.set_bias_copy(bv.data(), bv.size());
        }
        return L;
    }
};

// rows [n0, n0+n) of a Linear as a borrowed Linear (used for top-k expert subsets)
inline Linear row_view(const Linear& L, size_t n0, size_t n) {
    Linear v; v.type = L.type; v.N = n; v.K = L.K; v.w = L.w + n0 * L.row_stride; v.row_stride = L.row_stride;
    v.bias = L.bias ? L.bias + n0 : nullptr;
    return v;
}

struct Options {
    bool        convert = false;                 // opt-in lossy re-encode of weights
    WType       weights = WType::F32;            // target type when convert==true
    int         moe_top_k = 0;                   // 0 = exact (all experts)
    util::ThreadPool* pool = nullptr;            // only worth it for very large nets
};

enum class Kind { PoseMixtureVAE, PoseVAE, MLP };

class MotionVAE {
public:
    // ---- model description (read from GGUF metadata) ----
    Kind   kind = Kind::PoseMixtureVAE;
    size_t frame_size = 0, latent_size = 0, n_cond = 1, n_future = 1, n_experts = 0;
    size_t hidden = 256, gate_hidden = 64;
    std::string norm_mode = "zscore";            // zscore | minmax | none

    void load(const loader::GGUFLoader& g, const Options& opt = {}) {
        opt_ = opt;
        auto str = [&](const char* k, const char* d) { return g.meta_string(k, d); };
        auto num = [&](const char* k, double d) { return g.meta_number(k, d); };
        if (str("general.architecture", "") != "nanity-mvae")
            throw std::runtime_error("mvae: general.architecture != 'nanity-mvae'");
        std::string k = str("nanity.mvae.kind", "pose_mixture_vae");
        if      (k == "pose_mixture_vae") kind = Kind::PoseMixtureVAE;
        else if (k == "pose_vae")         kind = Kind::PoseVAE;
        else if (k == "mlp")              kind = Kind::MLP;
        else throw std::runtime_error("mvae: unknown nanity.mvae.kind '" + k + "'");
        TensorFile tf(g);

        if (kind == Kind::MLP) {                 // controller / actor
            size_t nl = (size_t)num("nanity.mvae.n_layers", 0);
            if (!nl) throw std::runtime_error("mvae: mlp needs nanity.mvae.n_layers");
            act_ = nn::act_from_string(str("nanity.mvae.act", "relu"));
            final_act_ = nn::act_from_string(str("nanity.mvae.final_act", "none"));
            size_t maxw = 0;
            for (size_t i = 0; i < nl; ++i) {
                mlp_.push_back(tf.linear("mlp." + std::to_string(i)));
                if (i && mlp_[i].K != mlp_[i - 1].N) throw std::runtime_error("mvae: mlp layer width mismatch");
                maxw = std::max({maxw, mlp_[i].K, mlp_[i].N});
            }
            maybe_convert(mlp_);
            in_dim_ = mlp_.front().K; out_dim_ = mlp_.back().N;
            ping_.assign(maxw, 0.f); pong_.assign(maxw, 0.f);
            return;
        }

        frame_size  = (size_t)num("nanity.mvae.frame_size", 0);
        latent_size = (size_t)num("nanity.mvae.latent_size", 0);
        n_cond      = (size_t)num("nanity.mvae.num_condition_frames", 1);
        n_future    = (size_t)num("nanity.mvae.num_future_predictions", 1);
        n_experts   = (size_t)num("nanity.mvae.num_experts", 0);
        hidden      = (size_t)num("nanity.mvae.hidden", 256);
        gate_hidden = (size_t)num("nanity.mvae.gate_hidden", 64);
        norm_mode   = str("nanity.mvae.normalization", "zscore");
        if (!frame_size || !latent_size) throw std::runtime_error("mvae: frame_size / latent_size missing");
        if (kind == Kind::PoseMixtureVAE && n_experts < 1) throw std::runtime_error("mvae: num_experts missing");

        if (norm_mode == "zscore") { avg_ = tf.floats("norm.avg"); std_ = tf.floats("norm.std"); }
        else if (norm_mode == "minmax") { min_ = tf.floats("norm.min"); max_ = tf.floats("norm.max"); }
        else if (norm_mode != "none") throw std::runtime_error("mvae: unknown normalization '" + norm_mode + "'");
        auto chk = [&](const std::vector<float>& v) { if (!v.empty() && v.size() != frame_size) throw std::runtime_error("mvae: norm stat size != frame_size"); };
        chk(avg_); chk(std_); chk(min_); chk(max_);

        const size_t cond_dim = frame_size * n_cond, out_dim = frame_size * n_future;
        // encoder is optional (a deployment-only export may omit it)
        if (tf.find("enc.fc1.weight")) {
            enc1_ = tf.linear("enc.fc1", out_dim + cond_dim, hidden);
            enc2_ = tf.linear("enc.fc2", frame_size + hidden, hidden);
            enc_mu_ = tf.linear("enc.mu", frame_size + hidden, latent_size);
            enc_lv_ = tf.linear("enc.logvar", frame_size + hidden, latent_size);
            has_encoder_ = true;
        }
        if (kind == Kind::PoseMixtureVAE) {
            const size_t E = n_experts;
            l_[0] = tf.linear("dec.l0", latent_size + cond_dim, E * hidden);
            l_[1] = tf.linear("dec.l1", latent_size + hidden,   E * hidden);
            l_[2] = tf.linear("dec.l2", latent_size + hidden,   E * out_dim);
            g_[0] = tf.linear("dec.gate.0", latent_size + cond_dim, gate_hidden);
            g_[1] = tf.linear("dec.gate.2", gate_hidden, gate_hidden);
            g_[2] = tf.linear("dec.gate.4", gate_hidden, E);
            if (!l_[0].bias || !l_[1].bias || !l_[2].bias) throw std::runtime_error("mvae: MoE layers need biases");
        } else {
            l_[0] = tf.linear("dec.fc4", latent_size + cond_dim, hidden);
            l_[1] = tf.linear("dec.fc5", latent_size + hidden,   hidden);
            l_[2] = tf.linear("dec.out", latent_size + hidden,   out_dim);
        }
        { std::vector<Linear*> all{&l_[0], &l_[1], &l_[2], &g_[0], &g_[1], &g_[2], &enc1_, &enc2_, &enc_mu_, &enc_lv_};
          if (opt_.convert) for (auto* p : all) if (!p->empty()) p->convert_to(opt_.weights); }

        // scratch (sized once)
        const size_t E = std::max<size_t>(1, n_experts);
        zc_.assign(latent_size + std::max(cond_dim, hidden), 0.f);       // [z | c-or-h]
        stack_.assign(E * std::max(hidden, out_dim), 0.f);
        h_.assign(std::max(hidden, out_dim), 0.f);
        gate_a_.assign(gate_hidden, 0.f); gate_b_.assign(gate_hidden, 0.f);
        gate_in_.assign(latent_size + cond_dim, 0.f);
        coef_.assign(E, 0.f);
        enc_in_.assign(out_dim + cond_dim, 0.f);
        enc_h1_.assign(hidden, 0.f); enc_s_.assign(frame_size + hidden, 0.f);
        norm_cond_.assign(cond_dim, 0.f);
        out_norm_.assign(out_dim, 0.f);
        hist_.assign(cond_dim, 0.f);                                      // raw frames, newest first
    }

    // ---- MLP / controller ----
    size_t mlp_in() const { return in_dim_; }
    size_t mlp_out() const { return out_dim_; }
    // obs -> action (e.g. 269 -> 32 latent z in [-1,1]); no allocation
    void mlp_forward(const float* x, float* y) {
        const float* cur = x; float* bufs[2] = {ping_.data(), pong_.data()};
        for (size_t i = 0; i < mlp_.size(); ++i) {
            const bool last = (i + 1 == mlp_.size());
            float* dst = last ? y : bufs[i & 1];
            nn::linear(mlp_[i], cur, 1, mlp_[i].K, dst, mlp_[i].N, last ? final_act_ : act_, opt_.pool);
            cur = dst;
        }
    }

    // ---- normalisation ----
    void normalize(const float* raw, float* out, size_t n_frames = 1) const {
        for (size_t f = 0; f < n_frames; ++f)
            for (size_t i = 0; i < frame_size; ++i) {
                float v = raw[f * frame_size + i];
                if (norm_mode == "zscore") v = (v - avg_[i]) / std_[i];
                else if (norm_mode == "minmax") v = 2.f * (v - min_[i]) / (max_[i] - min_[i]) - 1.f;
                out[f * frame_size + i] = v;
            }
    }
    void denormalize(const float* in, float* raw, size_t n_frames = 1) const {
        for (size_t f = 0; f < n_frames; ++f)
            for (size_t i = 0; i < frame_size; ++i) {
                float v = in[f * frame_size + i];
                if (norm_mode == "zscore") v = v * std_[i] + avg_[i];
                else if (norm_mode == "minmax") v = (v + 1.f) * (max_[i] - min_[i]) * 0.5f + min_[i];
                raw[f * frame_size + i] = v;
            }
    }

    // ---- decoder (normalised space) ----
    // z: [latent]; cond: [n_cond*frame] normalised, newest frame first (matches
    // the reference env's history layout); out: [n_future*frame] normalised.
    void decode(const float* z, const float* cond, float* out) {
        if (kind == Kind::MLP) throw std::runtime_error("mvae: decode() on an MLP model");
        const size_t L = latent_size, cond_dim = frame_size * n_cond, out_dim = frame_size * n_future;
        std::memcpy(zc_.data(), z, L * sizeof(float));
        std::memcpy(zc_.data() + L, cond, cond_dim * sizeof(float));
        if (kind == Kind::PoseVAE) {
            nn::linear(l_[0], zc_.data(), 1, L + cond_dim, h_.data(), hidden, Act::ELU, opt_.pool);
            std::memcpy(zc_.data() + L, h_.data(), hidden * sizeof(float));
            nn::linear(l_[1], zc_.data(), 1, L + hidden, h_.data(), hidden, Act::ELU, opt_.pool);
            std::memcpy(zc_.data() + L, h_.data(), hidden * sizeof(float));
            nn::linear(l_[2], zc_.data(), 1, L + hidden, out, out_dim, Act::None, opt_.pool);
            return;
        }
        // --- mixture decoder ---
        // gate input is [z | cond] — exactly zc_ at this point
        nn::linear(g_[0], zc_.data(), 1, L + cond_dim, gate_a_.data(), gate_hidden, Act::ELU);
        nn::linear(g_[1], gate_a_.data(), 1, gate_hidden, gate_b_.data(), gate_hidden, Act::ELU);
        nn::linear(g_[2], gate_b_.data(), 1, gate_hidden, coef_.data(), n_experts, Act::None);
        softmax_inplace(coef_.data(), n_experts);
        select_experts();

        moe_layer(0, L + cond_dim, hidden, Act::ELU, h_.data());
        std::memcpy(zc_.data() + L, h_.data(), hidden * sizeof(float));
        moe_layer(1, L + hidden, hidden, Act::ELU, h_.data());
        std::memcpy(zc_.data() + L, h_.data(), hidden * sizeof(float));
        moe_layer(2, L + hidden, out_dim, Act::None, out);
    }

    // ---- encoder (normalised space): x = target frame(s), cond as above ----
    void encode(const float* x, const float* cond, float* mu, float* logvar) {
        if (!has_encoder_) throw std::runtime_error("mvae: this file has no encoder tensors");
        const size_t cond_dim = frame_size * n_cond, out_dim = frame_size * n_future;
        std::memcpy(enc_in_.data(), x, out_dim * sizeof(float));
        std::memcpy(enc_in_.data() + out_dim, cond, cond_dim * sizeof(float));
        nn::linear(enc1_, enc_in_.data(), 1, out_dim + cond_dim, enc_h1_.data(), hidden, Act::ELU, opt_.pool);
        // fc2 input = [x | h1]  (x = the n_future*frame vector; reference uses frame_size, n_future==1)
        std::memcpy(enc_s_.data(), x, frame_size * sizeof(float));
        std::memcpy(enc_s_.data() + frame_size, enc_h1_.data(), hidden * sizeof(float));
        nn::linear(enc2_, enc_s_.data(), 1, frame_size + hidden, enc_h1_.data(), hidden, Act::ELU, opt_.pool);
        std::memcpy(enc_s_.data() + frame_size, enc_h1_.data(), hidden * sizeof(float));
        nn::linear(enc_mu_, enc_s_.data(), 1, frame_size + hidden, mu, latent_size, Act::None, opt_.pool);
        nn::linear(enc_lv_, enc_s_.data(), 1, frame_size + hidden, logvar, latent_size, Act::None, opt_.pool);
    }

    // ---- stateful rollout (what the animation engine calls every frame) ----
    // Seed the condition history with `n_cond` raw frames, NEWEST FIRST.
    void reset_history(const float* raw_frames_newest_first) {
        std::memcpy(hist_.data(), raw_frames_newest_first, frame_size * n_cond * sizeof(float));
    }
    // One autoregressive step: normalise history -> decode(z) -> denormalise.
    // next_raw: [n_future*frame]; the first predicted frame is pushed into the history.
    void step(const float* z, float* next_raw) {
        normalize(hist_.data(), norm_cond_.data(), n_cond);
        decode(z, norm_cond_.data(), out_norm_.data());
        denormalize(out_norm_.data(), next_raw, n_future);
        // shift history (newest first)
        if (n_cond > 1) std::memmove(hist_.data() + frame_size, hist_.data(), frame_size * (n_cond - 1) * sizeof(float));
        std::memcpy(hist_.data(), next_raw, frame_size * sizeof(float));
    }
    const float* history() const { return hist_.data(); }

    bool has_encoder() const { return has_encoder_; }

    // Standard-normal prior sample, for free-running generation.
    template <typename Rng>
    void sample_prior(Rng& rng, float* z) const {
        std::normal_distribution<float> nd(0.f, 1.f);
        for (size_t i = 0; i < latent_size; ++i) z[i] = nd(rng);
    }

private:
    static void softmax_inplace(float* x, size_t n) {
        float mx = x[0]; for (size_t i = 1; i < n; ++i) mx = std::max(mx, x[i]);
        float s = 0; for (size_t i = 0; i < n; ++i) { x[i] = std::exp(x[i] - mx); s += x[i]; }
        for (size_t i = 0; i < n; ++i) x[i] /= s;
    }

    // Fill sel_ with experts to evaluate, and re-normalise coef_ over them if pruned.
    void select_experts() {
        sel_.clear();
        const size_t E = n_experts;
        if (opt_.moe_top_k <= 0 || (size_t)opt_.moe_top_k >= E) { for (size_t e = 0; e < E; ++e) sel_.push_back(e); return; }
        std::array<size_t, 64> idx{}; if (E > idx.size()) throw std::runtime_error("mvae: too many experts for top-k");
        for (size_t e = 0; e < E; ++e) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + opt_.moe_top_k, idx.begin() + E,
                          [&](size_t a, size_t b) { return coef_[a] > coef_[b]; });
        float s = 0; for (int i = 0; i < opt_.moe_top_k; ++i) s += coef_[idx[i]];
        for (int i = 0; i < opt_.moe_top_k; ++i) sel_.push_back(idx[i]);
        std::sort(sel_.begin(), sel_.end());
        for (size_t e : sel_) coef_[e] /= s;
    }

    // out[j] = act( sum_e a_e * (x . W_e[j,:] + b_e[j]) )   — see header comment.
    void moe_layer(int li, size_t in_k, size_t out_n, Act act, float* out) {
        const Linear& S = l_[li];
        if (sel_.size() == n_experts) {
            nn::linear(S, zc_.data(), 1, in_k, stack_.data(), n_experts * out_n, Act::None, opt_.pool);
            blend(stack_.data(), out_n, out, /*all=*/true);
        } else {
            size_t slot = 0;
            for (size_t e : sel_) {
                Linear v = row_view(S, e * out_n, out_n);
                nn::linear(v, zc_.data(), 1, in_k, stack_.data() + slot * out_n, out_n, Act::None);
                ++slot;
            }
            blend(stack_.data(), out_n, out, /*all=*/false);
        }
        nn::apply_act(act, out, out_n);
    }
    void blend(const float* Y, size_t out_n, float* out, bool all) const {
        const size_t cnt = sel_.size();
        for (size_t slot = 0; slot < cnt; ++slot) {
            const float a = coef_[all ? sel_[slot] : sel_[slot]];
            const float* y = Y + (all ? sel_[slot] : slot) * out_n;
            size_t j = 0;
#ifdef NN_AVX2
            __m256 va = _mm256_set1_ps(a);
            if (slot == 0) for (; j + 8 <= out_n; j += 8) _mm256_storeu_ps(out + j, _mm256_mul_ps(va, _mm256_loadu_ps(y + j)));
            else           for (; j + 8 <= out_n; j += 8) _mm256_storeu_ps(out + j, _mm256_fmadd_ps(va, _mm256_loadu_ps(y + j), _mm256_loadu_ps(out + j)));
#endif
            for (; j < out_n; ++j) out[j] = (slot == 0 ? 0.f : out[j]) + a * y[j];
        }
    }

    void maybe_convert(std::vector<Linear>& v) { if (opt_.convert) for (auto& l : v) l.convert_to(opt_.weights); }

    Options opt_;
    Linear l_[3], g_[3], enc1_, enc2_, enc_mu_, enc_lv_;
    std::vector<Linear> mlp_;
    Act act_ = Act::ReLU, final_act_ = Act::None;
    size_t in_dim_ = 0, out_dim_ = 0;
    bool has_encoder_ = false;
    std::vector<float> avg_, std_, min_, max_;
    std::vector<float> zc_, stack_, h_, gate_a_, gate_b_, gate_in_, coef_, enc_in_, enc_h1_, enc_s_, norm_cond_, out_norm_, hist_, ping_, pong_;
    std::vector<size_t> sel_;
};

} // namespace mvae
