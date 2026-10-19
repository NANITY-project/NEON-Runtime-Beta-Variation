#pragma once
// =============================================================================
// rawllm_nn.hpp — NN = neural-network kernels: GEMM/GEMV, LayerNorm, softmax, activations
// (rawllm_mvae.hpp) and the vision encoder (rawllm_vit.hpp).
//
// Scope: inference only, CPU only, no dependencies beyond rawllm_util.hpp's
// ThreadPool. Written in the same spirit as the rest of NEON: explicit,
// from scratch, with a scalar fallback for every SIMD path so the result can
// be diffed against the vector path (build once with -mavx2 -mfma -mf16c, once
// without).
//
// Weight layout (everything here): W is [N rows = out_features][K cols =
// in_features], K contiguous — i.e. exactly what GGUF stores for a PyTorch
// nn.Linear (GGUF shape = [K, N]). Y[m,n] = sum_k X[m,k] * W[n,k] (+ bias[n]).
//
// Two execution paths, picked by M (number of input rows):
//   * M <= kGemvMaxM : "GEMV" path. Memory-bound; reads each weight row once,
//     converting F16/BF16/Q8_0 -> F32 in registers, sharing the activation
//     loads across 4 weight rows. This is the Motion-VAE per-frame path.
//   * M  > kGemvMaxM : "GEMM" path. Compute-bound; cache-blocked 6x16 AVX2
//     FMA microkernel over a packed [K][16] weight panel. The panel is built
//     from F32/F16/BF16/Q8_0 on the fly (KC x 16 floats = 16 KB, L1/L2
//     resident), so a 400M-param F16 vision tower never needs a full F32
//     copy in RAM. This is the vision-encoder path.
//
// Lossy conversions are never silent: Linear::convert_to() is an explicit
// call, and the callers expose it behind opt-in flags.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#  include <immintrin.h>
#  define NN_AVX2 1
#endif
#if defined(NN_AVX2) && defined(__F16C__)
#  define NN_F16C 1
#endif

#include "rawllm_util.hpp"

namespace nn {

// ----------------------------------------------------------------------------
// scalar helpers
// ----------------------------------------------------------------------------
inline float f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) out = sign;
        else {                                   // subnormal: renormalise
            int e = -1;
            do { ++e; man <<= 1; } while (!(man & 0x400u));
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 31) out = sign | 0x7F800000u | (man << 13);
    else out = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    float f; std::memcpy(&f, &out, 4); return f;
}
inline uint16_t f32_to_f16_scalar(float f) {     // round-to-nearest-even
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e    = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0));
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t h = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t h = sign | ((uint32_t)e << 10) | (man >> 13);
    uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t)h;
}
inline float bf16_to_f32_scalar(uint16_t b) {
    uint32_t x = (uint32_t)b << 16; float f; std::memcpy(&f, &x, 4); return f;
}

// ----------------------------------------------------------------------------
// vector exp / tanh (AVX2) — Cephes-style, ~1e-7 relative error on [-87, 88]
// ----------------------------------------------------------------------------
#ifdef NN_AVX2
inline __m256 exp256(__m256 x) {
    const __m256 hi = _mm256_set1_ps(88.3762626647949f), lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
inline float hmax256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
#endif

// ----------------------------------------------------------------------------
// activations (in place over a contiguous range)
// ----------------------------------------------------------------------------
enum class Act : uint8_t { None, ReLU, ELU, SiLU, GeluTanh, GeluQuick, GeluErf, Tanh };

inline Act act_from_string(const std::string& s) {
    if (s.empty() || s == "none" || s == "linear") return Act::None;
    if (s == "relu") return Act::ReLU;
    if (s == "elu") return Act::ELU;
    if (s == "silu" || s == "swish") return Act::SiLU;
    if (s == "gelu_tanh" || s == "gelu_pytorch_tanh" || s == "gelu_new") return Act::GeluTanh;
    if (s == "gelu_quick" || s == "quick_gelu") return Act::GeluQuick;
    if (s == "gelu" || s == "gelu_erf") return Act::GeluErf;
    if (s == "tanh") return Act::Tanh;
    throw std::runtime_error("nn: unknown activation '" + s + "'");
}

inline float act_scalar(Act a, float x) {
    switch (a) {
        case Act::None: return x;
        case Act::ReLU: return x > 0.f ? x : 0.f;
        case Act::ELU:  return x > 0.f ? x : std::expm1(x);
        case Act::SiLU: return x / (1.f + std::exp(-x));
        case Act::GeluTanh: {
            float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.f + std::tanh(u));
        }
        case Act::GeluQuick: return x / (1.f + std::exp(-1.702f * x));
        case Act::GeluErf: return 0.5f * x * (1.f + std::erf(x * 0.70710678118654752f));
        case Act::Tanh: return std::tanh(x);
    }
    return x;
}

inline void apply_act(Act a, float* p, size_t n) {
    if (a == Act::None) return;
    size_t i = 0;
#ifdef NN_AVX2
    const __m256 one = _mm256_set1_ps(1.f), zero = _mm256_setzero_ps();
    switch (a) {
        case Act::ReLU:
            for (; i + 8 <= n; i += 8) _mm256_storeu_ps(p + i, _mm256_max_ps(_mm256_loadu_ps(p + i), zero));
            break;
        case Act::ELU:
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(p + i);
                __m256 em1 = _mm256_sub_ps(exp256(_mm256_min_ps(x, zero)), one);   // exp(min(x,0))-1
                // exp(x)-1 loses precision near 0; the 1e-7 rel-error of exp256 is
                // absolute ~1e-7 here, fine for activations.
                __m256 m = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
                _mm256_storeu_ps(p + i, _mm256_blendv_ps(em1, x, m));
            }
            break;
        case Act::SiLU:
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(p + i);
                __m256 e = exp256(_mm256_sub_ps(zero, x));
                _mm256_storeu_ps(p + i, _mm256_div_ps(x, _mm256_add_ps(one, e)));
            }
            break;
        case Act::GeluQuick:
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(p + i);
                __m256 e = exp256(_mm256_mul_ps(x, _mm256_set1_ps(-1.702f)));
                _mm256_storeu_ps(p + i, _mm256_div_ps(x, _mm256_add_ps(one, e)));
            }
            break;
        case Act::GeluTanh:
            // 0.5*x*(1+tanh(u)) = x * sigmoid(2u) = x / (1 + exp(-2u))
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(p + i);
                __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
                __m256 u = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_set1_ps(0.044715f), x3, x),
                                          _mm256_set1_ps(0.7978845608028654f));
                __m256 e = exp256(_mm256_mul_ps(u, _mm256_set1_ps(-2.f)));
                _mm256_storeu_ps(p + i, _mm256_div_ps(x, _mm256_add_ps(one, e)));
            }
            break;
        case Act::Tanh:
            // tanh(x) = 2/(1+exp(-2x)) - 1
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(p + i);
                __m256 e = exp256(_mm256_mul_ps(x, _mm256_set1_ps(-2.f)));
                _mm256_storeu_ps(p + i, _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.f), _mm256_add_ps(one, e)), one));
            }
            break;
        default: break;   // GeluErf: scalar below (only used by small projector layers)
    }
#endif
    for (; i < n; ++i) p[i] = act_scalar(a, p[i]);
}

// ----------------------------------------------------------------------------
// Weight storage
// ----------------------------------------------------------------------------
enum class WType : uint8_t { F32, F16, BF16, Q8_0 };

inline size_t wtype_row_bytes(WType t, size_t K) {
    switch (t) {
        case WType::F32:  return K * 4;
        case WType::F16:  case WType::BF16: return K * 2;
        case WType::Q8_0: return (K / 32) * 34;      // fp16 scale + 32 x int8
    }
    return 0;
}

struct Linear {
    WType type = WType::F32;
    size_t N = 0, K = 0;
    const uint8_t* w = nullptr;        // borrowed (mmap) or points into `own`
    size_t row_stride = 0;             // bytes between weight rows
    std::vector<uint8_t> own;          // set when this Linear owns its weights
    const float* bias = nullptr;       // borrowed, or points into bias_own
    std::vector<float> bias_own;

    bool empty() const { return w == nullptr; }

    // Borrow a tensor straight out of the mmapped file (zero copy).
    void borrow(WType t, const uint8_t* data, size_t n_out, size_t k_in) {
        if (t == WType::Q8_0 && (k_in % 32)) throw std::runtime_error("nn::Linear: Q8_0 needs K % 32 == 0");
        type = t; N = n_out; K = k_in; w = data; row_stride = wtype_row_bytes(t, k_in);
        own.clear();
    }
    void set_bias_copy(const float* b, size_t n) { bias_own.assign(b, b + n); bias = bias_own.data(); }

    // Explicit, opt-in lossy re-encode (F32 -> F16 / Q8_0, etc.). Never called
    // implicitly. Returns false (no change) if the target type isn't possible.
    bool convert_to(WType target) {
        if (target == type) return true;
        if (target == WType::Q8_0 && (K % 32)) return false;
        std::vector<float> f((size_t)N * K);
        dequant_all(f.data());
        std::vector<uint8_t> buf(N * wtype_row_bytes(target, K));
        size_t rb = wtype_row_bytes(target, K);
        for (size_t n = 0; n < N; ++n) {
            const float* r = f.data() + n * K; uint8_t* o = buf.data() + n * rb;
            switch (target) {
                case WType::F32: std::memcpy(o, r, K * 4); break;
                case WType::F16: for (size_t k = 0; k < K; ++k) { uint16_t h = f32_to_f16_scalar(r[k]); std::memcpy(o + 2 * k, &h, 2); } break;
                case WType::BF16: for (size_t k = 0; k < K; ++k) {
                        uint32_t x; std::memcpy(&x, r + k, 4);
                        x += 0x7FFFu + ((x >> 16) & 1u);          // RNE
                        uint16_t h = (uint16_t)(x >> 16); std::memcpy(o + 2 * k, &h, 2); } break;
                case WType::Q8_0:
                    for (size_t b = 0; b < K / 32; ++b) {
                        float amax = 0; for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(r[b * 32 + j]));
                        float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
                        uint16_t dh = f32_to_f16_scalar(d); std::memcpy(o + b * 34, &dh, 2);
                        for (int j = 0; j < 32; ++j) o[b * 34 + 2 + j] = (uint8_t)(int8_t)std::lrintf(r[b * 32 + j] * id);
                    } break;
            }
        }
        own = std::move(buf); w = own.data(); type = target; row_stride = rb;
        return true;
    }

    // Row n -> F32 (n in [0,N)); out has K floats.
    void dequant_row(size_t n, float* out) const {
        const uint8_t* r = w + n * row_stride;
        switch (type) {
            case WType::F32: std::memcpy(out, r, K * 4); break;
            case WType::F16:
                for (size_t k = 0; k < K; ++k) { uint16_t h; std::memcpy(&h, r + 2 * k, 2); out[k] = f16_to_f32_scalar(h); }
                break;
            case WType::BF16:
                for (size_t k = 0; k < K; ++k) { uint16_t h; std::memcpy(&h, r + 2 * k, 2); out[k] = bf16_to_f32_scalar(h); }
                break;
            case WType::Q8_0:
                for (size_t b = 0; b < K / 32; ++b) {
                    uint16_t dh; std::memcpy(&dh, r + b * 34, 2); float d = f16_to_f32_scalar(dh);
                    for (int j = 0; j < 32; ++j) out[b * 32 + j] = d * (float)(int8_t)r[b * 34 + 2 + j];
                }
                break;
        }
    }
    void dequant_all(float* out) const { for (size_t n = 0; n < N; ++n) dequant_row(n, out + n * K); }
};

// ----------------------------------------------------------------------------
// weight-row -> float conversion for a contiguous span [k0, k0+len) of one row
// (the inner loop of both GEMV and the GEMM panel packer)
// ----------------------------------------------------------------------------
inline void row_span_to_f32(const Linear& L, size_t n, size_t k0, size_t len, float* out) {
    const uint8_t* r = L.w + n * L.row_stride;
    switch (L.type) {
        case WType::F32: std::memcpy(out, r + k0 * 4, len * 4); return;
        case WType::F16: {
            size_t i = 0;
#ifdef NN_F16C
            for (; i + 8 <= len; i += 8)
                _mm256_storeu_ps(out + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r + (k0 + i) * 2))));
#endif
            for (; i < len; ++i) { uint16_t h; std::memcpy(&h, r + (k0 + i) * 2, 2); out[i] = f16_to_f32_scalar(h); }
            return;
        }
        case WType::BF16: {
            size_t i = 0;
#ifdef NN_AVX2
            for (; i + 8 <= len; i += 8) {
                __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(r + (k0 + i) * 2)));
                _mm256_storeu_ps(out + i, _mm256_castsi256_ps(_mm256_slli_epi32(v, 16)));
            }
#endif
            for (; i < len; ++i) { uint16_t h; std::memcpy(&h, r + (k0 + i) * 2, 2); out[i] = bf16_to_f32_scalar(h); }
            return;
        }
        case WType::Q8_0: {
            // k0 and len are multiples of 32 on this path (KC=256, K%32==0 enforced)
            for (size_t b = k0 / 32; b < (k0 + len) / 32; ++b) {
                uint16_t dh; std::memcpy(&dh, r + b * 34, 2);
                float d = f16_to_f32_scalar(dh);
                const int8_t* q = (const int8_t*)(r + b * 34 + 2);
                float* o = out + (b * 32 - k0);
#ifdef NN_AVX2
                __m256 vd = _mm256_set1_ps(d);
                for (int j = 0; j < 32; j += 8) {
                    __m256i q32 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + j)));
                    _mm256_storeu_ps(o + j, _mm256_mul_ps(vd, _mm256_cvtepi32_ps(q32)));
                }
#else
                for (int j = 0; j < 32; ++j) o[j] = d * (float)q[j];
#endif
            }
            return;
        }
    }
}

// ----------------------------------------------------------------------------
// GEMV-path: M <= kGemvMaxM rows of X against all N weight rows
// ----------------------------------------------------------------------------
constexpr size_t kGemvMaxM = 4;

// dot of one X row with a weight row span, F32 weights fast path
#ifdef NN_AVX2
// 4 weight rows (F32) x 1 activation row
inline void dot4_f32(const float* x, const float* w0, const float* w1, const float* w2, const float* w3,
                     size_t K, float out[4]) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 xv = _mm256_loadu_ps(x + k);
        a0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w0 + k), a0);
        a1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w1 + k), a1);
        a2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w2 + k), a2);
        a3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w3 + k), a3);
    }
    out[0] = hsum256(a0); out[1] = hsum256(a1); out[2] = hsum256(a2); out[3] = hsum256(a3);
    for (; k < K; ++k) { out[0] += x[k]*w0[k]; out[1] += x[k]*w1[k]; out[2] += x[k]*w2[k]; out[3] += x[k]*w3[k]; }
}
#endif

inline float dot_f32(const float* a, const float* b, size_t n) {
    size_t k = 0; float s = 0.f;
#ifdef NN_AVX2
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    for (; k + 16 <= n; k += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),     _mm256_loadu_ps(b + k),     a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), a1);
    }
    for (; k + 8 <= n; k += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), a0);
    s = hsum256(_mm256_add_ps(a0, a1));
#endif
    for (; k < n; ++k) s += a[k] * b[k];
    return s;
}


#ifdef NN_F16C
// 4 F16 weight rows x 1 activation row, convert-in-register (no scratch row)
inline void dot4_f16(const float* x, const uint8_t* r0, const uint8_t* r1, const uint8_t* r2, const uint8_t* r3,
                     size_t K, float out[4]) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 xv = _mm256_loadu_ps(x + k);
        a0 = _mm256_fmadd_ps(xv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r0 + 2 * k))), a0);
        a1 = _mm256_fmadd_ps(xv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r1 + 2 * k))), a1);
        a2 = _mm256_fmadd_ps(xv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r2 + 2 * k))), a2);
        a3 = _mm256_fmadd_ps(xv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r3 + 2 * k))), a3);
    }
    out[0] = hsum256(a0); out[1] = hsum256(a1); out[2] = hsum256(a2); out[3] = hsum256(a3);
    const uint8_t* rs[4] = {r0, r1, r2, r3};
    for (; k < K; ++k) for (int j = 0; j < 4; ++j) { uint16_t h; std::memcpy(&h, rs[j] + 2 * k, 2); out[j] += x[k] * f16_to_f32_scalar(h); }
}
#endif
#ifdef NN_AVX2
// 4 Q8_0 weight rows x 1 activation row; K % 32 == 0. Per block: int8->f32 FMA chain,
// then one scale multiply-accumulate — no horizontal sum until the end.
inline void dot4_q8(const float* x, const uint8_t* r0, const uint8_t* r1, const uint8_t* r2, const uint8_t* r3,
                    size_t K, float out[4]) {
    const uint8_t* rs[4] = {r0, r1, r2, r3};
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (size_t b = 0; b < K / 32; ++b) {
        __m256 x0 = _mm256_loadu_ps(x + b * 32),      x1 = _mm256_loadu_ps(x + b * 32 + 8),
               x2 = _mm256_loadu_ps(x + b * 32 + 16), x3 = _mm256_loadu_ps(x + b * 32 + 24);
        for (int j = 0; j < 4; ++j) {
            const uint8_t* blk = rs[j] + b * 34;
            uint16_t dh; std::memcpy(&dh, blk, 2);
#ifdef NN_F16C
            float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(dh)));
#else
            float d = f16_to_f32_scalar(dh);
#endif
            const int8_t* q = (const int8_t*)(blk + 2);
            __m256 s = _mm256_mul_ps(x0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q)))));
            s = _mm256_fmadd_ps(x1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + 8)))), s);
            s = _mm256_fmadd_ps(x2, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + 16)))), s);
            s = _mm256_fmadd_ps(x3, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + 24)))), s);
            acc[j] = _mm256_fmadd_ps(_mm256_set1_ps(d), s, acc[j]);
        }
    }
    for (int j = 0; j < 4; ++j) out[j] = hsum256(acc[j]);
}
#endif

// rows [n0,n1) of Y for up to kGemvMaxM activation rows. Y[m*ldy + n].
inline void gemv_range(const Linear& L, const float* X, size_t M, size_t ldx,
                       float* Y, size_t ldy, size_t n0, size_t n1) {
    thread_local std::vector<float> wbuf;
    const size_t K = L.K;
    if (L.type == WType::F32) {
        size_t n = n0;
#ifdef NN_AVX2
        for (; n + 4 <= n1; n += 4) {
            const float* w0 = (const float*)(L.w + (n + 0) * L.row_stride);
            const float* w1 = (const float*)(L.w + (n + 1) * L.row_stride);
            const float* w2 = (const float*)(L.w + (n + 2) * L.row_stride);
            const float* w3 = (const float*)(L.w + (n + 3) * L.row_stride);
            for (size_t m = 0; m < M; ++m) {
                float o[4]; dot4_f32(X + m * ldx, w0, w1, w2, w3, K, o);
                for (int j = 0; j < 4; ++j) Y[m * ldy + n + j] = o[j];
            }
        }
#endif
        for (; n < n1; ++n) {
            const float* w = (const float*)(L.w + n * L.row_stride);
            for (size_t m = 0; m < M; ++m) Y[m * ldy + n] = dot_f32(X + m * ldx, w, K);
        }
    } else {
        size_t n = n0;
#if defined(NN_F16C)
        if (L.type == WType::F16) {
            for (; n + 4 <= n1; n += 4) {
                const uint8_t* r = L.w + n * L.row_stride;
                for (size_t m = 0; m < M; ++m) {
                    float o[4]; dot4_f16(X + m * ldx, r, r + L.row_stride, r + 2 * L.row_stride, r + 3 * L.row_stride, K, o);
                    for (int j = 0; j < 4; ++j) Y[m * ldy + n + j] = o[j];
                }
            }
        }
#endif
#if defined(NN_AVX2)
        if (L.type == WType::Q8_0) {
            for (; n + 4 <= n1; n += 4) {
                const uint8_t* r = L.w + n * L.row_stride;
                for (size_t m = 0; m < M; ++m) {
                    float o[4]; dot4_q8(X + m * ldx, r, r + L.row_stride, r + 2 * L.row_stride, r + 3 * L.row_stride, K, o);
                    for (int j = 0; j < 4; ++j) Y[m * ldy + n + j] = o[j];
                }
            }
        }
#endif
        // remaining rows (and BF16 / non-SIMD builds): convert one weight row at a time into an L1-resident
        // scratch row, then dot against every activation row (conversion cost is
        // amortised over M; for M==1 the conversion is ~1 op per weight, still
        // cheaper than the 2-4x extra memory traffic of storing F32).
        if (wbuf.size() < K) wbuf.resize(K);
        for (; n < n1; ++n) {
            row_span_to_f32(L, n, 0, K, wbuf.data());
            for (size_t m = 0; m < M; ++m) Y[m * ldy + n] = dot_f32(X + m * ldx, wbuf.data(), K);
        }
    }
}

// ----------------------------------------------------------------------------
// GEMM-path: packed-panel 6x16 AVX2 microkernel
// ----------------------------------------------------------------------------
constexpr size_t kNR = 16;      // panel width (output columns)
constexpr size_t kMR = 6;       // rows of X per microkernel call
constexpr size_t kKC = 256;     // K block

#ifdef NN_AVX2
template <int MR>
inline void micro_kernel(const float* X, size_t ldx, const float* P /*[kc][16]*/, size_t kc,
                         float* C, size_t ldc, bool accumulate) {
    __m256 c0[MR], c1[MR];
#pragma GCC unroll 6
    for (int i = 0; i < MR; ++i) {
        if (accumulate) { c0[i] = _mm256_loadu_ps(C + i * ldc); c1[i] = _mm256_loadu_ps(C + i * ldc + 8); }
        else { c0[i] = _mm256_setzero_ps(); c1[i] = _mm256_setzero_ps(); }
    }
    for (size_t k = 0; k < kc; ++k) {
        __m256 b0 = _mm256_loadu_ps(P + k * kNR), b1 = _mm256_loadu_ps(P + k * kNR + 8);
#pragma GCC unroll 6
        for (int i = 0; i < MR; ++i) {
            __m256 a = _mm256_broadcast_ss(X + i * ldx + k);
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
        }
    }
#pragma GCC unroll 6
    for (int i = 0; i < MR; ++i) { _mm256_storeu_ps(C + i * ldc, c0[i]); _mm256_storeu_ps(C + i * ldc + 8, c1[i]); }
}
#endif

template <int MR>
inline void micro_kernel_scalar(const float* X, size_t ldx, const float* P, size_t kc,
                                float* C, size_t ldc, bool accumulate) {
    for (int i = 0; i < MR; ++i)
        for (size_t j = 0; j < kNR; ++j) {
            float s = accumulate ? C[i * ldc + j] : 0.f;
            for (size_t k = 0; k < kc; ++k) s += X[i * ldx + k] * P[k * kNR + j];
            C[i * ldc + j] = s;
        }
}

inline void micro_dispatch(int mr, const float* X, size_t ldx, const float* P, size_t kc,
                           float* C, size_t ldc, bool acc) {
#ifdef NN_AVX2
#  define NN_MK(R) case R: micro_kernel<R>(X, ldx, P, kc, C, ldc, acc); return;
#else
#  define NN_MK(R) case R: micro_kernel_scalar<R>(X, ldx, P, kc, C, ldc, acc); return;
#endif
    switch (mr) { NN_MK(1) NN_MK(2) NN_MK(3) NN_MK(4) NN_MK(5) NN_MK(6) }
#undef NN_MK
}

// Compute columns [n0, n1) of Y (n0 multiple of 16) for all M rows.
inline void gemm_range(const Linear& L, const float* X, size_t M, size_t ldx,
                       float* Y, size_t ldy, size_t n0, size_t n1, Act act) {
    thread_local std::vector<float> panel, tmp, rowbuf;
    const size_t K = L.K;
    if (panel.size() < kKC * kNR) panel.resize(kKC * kNR);
    if (rowbuf.size() < kKC) rowbuf.resize(kKC);
    if (tmp.size() < kMR * kNR) tmp.resize(kMR * kNR);

    for (size_t p0 = n0; p0 < n1; p0 += kNR) {
        const size_t ncols = std::min(kNR, n1 - p0);
        for (size_t k0 = 0; k0 < K; k0 += kKC) {
            const size_t kc = std::min(kKC, K - k0);
            // pack: panel[k][j] = W[p0+j][k0+k]; zero-pad j >= ncols
            for (size_t j = 0; j < kNR; ++j) {
                if (j < ncols) {
                    row_span_to_f32(L, p0 + j, k0, kc, rowbuf.data());
                    for (size_t k = 0; k < kc; ++k) panel[k * kNR + j] = rowbuf[k];
                } else {
                    for (size_t k = 0; k < kc; ++k) panel[k * kNR + j] = 0.f;
                }
            }
            const bool acc = (k0 != 0);
            for (size_t m0 = 0; m0 < M; m0 += kMR) {
                int mr = (int)std::min(kMR, M - m0);
                const float* xp = X + m0 * ldx + k0;
                if (ncols == kNR) {
                    micro_dispatch(mr, xp, ldx, panel.data(), kc, Y + m0 * ldy + p0, ldy, acc);
                } else {
                    // narrow tail: compute into a 16-wide temp, copy valid columns
                    for (int i = 0; i < mr; ++i)
                        for (size_t j = 0; j < kNR; ++j) tmp[i * kNR + j] = (acc && j < ncols) ? Y[(m0 + i) * ldy + p0 + j] : 0.f;
                    micro_dispatch(mr, xp, ldx, panel.data(), kc, tmp.data(), kNR, acc);
                    for (int i = 0; i < mr; ++i)
                        for (size_t j = 0; j < ncols; ++j) Y[(m0 + i) * ldy + p0 + j] = tmp[i * kNR + j];
                }
            }
        }
        // epilogue for this panel: bias + activation (data is cache-hot)
        for (size_t m = 0; m < M; ++m) {
            float* y = Y + m * ldy + p0;
            if (L.bias) for (size_t j = 0; j < ncols; ++j) y[j] += L.bias[p0 + j];
            apply_act(act, y, ncols);
        }
    }
}

// ----------------------------------------------------------------------------
// public entry point: Y[M,N] = act(X[M,K] * W^T + bias)
// ----------------------------------------------------------------------------
// pool == nullptr  -> single-threaded (use inside parallel regions, and for the
//                     Motion-VAE where a barrier costs more than the work)
inline void linear(const Linear& L, const float* X, size_t M, size_t ldx,
                   float* Y, size_t ldy, Act act = Act::None, util::ThreadPool* pool = nullptr) {
    if (L.empty()) throw std::runtime_error("nn::linear: empty weight");
    const size_t N = L.N;
    const bool gemv = (M <= kGemvMaxM);
    const size_t work = N * L.K * M;
    const int nth = (pool && work >= (1u << 20)) ? std::max(1, pool->size() + 1) : 1;   // +1: wait() helps

    auto run = [&](size_t n0, size_t n1) {
        if (gemv) {
            gemv_range(L, X, M, ldx, Y, ldy, n0, n1);
            for (size_t m = 0; m < M; ++m) {
                float* y = Y + m * ldy;
                if (L.bias) for (size_t n = n0; n < n1; ++n) y[n] += L.bias[n];
                apply_act(act, y + n0, n1 - n0);
            }
        } else gemm_range(L, X, M, ldx, Y, ldy, n0, n1, act);
    };
    if (nth == 1) { run(0, N); return; }

    // split N in multiples of 16 (GEMM panels) / 4 (GEMV row groups)
    size_t chunk = ((N + (size_t)nth - 1) / (size_t)nth + 15) & ~(size_t)15;
    for (size_t n0 = 0; n0 < N; n0 += chunk) {
        size_t n1 = std::min(N, n0 + chunk);
        pool->submit([=, &run] { run(n0, n1); });
    }
    pool->wait();
}

// ----------------------------------------------------------------------------
// LayerNorm / RMSNorm / softmax / misc, row-wise over [rows, n]
// ----------------------------------------------------------------------------
inline void layernorm_row(float* x, size_t n, const float* g, const float* b, float eps) {
    float mean = 0.f, var = 0.f;
    size_t i = 0;
#ifdef NN_AVX2
    __m256 s = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) s = _mm256_add_ps(s, _mm256_loadu_ps(x + i));
    mean = hsum256(s);
#endif
    for (; i < n; ++i) mean += x[i];
    mean /= (float)n;
    i = 0;
#ifdef NN_AVX2
    __m256 vm = _mm256_set1_ps(mean), sv = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) { __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + i), vm); sv = _mm256_fmadd_ps(d, d, sv); }
    var = hsum256(sv);
#endif
    for (; i < n; ++i) { float d = x[i] - mean; var += d * d; }
    var /= (float)n;
    const float inv = 1.f / std::sqrt(var + eps);
    i = 0;
#ifdef NN_AVX2
    __m256 vi = _mm256_set1_ps(inv);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vm), vi);
        if (g) v = _mm256_mul_ps(v, _mm256_loadu_ps(g + i));
        if (b) v = _mm256_add_ps(v, _mm256_loadu_ps(b + i));
        _mm256_storeu_ps(x + i, v);
    }
#endif
    for (; i < n; ++i) { float v = (x[i] - mean) * inv; if (g) v *= g[i]; if (b) v += b[i]; x[i] = v; }
}

// softmax(x * scale) over n, in place
inline void softmax_scaled(float* x, size_t n, float scale) {
    float mx = -INFINITY; size_t i = 0;
#ifdef NN_AVX2
    __m256 vm = _mm256_set1_ps(-INFINITY);
    for (; i + 8 <= n; i += 8) vm = _mm256_max_ps(vm, _mm256_loadu_ps(x + i));
    mx = hmax256(vm);
#endif
    for (; i < n; ++i) mx = std::max(mx, x[i]);
    float sum = 0.f; i = 0;
#ifdef NN_AVX2
    __m256 vs = _mm256_setzero_ps(), vmx = _mm256_set1_ps(mx), vsc = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        __m256 e = exp256(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmx), vsc));
        _mm256_storeu_ps(x + i, e); vs = _mm256_add_ps(vs, e);
    }
    sum = hsum256(vs);
#endif
    for (; i < n; ++i) { float e = std::exp((x[i] - mx) * scale); x[i] = e; sum += e; }
    const float inv = 1.f / sum; i = 0;
#ifdef NN_AVX2
    __m256 vinv = _mm256_set1_ps(inv);
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
#endif
    for (; i < n; ++i) x[i] *= inv;
}

inline void add_inplace(float* a, const float* b, size_t n) {
    size_t i = 0;
#ifdef NN_AVX2
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
#endif
    for (; i < n; ++i) a[i] += b[i];
}

// split [0,n) over the pool; inline if tiny. fn(begin,end)
template <typename Fn>
inline void parallel_for(util::ThreadPool* pool, size_t n, size_t min_per_task, Fn fn) {
    int nth = pool ? std::max(1, pool->size() + 1) : 1;
    if (nth == 1 || n < 2 * min_per_task) { fn((size_t)0, n); return; }
    size_t chunk = std::max(min_per_task, (n + (size_t)nth - 1) / (size_t)nth);
    for (size_t b = 0; b < n; b += chunk) {
        size_t e = std::min(n, b + chunk);
        pool->submit([=, &fn] { fn(b, e); });
    }
    pool->wait();
}

} // namespace nn
