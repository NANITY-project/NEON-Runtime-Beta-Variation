#pragma once
// =============================================================================
// rawllm_repack.hpp — "rows-in-lanes" repacked Q4_0 / Q8_0 weight panels and the
// GEMM kernels that consume them (the llama.cpp "Q4_0_8x8 / repack" idea, written
// from scratch for NEON's own layout).
//
// WHY THIS EXISTS
//   The unpacked fused kernels (dot_q4_0_q8_0 / tile4_*) keep the 8 int32 lanes of
//   a block's dot product as K-partial sums of ONE output, so every block pays:
//   sign/abs trick, per-row scale multiply, int->float convert, FMA, and a final
//   horizontal sum per output. At ~6 vector uops per (row, position, block) the
//   AVX2 port budget (p0/p1) caps them at ~47 GOP/s per 2.2 GHz core, no matter
//   how the loops are tiled.
//
//   Here 8 consecutive weight rows are interleaved at 4-byte granularity so that
//   ONE 256-bit register holds [row0 k0..k3 | row1 k0..k3 | ... | row7 k0..k3].
//   Broadcasting 4 activation bytes against it makes every int32 lane a different
//   OUTPUT ROW, which means:
//     * no horizontal reduction anywhere (the 8 accumulators are the 8 outputs and
//       are stored contiguously, Y[pos*stride + r0 .. r0+7]);
//     * int->float convert + scale + FMA happen once per 8 rows per block, not once
//       per row per block;
//     * weights are streamed strictly sequentially (one panel = one linear read);
//     * Q4_0 nibbles stay UNSIGNED (0..15) so no sign/abs fix-up is needed: the -8
//       bias is removed with one integer subtract of 8*sum(x_block) per block.
//   On AVX2 the 16 int16 partial sums of a block are accumulated in int16 across all
//   8 maddubs of the block (|sum| <= 8*2*15*127 = 30480 < 32767 as long as the
//   activation codes are clamped to +-127, which quantize_row_q8_0 guarantees) and
//   widened once per block. On AVX-512-VNNI (256-bit dpbusd, AVX512VL) the whole
//   block accumulates straight into int32.
//
//   Measured single-core (2.2 GHz Sapphire Rapids VM, 3072-col Q4_0, 60 positions):
//     old tile4 AVX2 47 GOP/s | repack AVX2 149 GOP/s | repack VNNI 234 GOP/s
//   and the result is BIT-IDENTICAL to dot_q4_0_q8_0 (same int sums, same
//   wd*xd scale, same sequential-over-blocks float accumulation).
//
// LAYOUTS (per panel of 8 rows, per 32-element block; same bytes/row as GGUF)
//   Q4_0x8 (144 B): [8 x fp16 scale][4 x 32 B: vector j = for row i, qs[4j..4j+3]]
//        qs byte k holds element k (low nibble) and element k+16 (high nibble), so
//        lo-nibbles of vector j pair with x[4j..4j+3], hi-nibbles with x[16+4j..].
//   Q8_0x8 (272 B): [8 x fp16 scale][8 x 32 B: vector j = for row i, w[4j..4j+3]]
//
// COST / TRADE-OFFS (be aware)
//   * The repacked copy is an EXTRA heap copy of every Q4_0/Q8_0 matrix that goes
//     through proj_all_positions[_multi] (including a tied lm_head). It is built
//     lazily on first use (parallel, ~one pass over the weights) or eagerly via
//     fwd::prepack_model(). Resident memory therefore grows by roughly the model
//     size unless the OS drops the (clean, mmap'd) original pages.
//     Set RAWLLM_NO_REPACK=1 to disable and use the previous unpacked kernels.
//   * Rows beyond the last multiple of 8 stay on the legacy path (original data).
//   * RAWLLM_ISA=scalar|avx2|vnni forces a kernel tier (clamped to what the CPU has).
// =============================================================================
#include "rawllm_common.hpp"
#include "rawllm_util.hpp"
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace repack {

constexpr size_t PR = 8;                 // rows per panel
constexpr size_t Q4_BLK = 144;           // bytes per (panel, block) for Q4_0x8
constexpr size_t Q8_BLK = 272;           // bytes per (panel, block) for Q8_0x8

enum class Kind : uint8_t { Q4_0, Q8_0 };
enum class Isa : uint8_t { Scalar = 0, Avx2 = 1, Vnni = 2 };

struct Packed {
    Kind    kind = Kind::Q4_0;
    uint8_t* data = nullptr;    // 64B-aligned; nullptr => "repack failed/declined" (negative cache entry)
    size_t  rows = 0;           // packed rows (multiple of 8; may be < tensor rows)
    size_t  cols = 0, nb = 0;
    size_t  panel_bytes = 0;    // nb * (Q4_BLK | Q8_BLK)
    Packed() = default;
    Packed(const Packed&) = delete;
    Packed& operator=(const Packed&) = delete;
    ~Packed() { std::free(data); }
    size_t total_bytes() const { return (rows / PR) * panel_bytes; }
};

inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("RAWLLM_NO_REPACK");
        return !(e && *e && *e != '0');
    }();
    return on;
}

inline Isa detect_isa() {
    __builtin_cpu_init();
    const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    const bool vnni = avx2 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                      __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
    Isa best = vnni ? Isa::Vnni : (avx2 ? Isa::Avx2 : Isa::Scalar);
    if (const char* f = std::getenv("RAWLLM_ISA")) {
        std::string s(f);
        Isa want = best;
        if (s == "scalar") want = Isa::Scalar; else if (s == "avx2") want = Isa::Avx2; else if (s == "vnni") want = Isa::Vnni;
        if ((int)want < (int)best) best = want;
    }
    return best;
}
inline Isa current_isa() { static const Isa i = detect_isa(); return i; }
inline const char* isa_name(Isa i) { return i == Isa::Vnni ? "avx512-vnni" : (i == Isa::Avx2 ? "avx2" : "scalar"); }

// ───────────────────────────── packing (plain C++) ──────────────────────────────
// src: 8 consecutive rows in GGUF layout, row stride rb; dst: nb*144 bytes.
inline void pack_q4_panel(const uint8_t* src, size_t rb, size_t nb, uint8_t* dst) {
    for (size_t b = 0; b < nb; ++b) {
        uint8_t* d = dst + b * Q4_BLK;
        for (size_t i = 0; i < PR; ++i) {
            const uint8_t* blk = src + i * rb + b * 18;
            std::memcpy(d + i * 2, blk, 2);
            for (size_t j = 0; j < 4; ++j) std::memcpy(d + 16 + j * 32 + i * 4, blk + 2 + j * 4, 4);
        }
    }
}
inline void pack_q8_panel(const uint8_t* src, size_t rb, size_t nb, uint8_t* dst) {
    for (size_t b = 0; b < nb; ++b) {
        uint8_t* d = dst + b * Q8_BLK;
        for (size_t i = 0; i < PR; ++i) {
            const uint8_t* blk = src + i * rb + b * 34;
            std::memcpy(d + i * 2, blk, 2);
            for (size_t j = 0; j < 8; ++j) std::memcpy(d + 16 + j * 32 + i * 4, blk + 2 + j * 4, 4);
        }
    }
}

// ───────────────────────────────── kernels ──────────────────────────────────────
// Common contract for every gemm_* below:
//   pnl         first panel (panels are panel_bytes apart), npanels panels
//   xd/xs/xq    quantize_rows_q8_0() output for `np` consecutive positions starting at
//               the block's position 0 (xd,xs: nb per position; xq: cols per position)
//   Y           points at Y[pos0*y_stride + r0]; panel k writes columns r0+8k..r0+8k+7
//   bias        nullable, points at bias[r0]
using GemmFn = void (*)(const uint8_t* pnl, size_t npanels, size_t panel_bytes, size_t nb,
                        const float* xd, const int32_t* xs, const int8_t* xq, size_t cols,
                        size_t np, float* Y, size_t y_stride, const float* bias);

inline float half_to_float(uint16_t dh) {   // portable fp16 -> fp32 (scalar tier only)
    uint32_t sign = (dh & 0x8000u) << 16, exp = (dh >> 10) & 0x1F, mant = dh & 0x3FF, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; --exp; } mant &= 0x3FF; bits = sign | ((exp + 112) << 23) | (mant << 13); }
    } else if (exp == 0x1F) bits = sign | 0x7F800000u | (mant << 13);
    else bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

// ---- scalar tier (portable reference; also what runs on CPUs without AVX2) ----
inline void gemm_q4_scalar(const uint8_t* pnl, size_t npanels, size_t panel_bytes, size_t nb,
                           const float* xd, const int32_t* xs, const int8_t* xq, size_t cols,
                           size_t np, float* Y, size_t y_stride, const float* bias) {
    for (size_t pn = 0; pn < npanels; ++pn) {
        const uint8_t* P = pnl + pn * panel_bytes;
        for (size_t t = 0; t < np; ++t) {
            for (size_t i = 0; i < PR; ++i) {
                float total = 0.f;
                for (size_t b = 0; b < nb; ++b) {
                    const uint8_t* pb = P + b * Q4_BLK;
                    uint16_t h; std::memcpy(&h, pb + i * 2, 2);
                    const int8_t* x = xq + t * cols + b * 32;
                    int32_t isum = 0;
                    for (int k = 0; k < 16; ++k) {
                        uint8_t q = pb[16 + (k / 4) * 32 + i * 4 + (k % 4)];
                        isum += (int)(q & 0xF) * x[k] + (int)(q >> 4) * x[k + 16];
                    }
                    isum -= 8 * xs[t * nb + b];
                    total = std::fmaf(half_to_float(h) * xd[t * nb + b], (float)isum, total);
                }
                Y[t * y_stride + pn * PR + i] = total + (bias ? bias[pn * PR + i] : 0.f);
            }
        }
    }
}
inline void gemm_q8_scalar(const uint8_t* pnl, size_t npanels, size_t panel_bytes, size_t nb,
                           const float* xd, const int32_t* xs, const int8_t* xq, size_t cols,
                           size_t np, float* Y, size_t y_stride, const float* bias) {
    (void)xs;
    for (size_t pn = 0; pn < npanels; ++pn) {
        const uint8_t* P = pnl + pn * panel_bytes;
        for (size_t t = 0; t < np; ++t) {
            for (size_t i = 0; i < PR; ++i) {
                float total = 0.f;
                for (size_t b = 0; b < nb; ++b) {
                    const uint8_t* pb = P + b * Q8_BLK;
                    uint16_t h; std::memcpy(&h, pb + i * 2, 2);
                    const int8_t* x = xq + t * cols + b * 32;
                    int32_t isum = 0;
                    for (int k = 0; k < 32; ++k)
                        isum += (int)(int8_t)pb[16 + (k / 4) * 32 + i * 4 + (k % 4)] * x[k];
                    total = std::fmaf(half_to_float(h) * xd[t * nb + b], (float)isum, total);
                }
                Y[t * y_stride + pn * PR + i] = total + (bias ? bias[pn * PR + i] : 0.f);
            }
        }
    }
}

// ---- AVX2 tier ----------------------------------------------------------------
#define RP_AVX2 __attribute__((target("avx2,fma,f16c")))
#define RP_VNNI __attribute__((target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")))

namespace detail {
// Decode (P==1) streams every weight byte exactly once from DRAM; one core's hardware prefetcher
// alone doesn't keep enough misses in flight, so also touch the panel ~PF_AHEAD bytes ahead.
#ifndef RAWLLM_PF_AHEAD
#define RAWLLM_PF_AHEAD 4096
#endif
constexpr size_t PF_AHEAD = RAWLLM_PF_AHEAD;
template <size_t BLK> RP_AVX2 inline void prefetch_ahead(const uint8_t* pb) {
    _mm_prefetch((const char*)(pb + PF_AHEAD), _MM_HINT_T0);
    if (BLK > 128) _mm_prefetch((const char*)(pb + PF_AHEAD + 128), _MM_HINT_T0);
    if (BLK > 192) _mm_prefetch((const char*)(pb + PF_AHEAD + 192), _MM_HINT_T0);
}
RP_AVX2 inline int32_t ld32(const int8_t* p) { int32_t v; std::memcpy(&v, p, 4); return v; }

// Q4_0x8, P positions. 8 unpacked weight vectors are hoisted across the P positions.
template <int P>
RP_AVX2 inline void q4_panel_avx2(const uint8_t* pnl, size_t nb, const float* xd, const int32_t* xs,
                                  const int8_t* xq, size_t cols, float* out, size_t os, const float* bias) {
    __m256 acc[P];
    for (int p = 0; p < P; ++p) acc[p] = _mm256_setzero_ps();
    const __m256i m0F = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* pb = pnl + b * Q4_BLK;
        if (P == 1) prefetch_ahead<Q4_BLK>(pb);
        const __m256 wd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)pb));
        const __m256i v0 = _mm256_loadu_si256((const __m256i*)(pb + 16)),  v1 = _mm256_loadu_si256((const __m256i*)(pb + 48));
        const __m256i v2 = _mm256_loadu_si256((const __m256i*)(pb + 80)),  v3 = _mm256_loadu_si256((const __m256i*)(pb + 112));
        const __m256i lo0 = _mm256_and_si256(v0, m0F), hi0 = _mm256_and_si256(_mm256_srli_epi16(v0, 4), m0F);
        const __m256i lo1 = _mm256_and_si256(v1, m0F), hi1 = _mm256_and_si256(_mm256_srli_epi16(v1, 4), m0F);
        const __m256i lo2 = _mm256_and_si256(v2, m0F), hi2 = _mm256_and_si256(_mm256_srli_epi16(v2, 4), m0F);
        const __m256i lo3 = _mm256_and_si256(v3, m0F), hi3 = _mm256_and_si256(_mm256_srli_epi16(v3, 4), m0F);
        for (int p = 0; p < P; ++p) {
            const int8_t* x = xq + (size_t)p * cols + b * 32;
            __m256i a = _mm256_maddubs_epi16(lo0, _mm256_set1_epi32(ld32(x)));
            a = _mm256_add_epi16(a, _mm256_maddubs_epi16(hi0, _mm256_set1_epi32(ld32(x + 16))));
            a = _mm256_add_epi16(a, _mm256_maddubs_epi16(lo1, _mm256_set1_epi32(ld32(x + 4))));
            a = _mm256_add_epi16(a, _mm256_maddubs_epi16(hi1, _mm256_set1_epi32(ld32(x + 20))));
            a = _mm256_add_epi16(a, _mm256_maddubs_epi16(lo2, _mm256_set1_epi32(ld32(x + 8))));
            a = _mm256_add_epi16(a, _mm256_maddubs_epi16(hi2, _mm256_set1_epi32(ld32(x + 24))));
            a = _mm256_add_epi16(a, _mm256_maddubs_epi16(lo3, _mm256_set1_epi32(ld32(x + 12))));
            a = _mm256_add_epi16(a, _mm256_maddubs_epi16(hi3, _mm256_set1_epi32(ld32(x + 28))));
            const __m256i t = _mm256_sub_epi32(_mm256_madd_epi16(a, ones), _mm256_set1_epi32(8 * xs[(size_t)p * nb + b]));
            const __m256 sc = _mm256_mul_ps(wd, _mm256_set1_ps(xd[(size_t)p * nb + b]));
            acc[p] = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(t), acc[p]);
        }
    }
    const __m256 bv = bias ? _mm256_loadu_ps(bias) : _mm256_setzero_ps();
    for (int p = 0; p < P; ++p) _mm256_storeu_ps(out + (size_t)p * os, _mm256_add_ps(acc[p], bv));
}

// Q8_0x8, P positions. Signed x signed via |w| * sign(x, w); int32 accumulate per slice.
template <int P>
RP_AVX2 inline void q8_panel_avx2(const uint8_t* pnl, size_t nb, const float* xd, const int32_t* xs,
                                  const int8_t* xq, size_t cols, float* out, size_t os, const float* bias) {
    (void)xs;
    __m256 acc[P];
    for (int p = 0; p < P; ++p) acc[p] = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* pb = pnl + b * Q8_BLK;
        if (P == 1) prefetch_ahead<Q8_BLK>(pb);
        const __m256 wd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)pb));
        __m256i a[P];
        for (int p = 0; p < P; ++p) a[p] = _mm256_setzero_si256();
        for (int j = 0; j < 8; ++j) {
            const __m256i wj = _mm256_loadu_si256((const __m256i*)(pb + 16 + j * 32));
            const __m256i aw = _mm256_abs_epi8(wj);
            for (int p = 0; p < P; ++p) {
                const __m256i xb = _mm256_set1_epi32(ld32(xq + (size_t)p * cols + b * 32 + 4 * j));
                const __m256i m = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(xb, wj));   // <= 2*128*127 < 32767
                a[p] = _mm256_add_epi32(a[p], _mm256_madd_epi16(m, ones));
            }
        }
        for (int p = 0; p < P; ++p) {
            const __m256 sc = _mm256_mul_ps(wd, _mm256_set1_ps(xd[(size_t)p * nb + b]));
            acc[p] = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(a[p]), acc[p]);
        }
    }
    const __m256 bv = bias ? _mm256_loadu_ps(bias) : _mm256_setzero_ps();
    for (int p = 0; p < P; ++p) _mm256_storeu_ps(out + (size_t)p * os, _mm256_add_ps(acc[p], bv));
}

// ---- AVX-512-VNNI tier (256-bit dpbusd; needs AVX512VL + AVX512_VNNI) ----------
template <int P>
RP_VNNI inline void q4_panel_vnni(const uint8_t* pnl, size_t nb, const float* xd, const int32_t* xs,
                                  const int8_t* xq, size_t cols, float* out, size_t os, const float* bias) {
    __m256 acc[P];
    for (int p = 0; p < P; ++p) acc[p] = _mm256_setzero_ps();
    const __m256i m0F = _mm256_set1_epi8(0x0F);
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* pb = pnl + b * Q4_BLK;
        if (P == 1) prefetch_ahead<Q4_BLK>(pb);
        const __m256 wd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)pb));
        __m256i a[P];
        for (int p = 0; p < P; ++p) a[p] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i v = _mm256_loadu_si256((const __m256i*)(pb + 16 + j * 32));
            const __m256i lo = _mm256_and_si256(v, m0F), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0F);
            for (int p = 0; p < P; ++p) {
                const int8_t* x = xq + (size_t)p * cols + b * 32;
                a[p] = _mm256_dpbusd_epi32(a[p], lo, _mm256_set1_epi32(ld32(x + 4 * j)));
                a[p] = _mm256_dpbusd_epi32(a[p], hi, _mm256_set1_epi32(ld32(x + 16 + 4 * j)));
            }
        }
        for (int p = 0; p < P; ++p) {
            const __m256i t = _mm256_sub_epi32(a[p], _mm256_set1_epi32(8 * xs[(size_t)p * nb + b]));
            const __m256 sc = _mm256_mul_ps(wd, _mm256_set1_ps(xd[(size_t)p * nb + b]));
            acc[p] = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(t), acc[p]);
        }
    }
    const __m256 bv = bias ? _mm256_loadu_ps(bias) : _mm256_setzero_ps();
    for (int p = 0; p < P; ++p) _mm256_storeu_ps(out + (size_t)p * os, _mm256_add_ps(acc[p], bv));
}

// Q8: w is signed, dpbusd wants an unsigned first operand -> use (w ^ 0x80) = w + 128 and
// remove the bias afterwards: sum((w+128)*x) - 128*sum(x). (Same trick as the existing VNNI row kernel.)
template <int P>
RP_VNNI inline void q8_panel_vnni(const uint8_t* pnl, size_t nb, const float* xd, const int32_t* xs,
                                  const int8_t* xq, size_t cols, float* out, size_t os, const float* bias) {
    __m256 acc[P];
    for (int p = 0; p < P; ++p) acc[p] = _mm256_setzero_ps();
    const __m256i flip = _mm256_set1_epi8((char)0x80);
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* pb = pnl + b * Q8_BLK;
        if (P == 1) prefetch_ahead<Q8_BLK>(pb);
        const __m256 wd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)pb));
        __m256i a[P];
        for (int p = 0; p < P; ++p) a[p] = _mm256_setzero_si256();
        for (int j = 0; j < 8; ++j) {
            const __m256i wu = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(pb + 16 + j * 32)), flip);
            for (int p = 0; p < P; ++p)
                a[p] = _mm256_dpbusd_epi32(a[p], wu, _mm256_set1_epi32(ld32(xq + (size_t)p * cols + b * 32 + 4 * j)));
        }
        for (int p = 0; p < P; ++p) {
            const __m256i t = _mm256_sub_epi32(a[p], _mm256_set1_epi32(128 * xs[(size_t)p * nb + b]));
            const __m256 sc = _mm256_mul_ps(wd, _mm256_set1_ps(xd[(size_t)p * nb + b]));
            acc[p] = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(t), acc[p]);
        }
    }
    const __m256 bv = bias ? _mm256_loadu_ps(bias) : _mm256_setzero_ps();
    for (int p = 0; p < P; ++p) _mm256_storeu_ps(out + (size_t)p * os, _mm256_add_ps(acc[p], bv));
}
} // namespace detail

// Drivers: loop panels, then position groups (largest P first, then the remainder).
#define RP_DRIVER(NAME, ATTR, PMAX, KERNEL, BLK)                                                         \
    ATTR inline void NAME(const uint8_t* pnl, size_t npanels, size_t panel_bytes, size_t nb,             \
                          const float* xd, const int32_t* xs, const int8_t* xq, size_t cols,             \
                          size_t np, float* Y, size_t y_stride, const float* bias) {                     \
        for (size_t pn = 0; pn < npanels; ++pn) {                                                        \
            const uint8_t* P_ = pnl + pn * panel_bytes;                                                  \
            const float* bz = bias ? bias + pn * PR : nullptr;                                           \
            size_t t = 0;                                                                                \
            while (t < np) {                                                                             \
                const size_t rem = np - t;                                                               \
                float* o = Y + t * y_stride + pn * PR;                                                   \
                const float* d_ = xd + t * nb; const int32_t* s_ = xs + t * nb; const int8_t* q_ = xq + t * cols; \
                if (rem >= (size_t)PMAX)      { detail::KERNEL<PMAX>(P_, nb, d_, s_, q_, cols, o, y_stride, bz); t += PMAX; } \
                else switch (rem) {                                                                      \
                    case 5: detail::KERNEL<5 <= PMAX ? 5 : PMAX>(P_, nb, d_, s_, q_, cols, o, y_stride, bz); t += (5 <= PMAX ? 5 : PMAX); break; \
                    case 4: detail::KERNEL<4 <= PMAX ? 4 : PMAX>(P_, nb, d_, s_, q_, cols, o, y_stride, bz); t += (4 <= PMAX ? 4 : PMAX); break; \
                    case 3: detail::KERNEL<3>(P_, nb, d_, s_, q_, cols, o, y_stride, bz); t += 3; break; \
                    case 2: detail::KERNEL<2>(P_, nb, d_, s_, q_, cols, o, y_stride, bz); t += 2; break; \
                    default: detail::KERNEL<1>(P_, nb, d_, s_, q_, cols, o, y_stride, bz); t += 1; break; \
                }                                                                                        \
            }                                                                                            \
        }                                                                                                \
    }
RP_DRIVER(gemm_q4_avx2, RP_AVX2, 4, q4_panel_avx2, Q4_BLK)
RP_DRIVER(gemm_q8_avx2, RP_AVX2, 4, q8_panel_avx2, Q8_BLK)
RP_DRIVER(gemm_q4_vnni, RP_VNNI, 6, q4_panel_vnni, Q4_BLK)
RP_DRIVER(gemm_q8_vnni, RP_VNNI, 6, q8_panel_vnni, Q8_BLK)
#undef RP_DRIVER

inline GemmFn select_gemm(Kind k) {
    switch (current_isa()) {
        case Isa::Vnni: return k == Kind::Q4_0 ? gemm_q4_vnni : gemm_q8_vnni;
        case Isa::Avx2: return k == Kind::Q4_0 ? gemm_q4_avx2 : gemm_q8_avx2;
        default:        return k == Kind::Q4_0 ? gemm_q4_scalar : gemm_q8_scalar;
    }
}
// Hot-path entry point: panels [panel0, panel0+npanels) of `pk` against np positions.
inline void gemm(const Packed& pk, size_t panel0, size_t npanels, const float* xd, const int32_t* xs,
                 const int8_t* xq, size_t np, float* Y /* = Y_base + pos0*stride + panel0*8 */,
                 size_t y_stride, const float* bias /* = bias + panel0*8 or nullptr */) {
    static const GemmFn q4 = select_gemm(Kind::Q4_0), q8 = select_gemm(Kind::Q8_0);
    (pk.kind == Kind::Q4_0 ? q4 : q8)(pk.data + panel0 * pk.panel_bytes, npanels, pk.panel_bytes, pk.nb,
                                     xd, xs, xq, pk.cols, np, Y, y_stride, bias);
}

// ───────────────────────────── cache of packed tensors ──────────────────────────
inline std::mutex& cache_mutex() { static std::mutex m; return m; }
inline std::unordered_map<const void*, std::unique_ptr<Packed>>& cache_map() {
    static std::unordered_map<const void*, std::unique_ptr<Packed>> m; return m;
}

// Decode streams every packed byte once per token from DRAM, so TLB misses on 4 KB pages cost real
// bandwidth (measured ~15-20% on a 110 MB Q4 matrix). Large buffers are 2 MB-aligned and marked
// MADV_HUGEPAGE before first touch (needs THP "madvise" or "always"; silently a no-op otherwise).
// Release with std::free as before. Set RAWLLM_NO_THP=1 to opt out.
inline uint8_t* alloc_weights(size_t total) {
    constexpr size_t HP = 2u << 20;
#if defined(__linux__)
    static const bool thp = [] { const char* e = std::getenv("RAWLLM_NO_THP"); return !(e && *e && *e != '0'); }();
    if (thp && total >= 2 * HP) {
        const size_t sz = (total + HP - 1) & ~(HP - 1);
        if (auto* p = static_cast<uint8_t*>(std::aligned_alloc(HP, sz))) {
            madvise(p, sz, MADV_HUGEPAGE);
            return p;
        }
    }
#endif
    return static_cast<uint8_t*>(std::aligned_alloc(64, (total + 63) & ~(size_t)63));
}

// Returns the packed form of the tensor whose GGUF-layout data starts at `src`, building it
// (in parallel on `pool`) the first time. Returns nullptr when repacking is disabled, the
// tensor has < 8 rows, allocation failed, or the same pointer was previously seen with a
// different shape/kind. Call from the thread that owns the pool (not from inside a pool task).
inline const Packed* get_or_build(const uint8_t* src, Kind kind, size_t rows, size_t cols,
                                  size_t row_bytes, util::ThreadPool& pool) {
    if (!enabled()) return nullptr;
    const size_t prow = rows & ~(PR - 1);
    if (prow == 0 || cols % 32 != 0) return nullptr;
    {
        std::lock_guard<std::mutex> lk(cache_mutex());
        auto it = cache_map().find((const void*)src);
        if (it != cache_map().end()) {
            const Packed* p = it->second.get();
            return (p->data && p->kind == kind && p->cols == cols && p->rows == prow) ? p : nullptr;
        }
    }
    auto pk = std::make_unique<Packed>();
    pk->kind = kind; pk->rows = prow; pk->cols = cols; pk->nb = cols / 32;
    pk->panel_bytes = pk->nb * (kind == Kind::Q4_0 ? Q4_BLK : Q8_BLK);
    const size_t total = (prow / PR) * pk->panel_bytes;
    pk->data = alloc_weights(total);
    if (pk->data) {
        const size_t npan = prow / PR;
        auto build = [&, kind](size_t p0, size_t p1) {
            for (size_t pn = p0; pn < p1; ++pn) {
                const uint8_t* s = src + pn * PR * row_bytes;
                uint8_t* d = pk->data + pn * pk->panel_bytes;
                if (kind == Kind::Q4_0) pack_q4_panel(s, row_bytes, pk->nb, d);
                else                    pack_q8_panel(s, row_bytes, pk->nb, d);
            }
        };
        const int nth = std::max(1, pool.size());
        if (nth == 1 || npan < 16) build(0, npan);
        else {
            const size_t chunk = (npan + (size_t)nth - 1) / (size_t)nth;
            for (size_t p0 = 0; p0 < npan; p0 += chunk) { const size_t p1 = std::min(npan, p0 + chunk); pool.submit([=, &build] { build(p0, p1); }); }
            pool.wait();
        }
    }
    std::lock_guard<std::mutex> lk(cache_mutex());
    auto& slot = cache_map()[(const void*)src];
    if (!slot) slot = std::move(pk);          // another caller may have raced us; first one wins
    return slot->data ? slot.get() : nullptr;
}

// The cache is keyed by data pointer ONLY and is never invalidated: if a model is unloaded and another
// loaded in the same process (or any buffer address gets reused), a same-shaped tensor at the same
// address silently gets the OLD packed panels. Call this whenever model weights are freed/reloaded.
inline void clear_cache() {
    std::lock_guard<std::mutex> lk(cache_mutex());
    cache_map().clear();
}

inline size_t cached_bytes() {
    std::lock_guard<std::mutex> lk(cache_mutex());
    size_t n = 0; for (auto& kv : cache_map()) if (kv.second->data) n += kv.second->total_bytes();
    return n;
}

} // namespace repack
