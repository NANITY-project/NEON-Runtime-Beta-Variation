#pragma once
// =============================================================================
// rawllm_simd_avx512.hpp — AVX-512 SIMD kernels, with an optional
// AVX-512-VNNI fast path for the int8 block dot products.
//
// Only compiled in when RAWLLM_AVX512 is defined (rawllm_common.hpp sets
// this from __AVX512F__ && USE_AVX512). The VNNI path additionally requires
// __AVX512VNNI__ (i.e. -mavx512vnni on the compile line) and is guarded
// independently, so an AVX-512F-only build (no VNNI) still gets the
// f32 dot/axpy speedup and falls back to a portable madd-based int8 path
// for the quantized kernels.
// =============================================================================

#include "rawllm_common.hpp"

#if defined(RAWLLM_AVX512)
#include <immintrin.h>
#include <cstdint>
#include <cstddef>

namespace simd {
namespace avx512 {

inline float dot_f32(const float* a, const float* b, size_t n) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline void axpy_f32(float* out, float w, const float* v, size_t n) {
    __m512 vw = _mm512_set1_ps(w);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vo = _mm512_loadu_ps(out + i);
        __m512 vv = _mm512_loadu_ps(v + i);
        vo = _mm512_fmadd_ps(vw, vv, vo);
        _mm512_storeu_ps(out + i, vo);
    }
    for (; i < n; ++i) out[i] += w * v[i];
}

// Q4_0 block (32 elements): unpack the 16 packed nibble bytes into two
// 128-bit unsigned-nibble lanes, multiply against the matching int8
// activation halves. This stays at 128-bit width deliberately — a single
// Q4_0 block is only 32 elements, well under one 512-bit register's worth
// of int8 lanes (64), so widening to 512 here would mean mostly-idle lanes
// and extra shuffle overhead for no benefit; AVX-512's actual win for this
// kernel is in the caller's ability to pull two blocks per iteration
// (see dispatch header), not in per-block width.
inline int32_t block_isum_q4_0(const uint8_t* qs, const int8_t* xq) {
    __m128i raw  = _mm_loadu_si128((const __m128i*)qs);
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i lo_n = _mm_and_si128(raw, mask);
    __m128i hi_n = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
    __m128i x_lo = _mm_loadu_si128((const __m128i*)(xq));
    __m128i x_hi = _mm_loadu_si128((const __m128i*)(xq + 16));
    __m128i p_lo = _mm_maddubs_epi16(lo_n, x_lo);
    __m128i p_hi = _mm_maddubs_epi16(hi_n, x_hi);
    __m128i sum16 = _mm_add_epi16(p_lo, p_hi);
    __m128i sum32 = _mm_madd_epi16(sum16, _mm_set1_epi16(1));
    __m128i hi64  = _mm_unpackhi_epi64(sum32, sum32);
    __m128i sum2  = _mm_add_epi32(sum32, hi64);
    __m128i sum1  = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 0x1));
    return _mm_cvtsi128_si32(sum1);
}

// Q8_0-vs-Q8_0 block (32 elements), portable AVX-512F path: same
// widen-to-int16-then-madd approach as the AVX2 backend, just done in one
// 256-bit widen instead of two 128-bit halves, since AVX-512F still gives
// us _mm256_cvtepi8_epi16 for free. Used when VNNI isn't available.
inline int32_t block_isum_q8_0_f(const int8_t* w, const int8_t* xq) {
    __m256i w32 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)w));
    __m256i x32 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)xq));
    __m256i w32b = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + 16)));
    __m256i x32b = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xq + 16)));
    __m256i prod = _mm256_add_epi32(_mm256_madd_epi16(w32, x32), _mm256_madd_epi16(w32b, x32b));
    __m128i lo = _mm256_castsi256_si128(prod);
    __m128i hi = _mm256_extracti128_si256(prod, 1);
    __m128i sum4 = _mm_add_epi32(lo, hi);
    __m128i sh1  = _mm_add_epi32(sum4, _mm_unpackhi_epi64(sum4, sum4));
    __m128i sh2  = _mm_add_epi32(sh1, _mm_shuffle_epi32(sh1, 0x1));
    return _mm_cvtsi128_si32(sh2);
}

#if defined(__AVX512VNNI__)
// VNNI fast path: _mm512_dpbusd_epi32 computes, per 32-bit lane, the sum of
// four u8*i8 products accumulated directly into an existing int32 lane —
// exactly the primitive int8 GEMM kernels are built around. It requires
// operand 1 UNSIGNED and operand 2 signed, so the weight bytes (full-range
// signed int8) are offset to unsigned by XOR 0x80 (equivalent to +128) and
// the activation is bias-corrected for that shift the same way the Q4_0
// path corrects for its nibble bias: sum(un-signed) = sum((w+128)*x) =
// sum(w*x) + 128*sum(x), so subtracting 128*sum_x recovers the true dot.
// One 512-bit dpbusd call covers a full 64-byte span, i.e. two Q8_0 blocks
// (32 elements each) at once — the caller (dispatch header) is the one
// that takes advantage of that by pairing blocks.
inline __m512i dpbusd_biased(const int8_t* w64, const int8_t* x64) {
    __m512i w_raw = _mm512_loadu_si512((const void*)w64);
    __m512i x_raw = _mm512_loadu_si512((const void*)x64);
    __m512i bias  = _mm512_set1_epi8((char)0x80);
    __m512i w_u   = _mm512_xor_si512(w_raw, bias); // (w + 128) as unsigned
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_u, x_raw);
}

// Sum of int8 activation lanes across a 64-byte span (needed for the
// dpbusd bias correction above); done via madd against an all-ones int8
// vector so it stays in integer domain and vectorizes the same way.
inline int32_t sum_i8_64(const int8_t* x64) {
    __m512i x = _mm512_loadu_si512((const void*)x64);
    __m512i ones = _mm512_set1_epi8(1);
    // dpbusd needs an unsigned first operand; ones is already in [0,1] so
    // this is safe without an offset.
    __m512i sum32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, x);
    return _mm512_reduce_add_epi32(sum32);
}

// Per-block VNNI Q8_0 dot, 256-bit width (_mm256_dpbusd_epi32 — VNNI
// isn't only a 512-bit instruction; AVX-512VL exposes 128/256-bit forms
// too). This supersedes an earlier attempt at pairing two blocks into one
// 512-bit dpbusd_biased() call: that required staging the two blocks'
// int8 payloads into a contiguous 64-byte scratch buffer first (Q8_0's
// on-disk layout interleaves a 2-byte fp16 scale between every 32-byte
// payload, so blk1's payload isn't adjacent to blk0's), and the resulting
// store-then-immediately-reload created a store-forwarding stall that
// measured ~4x SLOWER end to end than the plain AVX-512F path despite the
// VNNI math itself being cheap in isolation. Operating on one block at a
// time at 256-bit width needs no staging (each block's 32-byte payload is
// read directly via loadu).
//
// Takes the block's activation sum as a precomputed scalar (sum_x) rather
// than computing it here via a second dpbusd — that second pass was the
// other thing eating the VNNI path's advantage: measured back-to-back
// against the AVX-512F widen+madd path, computing xsum in-kernel made
// this ~15% SLOWER despite doing "the same amount of work" on paper,
// because it's a second full load+dpbusd+reduce dependency chain per
// block on top of the first. sum_x depends only on the activation block —
// identical across every weight row that block gets dotted against — so
// having quantize_rows_q8_0() compute it once (as a free byproduct of the
// pass it already makes over those same 32 elements for the scale) and
// passing it in turns this into one dpbusd + one reduce + one scalar
// subtract, no second vector pass at all.
inline int32_t block_isum_q8_0_vnni(const int8_t* w32, const int8_t* xq, int32_t sum_x) {
    __m256i w_raw = _mm256_loadu_si256((const __m256i*)w32);
    __m256i x_raw = _mm256_loadu_si256((const __m256i*)xq);
    __m256i bias  = _mm256_set1_epi8((char)0x80);
    __m256i w_u   = _mm256_xor_si256(w_raw, bias);
    __m256i prod  = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_u, x_raw);

    __m128i lo = _mm256_castsi256_si128(prod);
    __m128i hi = _mm256_extracti128_si256(prod, 1);
    __m128i sum4 = _mm_add_epi32(lo, hi);
    __m128i sh1  = _mm_add_epi32(sum4, _mm_unpackhi_epi64(sum4, sum4));
    __m128i sh2  = _mm_add_epi32(sh1, _mm_shuffle_epi32(sh1, 0x1));
    int32_t uncentered = _mm_cvtsi128_si32(sh2);
    return uncentered - 128 * sum_x;
}
#endif // __AVX512VNNI__

} // namespace avx512
} // namespace simd

#endif // RAWLLM_AVX512
