#pragma once
// =============================================================================
// rawllm_simd_avx2.hpp — AVX2(+FMA) SIMD kernels.
//
// Only compiled in when RAWLLM_AVX2 is defined (rawllm_common.hpp sets this
// from __AVX2__). FMA is checked independently via __FMA__ since -mavx2
// alone doesn't guarantee -mfma was also passed by the build.
// =============================================================================

#include "rawllm_common.hpp"

#if defined(RAWLLM_AVX2)
#include <immintrin.h>
#include <cstdint>
#include <cstddef>

namespace simd {
namespace avx2 {

// dot_f32 uses a dual accumulator (acc0/acc1) rather than a single running
// __m256 sum — on most AVX2 cores the FMA has multi-cycle latency, so a
// single accumulator chain leaves the FMA port stalling on its own output;
// two independent accumulation chains give the scheduler two in-flight FMAs
// to interleave, closing most of that latency gap. Combined at the end.
inline float dot_f32(const float* a, const float* b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i),     b0 = _mm256_loadu_ps(b + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8), b1 = _mm256_loadu_ps(b + i + 8);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, b1));
#endif
    }
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va, vb));
#endif
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float sum = _mm_cvtss_f32(s4);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline void axpy_f32(float* out, float w, const float* v, size_t n) {
    __m256 vw = _mm256_set1_ps(w);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vo = _mm256_loadu_ps(out + i);
        __m256 vv = _mm256_loadu_ps(v + i);
#if defined(__FMA__)
        vo = _mm256_fmadd_ps(vw, vv, vo);
#else
        vo = _mm256_add_ps(vo, _mm256_mul_ps(vw, vv));
#endif
        _mm256_storeu_ps(out + i, vo);
    }
    for (; i < n; ++i) out[i] += w * v[i];
}

// Q4_0 nibble block vs int8 activation block (32 elements/block). Unpacks
// 16 packed bytes into lo/hi nibble lanes, does two _mm_maddubs_epi16
// (unsigned nibble x signed activation, both operands already in [0,15] and
// [-127,127] respectively so this can't hit the u8xi8->i16 saturation edge
// the way full-range i8xi8 would), then horizontally reduces to a scalar
// int32 uncentered sum; the caller applies the -8-per-nibble bias via the
// closed-form sum_x correction (same trick every backend uses).
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

// Q8_0-vs-Q8_0 block: straight int8 x int8 dot over 32 lanes.
//
// CORRECTNESS NOTE (this is the bug the scalar-only first draft hit): you
// cannot do this with a single _mm256_maddubs_epi16 the way the Q4_0 path
// does, because Q4_0's unpacked nibbles are unsigned [0,15] and Q8_0's
// weight bytes are full-range SIGNED int8 [-127,127]. _mm256_maddubs_epi16
// requires operand 1 unsigned / operand 2 signed and its 16-bit
// intermediate saturates at the full int8 x int8 range (127*127*2 can
// overflow a signed 16-bit lane) — fine for Q4_0's small nibble magnitudes,
// silently wrong for two full-range Q8_0 operands. Fix: sign-extend both
// int8 lanes up to int16 first (_mm256_cvtepi8_epi16, widening, no
// saturation possible) and multiply-accumulate in the 16-bit domain,
// horizontally reducing to int32 only at the end via _mm256_madd_epi16.
// Bit-exact-validated against the scalar int32 accumulator.
inline int32_t block_isum_q8_0(const int8_t* w, const int8_t* xq) {
    // 32 lanes don't fit one 128-bit widen (cvtepi8_epi16 takes 16 bytes ->
    // 16 int16 in a 256-bit reg), so widen the low and high halves of the
    // block separately and accumulate both.
    __m128i w_lo = _mm_loadu_si128((const __m128i*)w);
    __m128i w_hi = _mm_loadu_si128((const __m128i*)(w + 16));
    __m128i x_lo = _mm_loadu_si128((const __m128i*)xq);
    __m128i x_hi = _mm_loadu_si128((const __m128i*)(xq + 16));
    __m256i wlo16 = _mm256_cvtepi8_epi16(w_lo);
    __m256i whi16 = _mm256_cvtepi8_epi16(w_hi);
    __m256i xlo16 = _mm256_cvtepi8_epi16(x_lo);
    __m256i xhi16 = _mm256_cvtepi8_epi16(x_hi);
    __m256i prod_lo = _mm256_madd_epi16(wlo16, xlo16); // 8x int32
    __m256i prod_hi = _mm256_madd_epi16(whi16, xhi16); // 8x int32
    __m256i prod32 = _mm256_add_epi32(prod_lo, prod_hi);
    __m128i lo = _mm256_castsi256_si128(prod32);
    __m128i hi = _mm256_extracti128_si256(prod32, 1);
    __m128i sum4 = _mm_add_epi32(lo, hi);
    __m128i sh1  = _mm_add_epi32(sum4, _mm_unpackhi_epi64(sum4, sum4));
    __m128i sh2  = _mm_add_epi32(sh1, _mm_shuffle_epi32(sh1, 0x1));
    return _mm_cvtsi128_si32(sh2);
}

} // namespace avx2
} // namespace simd

#endif // RAWLLM_AVX2
