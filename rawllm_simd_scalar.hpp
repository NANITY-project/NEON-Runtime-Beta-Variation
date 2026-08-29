#pragma once
// =============================================================================
// rawllm_simd_scalar.hpp — portable scalar fallback for the NEON SIMD kernels.
//
// This file is deliberately backend-agnostic: no intrinsics, no #ifdefs on
// __AVX2__/__AVX512F__. It is included unconditionally by
// rawllm_simd_dispatch.hpp and serves double duty as:
//   1. the fallback path for any build without AVX2/AVX-512, and
//   2. the correctness REFERENCE that rawllm_simd_dispatch.hpp's opt-in
//      self-test (-DRAWLLM_SIMD_SELFTEST) checks every vectorized backend
//      against.
//
// Everything here lives in `simd::scalar`, mirroring the sibling
// `simd::avx2` / `simd::avx512` namespaces in the other headers.
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>

namespace simd {
namespace scalar {

inline float dot_f32(const float* a, const float* b, size_t n) {
    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline void axpy_f32(float* out, float w, const float* v, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] += w * v[i];
}

// Q4_0 nibble-vs-int8 block inner product, UNCENTERED: nibbles are used
// as-is in [0,15] rather than bias-shifted to [-8,7]. Every backend
// (scalar/AVX2/AVX-512) returns this same uncentered form; the shared
// caller in rawllm_simd_dispatch.hpp's dot_q4_0_q8_0() applies the -8-per-
// nibble bias once via the closed-form `uncentered - 8*sum_x` correction.
// Returning the pre-correction value here (rather than baking the bias in)
// is what lets the self-test compare this scalar reference directly
// against the vectorized backends' block_isum_q4_0 — they'd disagree by
// construction if one centered and the other didn't.
inline int32_t block_isum_q4_0(const uint8_t* qs, const int8_t* xq) {
    int32_t isum = 0;
    for (int j = 0; j < 16; ++j) {
        int lo = (qs[j] & 0x0F);
        int hi = (qs[j] >> 4);
        isum += lo * (int)xq[j] + hi * (int)xq[j + 16];
    }
    return isum;
}

// Q8_0-vs-Q8_0 block inner product: straight int8 x int8 dot over 32 lanes,
// no dequant, no nibble unpack, no bias correction needed (Q8_0 has no
// zero-point offset).
inline int32_t block_isum_q8_0(const int8_t* w, const int8_t* xq) {
    int32_t isum = 0;
    for (int j = 0; j < 32; ++j) isum += (int)w[j] * (int)xq[j];
    return isum;
}

} // namespace scalar
} // namespace simd
