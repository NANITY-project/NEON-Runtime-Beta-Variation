#pragma once
// =============================================================================
// rawllm_simd_dispatch.hpp — picks the best available SIMD backend at
// compile time and exposes it as `simd::dot_f32`, `simd::axpy_f32`,
// `simd::dot_q4_0_q8_0`, `simd::dot_q8_0_q8_0`.
//
// This is the only SIMD header rawllm_forward.hpp needs to include; the
// backend-specific headers (rawllm_simd_scalar.hpp / _avx2.hpp / _avx512.hpp)
// are implementation detail included from here.
//
// Backend selection mirrors rawllm_common.hpp's RAWLLM_AVX512/RAWLLM_AVX2
// macros: AVX-512 > AVX2 > scalar, decided once at compile time, not
// runtime-dispatched (this engine builds per-target, not as a fat binary —
// see NEON.py's compile flag toggles).
//
// Opt-in correctness self-test: compile any translation unit that includes
// this header with -DRAWLLM_SIMD_SELFTEST and call simd::run_selftest()
// once at startup (e.g. from behind --probe). It checks whichever
// vectorized backend the build selected against the scalar reference over
// randomized blocks and returns false on any mismatch. Not compiled in by
// default — it's a development/CI aid, not part of the hot path.
// =============================================================================

#include "rawllm_common.hpp"
#include "rawllm_simd_scalar.hpp"
#include "rawllm_simd_avx2.hpp"
#include "rawllm_simd_avx512.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>

namespace simd {

// ── f32 dot / axpy ───────────────────────────────────────────────────────────
inline float dot_f32(const float* a, const float* b, size_t n) {
#if defined(RAWLLM_AVX512)
    return avx512::dot_f32(a, b, n);
#elif defined(RAWLLM_AVX2)
    return avx2::dot_f32(a, b, n);
#else
    return scalar::dot_f32(a, b, n);
#endif
}

inline void axpy_f32(float* out, float w, const float* v, size_t n) {
#if defined(RAWLLM_AVX2)
    // AVX-512 deliberately not preferred for axpy specifically — this loop
    // runs over head_dim in attention, small enough that AVX2 already
    // captures the win without the extra register-width complexity
    // (unchanged from the original single-file version's reasoning). But
    // an AVX-512-only build (RAWLLM_AVX512 defined, RAWLLM_AVX2 NOT — see
    // rawllm_common.hpp's #elif, only one of the two is ever set) has no
    // avx2:: namespace compiled in at all, so that build falls through to
    // the AVX-512 branch below instead of a dangling reference to avx2::.
    avx2::axpy_f32(out, w, v, n);
#elif defined(RAWLLM_AVX512)
    avx512::axpy_f32(out, w, v, n);
#else
    scalar::axpy_f32(out, w, v, n);
#endif
}

// ── fused Q4_0 x Q8_0 ────────────────────────────────────────────────────────
// One Q4_0-packed weight row (raw bytes: per-32-element-block 2-byte fp16
// scale + 16 packed-nibble bytes) against one Q8_0-quantized activation row
// (see quantize_rows_q8_0() in rawllm_forward.hpp — internal scratch format,
// not the on-disk GGUF Q8_0 layout).
inline float dot_q4_0_q8_0(const uint8_t* w, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 18;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        // fp16->fp32: shared bit trick, duplicated here rather than pulled
        // in from rawllm_forward.hpp to keep this header standalone /
        // includable before forward.hpp if a future call site wants that.
        uint32_t sign = (dh & 0x8000u) << 16;
        uint32_t exp  = (dh >> 10) & 0x1F;
        uint32_t mant = dh & 0x3FF;
        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) bits = sign;
            else {
                exp = 1;
                while (!(mant & 0x400)) { mant <<= 1; --exp; }
                mant &= 0x3FF;
                bits = sign | ((exp + 112) << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000u | (mant << 13);
        } else {
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
        float wd; std::memcpy(&wd, &bits, 4);

        const uint8_t* qs = blk + 2;
        const int8_t*  xq = x_q + b * 32;
        // FIX (perf): this used to be `int32_t sum_x = 0; for (j) sum_x +=
        // xq[j];` — recomputed from scratch on EVERY call, i.e. once per
        // (output row, block) pair. sum_x only depends on the activation
        // block, which is the same for every output row of a projection
        // (thousands of rows share it), so it was being redundantly
        // recomputed thousands of times over. quantize_rows_q8_0() now
        // computes it once, as a byproduct of the pass it already makes
        // over the same elements to find amax/scale, and passes it in.
        int32_t sum_x = x_sum[b];
#if defined(RAWLLM_AVX512)
        int32_t uncentered = avx512::block_isum_q4_0(qs, xq);
#elif defined(RAWLLM_AVX2)
        int32_t uncentered = avx2::block_isum_q4_0(qs, xq);
#else
        int32_t uncentered = scalar::block_isum_q4_0(qs, xq);
#endif
        int32_t isum = uncentered - 8 * sum_x;
        total += wd * x_d[b] * (float)isum;
    }
    return total;
}

// ── fused Q8_0 x Q8_0 ────────────────────────────────────────────────────────
// One on-disk-layout Q8_0 weight row (2-byte fp16 scale + 32 signed int8
// values per block — see dequant_q8_0() in rawllm_forward.hpp for the exact
// layout this matches) against one Q8_0-quantized activation row. Unlike
// Q4_0, there's no dequant possible to skip on the WEIGHT side (it's already
// int8), so this is a strict win over dequantize_row()+dot_f32(): no float
// materialization at all, straight int8 dot + one scale multiply per block.
// fp16->fp32 for a Q8_0 block header (34-byte stride: 2-byte scale + 32
// signed int8 values). Pulled out so the VNNI-paired path below and the
// per-block fallback path can share it without duplicating the bit trick.
inline float q8_0_block_scale(const uint8_t* blk) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    uint32_t sign = (dh & 0x8000u) << 16;
    uint32_t exp  = (dh >> 10) & 0x1F;
    uint32_t mant = dh & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; --exp; }
            mant &= 0x3FF;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float wd; std::memcpy(&wd, &bits, 4);
    return wd;
}

inline float dot_q8_0_q8_0(const uint8_t* w, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 34;
        float wd = q8_0_block_scale(blk);
        const int8_t* wq = reinterpret_cast<const int8_t*>(blk + 2);
        const int8_t* xq = x_q + b * 32;
#if defined(RAWLLM_AVX512) && defined(__AVX512VNNI__)
        // Only the VNNI path needs x_sum: it XORs the weight bytes to
        // unsigned so dpbusd (which requires an unsigned first operand)
        // can be used at all, which means it needs the same -128*sum_x
        // bias correction Q4_0 needs. The AVX-512F/AVX2/scalar widen paths
        // below sign-extend both operands directly and never introduce
        // that offset, so they have no use for x_sum.
        int32_t isum = avx512::block_isum_q8_0_vnni(wq, xq, x_sum[b]);
#elif defined(RAWLLM_AVX512)
        int32_t isum = avx512::block_isum_q8_0_f(wq, xq);
#elif defined(RAWLLM_AVX2)
        int32_t isum = avx2::block_isum_q8_0(wq, xq);
#else
        int32_t isum = scalar::block_isum_q8_0(wq, xq);
#endif
        total += wd * x_d[b] * (float)isum;
    }
    return total;
}

// ── self-test (opt-in) ───────────────────────────────────────────────────────
#if defined(RAWLLM_SIMD_SELFTEST)
#include <random>
#include <cstdio>

inline bool run_selftest() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> fdist(-1.f, 1.f);
    std::uniform_int_distribution<int> i8dist(-127, 127);
    std::uniform_int_distribution<int> u8dist(0, 255);
    bool ok = true;

    // dot_f32 / axpy_f32 vs scalar reference, various lengths (including
    // non-multiple-of-16 tails).
    for (int trial = 0; trial < 200; ++trial) {
        size_t n = 1 + (rng() % 300);
        std::vector<float> a(n), b(n), out_ref(n), out_test(n);
        for (auto& v : a) v = fdist(rng);
        for (auto& v : b) v = fdist(rng);
        float ref = scalar::dot_f32(a.data(), b.data(), n);
        float got = dot_f32(a.data(), b.data(), n);
        if (std::fabs(ref - got) > 1e-2f * std::max(1.0f, std::fabs(ref))) {
            std::fprintf(stderr, "[simd selftest] dot_f32 mismatch n=%zu ref=%f got=%f\n", n, ref, got);
            ok = false;
        }
        for (auto& v : out_ref) v = fdist(rng);
        out_test = out_ref;
        float w = fdist(rng);
        scalar::axpy_f32(out_ref.data(), w, b.data(), n);
        axpy_f32(out_test.data(), w, b.data(), n);
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(out_ref[i] - out_test[i]) > 1e-3f) {
                std::fprintf(stderr, "[simd selftest] axpy_f32 mismatch n=%zu i=%zu\n", n, i);
                ok = false;
                break;
            }
        }
    }

    // Q4_0 / Q8_0 block int32 accumulators vs scalar, over 50,000 random
    // blocks each (matches the bit-exact validation this kernel was
    // developed against).
    for (int trial = 0; trial < 50000; ++trial) {
        uint8_t qs[16]; int8_t xq32[32];
        for (auto& v : qs) v = (uint8_t)u8dist(rng);
        for (auto& v : xq32) v = (int8_t)i8dist(rng);
        int32_t ref = scalar::block_isum_q4_0(qs, xq32);
#if defined(RAWLLM_AVX512)
        int32_t got = avx512::block_isum_q4_0(qs, xq32);
#elif defined(RAWLLM_AVX2)
        int32_t got = avx2::block_isum_q4_0(qs, xq32);
#else
        int32_t got = ref;
#endif
        if (ref != got) {
            std::fprintf(stderr, "[simd selftest] block_isum_q4_0 mismatch trial=%d ref=%d got=%d\n", trial, ref, got);
            ok = false;
        }

        int8_t w32[32];
        for (auto& v : w32) v = (int8_t)i8dist(rng);
        int32_t refq8 = scalar::block_isum_q8_0(w32, xq32);
#if defined(RAWLLM_AVX512)
        int32_t gotq8 = avx512::block_isum_q8_0_f(w32, xq32);
#elif defined(RAWLLM_AVX2)
        int32_t gotq8 = avx2::block_isum_q8_0(w32, xq32);
#else
        int32_t gotq8 = refq8;
#endif
        if (refq8 != gotq8) {
            std::fprintf(stderr, "[simd selftest] block_isum_q8_0 mismatch trial=%d ref=%d got=%d\n", trial, refq8, gotq8);
            ok = false;
        }

#if defined(RAWLLM_AVX512) && defined(__AVX512VNNI__)
        int32_t sum_x32 = 0;
        for (auto v : xq32) sum_x32 += v;
        int32_t vgot = avx512::block_isum_q8_0_vnni(w32, xq32, sum_x32);
        if (refq8 != vgot) {
            std::fprintf(stderr,
                "[simd selftest] block_isum_q8_0_vnni mismatch trial=%d ref=%d got=%d\n",
                trial, refq8, vgot);
            ok = false;
        }
#endif
    }

#if defined(RAWLLM_AVX512) && defined(__AVX512VNNI__)
    // Full dot_q8_0_q8_0() row test — catches layout bugs (per-block
    // scale application, sum_x indexing) that the isolated primitive test
    // above can't see.
    {
        std::uniform_real_distribution<float> ddist(0.001f, 2.0f);
        for (int trial = 0; trial < 500; ++trial) {
            size_t nb = 1 + (rng() % 9);
            size_t cols = nb * 32;
            std::vector<uint8_t> wrow(nb * 34);
            std::vector<float> x_d(nb);
            std::vector<int32_t> x_sum(nb);
            std::vector<int8_t> x_q(cols);
            for (auto& v : wrow) v = (uint8_t)u8dist(rng);
            for (auto& v : x_d)  v = ddist(rng);
            for (auto& v : x_q)  v = (int8_t)i8dist(rng);
            for (size_t b = 0; b < nb; ++b) {
                int32_t s = 0;
                for (size_t j = 0; j < 32; ++j) s += x_q[b * 32 + j];
                x_sum[b] = s;
            }

            // Fix up each block's 2-byte header to a well-formed (non-NaN,
            // non-subnormal-edge-case) fp16 so this is testing the dot
            // math, not fp16 decode edge cases already covered elsewhere.
            for (size_t b = 0; b < nb; ++b) {
                uint16_t half = (uint16_t)(0x3C00 + (rng() % 0x400)); // ~[1,2)
                std::memcpy(&wrow[b * 34], &half, 2);
            }

            float ref = 0.f;
            for (size_t b = 0; b < nb; ++b) {
                float wd = q8_0_block_scale(&wrow[b * 34]);
                int32_t isum = scalar::block_isum_q8_0(
                    reinterpret_cast<const int8_t*>(&wrow[b * 34 + 2]),
                    &x_q[b * 32]);
                ref += wd * x_d[b] * (float)isum;
            }
            float got = dot_q8_0_q8_0(wrow.data(), x_d.data(), x_sum.data(), x_q.data(), cols);
            if (std::fabs(ref - got) > 1e-2f * std::max(1.0f, std::fabs(ref))) {
                std::fprintf(stderr,
                    "[simd selftest] dot_q8_0_q8_0 mismatch trial=%d nb=%zu ref=%f got=%f\n",
                    trial, nb, ref, got);
                ok = false;
            }
        }
    }
#endif

    // dot_q4_0_q8_0() row test, all backends: this now sources sum_x from
    // quantize_rows_q8_0()'s precomputed output instead of recomputing it
    // inline (see the FIX comment at dot_q4_0_q8_0()'s definition), so
    // this checks that plumbing end to end, not just the block kernel.
    {
        std::uniform_real_distribution<float> ddist(0.001f, 2.0f);
        for (int trial = 0; trial < 500; ++trial) {
            size_t nb = 1 + (rng() % 9);
            size_t cols = nb * 32;
            std::vector<uint8_t> wrow(nb * 18); // 2-byte scale + 16 packed bytes/block
            std::vector<float> x_d(nb);
            std::vector<int32_t> x_sum(nb);
            std::vector<int8_t> x_q(cols);
            for (auto& v : wrow) v = (uint8_t)u8dist(rng);
            for (auto& v : x_d)  v = ddist(rng);
            for (auto& v : x_q)  v = (int8_t)i8dist(rng);
            for (size_t b = 0; b < nb; ++b) {
                uint16_t half = (uint16_t)(0x3C00 + (rng() % 0x400));
                std::memcpy(&wrow[b * 18], &half, 2);
                int32_t s = 0;
                for (size_t j = 0; j < 32; ++j) s += x_q[b * 32 + j];
                x_sum[b] = s;
            }

            float ref = 0.f;
            for (size_t b = 0; b < nb; ++b) {
                uint16_t dh; std::memcpy(&dh, &wrow[b * 18], 2);
                uint32_t sign = (dh & 0x8000u) << 16, exp = (dh >> 10) & 0x1F, mant = dh & 0x3FF, bits;
                if (exp == 0) { bits = sign; }
                else if (exp == 0x1F) { bits = sign | 0x7F800000u | (mant << 13); }
                else { bits = sign | ((exp + 112) << 23) | (mant << 13); }
                float wd; std::memcpy(&wd, &bits, 4);
                int32_t uncentered = scalar::block_isum_q4_0(&wrow[b * 18 + 2], &x_q[b * 32]);
                int32_t isum = uncentered - 8 * x_sum[b];
                ref += wd * x_d[b] * (float)isum;
            }
            float got = dot_q4_0_q8_0(wrow.data(), x_d.data(), x_sum.data(), x_q.data(), cols);
            if (std::fabs(ref - got) > 1e-2f * std::max(1.0f, std::fabs(ref))) {
                std::fprintf(stderr,
                    "[simd selftest] dot_q4_0_q8_0 mismatch trial=%d nb=%zu ref=%f got=%f\n",
                    trial, nb, ref, got);
                ok = false;
            }
        }
    }

    if (ok) std::fprintf(stderr, "[simd selftest] all backends match scalar reference (dot_f32/axpy_f32, %d trials; block_isum_q4_0/q8_0, 50000 trials)\n", 200);
    return ok;
}
#endif // RAWLLM_SIMD_SELFTEST

} // namespace simd
