#pragma once
// =============================================================================
// rawllm_simd_dispatch.hpp — picks the best available SIMD backend AT
// RUNTIME and exposes it as `simd::dot_f32`, `simd::axpy_f32`,
// `simd::dot_q4_0_q8_0`, `simd::dot_q8_0_q8_0`.
//
// PREVIOUSLY: backend selection was `#if defined(RAWLLM_AVX512) / elif
// RAWLLM_AVX2 / else`, decided once at compile time from whatever -m flags
// the whole translation unit was built with. That meant this engine had to
// be built per-target (a "-DUSE_AVX512 build" and a separate "-DUSE_AVX2
// build" were genuinely different binaries) and running a build's binary on
// a CPU one tier down than what it was compiled for was a SIGILL, not a
// graceful fallback — there was no way to ship one binary that's fast on an
// AVX-512 machine and still merely works (rather than crashes) on an
// AVX2-only one.
//
// NOW: dot_f32/axpy_f32/dot_q4_0_q8_0/dot_q8_0_q8_0 below are each defined
// several times with different `__attribute__((target(...)))` tags on the
// SAME function name — this is GCC/Clang function multiversioning. The
// compiler emits one code-genned copy per tag, plus an ifunc resolver that
// runs `cpuid` once (the first time the symbol is actually referenced, not
// on every call) and rewrites the call site to jump straight to the best
// match forever after. One binary, built with NO -march/-mavx* flags at
// all, correctly runs the AVX-512-VNNI path on hardware that has it and the
// portable scalar path on hardware that doesn't — verified empirically (see
// the compile log this was developed against): building this exact header
// with zero -m flags and running on an AVX-512-VNNI CPU selects the VNNI
// clone; the same binary's ifunc resolver would pick AVX2 or scalar on
// lesser hardware without needing a rebuild.
//
// The backend-specific headers (rawllm_simd_scalar.hpp / _avx2.hpp /
// _avx512.hpp) now define their functions with per-function target
// attributes too and are ALWAYS compiled in (no more file-level
// RAWLLM_AVX2/RAWLLM_AVX512 gate) — every backend's machine code exists in
// every build; only the resolver decides which runs.
//
// Opt-in correctness self-test: compile any translation unit that includes
// this header with -DRAWLLM_SIMD_SELFTEST and call simd::run_selftest()
// once at startup (e.g. from behind --probe). Because every backend is now
// always compiled in regardless of the running CPU, the self-test calls
// each backend's specific implementation directly (bypassing the
// multiversioning resolver, which only ever calls what run on the current
// CPU) — so it guards each one with `__builtin_cpu_supports(...)` rather
// than the old compile-time `#if defined(RAWLLM_AVX2)`, since unconditionally
// executing e.g. the AVX-512 clone on a CPU without AVX-512 would SIGILL.
// This also means the self-test is now exhaustive on every build regardless
// of what CPU it's compiled on — previously it could only test whichever
// single backend that TU happened to be compiled for.
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
__attribute__((target("default")))
inline float dot_f32(const float* a, const float* b, size_t n) {
    return scalar::dot_f32(a, b, n);
}
__attribute__((target("avx2,fma")))
inline float dot_f32(const float* a, const float* b, size_t n) {
    return avx2::dot_f32(a, b, n);
}
__attribute__((target("avx512f")))
inline float dot_f32(const float* a, const float* b, size_t n) {
    return avx512::dot_f32(a, b, n);
}

__attribute__((target("default")))
inline void axpy_f32(float* out, float w, const float* v, size_t n) {
    scalar::axpy_f32(out, w, v, n);
}
__attribute__((target("avx2,fma")))
inline void axpy_f32(float* out, float w, const float* v, size_t n) {
    // AVX-512 deliberately not used for axpy specifically — this loop runs
    // over head_dim in attention, small enough that AVX2 already captures
    // the win without the extra register-width complexity (unchanged
    // reasoning from the original single-file version). There is no
    // avx512-tagged overload of axpy_f32 at all below, so the resolver's
    // only choice on an AVX-512 CPU that also has AVX2 (i.e. every real
    // AVX-512 CPU) is this AVX2 clone or the plain default — it correctly
    // prefers this one, same outcome as the old file's explicit AVX2-over-
    // AVX512 preference, just arrived at by omission instead of an #elif.
    avx2::axpy_f32(out, w, v, n);
}

// ── fused Q4_0 x Q8_0 ────────────────────────────────────────────────────────
// One Q4_0-packed weight row (raw bytes: per-32-element-block 2-byte fp16
// scale + 16 packed-nibble bytes) against one Q8_0-quantized activation row
// (see quantize_rows_q8_0() in rawllm_forward.hpp — internal scratch format,
// not the on-disk GGUF Q8_0 layout).
//
// fp16->fp32 bit trick — target-agnostic (plain integer ops, no intrinsics),
// so it needs no target attribute and can be called from any of the
// multiversioned clones below without restriction.
inline float f16_to_f32(uint16_t dh) {
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

__attribute__((target("default")))
inline float dot_q4_0_q8_0(const uint8_t* w, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 18;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        float wd = f16_to_f32(dh);
        // FIX (perf): x_sum used to be recomputed from scratch on EVERY
        // call, i.e. once per (output row, block) pair, even though it only
        // depends on the activation block — the SAME value for every row
        // of a projection (thousands of rows share it). quantize_rows_q8_0()
        // now computes it once, as a byproduct of the pass it already makes
        // over the same elements to find amax/scale, and passes it in.
        int32_t uncentered = scalar::block_isum_q4_0(blk + 2, x_q + b * 32);
        int32_t isum = uncentered - 8 * x_sum[b];
        total += wd * x_d[b] * (float)isum;
    }
    return total;
}
__attribute__((target("avx2,fma")))
inline float dot_q4_0_q8_0(const uint8_t* w, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 18;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        float wd = f16_to_f32(dh);
        int32_t uncentered = avx2::block_isum_q4_0(blk + 2, x_q + b * 32);
        int32_t isum = uncentered - 8 * x_sum[b];
        total += wd * x_d[b] * (float)isum;
    }
    return total;
}
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline float dot_q4_0_q8_0(const uint8_t* w, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 18;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        float wd = f16_to_f32(dh);
        int32_t uncentered = avx512::block_isum_q4_0(blk + 2, x_q + b * 32);
        int32_t isum = uncentered - 8 * x_sum[b];
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
inline float q8_0_block_scale(const uint8_t* blk) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    return f16_to_f32(dh);
}

__attribute__((target("default")))
inline float dot_q8_0_q8_0(const uint8_t* w, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 34;
        float wd = q8_0_block_scale(blk);
        int32_t isum = scalar::block_isum_q8_0(
            reinterpret_cast<const int8_t*>(blk + 2), x_q + b * 32);
        total += wd * x_d[b] * (float)isum;
    }
    return total;
}
__attribute__((target("avx2,fma")))
inline float dot_q8_0_q8_0(const uint8_t* w, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 34;
        float wd = q8_0_block_scale(blk);
        int32_t isum = avx2::block_isum_q8_0(
            reinterpret_cast<const int8_t*>(blk + 2), x_q + b * 32);
        total += wd * x_d[b] * (float)isum;
    }
    return total;
}
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline float dot_q8_0_q8_0(const uint8_t* w, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 34;
        float wd = q8_0_block_scale(blk);
        int32_t isum = avx512::block_isum_q8_0_f(
            reinterpret_cast<const int8_t*>(blk + 2), x_q + b * 32);
        total += wd * x_d[b] * (float)isum;
    }
    return total;
}
// Only this clone (the one requiring avx512vnni specifically, on top of
// avx512f/bw/vl) actually reads x_sum: block_isum_q8_0_vnni's dpbusd
// approach XORs the weight bytes to unsigned so dpbusd (which requires an
// unsigned first operand) can be used at all, which needs the same
// -128*sum_x bias correction Q4_0 needs. Every other clone above sign-
// extends both operands directly and never introduces that offset, so
// they never touch x_sum. The resolver picks this one automatically on
// VNNI-capable CPUs, ranking it above the plain avx512f clone above.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
inline float dot_q8_0_q8_0(const uint8_t* w, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, size_t cols) {
    size_t nb = cols / 32;
    float total = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w + b * 34;
        float wd = q8_0_block_scale(blk);
        int32_t isum = avx512::block_isum_q8_0_vnni(
            reinterpret_cast<const int8_t*>(blk + 2), x_q + b * 32, x_sum[b]);
        total += wd * x_d[b] * (float)isum;
    }
    return total;
}

// ── 4-row-batched fused dots (register tiling) ──────────────────────────────
// PREVIOUSLY: proj_all_positions() called dot_q4_0_q8_0()/dot_q8_0_q8_0()
// once per output row, independently. Each call re-reads the SAME
// activation block (x_q/x_d/x_sum) from memory that every other row's call
// also reads — the same handful of cache lines get reloaded once per row,
// thousands of times per projection, even though L1 keeps them hot (so
// this isn't a correctness or even a cache-miss problem, it's redundant
// load *instructions* and, more importantly, a missed ILP opportunity: one
// row's dot is a single dependent accumulation chain — load, multiply-add,
// load, multiply-add — that leaves most CPUs' multiple FMA/ALU ports idle
// waiting on that chain's own latency, the same reason dot_f32's dual-
// accumulator trick exists.
//
// The dot4 functions below process 4 output rows in one call, loading each
// block's activation data once and reusing it across 4 independent
// per-row accumulator chains that the compiler can interleave — closing
// the same kind of latency gap dot_f32 already closes, just for the
// int8 fused paths, and without needing any change to the on-disk weight
// layout (no interleaved repack format, no changes to the Python
// conversion tool): the 4 weight rows are read from wherever they already
// live, just passed in as 4 separate pointers instead of looping 4 times.
//
// proj_all_positions() below calls these for every group of 4 consecutive
// output rows, falling back to the single-row dot for whatever remainder
// doesn't divide evenly by 4.
__attribute__((target("default")))
inline void dot4_q4_0_q8_0(const uint8_t* w0, const uint8_t* w1, const uint8_t* w2, const uint8_t* w3,
                            const float* x_d, const int32_t* x_sum, const int8_t* x_q, size_t cols,
                            float out[4]) {
    size_t nb = cols / 32;
    float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const int8_t* xq = x_q + b * 32;
        int32_t sx = x_sum[b];
        float xd = x_d[b];
        uint16_t dh0, dh1, dh2, dh3;
        std::memcpy(&dh0, w0 + b * 18, 2); std::memcpy(&dh1, w1 + b * 18, 2);
        std::memcpy(&dh2, w2 + b * 18, 2); std::memcpy(&dh3, w3 + b * 18, 2);
        int32_t i0 = scalar::block_isum_q4_0(w0 + b * 18 + 2, xq) - 8 * sx;
        int32_t i1 = scalar::block_isum_q4_0(w1 + b * 18 + 2, xq) - 8 * sx;
        int32_t i2 = scalar::block_isum_q4_0(w2 + b * 18 + 2, xq) - 8 * sx;
        int32_t i3 = scalar::block_isum_q4_0(w3 + b * 18 + 2, xq) - 8 * sx;
        t0 += f16_to_f32(dh0) * xd * (float)i0;
        t1 += f16_to_f32(dh1) * xd * (float)i1;
        t2 += f16_to_f32(dh2) * xd * (float)i2;
        t3 += f16_to_f32(dh3) * xd * (float)i3;
    }
    out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
}
__attribute__((target("avx2,fma")))
inline void dot4_q4_0_q8_0(const uint8_t* w0, const uint8_t* w1, const uint8_t* w2, const uint8_t* w3,
                            const float* x_d, const int32_t* x_sum, const int8_t* x_q, size_t cols,
                            float out[4]) {
    size_t nb = cols / 32;
    float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const int8_t* xq = x_q + b * 32;
        int32_t sx = x_sum[b];
        float xd = x_d[b];
        uint16_t dh0, dh1, dh2, dh3;
        std::memcpy(&dh0, w0 + b * 18, 2); std::memcpy(&dh1, w1 + b * 18, 2);
        std::memcpy(&dh2, w2 + b * 18, 2); std::memcpy(&dh3, w3 + b * 18, 2);
        int32_t i0 = avx2::block_isum_q4_0(w0 + b * 18 + 2, xq) - 8 * sx;
        int32_t i1 = avx2::block_isum_q4_0(w1 + b * 18 + 2, xq) - 8 * sx;
        int32_t i2 = avx2::block_isum_q4_0(w2 + b * 18 + 2, xq) - 8 * sx;
        int32_t i3 = avx2::block_isum_q4_0(w3 + b * 18 + 2, xq) - 8 * sx;
        t0 += f16_to_f32(dh0) * xd * (float)i0;
        t1 += f16_to_f32(dh1) * xd * (float)i1;
        t2 += f16_to_f32(dh2) * xd * (float)i2;
        t3 += f16_to_f32(dh3) * xd * (float)i3;
    }
    out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
}
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline void dot4_q4_0_q8_0(const uint8_t* w0, const uint8_t* w1, const uint8_t* w2, const uint8_t* w3,
                            const float* x_d, const int32_t* x_sum, const int8_t* x_q, size_t cols,
                            float out[4]) {
    size_t nb = cols / 32;
    float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const int8_t* xq = x_q + b * 32;
        int32_t sx = x_sum[b];
        float xd = x_d[b];
        uint16_t dh0, dh1, dh2, dh3;
        std::memcpy(&dh0, w0 + b * 18, 2); std::memcpy(&dh1, w1 + b * 18, 2);
        std::memcpy(&dh2, w2 + b * 18, 2); std::memcpy(&dh3, w3 + b * 18, 2);
        int32_t i0 = avx512::block_isum_q4_0(w0 + b * 18 + 2, xq) - 8 * sx;
        int32_t i1 = avx512::block_isum_q4_0(w1 + b * 18 + 2, xq) - 8 * sx;
        int32_t i2 = avx512::block_isum_q4_0(w2 + b * 18 + 2, xq) - 8 * sx;
        int32_t i3 = avx512::block_isum_q4_0(w3 + b * 18 + 2, xq) - 8 * sx;
        t0 += f16_to_f32(dh0) * xd * (float)i0;
        t1 += f16_to_f32(dh1) * xd * (float)i1;
        t2 += f16_to_f32(dh2) * xd * (float)i2;
        t3 += f16_to_f32(dh3) * xd * (float)i3;
    }
    out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
}

__attribute__((target("default")))
inline void dot4_q8_0_q8_0(const uint8_t* w0, const uint8_t* w1, const uint8_t* w2, const uint8_t* w3,
                            const float* x_d, const int32_t* x_sum, const int8_t* x_q, size_t cols,
                            float out[4]) {
    size_t nb = cols / 32;
    float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const int8_t* xq = x_q + b * 32;
        float xd = x_d[b];
        float wd0 = q8_0_block_scale(w0 + b * 34), wd1 = q8_0_block_scale(w1 + b * 34);
        float wd2 = q8_0_block_scale(w2 + b * 34), wd3 = q8_0_block_scale(w3 + b * 34);
        int32_t i0 = scalar::block_isum_q8_0(reinterpret_cast<const int8_t*>(w0 + b * 34 + 2), xq);
        int32_t i1 = scalar::block_isum_q8_0(reinterpret_cast<const int8_t*>(w1 + b * 34 + 2), xq);
        int32_t i2 = scalar::block_isum_q8_0(reinterpret_cast<const int8_t*>(w2 + b * 34 + 2), xq);
        int32_t i3 = scalar::block_isum_q8_0(reinterpret_cast<const int8_t*>(w3 + b * 34 + 2), xq);
        t0 += wd0 * xd * (float)i0; t1 += wd1 * xd * (float)i1;
        t2 += wd2 * xd * (float)i2; t3 += wd3 * xd * (float)i3;
    }
    out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
}
__attribute__((target("avx2,fma")))
inline void dot4_q8_0_q8_0(const uint8_t* w0, const uint8_t* w1, const uint8_t* w2, const uint8_t* w3,
                            const float* x_d, const int32_t* x_sum, const int8_t* x_q, size_t cols,
                            float out[4]) {
    size_t nb = cols / 32;
    float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const int8_t* xq = x_q + b * 32;
        float xd = x_d[b];
        float wd0 = q8_0_block_scale(w0 + b * 34), wd1 = q8_0_block_scale(w1 + b * 34);
        float wd2 = q8_0_block_scale(w2 + b * 34), wd3 = q8_0_block_scale(w3 + b * 34);
        int32_t i0 = avx2::block_isum_q8_0(reinterpret_cast<const int8_t*>(w0 + b * 34 + 2), xq);
        int32_t i1 = avx2::block_isum_q8_0(reinterpret_cast<const int8_t*>(w1 + b * 34 + 2), xq);
        int32_t i2 = avx2::block_isum_q8_0(reinterpret_cast<const int8_t*>(w2 + b * 34 + 2), xq);
        int32_t i3 = avx2::block_isum_q8_0(reinterpret_cast<const int8_t*>(w3 + b * 34 + 2), xq);
        t0 += wd0 * xd * (float)i0; t1 += wd1 * xd * (float)i1;
        t2 += wd2 * xd * (float)i2; t3 += wd3 * xd * (float)i3;
    }
    out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
}
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline void dot4_q8_0_q8_0(const uint8_t* w0, const uint8_t* w1, const uint8_t* w2, const uint8_t* w3,
                            const float* x_d, const int32_t* x_sum, const int8_t* x_q, size_t cols,
                            float out[4]) {
    size_t nb = cols / 32;
    float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const int8_t* xq = x_q + b * 32;
        float xd = x_d[b];
        float wd0 = q8_0_block_scale(w0 + b * 34), wd1 = q8_0_block_scale(w1 + b * 34);
        float wd2 = q8_0_block_scale(w2 + b * 34), wd3 = q8_0_block_scale(w3 + b * 34);
        int32_t i0 = avx512::block_isum_q8_0_f(reinterpret_cast<const int8_t*>(w0 + b * 34 + 2), xq);
        int32_t i1 = avx512::block_isum_q8_0_f(reinterpret_cast<const int8_t*>(w1 + b * 34 + 2), xq);
        int32_t i2 = avx512::block_isum_q8_0_f(reinterpret_cast<const int8_t*>(w2 + b * 34 + 2), xq);
        int32_t i3 = avx512::block_isum_q8_0_f(reinterpret_cast<const int8_t*>(w3 + b * 34 + 2), xq);
        t0 += wd0 * xd * (float)i0; t1 += wd1 * xd * (float)i1;
        t2 += wd2 * xd * (float)i2; t3 += wd3 * xd * (float)i3;
    }
    out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
}
// VNNI clone: the only one of the four that reads x_sum, same reason as
// the single-row dot_q8_0_q8_0's VNNI clone above.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
inline void dot4_q8_0_q8_0(const uint8_t* w0, const uint8_t* w1, const uint8_t* w2, const uint8_t* w3,
                            const float* x_d, const int32_t* x_sum, const int8_t* x_q, size_t cols,
                            float out[4]) {
    size_t nb = cols / 32;
    float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        const int8_t* xq = x_q + b * 32;
        float xd = x_d[b];
        int32_t sx = x_sum[b];
        float wd0 = q8_0_block_scale(w0 + b * 34), wd1 = q8_0_block_scale(w1 + b * 34);
        float wd2 = q8_0_block_scale(w2 + b * 34), wd3 = q8_0_block_scale(w3 + b * 34);
        int32_t i0 = avx512::block_isum_q8_0_vnni(reinterpret_cast<const int8_t*>(w0 + b * 34 + 2), xq, sx);
        int32_t i1 = avx512::block_isum_q8_0_vnni(reinterpret_cast<const int8_t*>(w1 + b * 34 + 2), xq, sx);
        int32_t i2 = avx512::block_isum_q8_0_vnni(reinterpret_cast<const int8_t*>(w2 + b * 34 + 2), xq, sx);
        int32_t i3 = avx512::block_isum_q8_0_vnni(reinterpret_cast<const int8_t*>(w3 + b * 34 + 2), xq, sx);
        t0 += wd0 * xd * (float)i0; t1 += wd1 * xd * (float)i1;
        t2 += wd2 * xd * (float)i2; t3 += wd3 * xd * (float)i3;
    }
    out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
}

// ── self-test (opt-in) ───────────────────────────────────────────────────────
// ───────────────────── register-tiled prefill kernels ───────────────────────
// PREVIOUSLY every weight row was re-read from L2/L3/DRAM once PER PROMPT
// POSITION (loop order: position outer, row inner), and the "multi" path
// used single-row dots with a per-block horizontal reduction. For a 256-token
// prefill that is 256 full passes over each thread's weight slice -- which is
// why prefill sat at ~0.2x of llama.cpp while decode (one pass) was close.
//
// tile4_*: computes a 4(rows) x npos(positions) block of outputs in one call.
// Each weight block is loaded + unpacked ONCE and reused for P=2 positions;
// the int32 block sums stay in vector lanes (converted to float and FMA'd
// with wd*xd), and the horizontal reduction happens once per output at the
// very end instead of once per 32-element block.
//   out[t*out_stride + r] for r in [0,4), t in [0,npos). No bias (caller adds).
//   w points at row 0; rows are row_bytes apart. x_* point at position 0 and
//   are laid out as quantize_rows_q8_0() produces them (x_d/x_sum: nb per
//   position, x_q: cols per position).
#ifndef TILE_R
#define TILE_R 4
#endif
#ifndef TILE_P
#define TILE_P 2
#endif
namespace tile_detail {

__attribute__((target("avx2,fma,f16c"), always_inline))
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
__attribute__((target("avx2,fma,f16c"), always_inline))
inline float load_f16(const uint8_t* p) {
    uint16_t h; std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

template <int R, int P>
__attribute__((target("avx2,fma,f16c"), always_inline))
inline void q4_tile(const uint8_t* w, size_t rb, const float* x_d,
                    const int8_t* x_q, size_t cols, float* out, size_t os) {
    const size_t nb = cols / 32;
    __m256 acc[R][P];
    for (int r = 0; r < R; ++r) for (int p = 0; p < P; ++p) acc[r][p] = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m256i eight = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t b = 0; b < nb; ++b) {
        __m256i xv[P]; __m256 xd[P];
        for (int p = 0; p < P; ++p) {
            xv[p] = _mm256_loadu_si256((const __m256i*)(x_q + p * cols + b * 32));
            xd[p] = _mm256_broadcast_ss(x_d + p * nb + b);
        }
        for (int r = 0; r < R; ++r) {
            const uint8_t* blk = w + r * rb + b * 18;
            uint16_t h; std::memcpy(&h, blk, 2);
            __m256 wd = _mm256_cvtph_ps(_mm_set1_epi16((short)h));   // lane-broadcast scale
            __m128i raw = _mm_loadu_si128((const __m128i*)(blk + 2));
            __m128i lo = _mm_and_si128(raw, mask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
            // centered weights in [-8,7]: removes the per-block "-8*sum(x)" correction
            __m256i wc = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);
            __m256i aw = _mm256_sign_epi8(wc, wc);
            for (int p = 0; p < P; ++p) {
                __m256i sx  = _mm256_sign_epi8(xv[p], wc);
                __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                acc[r][p] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s32), _mm256_mul_ps(wd, xd[p]), acc[r][p]);
            }
        }
    }
    for (int r = 0; r < R; ++r) for (int p = 0; p < P; ++p)
        out[p * os + r] = hsum256(acc[r][p]);
}

template <int R, int P>
__attribute__((target("avx2,fma,f16c"), always_inline))
inline void q8_tile(const uint8_t* w, size_t rb, const float* x_d,
                    const int8_t* x_q, size_t cols, float* out, size_t os) {
    const size_t nb = cols / 32;
    __m256 acc[R][P];
    for (int r = 0; r < R; ++r) for (int p = 0; p < P; ++p) acc[r][p] = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t b = 0; b < nb; ++b) {
        __m256i xv[P]; float xd[P];
        for (int p = 0; p < P; ++p) {
            xv[p] = _mm256_loadu_si256((const __m256i*)(x_q + p * cols + b * 32));
            xd[p] = x_d[p * nb + b];
        }
        for (int r = 0; r < R; ++r) {
            const uint8_t* blk = w + r * rb + b * 34;
            float wd = load_f16(blk);
            __m256i wv = _mm256_loadu_si256((const __m256i*)(blk + 2));
            __m256i aw = _mm256_sign_epi8(wv, wv);           // |w| (unsigned operand)
            for (int p = 0; p < P; ++p) {
                // |w|*sign(x,w): exact; max pair sum 2*128*127 < 32767 (x clamped to +-127)
                __m256i sx  = _mm256_sign_epi8(xv[p], wv);
                __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                acc[r][p] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s32),
                                            _mm256_set1_ps(wd * xd[p]), acc[r][p]);
            }
        }
    }
    for (int r = 0; r < R; ++r) for (int p = 0; p < P; ++p)
        out[p * os + r] = hsum256(acc[r][p]);
}
} // namespace tile_detail

__attribute__((target("default")))
inline void tile4_q4_0_q8_0(const uint8_t* w, size_t rb, const float* x_d, const int32_t* x_sum,
                             const int8_t* x_q, size_t cols, size_t npos, float* out, size_t os) {
    size_t nb = cols / 32;
    for (size_t t = 0; t < npos; ++t)
        for (int r = 0; r < 4; ++r)
            out[t * os + r] = dot_q4_0_q8_0(w + r * rb, x_d + t * nb, x_sum + t * nb, x_q + t * cols, cols);
}
__attribute__((target("avx2,fma,f16c")))
inline void tile4_q4_0_q8_0(const uint8_t* w, size_t rb, const float* x_d, const int32_t* x_sum,
                             const int8_t* x_q, size_t cols, size_t npos, float* out, size_t os) {
    size_t nb = cols / 32, t = 0;
    (void)x_sum;
    for (; t + TILE_P <= npos; t += TILE_P)
        for (int r = 0; r < 4; r += TILE_R)
            tile_detail::q4_tile<TILE_R, TILE_P>(w + r * rb, rb, x_d + t * nb, x_q + t * cols, cols, out + t * os + r, os);
    for (; t < npos; ++t)
        tile_detail::q4_tile<4, 1>(w, rb, x_d + t * nb, x_q + t * cols, cols, out + t * os, os);
}
__attribute__((target("default")))
inline void tile4_q8_0_q8_0(const uint8_t* w, size_t rb, const float* x_d, const int32_t* x_sum,
                             const int8_t* x_q, size_t cols, size_t npos, float* out, size_t os) {
    size_t nb = cols / 32;
    for (size_t t = 0; t < npos; ++t)
        for (int r = 0; r < 4; ++r)
            out[t * os + r] = dot_q8_0_q8_0(w + r * rb, x_d + t * nb, x_sum + t * nb, x_q + t * cols, cols);
}
__attribute__((target("avx2,fma,f16c")))
inline void tile4_q8_0_q8_0(const uint8_t* w, size_t rb, const float* x_d, const int32_t* x_sum,
                             const int8_t* x_q, size_t cols, size_t npos, float* out, size_t os) {
    size_t nb = cols / 32, t = 0;
    (void)x_sum;
    for (; t + TILE_P <= npos; t += TILE_P)
        for (int r = 0; r < 4; r += TILE_R)
            tile_detail::q8_tile<TILE_R, TILE_P>(w + r * rb, rb, x_d + t * nb, x_q + t * cols, cols, out + t * os + r, os);
    for (; t < npos; ++t)
        tile_detail::q8_tile<4, 1>(w, rb, x_d + t * nb, x_q + t * cols, cols, out + t * os, os);
}

// ───────────────── attention / activation-quantization kernels ──────────────
// attn_wsum_i8: out[d] = sum_s coeff[s] * (float)vq[s*vstride + d], d in [0,hd).
// The AVX2 version keeps 64 output lanes in 8 registers across the whole
// position loop (the old per-position axpy_i8 re-loaded and re-stored `out`
// for every position).
__attribute__((target("default")))
inline void attn_wsum_i8(float* out, const float* coeff, const int8_t* vq, size_t vstride,
                         size_t klen, size_t hd) {
    for (size_t d = 0; d < hd; ++d) out[d] = 0.f;
    for (size_t s = 0; s < klen; ++s) {
        const int8_t* v = vq + s * vstride; float c = coeff[s];
        for (size_t d = 0; d < hd; ++d) out[d] += c * (float)v[d];
    }
}
__attribute__((target("avx2,fma")))
inline void attn_wsum_i8(float* out, const float* coeff, const int8_t* vq, size_t vstride,
                         size_t klen, size_t hd) {
    size_t c0 = 0;
    for (; c0 + 64 <= hd; c0 += 64) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        for (size_t s = 0; s < klen; ++s) {
            const int8_t* v = vq + s * vstride + c0;
            __m256 c = _mm256_set1_ps(coeff[s]);
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(v +  0)))), c, a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(v +  8)))), c, a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(v + 16)))), c, a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(v + 24)))), c, a3);
            a4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(v + 32)))), c, a4);
            a5 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(v + 40)))), c, a5);
            a6 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(v + 48)))), c, a6);
            a7 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(v + 56)))), c, a7);
        }
        _mm256_storeu_ps(out + c0 +  0, a0); _mm256_storeu_ps(out + c0 +  8, a1);
        _mm256_storeu_ps(out + c0 + 16, a2); _mm256_storeu_ps(out + c0 + 24, a3);
        _mm256_storeu_ps(out + c0 + 32, a4); _mm256_storeu_ps(out + c0 + 40, a5);
        _mm256_storeu_ps(out + c0 + 48, a6); _mm256_storeu_ps(out + c0 + 56, a7);
    }
    if (c0 < hd) {   // head_dim not a multiple of 64: scalar tail columns
        for (size_t d = c0; d < hd; ++d) out[d] = 0.f;
        for (size_t s = 0; s < klen; ++s) {
            const int8_t* v = vq + s * vstride; float c = coeff[s];
            for (size_t d = c0; d < hd; ++d) out[d] += c * (float)v[d];
        }
    }
}

// quantize_row_q8_0: one activation row -> per-32-block {scale, int8[32], sum(int8)}.
// Replaces the scalar loop that called std::lround per element. (Rounding is
// round-to-nearest-even here vs. half-away-from-zero before; differs only on
// exact .5 ties, <= 1 LSB of the int8 code.)
__attribute__((target("default")))
inline void quantize_row_q8_0(const float* x, size_t cols, float* d_out, int8_t* q_out, int32_t* s_out) {
    size_t nb = cols / 32;
    for (size_t b = 0; b < nb; ++b) {
        const float* xb = x + b * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
        float d = amax / 127.0f, id = d > 0.f ? 1.0f / d : 0.f;
        d_out[b] = d;
        int8_t* qb = q_out + b * 32; int32_t sum = 0;
        for (int j = 0; j < 32; ++j) {
            int v = (int)std::lround(xb[j] * id);
            qb[j] = (int8_t)std::clamp(v, -127, 127); sum += qb[j];
        }
        s_out[b] = sum;
    }
}
__attribute__((target("avx2,fma")))
inline void quantize_row_q8_0(const float* x, size_t cols, float* d_out, int8_t* q_out, int32_t* s_out) {
    size_t nb = cols / 32;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (size_t b = 0; b < nb; ++b) {
        const float* xb = x + b * 32;
        __m256 v0 = _mm256_loadu_ps(xb), v1 = _mm256_loadu_ps(xb + 8), v2 = _mm256_loadu_ps(xb + 16), v3 = _mm256_loadu_ps(xb + 24);
        __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_and_ps(v0, absmask), _mm256_and_ps(v1, absmask)),
                                 _mm256_max_ps(_mm256_and_ps(v2, absmask), _mm256_and_ps(v3, absmask)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        float amax = _mm_cvtss_f32(m4);
        float d = amax / 127.0f, id = d > 0.f ? 1.0f / d : 0.f;
        d_out[b] = d;
        __m256 idv = _mm256_set1_ps(id);
        __m256i i0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v0, idv)), lo), hi);
        __m256i i1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v1, idv)), lo), hi);
        __m256i i2 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v2, idv)), lo), hi);
        __m256i i3 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v3, idv)), lo), hi);
        __m256i sum8 = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum8), _mm256_extracti128_si256(sum8, 1));
        s4 = _mm_add_epi32(s4, _mm_unpackhi_epi64(s4, s4));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 1));
        s_out[b] = _mm_cvtsi128_si32(s4);
        // pack 4x8 int32 -> 32 int8 (lane order fixed by the final permute)
        __m256i p01 = _mm256_packs_epi32(i0, i1), p23 = _mm256_packs_epi32(i2, i3);
        __m256i p = _mm256_packs_epi16(p01, p23);
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*)(q_out + b * 32), p);
    }
}

// ───────────────────────── attention score kernel (Q . K^T) ─────────────────────────
// out[s] = scale * dot(q, K + s*kstride, hd) for s in [0, klen); returns max(out).
// PREVIOUSLY the attention loop called dot_f32() once per key: an ifunc'd (non-inlinable) call plus
// a full horizontal reduction per key, ~50% of attention time. Here 8 keys are scored at once:
// q is loaded once per 8 floats, 8 independent FMA chains hide the FMA latency, and the 8 horizontal
// sums are produced together with a hadd tree. Summation order differs from dot_f32 (fp reassociation).
__attribute__((target("default")))
inline float attn_scores(const float* q, const float* K, size_t kstride, size_t klen, size_t hd,
                         float scale, float* out) {
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t s = 0; s < klen; ++s) {
        float v = scalar::dot_f32(q, K + s * kstride, hd) * scale;
        out[s] = v; if (v > mx) mx = v;
    }
    return mx;
}
__attribute__((target("avx2,fma")))
inline float attn_scores(const float* q, const float* K, size_t kstride, size_t klen, size_t hd,
                         float scale, float* out) {
    float mx = -std::numeric_limits<float>::infinity();
    const size_t hd8 = hd & ~(size_t)7;
    size_t s = 0;
    for (; s + 8 <= klen; s += 8) {
        const float* k[8];
        for (int j = 0; j < 8; ++j) k[j] = K + (s + j) * kstride;
        __m256 a[8];
        for (int j = 0; j < 8; ++j) a[j] = _mm256_setzero_ps();
        for (size_t i = 0; i < hd8; i += 8) {
            __m256 qv = _mm256_loadu_ps(q + i);
            for (int j = 0; j < 8; ++j) a[j] = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k[j] + i), a[j]);
        }
        __m256 h01 = _mm256_hadd_ps(a[0], a[1]), h23 = _mm256_hadd_ps(a[2], a[3]);
        __m256 h45 = _mm256_hadd_ps(a[4], a[5]), h67 = _mm256_hadd_ps(a[6], a[7]);
        __m256 hA = _mm256_hadd_ps(h01, h23), hB = _mm256_hadd_ps(h45, h67);   // per lane: [k0..k3] / [k4..k7] partials
        __m128 rA = _mm_add_ps(_mm256_castps256_ps128(hA), _mm256_extractf128_ps(hA, 1));
        __m128 rB = _mm_add_ps(_mm256_castps256_ps128(hB), _mm256_extractf128_ps(hB, 1));
        alignas(16) float r[8];
        _mm_store_ps(r, rA); _mm_store_ps(r + 4, rB);
        for (int j = 0; j < 8; ++j) {
            float t = r[j];
            for (size_t i = hd8; i < hd; ++i) t += q[i] * k[j][i];          // head_dim not a multiple of 8
            t *= scale; out[s + j] = t; if (t > mx) mx = t;
        }
    }
    for (; s < klen; ++s) {
        const float* kr = K + s * kstride;
        __m256 acc = _mm256_setzero_ps();
        for (size_t i = 0; i < hd8; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(kr + i), acc);
        __m128 v = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        v = _mm_add_ps(v, _mm_movehl_ps(v, v)); v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
        float t = _mm_cvtss_f32(v);
        for (size_t i = hd8; i < hd; ++i) t += q[i] * kr[i];
        t *= scale; out[s] = t; if (t > mx) mx = t;
    }
    return mx;
}

#if defined(RAWLLM_SIMD_SELFTEST)
#include <random>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

inline bool run_selftest() {
    __builtin_cpu_init();
    const bool has_avx2       = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    const bool has_avx512f    = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl");
    const bool has_avx512vnni = has_avx512f && __builtin_cpu_supports("avx512vnni");

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> fdist(-1.f, 1.f);
    std::uniform_int_distribution<int> i8dist(-127, 127);
    std::uniform_int_distribution<int> u8dist(0, 255);
    bool ok = true;

    // dot_f32 / axpy_f32: the multiversioned entry points vs scalar
    // reference, various lengths (including non-multiple-of-16 tails). This
    // exercises whatever clone the resolver picked on THIS machine — the
    // per-backend block_isum tests below are what exhaustively cover every
    // backend regardless of the running CPU.
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
    // blocks each. Every backend is always compiled in now (no more
    // RAWLLM_AVX2/RAWLLM_AVX512 file-level gate), so every backend gets
    // tested on every build — but only the ones the RUNNING cpu actually
    // supports get invoked, guarded by the __builtin_cpu_supports() checks
    // above, since calling e.g. the AVX-512 clone on a CPU without AVX-512
    // would SIGILL rather than gracefully fail.
    for (int trial = 0; trial < 50000; ++trial) {
        uint8_t qs[16]; int8_t xq32[32];
        for (auto& v : qs) v = (uint8_t)u8dist(rng);
        for (auto& v : xq32) v = (int8_t)i8dist(rng);
        int32_t ref = scalar::block_isum_q4_0(qs, xq32);

        if (has_avx2) {
            int32_t got = avx2::block_isum_q4_0(qs, xq32);
            if (ref != got) {
                std::fprintf(stderr, "[simd selftest] avx2::block_isum_q4_0 mismatch trial=%d ref=%d got=%d\n", trial, ref, got);
                ok = false;
            }
        }
        if (has_avx512f) {
            int32_t got = avx512::block_isum_q4_0(qs, xq32);
            if (ref != got) {
                std::fprintf(stderr, "[simd selftest] avx512::block_isum_q4_0 mismatch trial=%d ref=%d got=%d\n", trial, ref, got);
                ok = false;
            }
        }

        int8_t w32[32];
        for (auto& v : w32) v = (int8_t)i8dist(rng);
        int32_t refq8 = scalar::block_isum_q8_0(w32, xq32);

        if (has_avx2) {
            int32_t got = avx2::block_isum_q8_0(w32, xq32);
            if (refq8 != got) {
                std::fprintf(stderr, "[simd selftest] avx2::block_isum_q8_0 mismatch trial=%d ref=%d got=%d\n", trial, refq8, got);
                ok = false;
            }
        }
        if (has_avx512f) {
            int32_t got = avx512::block_isum_q8_0_f(w32, xq32);
            if (refq8 != got) {
                std::fprintf(stderr, "[simd selftest] avx512::block_isum_q8_0_f mismatch trial=%d ref=%d got=%d\n", trial, refq8, got);
                ok = false;
            }
        }
        if (has_avx512vnni) {
            int32_t sum_x32 = 0;
            for (auto v : xq32) sum_x32 += v;
            int32_t vgot = avx512::block_isum_q8_0_vnni(w32, xq32, sum_x32);
            if (refq8 != vgot) {
                std::fprintf(stderr,
                    "[simd selftest] avx512::block_isum_q8_0_vnni mismatch trial=%d ref=%d got=%d\n",
                    trial, refq8, vgot);
                ok = false;
            }
        }
    }

    // Full dot_q8_0_q8_0() row test — catches layout bugs (per-block scale
    // application, sum_x indexing) that the isolated primitive test above
    // can't see. Calls the multiversioned entry point directly, so this
    // exercises whichever clone the resolver picks on THIS machine.
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

    // dot_q4_0_q8_0() row test: sources sum_x from quantize_rows_q8_0()'s
    // precomputed output instead of recomputing it inline, so this checks
    // that plumbing end to end, not just the block kernel. Also calls the
    // multiversioned entry point directly (whichever clone the resolver
    // picks on this machine).
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
                float wd = f16_to_f32(dh);
                int32_t uncentered = scalar::block_isum_q4_0(
                    reinterpret_cast<const uint8_t*>(&wrow[b * 18 + 2]), &x_q[b * 32]);
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

    // attn_scores vs scalar reference (various klen / head_dim incl. non-multiples of 8)
    for (int trial = 0; trial < 200; ++trial) {
        size_t hd = 8 + (rng() % 140), klen = 1 + (rng() % 40), stride = hd + (rng() % 16);
        std::vector<float> q(hd), K(klen * stride), o(klen);
        for (auto& v : q) v = fdist(rng);
        for (auto& v : K) v = fdist(rng);
        float mx = attn_scores(q.data(), K.data(), stride, klen, hd, 0.37f, o.data());
        float rmx = -1e30f;
        for (size_t s2 = 0; s2 < klen; ++s2) {
            float ref = scalar::dot_f32(q.data(), K.data() + s2 * stride, hd) * 0.37f;
            rmx = std::max(rmx, ref);
            if (std::fabs(ref - o[s2]) > 1e-3f * std::max(1.0f, std::fabs(ref))) { std::fprintf(stderr, "[simd selftest] attn_scores mismatch hd=%zu klen=%zu s=%zu\n", hd, klen, s2); ok = false; break; }
        }
        if (std::fabs(rmx - mx) > 1e-3f * std::max(1.0f, std::fabs(rmx))) { std::fprintf(stderr, "[simd selftest] attn_scores max mismatch\n"); ok = false; }
    }

    // dot4_q4_0_q8_0 / dot4_q8_0_q8_0 (register-tiled 4-row batch) vs 4
    // independent single-row calls — proves the batched kernels compute
    // the exact same result as the path they replace in proj_all_positions.
    {
        std::uniform_real_distribution<float> ddist(0.001f, 2.0f);
        for (int trial = 0; trial < 200; ++trial) {
            size_t nb = 1 + (rng() % 9);
            size_t cols = nb * 32;
            std::vector<float> x_d(nb);
            std::vector<int32_t> x_sum(nb);
            std::vector<int8_t> x_q(cols);
            for (auto& v : x_d) v = ddist(rng);
            for (auto& v : x_q) v = (int8_t)i8dist(rng);
            for (size_t b = 0; b < nb; ++b) {
                int32_t s = 0;
                for (size_t j = 0; j < 32; ++j) s += x_q[b * 32 + j];
                x_sum[b] = s;
            }

            // Q8_0
            std::vector<uint8_t> rows8[4];
            for (auto& row : rows8) { row.resize(nb * 34); for (auto& v : row) v = (uint8_t)u8dist(rng); }
            float ref8[4], got8[4];
            for (int i = 0; i < 4; ++i)
                ref8[i] = dot_q8_0_q8_0(rows8[i].data(), x_d.data(), x_sum.data(), x_q.data(), cols);
            dot4_q8_0_q8_0(rows8[0].data(), rows8[1].data(), rows8[2].data(), rows8[3].data(),
                           x_d.data(), x_sum.data(), x_q.data(), cols, got8);
            for (int i = 0; i < 4; ++i) {
                if (std::fabs(ref8[i] - got8[i]) > 1e-2f * std::max(1.0f, std::fabs(ref8[i]))) {
                    std::fprintf(stderr, "[simd selftest] dot4_q8_0_q8_0 mismatch trial=%d i=%d ref=%f got=%f\n",
                                 trial, i, ref8[i], got8[i]);
                    ok = false;
                }
            }

            // Q4_0
            std::vector<uint8_t> rows4[4];
            for (auto& row : rows4) { row.resize(nb * 18); for (auto& v : row) v = (uint8_t)u8dist(rng); }
            float ref4[4], got4[4];
            for (int i = 0; i < 4; ++i)
                ref4[i] = dot_q4_0_q8_0(rows4[i].data(), x_d.data(), x_sum.data(), x_q.data(), cols);
            dot4_q4_0_q8_0(rows4[0].data(), rows4[1].data(), rows4[2].data(), rows4[3].data(),
                           x_d.data(), x_sum.data(), x_q.data(), cols, got4);
            for (int i = 0; i < 4; ++i) {
                if (std::fabs(ref4[i] - got4[i]) > 1e-2f * std::max(1.0f, std::fabs(ref4[i]))) {
                    std::fprintf(stderr, "[simd selftest] dot4_q4_0_q8_0 mismatch trial=%d i=%d ref=%f got=%f\n",
                                 trial, i, ref4[i], got4[i]);
                    ok = false;
                }
            }
        }
    }

    if (ok) {
        std::fprintf(stderr,
            "[simd selftest] all backends match scalar reference on this CPU "
            "(avx2=%d avx512f=%d avx512vnni=%d; dot_f32/axpy_f32 200 trials; "
            "block_isum_q4_0/q8_0 50000 trials; dot_q4_0_q8_0/dot_q8_0_q8_0 500 trials)\n",
            (int)has_avx2, (int)has_avx512f, (int)has_avx512vnni);
    }
    return ok;
}
#endif // RAWLLM_SIMD_SELFTEST

} // namespace simd
