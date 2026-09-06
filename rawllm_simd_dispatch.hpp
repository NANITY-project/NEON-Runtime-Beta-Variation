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
