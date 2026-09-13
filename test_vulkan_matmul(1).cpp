// test_vulkan_matmul.cpp — standalone correctness + throughput check for
// rawllm_vulkan.hpp's GEMM path (queue_matmul_f32 / queue_matmul_f16),
// independent of NEON-3.cpp/the rest of the engine.
//
// This exists to close two gaps VULKAN_BACKEND_NOTES.md calls out
// explicitly:
//   1. Every number captured so far (test_vulkan_matvec.cpp) is decode-
//      shaped (M=1, bandwidth-bound) — the easiest case for a generic
//      shader to look competitive with a vendor GEMV. Prefill (M>1,
//      compute-bound) is untested and is where a hand-tuned kernel's
//      tiling/tensor-core usage should matter more.
//   2. matmul_coopmat_f16.comp compiled and pipeline construction
//      accepted it on the T4, but it was never actually dispatched or
//      checked for correctness.
//
// Only F32 (matmul_tiled_f32.comp, no capability gate) and F16
// (matmul_coopmat_f16.comp, only if coop_matrix_supported()) are covered.
// Q4_0/Q8_0 GEMM (queue_matmul_q4_0/_q8_0) is deliberately NOT included —
// same reason test_vulkan_matvec.cpp gives for skipping the quantized
// GEVM path: a correct reference needs this repo's real x_q/x_d/x_sum
// production (quantize_rows_q8_0() et al. in rawllm_forward.hpp), which
// isn't reachable standalone without pulling in the rest of the engine.
// Wire those in the same way once this passes, if you want quantized
// GEMM numbers too — reusing queue_matmul_quantized()'s existing
// packing contract (see that function's comment in rawllm_vulkan.hpp).
//
// Build (from the repo root; shaders are at the repo root here, not in a
// shaders/ subdir — see VULKAN_BACKEND_NOTES.md's "Build notes specific
// to this repo's layout"):
//   glslangValidator -V --target-env vulkan1.1 matmul_tiled_f32.comp   -o matmul_tiled_f32.spv
//   glslangValidator -V --target-env vulkan1.1 matmul_coopmat_f16.comp -o matmul_coopmat_f16.spv   # only if your device supports VK_KHR_cooperative_matrix
//   g++ -std=c++20 -O2 -pthread -DUSE_VULKAN test_vulkan_matmul.cpp -lvulkan -o test_vulkan_matmul
// Run:
//   ./test_vulkan_matmul
//
// NOTE: create_pipelines() also unconditionally builds pipeline_matmul_q4_0_/
// _q8_0_, so matmul_tiled_q4_0.spv and matmul_tiled_q8_0.spv (or their
// _dp4a variants — see this file's header comment on the dp4a fallback
// fix) must exist too even though this harness never dispatches them,
// same as test_vulkan_matvec.cpp needing matvec_q4_0.spv/matvec_q8_0.spv
// present for construction to succeed.

#include "rawllm_vulkan.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

// Standalone F32<->F16 round-trip, used ONLY to build the reference for the
// coopmat test (queue_matmul_f16()'s own conversion is
// VulkanMatvecBackend::float_to_half(), a private member we can't call
// from here — this is a separate, ordinary round-to-nearest-even
// implementation, not required to be bit-identical to it since the 5e-2
// relative tolerance in run_f16_case() already accounts for fp16
// precision, not exact-rounding equivalence).
uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if (shift > 0 && ((mant >> (shift - 1)) & 1u))
            if (((mant & ((1u << (shift - 1)) - 1u)) != 0u) || (half_mant & 1u)) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    uint32_t half_mant = mant >> 13;
    if (mant & 0x1000u)
        if ((mant & 0xFFFu) != 0u || (half_mant & 1u)) {
            half_mant++;
            if (half_mant == 0x400u) { half_mant = 0; exp++; if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u); }
        }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_mant);
}

float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            // subnormal half -> normal float
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Naive double-accumulate CPU reference GEMM: C[M x N] = A[M x K] * B[K x N],
// both row-major. Ground truth, not the engine's real path.
std::vector<float> cpu_reference_gemm(const std::vector<float>& A, const std::vector<float>& B,
                                       uint32_t M, uint32_t K, uint32_t N) {
    std::vector<float> C((size_t)M * N, 0.0f);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (uint32_t k = 0; k < K; ++k)
                acc += (double)A[(size_t)m * K + k] * (double)B[(size_t)k * N + n];
            C[(size_t)m * N + n] = (float)acc;
        }
    }
    return C;
}

struct ErrorStats { double max_abs = 0.0, max_rel = 0.0; };

ErrorStats compare(const std::vector<float>& got, const std::vector<float>& ref) {
    ErrorStats e;
    for (size_t i = 0; i < ref.size(); ++i) {
        double err = std::fabs((double)got[i] - (double)ref[i]);
        e.max_abs = std::max(e.max_abs, err);
        double denom = std::max(1e-6, std::fabs((double)ref[i]));
        e.max_rel = std::max(e.max_rel, err / denom);
    }
    return e;
}

// Prefill-representative M values: 1 is the decode edge case
// queue_matmul_f32()'s own comment says NOT to route here (use
// queue_matvec_f32() instead) — included anyway as a correctness/padding
// edge case, not as a "should be fast" claim. 8/32 are small-batch/short-
// prompt prefill. 128 is a more realistic single prompt chunk.
const uint32_t kMShapes[] = {1, 8, 32, 128};
constexpr uint32_t kK = 4096;  // hidden size, matches test_vulkan_matvec.cpp's shape
constexpr uint32_t kN = 4096;
constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 100;

void run_f32_case(vk_backend::VulkanMatvecBackend& backend, std::mt19937& rng,
                   std::uniform_real_distribution<float>& dist, uint32_t M) {
    using clock = std::chrono::steady_clock;

    std::vector<float> B(kK * kN);
    for (auto& v : B) v = dist(rng);
    vk_backend::GpuTensor Wt = backend.upload_weight_f32(B.data(), kK, kN);

    std::vector<float> A((size_t)M * kK);
    for (auto& v : A) v = dist(rng);

    auto ref = cpu_reference_gemm(A, B, M, kK, kN);
    std::vector<float> got((size_t)M * kN, 0.0f);

    // Single-shot correctness check.
    backend.begin_batch();
    backend.queue_matmul_f32(Wt, A.data(), got.data(), M, kK, kN);
    auto handle = backend.end_batch();
    backend.wait_batch(handle);

    auto err = compare(got, ref);
    bool ok = err.max_rel < 1e-2; // same bar as test_vulkan_matvec.cpp
    std::printf("  [F32 GEMM] M=%-4u correctness: max_abs=%.6g max_rel=%.6g -> %s\n",
                M, err.max_abs, err.max_rel, ok ? "PASS" : "FAIL");

    for (int i = 0; i < kWarmupIters; ++i) {
        backend.begin_batch();
        backend.queue_matmul_f32(Wt, A.data(), got.data(), M, kK, kN);
        backend.wait_batch(backend.end_batch());
    }
    auto t0 = clock::now();
    for (int i = 0; i < kTimedIters; ++i) {
        backend.begin_batch();
        backend.queue_matmul_f32(Wt, A.data(), got.data(), M, kK, kN);
        backend.wait_batch(backend.end_batch());
    }
    auto t1 = clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_call_us = (total_ms * 1000.0) / kTimedIters;
    double gflops = (2.0 * M * kK * kN) / (per_call_us * 1e-6) / 1e9;
    std::printf("  [F32 GEMM] M=%-4u %.2f us/call, ~%.2f GFLOP/s\n", M, per_call_us, gflops);

    backend.free_tensor(Wt);
}

void run_f16_case(vk_backend::VulkanMatvecBackend& backend, std::mt19937& rng,
                   std::uniform_real_distribution<float>& dist, uint32_t M) {
    using clock = std::chrono::steady_clock;

    std::vector<float> B(kK * kN);
    for (auto& v : B) v = dist(rng);
    vk_backend::GpuTensor Wt = backend.upload_weight_f16(B.data(), kK, kN);

    std::vector<float> A((size_t)M * kK);
    for (auto& v : A) v = dist(rng);

    // Reference must account for the fp16 round-trip on BOTH operands —
    // queue_matmul_f16() converts A host-side and B was converted at
    // upload_weight_f16() time — or every shape will show spurious
    // "error" that's actually just fp16 precision, not a bug.
    std::vector<float> A_h(A.size()), B_h(B.size());
    for (size_t i = 0; i < A.size(); ++i) A_h[i] = f16_to_f32(f32_to_f16(A[i]));
    for (size_t i = 0; i < B.size(); ++i) B_h[i] = f16_to_f32(f32_to_f16(B[i]));
    auto ref = cpu_reference_gemm(A_h, B_h, M, kK, kN);
    std::vector<float> got((size_t)M * kN, 0.0f);

    backend.begin_batch();
    backend.queue_matmul_f16(Wt, A.data(), got.data(), M, kK, kN);
    auto handle = backend.end_batch();
    backend.wait_batch(handle);

    auto err = compare(got, ref);
    // Looser bar than F32: fp16 has ~3 decimal digits of precision and
    // K=4096 accumulation terms, so drift here is expected, not a bug —
    // this threshold is a starting point, tighten it once you have a
    // real distribution of values from your model rather than uniform
    // [-1,1] noise.
    bool ok = err.max_rel < 5e-2;
    std::printf("  [F16 coopmat GEMM] M=%-4u correctness: max_abs=%.6g max_rel=%.6g -> %s\n",
                M, err.max_abs, err.max_rel, ok ? "PASS" : "FAIL (or just fp16 precision — inspect before trusting)");

    for (int i = 0; i < kWarmupIters; ++i) {
        backend.begin_batch();
        backend.queue_matmul_f16(Wt, A.data(), got.data(), M, kK, kN);
        backend.wait_batch(backend.end_batch());
    }
    auto t0 = clock::now();
    for (int i = 0; i < kTimedIters; ++i) {
        backend.begin_batch();
        backend.queue_matmul_f16(Wt, A.data(), got.data(), M, kK, kN);
        backend.wait_batch(backend.end_batch());
    }
    auto t1 = clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_call_us = (total_ms * 1000.0) / kTimedIters;
    double gflops = (2.0 * M * kK * kN) / (per_call_us * 1e-6) / 1e9;
    std::printf("  [F16 coopmat GEMM] M=%-4u %.2f us/call, ~%.2f GFLOP/s\n", M, per_call_us, gflops);

    backend.free_tensor(Wt);
}

} // namespace

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::printf("Constructing VulkanMatvecBackend...\n");
    std::unique_ptr<vk_backend::VulkanMatvecBackend> backend;
    try {
        backend = std::make_unique<vk_backend::VulkanMatvecBackend>(".");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Construction failed: %s\n", e.what());
        std::fprintf(stderr, "(Most likely cause: a required *.spv file is missing in the repo root "
                              "— see this file's header comment for the exact glslangValidator commands.)\n");
        return 1;
    }
    std::printf("Backend constructed OK.\n");
    std::printf("dot_product_supported=%d dp4a_pipelines_loaded=%d%s\n",
                backend->dot_product_supported(), backend->dp4a_pipelines_loaded(),
                backend->dp4a_pipelines_loaded() || !backend->dot_product_supported() ? "" :
                    (" (" + backend->dp4a_fallback_reason() + ")").c_str());
    std::printf("coop_matrix_supported=%d\n", backend->coop_matrix_supported());
    std::printf("push_descriptor_supported=%d\n\n", backend->push_descriptor_supported());

    bool all_ok = true;

    std::printf("== F32 tiled GEMM (matmul_tiled_f32.comp) ==\n");
    for (uint32_t M : kMShapes) {
        try {
            run_f32_case(*backend, rng, dist, M);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  M=%u threw: %s\n", M, e.what());
            all_ok = false;
        }
    }

    std::printf("\n== F16 cooperative-matrix GEMM (matmul_coopmat_f16.comp) ==\n");
    if (!backend->coop_matrix_supported()) {
        std::printf("  Skipped — this device doesn't report VK_KHR_cooperative_matrix "
                     "with the 16x16x16 fp16xfp16->f32 shape this backend requires.\n");
    } else {
        for (uint32_t M : kMShapes) {
            try {
                run_f16_case(*backend, rng, dist, M);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  M=%u threw: %s\n", M, e.what());
                all_ok = false;
            }
        }
    }

    std::printf("\nCompare these GFLOP/s numbers against test_cuda_matmul.cu's cublasSgemm/"
                "cublasGemmEx(FP16) figures on the same hardware — that's the actual "
                "compute-bound comparison VULKAN_BACKEND_NOTES.md was still missing.\n");

    return all_ok ? 0 : 1;
}
