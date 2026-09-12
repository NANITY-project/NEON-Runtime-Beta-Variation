// test_vulkan_matvec.cpp — standalone correctness + throughput check for
// rawllm_vulkan.hpp's F32 matvec path, independent of NEON-3.cpp/the rest
// of the engine. Exercises exactly the code path today's two perf changes
// touch (persistent block mapping in upload()/download(), descriptor-set
// dedup in queue_matvec_f32()) via the synchronous matvec_f32() wrapper.
//
// Build (from the repo root, after compiling every shader listed in the
// accompanying instructions into shaders/*.spv):
//   g++ -std=c++20 -O2 -pthread -DUSE_VULKAN test_vulkan_matvec.cpp -lvulkan -o test_vulkan_matvec
// Run:
//   ./test_vulkan_matvec
//
// Only tests the F32 GEMV path — Q4_0/Q8_0 need this repo's real
// quantization helpers (x_q/x_d/x_sum production) to build a correct
// reference, which live elsewhere in the codebase; wire those in the same
// way once this passes if you want quantized-path numbers too.

// USE_VULKAN is passed via -DUSE_VULKAN on the command line (matching your
// existing build invocation) rather than #define'd here.
#include "rawllm_vulkan.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    using clock = std::chrono::steady_clock;

    // Realistic-ish decode shape: a single-token activation against one
    // projection's weight matrix. Adjust to your actual model's hidden
    // size / projection dims if you want numbers that mean something for
    // your workload specifically.
    const uint32_t cols = 4096;   // hidden size (input dim)
    const uint32_t rows = 4096;   // output dim
    const int warmup_iters = 20;
    const int timed_iters = 500;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> W(static_cast<size_t>(rows) * cols);
    std::vector<float> x(cols);
    for (auto& v : W) v = dist(rng);
    for (auto& v : x) v = dist(rng);

    // CPU reference (naive, not the engine's AVX2 path — just ground truth).
    std::vector<float> y_ref(rows, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        const float* row = &W[static_cast<size_t>(r) * cols];
        for (uint32_t c = 0; c < cols; ++c) acc += static_cast<double>(row[c]) * x[c];
        y_ref[r] = static_cast<float>(acc);
    }

    std::printf("Constructing VulkanMatvecBackend...\n");
    std::unique_ptr<vk_backend::VulkanMatvecBackend> backend;
    try {
        backend = std::make_unique<vk_backend::VulkanMatvecBackend>("shaders");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Construction failed: %s\n", e.what());
        std::fprintf(stderr, "(Most likely cause: a required shaders/*.spv file is missing "
                              "— see the build instructions' full shader list.)\n");
        return 1;
    }
    std::printf("Backend constructed OK — a real Vulkan device accepted the pipelines.\n");

    vk_backend::GpuTensor Wt = backend->upload_weight_f32(W.data(), rows, cols);

    // ── Correctness ──────────────────────────────────────────────────
    std::vector<float> y_gpu(rows, 0.0f);
    backend->matvec_f32(Wt, x.data(), y_gpu.data(), cols, rows);

    double max_abs_err = 0.0, max_rel_err = 0.0;
    for (uint32_t r = 0; r < rows; ++r) {
        double err = std::fabs(static_cast<double>(y_gpu[r]) - static_cast<double>(y_ref[r]));
        max_abs_err = std::max(max_abs_err, err);
        double denom = std::max(1e-6, std::fabs(static_cast<double>(y_ref[r])));
        max_rel_err = std::max(max_rel_err, err / denom);
    }
    std::printf("Correctness: max abs err = %.6g, max rel err = %.6g\n", max_abs_err, max_rel_err);
    // Float accumulation order differs GPU vs CPU (see the file's own
    // validation notes), so exact bit equality isn't the right bar —
    // a few ulp of drift at this size is expected; anything past ~1e-2
    // relative on a value that isn't near zero suggests a real bug, not
    // just accumulation-order noise.
    bool correctness_ok = max_rel_err < 1e-2;
    std::printf("Correctness check: %s\n", correctness_ok ? "PASS" : "FAIL (investigate before trusting timing numbers)");

    // ── Throughput ───────────────────────────────────────────────────
    for (int i = 0; i < warmup_iters; ++i)
        backend->matvec_f32(Wt, x.data(), y_gpu.data(), cols, rows);

    auto t0 = clock::now();
    for (int i = 0; i < timed_iters; ++i)
        backend->matvec_f32(Wt, x.data(), y_gpu.data(), cols, rows);
    auto t1 = clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_call_us = (total_ms * 1000.0) / timed_iters;
    double gflops = (2.0 * rows * cols) / (per_call_us * 1e-6) / 1e9;

    std::printf("\n%d x %d matvec, %d timed calls:\n", rows, cols, timed_iters);
    std::printf("  %.2f us/call, %.2f matvec/sec, ~%.2f GFLOP/s\n",
                per_call_us, 1e6 / per_call_us, gflops);
    std::printf("  (compare this per-call figure against your AVX2 path's per-call time\n"
                "   for the same shape — that's the actual number that answers \"is this\n"
                "   worth routing to yet\", not GFLOP/s in isolation.)\n");

    backend->free_tensor(Wt);
    return correctness_ok ? 0 : 1;
}
