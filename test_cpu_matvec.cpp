// test_cpu_matvec.cpp — CPU-side F32 matvec throughput, using the exact
// runtime-dispatched simd::dot_f32() (rawllm_simd_dispatch.hpp) that
// proj_all_positions() calls per output row in the real engine. Same
// shape, same synthetic data generation, same warmup/timed-iteration
// structure as test_vulkan_matvec.cpp, so the two per-call numbers are
// directly comparable.
//
// Two variants are timed:
//   1. single_threaded_matvec(): one thread, straight loop over rows.
//      A true floor number, NOT what the engine achieves in practice —
//      proj_all_positions() parallelizes across rows via a ThreadPool.
//      Still useful as a baseline and to sanity-check dot_f32 itself.
//   2. threaded_matvec(): a naive std::thread row-split across
//      hardware_concurrency() cores. An approximation of the engine's
//      threaded path — NOT necessarily identical to whatever tuning your
//      real ThreadPool does (core pinning, spin-then-park), but a
//      reasonable stand-in without needing to wire up that type here.
//      This is the number that's actually fair to compare against the
//      Vulkan GFLOP/s figure, since the GPU is doing all 4096 rows in
//      parallel too.
//
// Build (try this first — GCC/Clang function multiversioning means
// dot_f32's AVX2/AVX-512 variants are selected by a runtime CPUID check
// inside the function itself, via per-function target() attributes, NOT
// a compile-wide -mavx2 flag — so a plain -O2 build should be enough for
// the dispatch to still pick the fastest available variant on this CPU):
//   g++ -std=c++20 -O2 -pthread test_cpu_matvec.cpp -o test_cpu_matvec
// Run:
//   ./test_cpu_matvec
//
// If this doesn't compile as-is, paste the error — I don't have
// rawllm_simd_dispatch.hpp's full contents (only grep output), so there
// may be an include path or a namespace detail I've gotten wrong.

#include "rawllm_simd_dispatch.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

int main() {
    using clock = std::chrono::steady_clock;

    // Same shape, same seed, same distribution as test_vulkan_matvec.cpp,
    // so the two runs' numbers are directly comparable.
    const uint32_t cols = 4096;
    const uint32_t rows = 4096;
    const int warmup_iters = 20;
    const int timed_iters = 500;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> W(static_cast<size_t>(rows) * cols);
    std::vector<float> x(cols);
    for (auto& v : W) v = dist(rng);
    for (auto& v : x) v = dist(rng);

    std::vector<float> y_ref(rows, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        const float* row = &W[static_cast<size_t>(r) * cols];
        for (uint32_t c = 0; c < cols; ++c) acc += static_cast<double>(row[c]) * x[c];
        y_ref[r] = static_cast<float>(acc);
    }

    std::vector<float> y(rows, 0.0f);

    auto single_threaded_matvec = [&]() {
        for (uint32_t r = 0; r < rows; ++r)
            y[r] = simd::dot_f32(&W[static_cast<size_t>(r) * cols], x.data(), cols);
    };

    unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
    auto threaded_matvec = [&]() {
        std::vector<std::thread> pool;
        uint32_t chunk = (rows + hw_threads - 1) / hw_threads;
        for (unsigned t = 0; t < hw_threads; ++t) {
            uint32_t r0 = t * chunk, r1 = std::min(rows, r0 + chunk);
            if (r0 >= r1) break;
            pool.emplace_back([&, r0, r1]() {
                for (uint32_t r = r0; r < r1; ++r)
                    y[r] = simd::dot_f32(&W[static_cast<size_t>(r) * cols], x.data(), cols);
            });
        }
        for (auto& th : pool) th.join();
    };

    auto check_correctness = [&](const char* label) {
        double max_abs_err = 0.0, max_rel_err = 0.0;
        for (uint32_t r = 0; r < rows; ++r) {
            double err = std::fabs(static_cast<double>(y[r]) - static_cast<double>(y_ref[r]));
            max_abs_err = std::max(max_abs_err, err);
            double denom = std::max(1e-6, std::fabs(static_cast<double>(y_ref[r])));
            max_rel_err = std::max(max_rel_err, err / denom);
        }
        bool ok = max_rel_err < 1e-2;
        std::printf("%s correctness: max abs err = %.6g, max rel err = %.6g -> %s\n",
                    label, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
        return ok;
    };

    auto bench = [&](const char* label, auto&& fn) {
        for (int i = 0; i < warmup_iters; ++i) fn();
        auto t0 = clock::now();
        for (int i = 0; i < timed_iters; ++i) fn();
        auto t1 = clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double per_call_us = (total_ms * 1000.0) / timed_iters;
        double gflops = (2.0 * rows * cols) / (per_call_us * 1e-6) / 1e9;
        std::printf("%s: %.2f us/call, %.2f matvec/sec, ~%.2f GFLOP/s\n",
                    label, per_call_us, 1e6 / per_call_us, gflops);
    };

    std::printf("hardware_concurrency() = %u\n\n", hw_threads);

    single_threaded_matvec();
    check_correctness("single-threaded");
    bench("single-threaded", single_threaded_matvec);

    std::printf("\n");
    threaded_matvec();
    check_correctness("threaded");
    bench("threaded", threaded_matvec);

    std::printf("\nCompare the 'threaded' per-call figure above against test_vulkan_matvec's\n"
                "326.21 us/call GPU result — that's the actual apples-to-apples number.\n");

    return 0;
}
