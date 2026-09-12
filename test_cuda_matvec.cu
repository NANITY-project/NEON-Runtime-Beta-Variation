// test_cuda_matvec.cu — cuBLAS SGEMV throughput on the SAME hardware
// (T4) the Vulkan test ran on, so this isolates the API/driver stack as
// the only variable — no hardware confound like the CPU comparison had.
// Same shape (4096x4096), same seed/data generation, same warmup/timed
// iteration counts, and — deliberately — the same "one blocking round
// trip per call" discipline as test_vulkan_matvec.cpp's matvec_f32()
// (upload x, compute, download y, synchronize — every single call, not
// batched). That matches what was actually measured on the Vulkan side,
// so the two us/call numbers are comparable apples-to-apples. It is NOT
// how you'd use cuBLAS for best throughput in a real pipeline (you'd
// keep everything device-resident and pipeline calls with streams) —
// same caveat the Vulkan test's own comment made about matvec_f32()
// being the worst-case single-op wrapper, not the batched form.
//
// Build:
//   nvcc -O2 -std=c++17 test_cuda_matvec.cu -lcublas -o test_cuda_matvec
// (nvcc already knows where the CUDA toolkit's headers/libs live in a
// Colab GPU runtime — if it's not on PATH, it's usually at
// /usr/local/cuda/bin/nvcc.)
// Run:
//   ./test_cuda_matvec

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st__ = (call); \
    if (st__ != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %s at %s:%d: status=%d\n", #call, __FILE__, __LINE__, (int)st__); \
        std::exit(1); \
    } \
} while (0)

int main() {
    using clock = std::chrono::steady_clock;

    // Same shape, same seed, same distribution as test_vulkan_matvec.cpp
    // and test_cpu_matvec.cpp.
    const int cols = 4096;
    const int rows = 4096;
    const int warmup_iters = 20;
    const int timed_iters = 500;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> W(static_cast<size_t>(rows) * cols);
    std::vector<float> x(cols);
    for (auto& v : W) v = dist(rng);
    for (auto& v : x) v = dist(rng);

    std::vector<float> y_ref(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        const float* row = &W[static_cast<size_t>(r) * cols];
        for (int c = 0; c < cols; ++c) acc += static_cast<double>(row[c]) * x[c];
        y_ref[r] = static_cast<float>(acc);
    }

    std::printf("Initializing cuBLAS...\n");
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Weight upload ONCE, device-resident for the backend's whole
    // lifetime — mirrors upload_weight_f32() in the Vulkan test. x/y are
    // re-transferred every call below, matching what matvec_f32()
    // actually does per call on the Vulkan side (activation upload +
    // output download every time, weight resident).
    float *d_W, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(rows) * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(cols) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(rows) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W, W.data(), static_cast<size_t>(rows) * cols * sizeof(float), cudaMemcpyHostToDevice));
    std::printf("Weight uploaded, device-resident.\n");

    // W is row-major (rows x cols), W[r*cols+c] = W(r,c). cuBLAS is
    // column-major. The same memory, read as column-major with
    // dimensions (m=cols, n=rows, lda=cols), IS W^T — so
    // cublasSgemv(OP_T, m=cols, n=rows, ...) computes (W^T)^T * x = W*x
    // directly against the row-major buffer, no host-side transpose
    // needed. Standard row-major-via-cuBLAS trick.
    const float alpha = 1.0f, beta = 0.0f;

    std::vector<float> y(rows, 0.0f);
    auto one_matvec = [&]() {
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), static_cast<size_t>(cols) * sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, cols, rows, &alpha, d_W, cols, d_x, 1, &beta, d_y, 1));
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, static_cast<size_t>(rows) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize()); // one full blocking round trip, matching matvec_f32()'s discipline
    };

    one_matvec();
    double max_abs_err = 0.0, max_rel_err = 0.0;
    for (int r = 0; r < rows; ++r) {
        double err = std::fabs(static_cast<double>(y[r]) - static_cast<double>(y_ref[r]));
        max_abs_err = std::max(max_abs_err, err);
        double denom = std::max(1e-6, std::fabs(static_cast<double>(y_ref[r])));
        max_rel_err = std::max(max_rel_err, err / denom);
    }
    bool ok = max_rel_err < 1e-2;
    std::printf("Correctness: max abs err = %.6g, max rel err = %.6g -> %s\n",
                max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");

    for (int i = 0; i < warmup_iters; ++i) one_matvec();

    auto t0 = clock::now();
    for (int i = 0; i < timed_iters; ++i) one_matvec();
    auto t1 = clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_call_us = (total_ms * 1000.0) / timed_iters;
    double gflops = (2.0 * rows * cols) / (per_call_us * 1e-6) / 1e9;
    double gbps = (static_cast<double>(rows) * cols * sizeof(float)) / (per_call_us * 1e-6) / 1e9;

    std::printf("\n%d x %d cuBLAS SGEMV, %d timed calls:\n", rows, cols, timed_iters);
    std::printf("  %.2f us/call, %.2f matvec/sec, ~%.2f GFLOP/s, ~%.2f GB/s effective bandwidth\n",
                per_call_us, 1e6 / per_call_us, gflops, gbps);
    std::printf("\nCompare this directly against test_vulkan_matvec's result on the SAME T4 —\n"
                "same hardware, so the ratio here isolates cuBLAS-vs-Vulkan-compute-shader\n"
                "overhead specifically, not a hardware difference.\n");

    cublasDestroy(handle);
    cudaFree(d_W); cudaFree(d_x); cudaFree(d_y);
    return ok ? 0 : 1;
}
