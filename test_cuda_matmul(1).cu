// test_cuda_matmul.cu — cuBLAS SGEMM (F32) and HGEMM (F16, Tensor Core
// path) throughput on the SAME hardware test_vulkan_matmul.cpp ran on, so
// the ratio isolates API/driver-stack overhead the way test_cuda_matvec.cu
// did for the GEVM path. Same shapes (K=N=4096, M in {1,8,32,128}), same
// seed/data generation as test_vulkan_matmul.cpp, so the two programs'
// numbers are directly comparable.
//
// This is the compute-bound counterpart to test_cuda_matvec.cu's
// bandwidth-bound comparison. VULKAN_BACKEND_NOTES.md's caveat on the
// GEVM result — "both sides are mostly just waiting on the same memory
// bus... the compute-bound GEMM/prefill path has not been tested and may
// show a larger gap, since that's where tensor-core usage and aggressive
// tiling matter more" — is exactly what this pair of files is for
// answering.
//
// Build:
//   nvcc -O2 -std=c++17 test_cuda_matmul.cu -lcublas -o test_cuda_matmul
// Run:
//   ./test_cuda_matmul

#include <cublas_v2.h>
#include <cuda_fp16.h>
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

namespace {

constexpr int kK = 4096;
constexpr int kN = 4096;
const int kMShapes[] = {1, 8, 32, 128};
constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 100;

// Same reference + comparison shape as test_vulkan_matmul.cpp, kept
// independent (no shared header) so this file stays a standalone,
// CUDA-only harness the way test_cuda_matvec.cu is.
std::vector<float> cpu_reference_gemm(const std::vector<float>& A, const std::vector<float>& B,
                                       int M, int K, int N) {
    std::vector<float> C((size_t)M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[(size_t)m * K + k] * (double)B[(size_t)k * N + n];
            C[(size_t)m * N + n] = (float)acc;
        }
    return C;
}

double max_rel_err(const std::vector<float>& got, const std::vector<float>& ref) {
    double e = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double err = std::fabs((double)got[i] - (double)ref[i]);
        double denom = std::max(1e-6, std::fabs((double)ref[i]));
        e = std::max(e, err / denom);
    }
    return e;
}

void run_sgemm_case(cublasHandle_t handle, std::mt19937& rng, std::uniform_real_distribution<float>& dist, int M) {
    using clock = std::chrono::steady_clock;

    std::vector<float> A((size_t)M * kK), B((size_t)kK * kN);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    auto ref = cpu_reference_gemm(A, B, M, kK, kN);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * kK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)kK * kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * kN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), (size_t)kK * kN * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    std::vector<float> C((size_t)M * kN, 0.0f);

    // Row-major A[M x K] * B[K x N] via the same "read row-major as its
    // transpose" trick test_cuda_matvec.cu uses for SGEMV: compute
    // C^T = B^T * A^T in column-major, which read back row-major IS
    // C = A * B. cublasSgemm(OP_N, OP_N, n=N, m=M, k=K, B, ldb=N, A, lda=K, C, ldc=N).
    auto one_gemm = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), (size_t)M * kK * sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  kN, M, kK, &alpha,
                                  d_B, kN,
                                  d_A, kK,
                                  &beta, d_C, kN));
        CUDA_CHECK(cudaMemcpy(C.data(), d_C, (size_t)M * kN * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    one_gemm();
    double err = max_rel_err(C, ref);
    bool ok = err < 1e-2;
    std::printf("  [cuBLAS SGEMM] M=%-4d correctness: max_rel=%.6g -> %s\n", M, err, ok ? "PASS" : "FAIL");

    for (int i = 0; i < kWarmupIters; ++i) one_gemm();
    auto t0 = clock::now();
    for (int i = 0; i < kTimedIters; ++i) one_gemm();
    auto t1 = clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_call_us = (total_ms * 1000.0) / kTimedIters;
    double gflops = (2.0 * M * kK * kN) / (per_call_us * 1e-6) / 1e9;
    std::printf("  [cuBLAS SGEMM] M=%-4d %.2f us/call, ~%.2f GFLOP/s\n", M, per_call_us, gflops);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

__global__ void float_to_half_kernel(const float* in, __half* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

void run_hgemm_case(cublasHandle_t handle, std::mt19937& rng, std::uniform_real_distribution<float>& dist, int M) {
    using clock = std::chrono::steady_clock;

    std::vector<float> A((size_t)M * kK), B((size_t)kK * kN);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    // fp16 round-trip both operands before building the reference, same
    // reasoning as test_vulkan_matmul.cpp's run_f16_case() — otherwise
    // every shape shows spurious "error" that's just fp16 precision.
    std::vector<__half> A_h(A.size()), B_h(B.size());
    for (size_t i = 0; i < A.size(); ++i) A_h[i] = __float2half(A[i]);
    for (size_t i = 0; i < B.size(); ++i) B_h[i] = __float2half(B[i]);
    std::vector<float> A_rt(A.size()), B_rt(B.size());
    for (size_t i = 0; i < A.size(); ++i) A_rt[i] = __half2float(A_h[i]);
    for (size_t i = 0; i < B.size(); ++i) B_rt[i] = __half2float(B_h[i]);
    auto ref = cpu_reference_gemm(A_rt, B_rt, M, kK, kN);

    __half *d_A, *d_B; float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * kK * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)kK * kN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * kN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_B, B_h.data(), (size_t)kK * kN * sizeof(__half), cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    std::vector<float> C((size_t)M * kN, 0.0f);

    // Same row-major-via-transpose-trick as SGEMM above, F32 accumulate
    // (CUBLAS_COMPUTE_32F) — cublasGemmEx dispatches to Tensor Core paths
    // on hardware that has them (T4's Turing Tensor Cores) automatically
    // when operand types are FP16 and the shape/alignment qualify; no
    // separate "Tensor Core API" to opt into on this generation.
    auto one_gemm = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A, A_h.data(), (size_t)M * kK * sizeof(__half), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    kN, M, kK, &alpha,
                                    d_B, CUDA_R_16F, kN,
                                    d_A, CUDA_R_16F, kK,
                                    &beta, d_C, CUDA_R_32F, kN,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaMemcpy(C.data(), d_C, (size_t)M * kN * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    one_gemm();
    double err = max_rel_err(C, ref);
    bool ok = err < 5e-2; // same looser bar as test_vulkan_matmul.cpp's F16 case
    std::printf("  [cuBLAS HGEMM/TensorCore] M=%-4d correctness: max_rel=%.6g -> %s\n", M, err, ok ? "PASS" : "FAIL (or fp16 precision)");

    for (int i = 0; i < kWarmupIters; ++i) one_gemm();
    auto t0 = clock::now();
    for (int i = 0; i < kTimedIters; ++i) one_gemm();
    auto t1 = clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_call_us = (total_ms * 1000.0) / kTimedIters;
    double gflops = (2.0 * M * kK * kN) / (per_call_us * 1e-6) / 1e9;
    std::printf("  [cuBLAS HGEMM/TensorCore] M=%-4d %.2f us/call, ~%.2f GFLOP/s\n", M, per_call_us, gflops);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

} // namespace

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::printf("Initializing cuBLAS...\n");
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::printf("\n== cuBLAS SGEMM (F32) ==\n");
    for (int M : kMShapes) run_sgemm_case(handle, rng, dist, M);

    std::printf("\n== cuBLAS GemmEx FP16 (Tensor Core path on T4) ==\n");
    for (int M : kMShapes) run_hgemm_case(handle, rng, dist, M);

    std::printf("\nCompare directly against test_vulkan_matmul's F32/F16-coopmat numbers on\n"
                "the SAME T4 — this is the compute-bound comparison VULKAN_BACKEND_NOTES.md\n"
                "flagged as untested; expect a bigger Vulkan-vs-cuBLAS gap here than the\n"
                "GEVM result showed, especially once M grows past a few tiles.\n");

    cublasDestroy(handle);
    return 0;
}
