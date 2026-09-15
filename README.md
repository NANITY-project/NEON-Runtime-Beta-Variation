# NEON — Vulkan compute backend (experimental)

This branch is `main` plus everything needed to build and exercise
`rawllm_vulkan.hpp`, NEON's experimental Vulkan compute backend: the
`.comp` shader sources and the standalone Vulkan/CPU/CUDA test harnesses.
For the runtime itself, the model format, training, and general
build/usage docs, see `main`'s README — this one only covers the
Vulkan-specific pieces that live on this branch.

> **Status:** experimental, matvec-only, validated on real GPU hardware
> (NVIDIA T4) — see "What's validated vs. not" below.

This backend is deliberately **matvec-only, no GEMM/batched path**. NEON
is a single-user, single-prompt runtime — there's no batching workload
here that a GEMM kernel would actually serve, so that surface was
removed rather than carried as unused complexity.

## Files on this branch (beyond `main`)

| File | What it is |
|---|---|
| `matvec_f32.comp` | F32 GEVM (matrix-vector) shader — required |
| `matvec_f32_scalar.comp` | Scalar fallback variant of the F32 GEVM shader — required |
| `matvec_q4_0.comp` | Q4_0 quantized GEVM shader (manual unpack; `-DHAS_INT8_DOT=1`/`-DINT8_STORAGE=1` variants build from this same source) — required |
| `matvec_q8_0.comp` | Q8_0 quantized GEVM shader — required |
| `matvec_f16.comp` | Experimental F16 GEVM shader — optional, only loaded if its `.spv` is present |
| `matvec_f32_x2.comp` | Experimental 2-wide F32 GEVM shader — optional, only loaded if its `.spv` is present |
| `test_vulkan_matvec.cpp` | Standalone correctness + throughput check for the F32 matvec path, independent of `NEON-3.cpp` |
| `test_cpu_matvec.cpp` | CPU-side comparison: the real `simd::dot_f32()` dispatch, single-threaded and naive-threaded, same shape/seed as the Vulkan test so the numbers are comparable |
| `test_cuda_matvec.cu` | `cublasSgemv()` comparison on the same hardware — isolates the API/driver-stack gap from the Vulkan number, no hardware confound |
| `VULKAN_BACKEND_NOTES.md` | The full write-up: perf changes, real-hardware (NVIDIA T4) results, and the build recipe this README is based on |

`rawllm_vulkan.hpp` itself, `rawllm_rocm.hpp`, and the rest of the engine
are unchanged from `main` — see that branch's README for what they do.

## Prerequisites

- A Vulkan loader + headers (LunarG Vulkan SDK, or your distro's
  `vulkan-headers`/`libvulkan-dev`) and a working Vulkan device — real
  hardware or a software implementation like Mesa lavapipe for
  logic-only testing.
- `glslangValidator`, **built with `--target-env vulkan1.1` support** —
  the subgroup-reduction ops in `matvec_f32.comp`/`matvec_f32_scalar.comp`/
  `matvec_q4_0.comp`/`matvec_q8_0.comp` need SPIR-V 1.3, and the default
  target (SPIR-V 1.0) fails with `'subgroup op' : requires SPIR-V 1.3`
  without that flag.
- Optional, for the comparison harness only: a CUDA toolchain (`nvcc`,
  `cublas`) on NVIDIA hardware, to reproduce the cuBLAS numbers in
  `VULKAN_BACKEND_NOTES.md` alongside the Vulkan ones.

## Build: compiling the shaders

**Shader directory note.** `VULKAN_BACKEND_NOTES.md`'s original recipe
compiles straight to the repo root (the `.comp` sources sit there, not
in a `shaders/` subfolder) and passes `"."` as the shaders directory.
`test_vulkan_matvec.cpp` and `NEON-3.cpp`'s `--probe` path instead
hardcode `"shaders"` as the constructor argument. Pick one and be
consistent:

- **Easiest:** compile into a `shaders/` subdirectory, matching what the
  test harness and `NEON-3.cpp` already expect:
  ```bash
  mkdir -p shaders
  glslangValidator -V --target-env vulkan1.1 matvec_f32.comp        -o shaders/matvec_f32.spv
  glslangValidator -V --target-env vulkan1.1 matvec_f32_scalar.comp -o shaders/matvec_f32_scalar.spv
  glslangValidator -V --target-env vulkan1.1 matvec_q4_0.comp       -o shaders/matvec_q4_0.spv
  glslangValidator -V --target-env vulkan1.1 matvec_q8_0.comp       -o shaders/matvec_q8_0.spv
  # Optional, experimental, only loaded if present:
  glslangValidator -V --target-env vulkan1.1 matvec_f16.comp     -o shaders/matvec_f16.spv
  glslangValidator -V --target-env vulkan1.1 matvec_f32_x2.comp  -o shaders/matvec_f32_x2.spv
  ```
- **Alternative:** if you'd rather match `VULKAN_BACKEND_NOTES.md`
  exactly, compile straight to the repo root (drop `shaders/` from every
  `-o` path above) and change the constructor argument in
  `test_vulkan_matvec.cpp`/`NEON-3.cpp` from `"shaders"` to `"."`.

## Build & run: the Vulkan test harness

```bash
g++ -std=c++20 -O2 -pthread -DUSE_VULKAN test_vulkan_matvec.cpp -lvulkan -o test_vulkan_matvec
./test_vulkan_matvec
```
Exercises the F32 GEVM path end-to-end: constructs `VulkanMatvecBackend`,
uploads a random 4096×4096 weight matrix, checks GPU output against a
CPU double-precision reference (pass bar: max relative error < 1e-2,
looser than bit-exact since GPU/CPU float accumulation order differs),
then times 500 iterations after 20 warmup calls. Doesn't need a trained
model or the rest of the engine — it drives the backend directly with
synthetic data.

## Build & run: the comparison harnesses

For putting the Vulkan numbers in context against CPU and (on NVIDIA
hardware) cuBLAS, using the same shape/seed/iteration counts so the
`us/call` outputs line up:

```bash
g++ -std=c++20 -O2 -pthread test_cpu_matvec.cpp -o test_cpu_matvec
./test_cpu_matvec
```
Times `simd::dot_f32()` — the actual AVX2/AVX-512 dispatch
`proj_all_positions()` uses in the real engine — single-threaded and via
a naive `std::thread` row-split. GCC/Clang function multiversioning means
a plain `-O2` build already picks the fastest ISA variant for the host
CPU; no `-mavx2`/`-mavx512...` flags needed for this one.

```bash
nvcc -O2 -std=c++17 test_cuda_matvec.cu -lcublas -o test_cuda_matvec
./test_cuda_matvec
```
NVIDIA-only, needs the CUDA toolkit (`nvcc` is usually already on `PATH`
in a Colab/cloud GPU runtime; otherwise typically at
`/usr/local/cuda/bin/nvcc`). Matches `test_vulkan_matvec.cpp`'s "one
blocking round trip per call" discipline deliberately, so the two
`us/call` numbers are directly comparable — this is not how you'd use
cuBLAS for best throughput in a real pipeline.

## Building the full runtime with Vulkan enabled

Once the shaders above are compiled and in place:
```bash
g++ -std=c++20 -O2 -pthread -DUSE_VULKAN NEON-3.cpp -lvulkan -o neon
./neon --model your-model-nanity.gguf --probe
```
`--probe` constructs `VulkanMatvecBackend` and reports whether a usable
device + shader set was found (`[Hardware] Vulkan: ...`), but this is
informational only — every token is still generated on the CPU path
regardless of what `--probe` reports. Nothing here routes real
generation through the GPU yet; see `rawllm_vulkan.hpp`'s header comment
and `VULKAN_BACKEND_NOTES.md`'s "What's not done yet" for the actual
state of GPU-routed inference.

## What's validated vs. not (see `VULKAN_BACKEND_NOTES.md` for the full detail)

- **Real-hardware result on record:** NVIDIA T4, F32 4096×4096 matvec.
  The optimized backend (persistent block-level memory mapping +
  descriptor-set-write deduplication) came in **about 5–11% slower than
  cuBLAS's SGEMV** on the same hardware — a clean isolation of the
  API/driver-stack gap specifically, with hardware held constant.
- **Not tested on any hardware yet:** AMD/RDNA/CDNA/MI300X (the
  project's actual target — the T4 result used the only GPU available
  at the time) and the Q4_0/Q8_0 quantized GEVM path.

## Back to `main`

```bash
git checkout main
```
for the general runtime README — build flags, CPU SIMD backends,
training pipeline, quick start, and the full repository layout.
