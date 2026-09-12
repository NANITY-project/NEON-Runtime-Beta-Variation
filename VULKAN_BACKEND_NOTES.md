# Vulkan compute backend — perf work & first real-hardware validation

This covers two things done in this pass: (1) two host-side performance
fixes to `rawllm_vulkan.hpp`, and (2) the first time this backend has been
run against real GPU hardware rather than Mesa lavapipe (software Vulkan).

## Status at a glance

| | |
|---|---|
| Correctness | Validated on real hardware (NVIDIA T4, F32 GEMV path) |
| Perf changes below | Implemented, syntax-checked against real Vulkan headers — **not yet benchmarked on real hardware** (see "What's not done yet") |
| Hardware tested | NVIDIA T4 (Google Colab), F32 4096x4096 matvec only |
| Hardware NOT tested | Any AMD device, any MI300X/ROCm target, the Q4_0/Q8_0 quantized path, the GEMM/prefill path |

## Perf changes made this pass

Two host-side overhead cuts to `rawllm_vulkan.hpp`, no shader or dispatch
logic changed:

1. **Persistent block-level memory mapping.** `upload()`/`download()`
   previously called `vkMapMemory`/`vkUnmapMemory` on every single call —
   i.e. every token, every projection. Memory blocks (the suballocator's
   `MemBlock`s) are now mapped once, up front, if their type is
   host-visible; every `GpuBuffer`/`GpuTensor` suballocated from a block
   derives its pointer as `block.mapped + offset` and just `memcpy`s
   directly, for the buffer's whole lifetime.
2. **Descriptor-set write deduplication.** `queue_matvec_*()`/
   `queue_matmul_*()` previously called `vkUpdateDescriptorSets()`
   unconditionally on every call. Since a given batch slot is reused for
   the same logical projection every token, and both the weight tensor
   and the grow-never-shrink scratch buffers behind it are the same
   `VkBuffer` handles once warmed up, the descriptor write is now skipped
   whenever a slot's bound buffers haven't changed since last time
   (`update_descriptor_set3_if_changed()`/`_5_if_changed()`).

Rationale in full: a single decode-time matvec's actual GPU compute time
is on the order of microseconds, so the CPU-side bookkeeping around each
dispatch (map/unmap syscalls, redundant descriptor writes) was a larger
share of per-token cost than the shader itself for this workload shape.

**Not done yet, same category of win, ordered by expected impact:**
- Closing the dp4a gap for the actually-hot GEVM shaders
  (`matvec_q4_0.comp`/`matvec_q8_0.comp` still do a manual unpack loop;
  only the GEMM path got `dotPacked4x8EXT()` treatment) — blocked on
  toolchain (see below), not on device support.
- Pre-recording the command buffer once per fixed decode shape instead of
  re-recording identical `vkCmdBind*`/`vkCmdDispatch` calls every token
  (biggest remaining structural win, most invasive to build).
- `VK_KHR_push_descriptor` as a more thorough alternative to the
  dedup-caching above.
- True int8 activation upload for the quantized path instead of widening
  to int32 host-side (4x less upload bandwidth), gated on
  `VK_KHR_8bit_storage`/`shaderInt8` support.

## Real-hardware validation: NVIDIA T4 (Google Colab)

First time any part of this backend has run against a real GPU — every
prior validation was Mesa lavapipe (software rasterizer; proves logic
correctness, says nothing about throughput). Shape tested: 4096x4096 F32
matvec, single blocking round trip per call (upload activation, dispatch,
download output, synchronize) — the worst case for dispatch overhead, via
the `matvec_f32()` convenience wrapper, not the batched
`queue_matvec_*()`/`begin_batch()`/`end_batch()` form a real decode step
would use.

**This was the pre-optimization backend** — the two changes above were
not yet applied when these numbers were captured (confirmed via
`grep -n "update_descriptor_set3_if_changed" rawllm_vulkan.hpp` returning
no match in the tested checkout). Numbers below are a baseline, not a
"with optimizations" result.

| Path | us/call | matvec/sec | GFLOP/s | Effective bandwidth |
|---|---|---|---|---|
| Vulkan (this backend, baseline) | 326.21 | 3065.51 | ~102.86 | ~206 GB/s (~64% of T4's ~320 GB/s peak) |
| cuBLAS SGEMV (same T4) | 287.79 | 3474.78 | ~116.59 | ~233 GB/s |
| CPU AVX2, threaded (2-vCPU Colab VM) | 4858.78 | 205.81 | ~6.91 | — |

**Vulkan is ~13.3% slower than cuBLAS on identical hardware.** This
isolates the API/driver-stack gap specifically (generic portable compute
shader vs. vendor-tuned GEMV kernel), with hardware held constant — a
much cleaner comparison than the CPU number, which crosses both hardware
*and* API and shouldn't be read as more than a sanity check (see caveats
below).

**Caveats on each row:**
- **Vulkan/cuBLAS**: both memory-bandwidth-bound at this shape (matvec is
  dominated by reading the weight matrix, not compute), which is the
  *easiest* case for a generic shader to stay close to a tuned library —
  both sides are mostly just waiting on the same memory bus. The
  compute-bound GEMM/prefill path (`queue_matmul_f32`/`_q4_0`/`_q8_0` vs.
  `cublasSgemm`) has not been tested and may show a larger gap, since
  that's where tensor-core usage and aggressive tiling matter more.
- **CPU AVX2**: `hardware_concurrency()` on the Colab free tier was 2 —
  threading barely helped (5057.85 → 4858.78 us/call), consistent with
  this being bandwidth-bound on the CPU side too, on a shared/virtualized
  memory channel. This number reflects a 2-vCPU cloud VM, not a real
  desktop or server CPU's AVX2 ceiling, and should not be used to
  estimate performance on real target hardware (e.g. the project's local
  i5-class dev machine).

## What's not done yet

- The optimization pass above, benchmarked on real hardware (only
  lavapipe + syntax-checked so far).
- Any AMD hardware. Nothing here has run on RDNA/CDNA/an actual MI300X —
  this session used a Colab T4 because that's the only GPU currently
  available; the project's primary target is AMD ROCm/HIP.
- The Q4_0/Q8_0 quantized GEVM/GEMM paths on real hardware.
- The GEMM/prefill path (`queue_matmul_*`) on real hardware, and its
  comparison against `cublasSgemm`/`rocblas_sgemm`.
- Whether the T4's `VK_KHR_cooperative_matrix` path
  (`matmul_coopmat_f16`) is actually correct/fast — it compiled and
  construction accepted it, but it wasn't exercised in this test (only
  the plain F32 GEVM path was).

## Known toolchain issue: dp4a shader variants don't compile here

Ubuntu 24.04's `glslang-tools` 15.1.0 cannot parse
`#extension GL_EXT_shader_integer_dot_product` — confirmed by attempting
to compile `matmul_tiled_q4_0.comp`/`_q8_0.comp` with `-DHAS_INT8_DOT=1`,
which fails with `extension not supported`. This matches a limitation the
file's own header comments already documented.

The T4 *does* report `VK_KHR_shader_integer_dot_product` support
(`dot_product_supported_` comes back true from the device query), so
`create_pipelines()` tries to load the `_dp4a` variant and crashes
construction when it's missing. **Workaround applied for this test
session only** (not a real fix): `dot_product_supported_` forced to
`false` right after its assignment in the constructor, so the portable
(non-dp4a) `matmul_tiled_q4_0.spv`/`matmul_tiled_q8_0.spv` load instead.

**Real fix, not yet done:** `create_pipelines()` should catch a missing
optional-variant `.spv` file and fall back to the portable pipeline at
runtime, instead of (a) crashing outright, or (b) requiring this kind of
global capability override as a workaround. Worth doing before this
backend ships anywhere — a device correctly reporting a capability it
then can't get a shader for, because of a toolchain gap two build steps
away, is exactly the kind of thing that turns into a confusing crash
report from someone who isn't debugging it interactively.

## Reproducing this

Three standalone test harnesses, independent of the rest of the engine
(bypass `NEON-3.cpp` entirely — useful for isolating backend-level
questions without needing to understand the whole engine's CLI):

- `test_vulkan_matvec.cpp` — this backend's F32 matvec, correctness +
  throughput.
- `test_cpu_matvec.cpp` — `simd::dot_f32()` (the real AVX2/AVX-512
  dispatch `proj_all_positions()` uses), single-threaded and
  naive-`std::thread`-parallel variants.
- `test_cuda_matvec.cu` — `cublasSgemv()` on the same hardware, for an
  apples-to-apples API comparison.

All three use the same shape (4096x4096), the same RNG seed (42) and
distribution, and the same warmup(20)/timed(500) iteration structure, so
their `us/call` outputs are directly comparable.

### Build notes specific to this repo's layout

- `.comp` shader sources are at the repo root, **not** in a `shaders/`
  subdirectory — pass `"."` as the shaders directory to
  `VulkanMatvecBackend`'s constructor, not the default `"shaders"`.
- `glslangValidator` needs `--target-env vulkan1.1` (or newer) to compile
  `matvec_f32.comp`, `matvec_f32_scalar.comp`, `matvec_q4_0.comp`,
  `matvec_q8_0.comp` — the default target is SPIR-V 1.0, and this
  backend's subgroup-reduction shaders (`subgroupAdd()`) need SPIR-V 1.3.
  Without this flag: `'subgroup op' : requires SPIR-V 1.3`.
- Required `.spv` files for `create_pipelines()` to succeed (all
  unconditional except where noted):
  ```
  matvec_f32.spv
  matvec_f32_scalar.spv
  matvec_q4_0.spv
  matvec_q8_0.spv
  matmul_tiled_f32.spv
  matmul_tiled_q4_0.spv        (or matmul_tiled_q4_0_dp4a.spv, see below)
  matmul_tiled_q8_0.spv        (or matmul_tiled_q8_0_dp4a.spv, see below)
  matmul_coopmat_f16.spv       (only if the device reports
                                 VK_KHR_cooperative_matrix support)
  ```
- The `_dp4a` variants currently don't compile with this toolchain — see
  "Known toolchain issue" above. Compile the plain variants and apply the
  `dot_product_supported_` workaround (or the proper fallback fix, once
  written) if your device reports dot-product support.

```bash
glslangValidator -V --target-env vulkan1.1 matvec_f32.comp        -o matvec_f32.spv
glslangValidator -V --target-env vulkan1.1 matvec_f32_scalar.comp -o matvec_f32_scalar.spv
glslangValidator -V --target-env vulkan1.1 matvec_q4_0.comp       -o matvec_q4_0.spv
glslangValidator -V --target-env vulkan1.1 matvec_q8_0.comp       -o matvec_q8_0.spv
glslangValidator -V --target-env vulkan1.1 matmul_tiled_f32.comp  -o matmul_tiled_f32.spv
glslangValidator -V --target-env vulkan1.1 matmul_tiled_q4_0.comp -o matmul_tiled_q4_0.spv
glslangValidator -V --target-env vulkan1.1 matmul_tiled_q8_0.comp -o matmul_tiled_q8_0.spv
glslangValidator -V --target-env vulkan1.1 matmul_coopmat_f16.comp -o matmul_coopmat_f16.spv  # if needed

g++ -std=c++20 -O2 -pthread -DUSE_VULKAN test_vulkan_matvec.cpp -lvulkan -o test_vulkan_matvec
g++ -std=c++20 -O2 -pthread test_cpu_matvec.cpp -o test_cpu_matvec
nvcc -O2 -std=c++17 test_cuda_matvec.cu -lcublas -o test_cuda_matvec
```
