#pragma once
// =============================================================================
// rawllm_vulkan.hpp — Vulkan compute backend for matvec, covering F32 AND
// the fused Q4_0/Q8_0 quantized paths (proj_all_positions()'s per-tensor
// dequantize+dot / fused-int8-dot loops in rawllm_forward.hpp).
//
// STATUS: still not wired into the engine's hot path — rawllm_forward.hpp
// does not call into this yet, CPU SIMD (rawllm_simd_dispatch.hpp) remains
// the only path NEON-3.cpp actually exercises during generation. Deciding
// when GPU beats CPU for a given tensor size and doing the routing is still
// future work. What changed in this pass is everything about HOW a matvec
// runs once you do call in, taking cues from how llama.cpp's ggml-vulkan
// backend and rocBLAS's GEMV kernels structure this exact operation:
//
//   1. WEIGHTS ARE RESIDENT, NOT RE-UPLOADED PER TOKEN. The previous version
//      of this file re-uploaded the entire weight matrix — potentially
//      hundreds of MB — to the GPU on EVERY matvec() call, which would have
//      made autoregressive decoding (one token = one matvec per projection)
//      pay a full weight-upload cost per token per tensor. That's the
//      single biggest fix here: upload_weight_f32()/_q4_0()/_q8_0() upload a
//      tensor ONCE at model-load time into a device-local GpuTensor handle;
//      matvec_f32()/_q4_0()/_q8_0() then only transfer the small activation
//      vector (cols floats, not rows*cols) and download the small output
//      vector (rows floats) per call. This mirrors how ggml-vulkan keeps
//      the model's weight tensors resident in VRAM for the whole session.
//   2. QUANTIZED WEIGHTS STAY QUANTIZED ON THE GPU. matvec_q4_0()/_q8_0()
//      upload the on-disk Q4_0/Q8_0 block bytes as-is and do the nibble
//      unpack + fused int8 dot IN the shader (shaders/matvec_q4_0.comp,
//      matvec_q8_0.comp) — no per-row dequant-to-F32 pass, matching what
//      the CPU fused kernels in rawllm_simd_dispatch.hpp already do and
//      closing the gap this file's previous version called out as a
//      documented follow-up ("quantized tensors would need a per-row
//      dequant step before this shader could touch them").
//   3. SUBGROUP-PER-ROW REDUCTION, NOT SHARED-MEMORY TREE REDUCTION. All
//      three shaders now give one subgroup/wavefront its own output row and
//      combine that row's partial sums with subgroupAdd() — no shared
//      memory, no barrier() at all — and fit gl_NumSubgroups rows into one
//      workgroup dispatch instead of one. See shaders/matvec_f32.comp's
//      header comment for the full rationale (this is the same shape
//      rocBLAS's and llama.cpp's GEMV kernels use).
//   4. PERSISTENT GPU RESOURCES. Descriptor sets, the command buffer, and
//      the small per-call scratch buffers (activation upload, output
//      download) are now allocated ONCE and reused/grown across calls
//      instead of allocated-and-freed every matvec() (which is what forced
//      the old code's documented late fix: a descriptor pool created
//      without FREE_DESCRIPTOR_SET support that broke on a call's 2nd use).
//   5. BATCHED RECORDING: MULTIPLE DISPATCHES PER SUBMIT. matvec_f32() /
//      matvec_q4_0() / matvec_q8_0() are now thin convenience wrappers
//      around begin_batch()/queue_matvec_*()/end_batch() — a batch records
//      N dispatches into the ONE persistent command buffer and does exactly
//      one vkQueueSubmit + one vkQueueWaitIdle for however many ops it
//      contains, instead of a submit+wait round trip per op. This matters
//      because a transformer layer's Q/K/V projections (and separately its
//      MLP gate/up projections) are mutually independent — they read the
//      same input activation and write disjoint output buffers — so they
//      can be queued back-to-back with NO barrier between them and let the
//      GPU schedule them without a CPU round trip in between. A dependent
//      op (one whose input is itself GPU-computed output from an earlier
//      op in the same batch) needs an explicit barrier() call between the
//      two queue_*() calls; ops queued without an intervening barrier() are
//      assumed independent. NOTE ON WHAT THIS DOES NOT FIX: the elementwise
//      work between dependent matvecs in this codebase (softmax over
//      attention scores, SiLU(gate)*up) runs on the CPU today, not in a
//      shader — so batching mainly collapses the *independent* groups
//      (QKV together, gate+up together) from 3-or-2 submits into 1 each;
//      it does not turn a whole layer into a single submit, and won't
//      until that elementwise math has a GPU kernel of its own to bridge
//      the gap without leaving the GPU.
//   7. TIMELINE SEMAPHORES + A SEPARATE TRANSFER QUEUE. end_batch() used
//      to vkQueueWaitIdle() the compute queue before returning, which
//      meant the CPU sat idle for the whole GPU submission before it
//      could start preparing the NEXT batch's activations (quantizing
//      the next token's activations, running the CPU-side softmax/SiLU
//      elementwise math between dependent matvecs, etc). end_batch() now
//      submits with a VK_SEMAPHORE_TYPE_TIMELINE signal and returns
//      immediately (see BatchHandle, end_batch(), wait_batch()); the two
//      persistent command buffers / descriptor-set-slot arrays / scratch
//      buffers this backend already allocated are now double-buffered
//      (kFramesInFlight=2) so begin_batch() for the NEXT batch can record
//      into and upload activations for the OTHER slot while THIS slot's
//      submission is still in flight, only blocking (in begin_batch()'s
//      defensive sync_slot() call) if the caller tries to reuse a given
//      slot before its previous occupant's GPU work has actually
//      finished. Separately, upload_weight_bytes()'s one-time staging
//      copy now runs on a dedicated transfer-only queue family when the
//      device exposes one (see pick_physical_device()'s search and
//      copy_staging_to_weight()), instead of sharing the compute queue's
//      timeline for that copy — the DMA engine backing a real transfer
//      queue can move bytes independently of whatever the compute queue
//      is dispatching. Both pieces require a queue-family-ownership
//      release/acquire barrier pair around the cross-queue copy (see
//      copy_staging_to_weight()) since every buffer here is created
//      VK_SHARING_MODE_EXCLUSIVE.
//   8. SUBALLOCATION FROM A HANDFUL OF LARGE VkDeviceMemory BLOCKS. Every
//      GpuBuffer/GpuTensor used to get its own vkAllocateMemory call —
//      fine at a handful of tensors, but a real model has one weight
//      tensor per projection per layer (easily 100+ for a many-layer
//      model), and drivers cap the total number of live memory objects
//      a process may hold (VkPhysicalDeviceMaintenance3Properties::
//      maxMemoryAllocationCount — as low as 4096 on some drivers, far
//      less on others) — burning that budget one tensor at a time for
//      no benefit is exactly what VMA (and every other serious Vulkan
//      allocator) exists to avoid. make_buffer()/upload_weight_bytes()
//      now go through sub_alloc()/sub_free(): a handful of large
//      VkDeviceMemory blocks per memory-type index, each subdivided by a
//      simple offset+size free-list (grown by allocating a new block
//      only when no existing block has room). GpuBuffer/GpuTensor now
//      carry a SubAlloc (block index + offset + size) instead of their
//      own VkDeviceMemory; vkMapMemory calls take that offset into the
//      shared block instead of assuming offset 0.
//   9. BATCHING ACROSS CONCURRENT REQUESTS, NOT JUST ONE REQUEST'S
//      INDEPENDENT PROJECTIONS. Point 5 above (batched recording) only
//      ever showed one sequence's Q/K/V or gate/up sharing a batch —
//      independent ops of a SINGLE request/token. Continuous batching
//      needs the other axis too: N *different* concurrent requests each
//      running the SAME projection weight this step. queue_matvec_f32_
//      requests()/_q4_0_requests()/_q8_0_requests() queue N independent
//      GEMV dispatches (one per request, same GpuTensor W, different
//      per-request x/y) into the current batch — mechanically just N
//      calls to the existing queue_matvec_*() functions, but named and
//      documented so this is a first-class supported pattern rather
//      than something a caller has to intuit, and sized by the
//      constructor's max_ops_per_batch parameter (formerly a fixed
//      kMaxOpsPerBatch=8 constant) rather than a hardcoded one-request
//      budget. queue_matvec_f16_requests() is the coop-matrix-GEMM-path
//      equivalent — a thin, purely-documentation wrapper around
//      queue_matmul_f16(), which already treats its M activation rows
//      generically and therefore already handles this case; the wrapper
//      exists so a caller doing decode-time request batching doesn't
//      have to read queue_matmul_f16's prefill-oriented header comment
//      to realize it. See that wrapper's comment for the (unmeasured)
//      GEMV-vs-GEMM crossover question this raises.
//  10. OPTIONAL FP16 COOPERATIVE-MATRIX (MATRIX-CORE) GEMM PATH:
//      upload_weight_f16() + queue_matmul_f16(), used only where the
//      VK_KHR_cooperative_matrix extension AND a 16x16x16 fp16xfp16->f32
//      subgroup-scope shape are both present (see coop_matrix_supported())
//      — every other function in this file works identically with zero
//      device extensions if that's absent. This is a real matrix-matrix
//      GEMM path, not another matvec: it exists for prefill (many
//      activation rows at once), where there's enough work to fill a
//      16x16 matrix-core tile. It is NOT a replacement for the per-token
//      decode matvec_*() functions above — running this at M=1 wastes a
//      whole tile computing 15 rows nobody asked for (see
//      matmul_coopmat_f16.comp's header comment). Nothing in this file
//      decides prefill-vs-decode for you; that's a rawllm_forward.hpp
//      decision, same as every other scheduling choice this file leaves
//      to its caller.
//  11. PORTABLE TILED F32 GEMM: queue_matmul_f32() + matmul_tiled_f32.comp,
//      a plain shared-memory tiled GEMM with no extension and no
//      device-capability check — built unconditionally, runs everywhere
//      queue_matvec_f32() already runs. This exists because
//      queue_matmul_f16() only works on the (currently rare) hardware
//      with VK_KHR_cooperative_matrix support; without this, every other
//      device had NO batched-GEMM prefill path at all, only a loop of
//      per-token matvec_f32() calls. Same caveat as point 10: this is a
//      GEMM for prefill (M>1), not a GEMV replacement — see
//      queue_matmul_f32()'s own comment.
//  12. QUANTIZED BATCHED GEMM: queue_matmul_q4_0()/queue_matmul_q8_0() +
//      matmul_tiled_q4_0.comp/matmul_tiled_q8_0.comp. Closes the gap point
//      11 left open: before this, a Q4_0/Q8_0 model had queue_matmul_f16()
//      (needs coop-matrix hardware) or queue_matmul_f32() (needs an F32
//      copy of the weights — a dequant pass this backend was specifically
//      built to avoid, see point 2 above) for prefill, and otherwise had
//      to fall back to looping queue_matvec_q4_0()/_q8_0() once per prompt
//      token — exactly the per-token round-trip cost points 1/5/7 exist to
//      eliminate. Same shared-memory tiling as matmul_tiled_f32.comp, but
//      the tile load unpacks the Q4_0/Q8_0 block format (see
//      upload_weight_q4_0()/_q8_0()'s doc comments for the on-disk layout)
//      instead of reading plain floats — weights stay quantized on the
//      GPU the whole time, same as the GEMV path. Activations are NOT
//      quantized inside this file: queue_matmul_q4_0()/_q8_0() take
//      already-block-quantized xs_d/xs_sum/xs_q arrays, one row's worth
//      per M (same per-row layout queue_matvec_q4_0()/_q8_0() already
//      expect, just M of them back to back) — callers already have this
//      quantization code (it's what feeds the CPU fused kernels and the
//      GEMV path above), so this file doesn't duplicate it.
//
//      Two SPIR-V variants of each shader are built from the same .comp
//      source: a portable one compiled as-is, and a faster one compiled
//      with -DHAS_INT8_DOT=1 that uses GL_EXT_integer_dot_product's
//      dotPacked4x8EXT()/dotPacked4x8AccSatEXT() built-ins — the GLSL
//      surface for VK_KHR_shader_integer_dot_product, the single-
//      instruction 4xint8 dot this backend's manual unpack-and-multiply-
//      loop shaders were missing (the Vulkan analogue of the dp4a-style
//      instruction the ROCm/AVX2 CPU kernels already use — this was #1 on
//      this file's very first roadmap and is only closed now, for the
//      GEMM path; the GEMV shaders (matvec_q4_0.comp/matvec_q8_0.comp)
//      still do the manual unpack loop and are the natural next target for
//      the same treatment). dot_product_supported() reports whether the
//      device advertises VK_KHR_shader_integer_dot_product with 8-bit
//      signed-signed-accumulate support; create_pipelines() picks the
//      _dp4a pipeline when true and the portable one otherwise, exactly
//      the same shape as point 10's coop-matrix-or-not choice.
//
//      VALIDATED (portable variant only, this pass): matmul_tiled_q4_0.comp
//      and matmul_tiled_q8_0.comp (compiled WITHOUT -DHAS_INT8_DOT) both
//      compile cleanly with glslangValidator and pass spirv-val — actually
//      run in-sandbox, not just eyeballed. NOT run end-to-end against a CPU
//      quantized-GEMM reference yet (no device was available this pass,
//      same caveat matmul_tiled_f32.comp had before ITS harness run — see
//      queue_matmul_f32()'s comment).
//
//      NOT validated, and NOT EVEN COMPILABLE in this pass's sandbox: the
//      -DHAS_INT8_DOT dp4a variant. The glslang shipped by Ubuntu 24.04's
//      noble-updates archive (glslang-tools 15.1.0-2, checked this pass)
//      recognizes SPV_KHR_integer_dot_product as a SPIR-V-level string but
//      its GLSL front end does NOT yet parse `#extension
//      GL_EXT_shader_integer_dot_product` or resolve `dotPacked4x8EXT()` —
//      confirmed with a minimal standalone reproducer, not an assumption.
//      A newer glslang build (or building glslang from Khronos source) is
//      needed before the dp4a shader source below can even be turned into
//      SPIR-V, let alone run on real dot-product-capable hardware. The
//      source is written and kept in this file on the (reasonable but
//      unverified) assumption that a toolchain which DOES parse the
//      extension will accept it as written; treat it as unbuilt, not just
//      unvalidated, until that's confirmed.
//
// Requires the Vulkan SDK loader + headers (vulkan/vulkan.h), a device
// exposing VK_SUBGROUP_FEATURE_ARITHMETIC_BIT in compute shaders (checked
// at construction — see query_subgroup_properties()), and glslang/glslc
// having compiled shaders/matvec_f32.comp, matvec_f32_scalar.comp,
// matvec_q4_0.comp, matvec_q8_0.comp, matmul_tiled_f32.comp, and (point 12)
// matmul_tiled_q4_0.comp/matmul_tiled_q8_0.comp into .spv siblings before
// this runs (NEON.py's Vulkan build step; see README's build line). The
// last two need TWO .spv outputs each — matmul_tiled_q4_0.spv (plain) and
// matmul_tiled_q4_0_dp4a.spv (same source, glslc -DHAS_INT8_DOT=1), same for
// q8_0 — create_pipelines() picks between them at runtime based on
// dot_product_supported(); only the pair that's actually needed on a given
// device has to exist on disk for that device to work, but building both up
// front is simplest. Only compiled in when USE_VULKAN is defined.
//
// VALIDATED (this pass): all four shaders compile cleanly with
// glslangValidator, and the full construct → upload_weight → matvec →
// compare-against-CPU-reference round trip for F32, Q4_0, and Q8_0 was run
// against Mesa lavapipe (a real, if software, Vulkan 1.3 device — subgroup
// size 8, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT present) across several
// matrix sizes including non-multiple-of-4 cols (exercising the scalar
// fallback pipeline) and realistic transformer dimensions, and matched the
// CPU dot_f32()/dot_q4_0_q8_0()/dot_q8_0_q8_0() reference closely (float
// accumulation order differs from the CPU path, so exact bit equality
// isn't the right bar — see the test harness this was checked against for
// the tolerance used). NOT yet run against real discrete/integrated GPU
// hardware or AMD's RADV/amdgpu Vulkan driver specifically — that remains
// the next validation step before any real routing decision.
//
// NOT YET VALIDATED (this pass): the begin_batch()/queue_*()/barrier()/
// end_batch() machinery itself. matvec_f32()/matvec_q4_0()/matvec_q8_0()
// (single op per batch) go through the exact same recording path the
// lavapipe round trip above already covers, so those should carry the same
// confidence — but no test here has queued 2+ independent ops into one
// batch, or exercised barrier(), or driven the descriptor-set-pool slot
// rotation (see kMaxOpsPerBatch) past a single slot. That multi-op path is
// new code and needs its own round trip against the CPU reference before
// it's trusted, the same way the single-op path was validated above.
//
// NOT YET VALIDATED (this pass, points 7-9 above): the timeline-semaphore
// double-buffered batch submission (begin_batch()/end_batch()/wait_batch()),
// the dedicated-transfer-queue staging-copy path and its queue-family
// ownership-transfer barriers (copy_staging_to_weight()), the suballocator
// (sub_alloc()/sub_free()), and the request-batching convenience wrappers
// (queue_matvec_*_requests()) are all new in this pass and have NOT been
// run against anything — no device was available to test any of it,
// timeline-semaphore or dedicated-transfer-queue support included. The
// suballocator's free-list logic was checked by hand-tracing a few
// alloc/free/coalesce sequences on paper, not by running it. Treat all
// four of these the same way the fp16 cooperative-matrix path below is
// treated: compiles-and-should-be-right, not validated, until they've
// actually run against a real Vulkan implementation — ideally starting
// with the same Mesa lavapipe device the rest of this file's VALIDATED
// section above was checked against, since lavapipe reports a single
// combined queue family (no dedicated transfer queue to exercise the
// ownership-transfer barrier pair) but does support Vulkan 1.3 and should
// exercise the timeline-semaphore and suballocator paths directly.
//
// ALSO NOT VALIDATED: the fp16 cooperative-matrix path (upload_weight_f16(),
// queue_matmul_f16(), matmul_coopmat_f16.comp). The shader itself compiles
// and validates to correct SPIR-V (glslangValidator + spirv-val — real
// OpCooperativeMatrixLoadKHR/MulAddKHR/StoreKHR with the right capability,
// scope, and use operands, confirmed by disassembly) and float_to_half()
// was checked against the compiler's native _Float16 conversion across 2M
// random values plus every edge case (subnormals, overflow, NaN, +-0) with
// zero mismatches — those two pieces are solid. What's NOT been run at all:
// the extension/feature-enabling code in create_device_and_queue(), the
// vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR shape query in
// query_coop_matrix_shape(), or the actual GEMM end-to-end against a CPU
// reference — because no coop-matrix-capable GPU (or even a software
// implementation of the extension) was available to test any of that
// against. Treat this path as a compiles-and-should-be-right starting
// point, not a validated one, until it's been run for real.
//
// VALIDATED (point 11, matmul_tiled_f32.comp / queue_matmul_f32()): this
// is the one matmul path in this file that's had a real end-to-end check
// — a standalone harness (not part of this file) ran the shader on Mesa
// lavapipe against a CPU double-precision GEMM reference across 7 shapes,
// including deliberately non-multiple-of-16 M/K/N (37x40x29, 5x33x17,
// 17x17x17) and M=1, all within float-accumulation tolerance. That
// validates the shader's math, not queue_matmul_f32()'s wiring of it into
// this class (slot pool, descriptor updates, batching) — that integration
// layer carries the same "compiles and should be right" status as
// everything else marked as such above, since the standalone harness
// intentionally bypassed this class to test the shader in isolation.
// =============================================================================

#if defined(USE_VULKAN)

#include "rawllm_common.hpp"
#include <vulkan/vulkan.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace vk_backend {

inline std::vector<uint32_t> read_spirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("rawllm_vulkan: couldn't open SPIR-V file: " + path +
        " (did the build step run glslangValidator/glslc on the matching .comp file?)");
    size_t size = (size_t)f.tellg();
    if (size % 4 != 0)
        throw std::runtime_error("rawllm_vulkan: SPIR-V file size not a multiple of 4: " + path);
    std::vector<uint32_t> buf(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)size);
    return buf;
}

inline void vk_check(VkResult r, const std::string& what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error("rawllm_vulkan: " + what + " failed (VkResult=" + std::to_string((int)r) + ")");
}

// Opaque handle to a weight tensor uploaded ONCE and kept resident on the
// GPU (device-local when the device exposes pure device-local memory,
// direct host-visible+device-local combined memory otherwise — e.g. every
// integrated GPU and lavapipe, which only expose the combined type; see
// find_device_local_memory()). Caller owns the lifetime: every GpuTensor
// returned by upload_weight_*() must be passed to free_tensor() before the
// VulkanMatvecBackend that created it is destroyed.
struct GpuTensor {
    VkBuffer buf = VK_NULL_HANDLE;
    // These four mirror VulkanMatvecBackend's private SubAlloc exactly
    // (mem/offset/size/block) — duplicated here as plain fields, not the
    // nested type itself, only because SubAlloc is private and GpuTensor
    // is a public, class-external return type. free_tensor() reconstructs
    // a SubAlloc from these to call sub_free() (see that function) — mem
    // is a shared block's handle, NOT a dedicated per-tensor allocation,
    // so it must go through sub_free(), never vkFreeMemory() directly.
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    size_t block_index = SIZE_MAX;
    // The actual suballocated range's size (== the SubAlloc's `size`,
    // which is req.size from vkGetBufferMemoryRequirements — i.e. the
    // padded/aligned allocation, NOT the logical `size` above). Needed
    // so free_tensor() can hand back to sub_free() exactly the range
    // sub_alloc() carved out; using the logical `size` there would
    // return a smaller range than was actually reserved and leak the
    // tail of every freed tensor's allocation.
    VkDeviceSize alloc_size = 0;
};

// Experimental single-device, single-queue Vulkan compute backend for the
// F32/Q4_0/Q8_0 matvec kernels. One instance owns its own VkInstance/
// VkDevice plus all persistent pipeline/descriptor/command-buffer state —
// construct once, call upload_weight_*() once per tensor at load time, then
// matvec_*() repeatedly per token (mirrors util::ThreadPool's lifecycle for
// the CPU path).
class VulkanMatvecBackend {
public:
    // shaders_dir must contain matvec_f32.spv, matvec_f32_scalar.spv,
    // matvec_q4_0.spv, and matvec_q8_0.spv (compiled from this repo's
    // shaders/*.comp of the same names).
    //
    // max_ops_per_batch bounds how many independent dispatches a single
    // begin_batch()/end_batch() span may queue, PER layout family (3-
    // buffer F32, 5-buffer quantized), PER double-buffer slot — see
    // max_ops_per_batch_'s and kFramesInFlight's comments. The default
    // of 8 covers one request's Q+K+V+attn_out+gate+up+down (7) with a
    // little headroom; raise it if you're also using this batch to cover
    // N concurrent requests through the same weight (see
    // queue_matvec_f32_requests() and this file's header comment,
    // point 9) — e.g. 7 (one request's ops) + 15 (15 MORE concurrent
    // requests' worth of just their Q projection, say) needs at least 22.
    // pipeline_cache_path: where create_pipeline_cache() looks for a
    // previously saved VkPipelineCache blob to seed this run with, and
    // where the destructor writes the (possibly grown) cache back out —
    // see create_pipeline_cache()/save_pipeline_cache_to_disk(). Pass ""
    // to disable persistence entirely (in-memory cache only, same
    // behavior as before this existed). A missing, empty, or
    // driver-rejected file is not an error — see create_pipeline_cache()'s
    // comment — so it's safe to point this at a path that doesn't exist
    // yet on the very first run.
    explicit VulkanMatvecBackend(const std::string& shaders_dir = "shaders", uint32_t max_ops_per_batch = 8,
                                  const std::string& pipeline_cache_path = "shaders/vk_pipeline_cache.bin")
        : pipeline_cache_path_(pipeline_cache_path), max_ops_per_batch_(max_ops_per_batch) {
        create_instance();
        pick_physical_device();
        find_transfer_queue_family();
        query_subgroup_properties();
        create_device_and_queue();
        query_coop_matrix_shape();
        create_pipeline_cache();
        create_descriptor_layouts();
        create_pipelines(shaders_dir);
        create_descriptor_pool_and_sets();
        create_command_resources();
    }

    ~VulkanMatvecBackend() {
        if (device_ == VK_NULL_HANDLE) { if (instance_) vkDestroyInstance(instance_, nullptr); return; }
        vkDeviceWaitIdle(device_);
        save_pipeline_cache_to_disk();
        for (uint32_t f = 0; f < kFramesInFlight; ++f) {
            for (auto& s : slots3_[f]) { destroy_buffer(s.x); destroy_buffer(s.y); }
            for (auto& s : slots5_[f]) { destroy_buffer(s.xq); destroy_buffer(s.xd); destroy_buffer(s.xsum); destroy_buffer(s.y); }
        }
        destroy_buffer(staging_buf_);
        // Every GpuBuffer above has now had its VkBuffer destroyed
        // (destroy_buffer() also returns its range to the suballocator's
        // free-list via sub_free()) — safe to free the underlying
        // VkDeviceMemory blocks in bulk now. Caller-owned GpuTensor
        // weights (from upload_weight_*()) are NOT freed here — see
        // GpuTensor's/free_tensor()'s doc comment; the caller must have
        // freed every one of those before destroying this backend, same
        // contract as before suballocation existed.
        destroy_all_blocks();
        if (timeline_sem_) vkDestroySemaphore(device_, timeline_sem_, nullptr);
        if (transfer_cmd_pool_) vkDestroyCommandPool(device_, transfer_cmd_pool_, nullptr);
        if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
        if (pipeline_f32_) vkDestroyPipeline(device_, pipeline_f32_, nullptr);
        if (pipeline_f32_scalar_) vkDestroyPipeline(device_, pipeline_f32_scalar_, nullptr);
        if (pipeline_q4_0_) vkDestroyPipeline(device_, pipeline_q4_0_, nullptr);
        if (pipeline_q8_0_) vkDestroyPipeline(device_, pipeline_q8_0_, nullptr);
        if (pipeline_matmul_f16_) vkDestroyPipeline(device_, pipeline_matmul_f16_, nullptr);
        if (pipeline_matmul_f32_) vkDestroyPipeline(device_, pipeline_matmul_f32_, nullptr);
        if (pipeline_matmul_q4_0_) vkDestroyPipeline(device_, pipeline_matmul_q4_0_, nullptr);
        if (pipeline_matmul_q8_0_) vkDestroyPipeline(device_, pipeline_matmul_q8_0_, nullptr);
        if (pipeline_layout3_) vkDestroyPipelineLayout(device_, pipeline_layout3_, nullptr);
        if (pipeline_layout5_) vkDestroyPipelineLayout(device_, pipeline_layout5_, nullptr);
        if (pipeline_layout_coopmat_) vkDestroyPipelineLayout(device_, pipeline_layout_coopmat_, nullptr);
        if (pipeline_layout5_mm_) vkDestroyPipelineLayout(device_, pipeline_layout5_mm_, nullptr);
        if (desc_set_layout3_) vkDestroyDescriptorSetLayout(device_, desc_set_layout3_, nullptr);
        if (desc_set_layout5_) vkDestroyDescriptorSetLayout(device_, desc_set_layout5_, nullptr);
        if (pipeline_cache_) vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
        vkDestroyDevice(device_, nullptr);
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    VulkanMatvecBackend(const VulkanMatvecBackend&) = delete;
    VulkanMatvecBackend& operator=(const VulkanMatvecBackend&) = delete;

    // ── weight residency ────────────────────────────────────────────────
    // Upload a row-major F32 weight matrix (rows*cols floats) once; keep
    // calling matvec_f32() against the returned handle for as long as the
    // tensor is needed. Caller must free_tensor() it when done.
    GpuTensor upload_weight_f32(const float* W, size_t rows, size_t cols) {
        return upload_weight_bytes(W, (VkDeviceSize)rows * cols * sizeof(float));
    }

    // raw_blocks: rows*(cols/32)*18 bytes of on-disk Q4_0 block data
    // (2-byte fp16 scale + 16 packed-nibble bytes per 32-element block),
    // uploaded byte-for-byte — see shaders/matvec_q4_0.comp's header for
    // the exact layout the shader expects (matmul_tiled_q4_0.comp, the
    // GEMM path added by point 12, reads this same layout).
    GpuTensor upload_weight_q4_0(const uint8_t* raw_blocks, size_t rows, size_t cols) {
        if (cols % 32 != 0) throw std::runtime_error("rawllm_vulkan: Q4_0 weight cols must be a multiple of 32");
        return upload_weight_bytes(raw_blocks, (VkDeviceSize)rows * (cols / 32) * 18);
    }

    // raw_blocks: rows*(cols/32)*34 bytes of on-disk Q8_0 block data
    // (2-byte fp16 scale + 32 signed int8 values per block). Also the
    // layout matmul_tiled_q8_0.comp (point 12's GEMM path) expects.
    GpuTensor upload_weight_q8_0(const uint8_t* raw_blocks, size_t rows, size_t cols) {
        if (cols % 32 != 0) throw std::runtime_error("rawllm_vulkan: Q8_0 weight cols must be a multiple of 32");
        return upload_weight_bytes(raw_blocks, (VkDeviceSize)rows * (cols / 32) * 34);
    }

    void free_tensor(GpuTensor& t) {
        if (t.buf) vkDestroyBuffer(device_, t.buf, nullptr);
        if (t.mem) {
            // t.mem is a shared block's handle (see the GpuTensor comment
            // above), not a dedicated per-tensor allocation, so it must
            // go through sub_free() to return the range to that block's
            // free-list — never vkFreeMemory() directly, which would
            // free the whole block out from under every other tensor or
            // scratch buffer still suballocated from it, and then double-
            // free it again later when destroy_all_blocks() runs.
            SubAlloc a{t.mem, t.offset, t.alloc_size, t.block_index};
            sub_free(a);
        }
        t = GpuTensor{};
    }

    // Returned by end_batch(); pass to wait_batch() once you actually
    // need this batch's y pointers to contain real data — see
    // end_batch()'s comment. Copyable value type, no special lifetime of
    // its own (just an index + a target timeline value); waiting on a
    // stale/already-waited handle is a harmless no-op (sync_slot() below
    // treats last_signal_[slot]==0 as "nothing outstanding").
    struct BatchHandle { uint32_t slot; uint64_t signal_value; };

    // ── batched recording ───────────────────────────────────────────────
    // Typical use for one layer's Q/K/V (independent) followed by
    // attn_out (depends on all three):
    //
    //   backend.begin_batch();
    //   backend.queue_matvec_f32(Wq, x, yq, d_model, d_model);
    //   backend.queue_matvec_f32(Wk, x, yk, d_model, d_kv);
    //   backend.queue_matvec_f32(Wv, x, yv, d_model, d_kv);
    //   auto h = backend.end_batch();     // ONE submit for all 3, async — does NOT wait
    //   // ... prepare the NEXT token's/request's activations here on the
    //   //     CPU while the GPU is still chewing on the batch above ...
    //   backend.wait_batch(h);            // only NOW must yq/yk/yv be real
    //   // ... attention math on yq/yk/yv (CPU-side today) ...
    //   backend.matvec_f32(Wo, attn_out, y, d_model, d_model);
    //
    // begin_batch() resets ONE of the two double-buffered persistent
    // command buffers (see kFramesInFlight) and starts recording; each
    // queue_matvec_*() call below records one dispatch into it
    // (uploading its activation input immediately, but NOT submitting or
    // waiting); end_batch() ends recording, submits once, and returns a
    // BatchHandle WITHOUT waiting — see end_batch()'s own comment for why
    // that's the whole point of this pass's timeline-semaphore change
    // (this file's header comment, point 7). Ops queued with no
    // barrier() between them are assumed independent (disjoint
    // inputs/outputs) and may run in either order or overlap on the GPU;
    // call barrier() before queueing an op that reads another queued
    // op's output.
    //
    // Slot budget: each pipeline-layout family (the 3-buffer F32 layout,
    // the 5-buffer quantized layout), PER double-buffer slot, gets
    // max_ops_per_batch_ independent scratch-buffer-plus-descriptor-set
    // slots, cycled through in queue order and reset at every
    // begin_batch(). This exists because reusing ONE descriptor set
    // across multiple queued-but-not-yet-submitted dispatches would have
    // every dispatch in the batch execute against whatever that set was
    // LAST updated to point at — Vulkan does not snapshot descriptor
    // contents at vkCmdBindDescriptorSets time. Queueing more than
    // max_ops_per_batch_ ops of the same layout in one batch throws;
    // raise the constructor parameter if a real workload needs more.
    //
    // Double-buffering and the wait this still needs: begin_batch() picks
    // the OTHER slot from whichever end_batch() last used, so back-to-
    // back begin_batch()/end_batch() pairs (no explicit wait_batch() in
    // between) naturally alternate slots and never block on each other —
    // that's what lets the CPU get a full batch's worth of recording+
    // upload done while the PREVIOUS batch is still executing. It's only
    // once a THIRD begin_batch() comes around to reuse the FIRST slot
    // again that begin_batch() has to actually wait (via sync_slot()) —
    // and only if that slot's batch hasn't finished on the GPU yet by
    // then. This gives one full batch of overlap depth, same as a
    // typical double-buffered renderer.
    void begin_batch() {
        uint32_t slot = next_frame_ % kFramesInFlight;
        next_frame_++;
        // Defensive: about to reset this slot's command buffer and reuse
        // its descriptor sets/scratch buffers, which is only safe once
        // the GPU is actually done with whatever THIS slot last
        // submitted. If the caller already called wait_batch() on that
        // submission's handle, this is a no-op (last_signal_[slot] is
        // already 0); if not, this blocks here instead of corrupting a
        // still-in-flight dispatch's inputs.
        sync_slot(slot);
        cur_ = slot;
        batch_slot3_ = 0;
        batch_slot5_ = 0;
        pending_downloads_.clear();
        vk_check(vkResetCommandBuffer(cmd_[cur_], 0), "vkResetCommandBuffer(batch)");
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(cmd_[cur_], &cbbi), "vkBeginCommandBuffer(batch)");
        batch_open_ = true;
    }

    // Insert a compute-to-compute memory barrier: every dispatch queued
    // AFTER this call will wait for every write from dispatches queued
    // BEFORE it (in this batch) to finish and become visible. Needed
    // between e.g. a "gate"/"up" pair and a "down" that reads a
    // GPU-resident combination of their outputs; NOT needed between
    // "gate" and "up" themselves (independent — no barrier between them
    // lets the GPU schedule them concurrently instead of serializing).
    void barrier() {
        if (!batch_open_) throw std::runtime_error("rawllm_vulkan: barrier() called outside begin_batch()/end_batch()");
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_[cur_], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // y[r] = dot(W[r*cols:(r+1)*cols], x[0:cols]) for r in [0, rows).
    // Selects the vec4-vectorized pipeline when cols % 4 == 0 (always true
    // for realistic transformer hidden sizes) and the scalar fallback
    // otherwise. x is uploaded immediately; y is only valid after the
    // enclosing end_batch()'s BatchHandle has been passed to wait_batch().
    // W must be a handle from upload_weight_f32().
    void queue_matvec_f32(const GpuTensor& W, const float* x, float* y, uint32_t cols, uint32_t rows) {
        if (!batch_open_) throw std::runtime_error("rawllm_vulkan: queue_matvec_f32() called outside begin_batch()/end_batch()");
        bool vec4_ok = (cols % 4 == 0);
        Layout3Slot& slot = next_slot3();
        ensure_scratch(slot.x, (VkDeviceSize)cols * sizeof(float));
        ensure_scratch(slot.y, (VkDeviceSize)rows * sizeof(float));
        upload(slot.x, x, (VkDeviceSize)cols * sizeof(float));

        VkPipeline pipe = vec4_ok ? pipeline_f32_ : pipeline_f32_scalar_;
        update_descriptor_set3(slot.set, W.buf, slot.x.buf, slot.y.buf);
        record_dispatch(pipe, pipeline_layout3_, slot.set, cols, rows);
        pending_downloads_.push_back({&slot.y, y, (VkDeviceSize)rows * sizeof(float)});
    }

    // ── batching across concurrent requests ─────────────────────────────
    // See this file's header comment (point 9). Queues `count`
    // independent GEMV dispatches into the CURRENT batch — one per
    // request — all reading the SAME resident weight W but each its own
    // x/y: this is the "N concurrent requests, one shared projection"
    // axis, orthogonal to (and freely combinable with, subject to the
    // max_ops_per_batch_ budget) the "one request's independent
    // projections" batching queue_matvec_f32() already supported. xs[i]/
    // ys[i] follow the exact same contract as queue_matvec_f32()'s x/y.
    void queue_matvec_f32_requests(const GpuTensor& W, const float* const* xs, float* const* ys,
                                    uint32_t cols, uint32_t rows, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) queue_matvec_f32(W, xs[i], ys[i], cols, rows);
    }

    // x_d: activation per-block scales (cols/32 floats). x_sum: per-block
    // sum of x_q (cols/32 ints) — the same hoisted-out-of-the-row-loop
    // value rawllm_simd_dispatch.hpp's dot_q4_0_q8_0() takes, computed once
    // per token by quantize_rows_q8_0() on the CPU side, not recomputed
    // here. x_q: quantized activations (cols int8s, widened to int32 for
    // upload — see the widening loop below; this is a cols-sized, not
    // rows*cols-sized, cost). W must be a handle from upload_weight_q4_0().
    // y is only valid after the enclosing end_batch()'s BatchHandle has
    // been passed to wait_batch().
    void queue_matvec_q4_0(const GpuTensor& W, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, float* y, uint32_t cols, uint32_t rows) {
        queue_matvec_quantized(pipeline_q4_0_, W, x_d, x_sum, x_q, y, cols, rows);
    }

    // Request-batched form of queue_matvec_q4_0() — see
    // queue_matvec_f32_requests()'s comment; same "N requests through the
    // SAME weight" axis, per-request arrays follow queue_matvec_q4_0()'s
    // per-array contracts exactly.
    void queue_matvec_q4_0_requests(const GpuTensor& W, const float* const* xs_d, const int32_t* const* xs_sum,
                                     const int8_t* const* xs_q, float* const* ys, uint32_t cols, uint32_t rows, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) queue_matvec_q4_0(W, xs_d[i], xs_sum[i], xs_q[i], ys[i], cols, rows);
    }

    // Same activation-side contract as queue_matvec_q4_0(); x_sum is
    // accepted for API symmetry (and because the CPU reference
    // simd::dot_q8_0_q8_0() takes the same unused parameter) but ignored —
    // Q8_0 has no zero-point to correct for.
    void queue_matvec_q8_0(const GpuTensor& W, const float* x_d, const int32_t* x_sum,
                            const int8_t* x_q, float* y, uint32_t cols, uint32_t rows) {
        queue_matvec_quantized(pipeline_q8_0_, W, x_d, x_sum, x_q, y, cols, rows);
    }

    // Request-batched form of queue_matvec_q8_0() — see
    // queue_matvec_f32_requests()'s comment.
    void queue_matvec_q8_0_requests(const GpuTensor& W, const float* const* xs_d, const int32_t* const* xs_sum,
                                     const int8_t* const* xs_q, float* const* ys, uint32_t cols, uint32_t rows, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) queue_matvec_q8_0(W, xs_d[i], xs_sum[i], xs_q[i], ys[i], cols, rows);
    }

    // Ends recording and submits the whole batch ONCE with a timeline-
    // semaphore signal, then returns IMMEDIATELY — no wait, no download.
    // This is the change described in this file's header comment (point
    // 7): the old version of this function did a synchronous submit+
    // wait+download right here, which meant the CPU could not do
    // anything else (including preparing the NEXT batch) until the GPU
    // finished this one. Now: call end_batch(), keep the returned
    // BatchHandle, do whatever CPU-side work doesn't need this batch's
    // outputs (quantizing the next token's activations is the motivating
    // case), and only call wait_batch() once something actually needs to
    // read from a y pointer this batch wrote. If you never call
    // wait_batch() yourself, the NEXT begin_batch() that reuses this same
    // double-buffer slot will do it for you (see begin_batch()'s
    // comment) — correctness doesn't depend on the caller remembering,
    // only the AMOUNT of overlap achieved does.
    BatchHandle end_batch() {
        if (!batch_open_) throw std::runtime_error("rawllm_vulkan: end_batch() called without a matching begin_batch()");
        vk_check(vkEndCommandBuffer(cmd_[cur_]), "vkEndCommandBuffer(batch)");
        uint64_t signal_val = ++timeline_counter_;
        VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        tsi.signalSemaphoreValueCount = 1; tsi.pSignalSemaphoreValues = &signal_val;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &tsi;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd_[cur_];
        si.signalSemaphoreCount = 1; si.pSignalSemaphores = &timeline_sem_;
        vk_check(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit(batch)");
        last_signal_[cur_] = signal_val;
        batch_downloads_[cur_] = std::move(pending_downloads_);
        pending_downloads_.clear();
        batch_open_ = false;
        return BatchHandle{cur_, signal_val};
    }

    // Blocks until the batch identified by `h` has finished on the GPU
    // (no-op if it already has, or if this handle's slot was already
    // synced by a later begin_batch()/wait_batch() call — see
    // sync_slot()), then copies every one of that batch's queued outputs
    // back to the host pointers its queue_matvec_*() calls were given.
    // Call this once you actually need those y pointers to be valid —
    // e.g. right before the CPU-side attention/elementwise math that
    // reads them.
    void wait_batch(const BatchHandle& h) { sync_slot(h.slot); }

    // ── matvec (single-op convenience wrappers) ────────────────────────
    // Each is exactly begin_batch() + one queue_matvec_*() + end_batch()
    // + wait_batch() — fully synchronous, kept for callers that just want
    // one matvec with results immediately available and no batching.
    // Prefer the queue_matvec_*()/barrier()/end_batch()/wait_batch() form
    // directly when issuing several matvecs that don't all depend on
    // each other (e.g. Q/K/V, or gate+up, or N concurrent requests
    // through the same weight via queue_matvec_f32_requests()): batching
    // them cuts N submit round trips down to 1, and deferring wait_batch()
    // lets the CPU overlap its own work with this batch's GPU time.
    void matvec_f32(const GpuTensor& W, const float* x, float* y, uint32_t cols, uint32_t rows) {
        begin_batch();
        queue_matvec_f32(W, x, y, cols, rows);
        wait_batch(end_batch());
    }

    void matvec_q4_0(const GpuTensor& W, const float* x_d, const int32_t* x_sum,
                      const int8_t* x_q, float* y, uint32_t cols, uint32_t rows) {
        begin_batch();
        queue_matvec_q4_0(W, x_d, x_sum, x_q, y, cols, rows);
        wait_batch(end_batch());
    }

    void matvec_q8_0(const GpuTensor& W, const float* x_d, const int32_t* x_sum,
                      const int8_t* x_q, float* y, uint32_t cols, uint32_t rows) {
        begin_batch();
        queue_matvec_q8_0(W, x_d, x_sum, x_q, y, cols, rows);
        wait_batch(end_batch());
    }

    // ── fp16 cooperative-matrix matmul (matrix-core GEMM) ──────────────
    // True iff VK_KHR_cooperative_matrix is present AND the device reports
    // a 16x16x16 fp16xfp16->f32 subgroup-scope shape (checked once at
    // construction — see query_coop_matrix_shape()). Check this before
    // calling queue_matmul_f16() or upload_weight_f16(); every other
    // function in this class works the same regardless of what this
    // returns.
    bool coop_matrix_supported() const { return coop_matrix_supported_; }

    // True iff VK_KHR_shader_integer_dot_product is present with 8-bit
    // signed-signed-accumulate support (checked once at construction —
    // see create_device_and_queue()). Purely informational for callers;
    // create_pipelines() already uses this to pick between the two
    // matmul_tiled_q4_0/q8_0 SPIR-V variants (see this file's header
    // comment, point 12), so nothing outside this class needs to branch
    // on it — queue_matmul_q4_0()/queue_matmul_q8_0() work identically
    // either way.
    bool dot_product_supported() const { return dot_product_supported_; }

    // One-time resident upload of a K x N row-major weight matrix,
    // converted to fp16 on the host (this backend has no on-disk fp16
    // format of its own; callers hold F32 or quantized weights, so
    // conversion happens here). Same device-local-if-possible /
    // staging-buffer-if-not path as upload_weight_f32(). Only call this
    // if coop_matrix_supported() is true.
    GpuTensor upload_weight_f16(const float* W, size_t rows_k, size_t cols_n) {
        std::vector<uint16_t> half_data(rows_k * cols_n);
        for (size_t i = 0; i < half_data.size(); ++i) half_data[i] = float_to_half(W[i]);
        return upload_weight_bytes(half_data.data(), (VkDeviceSize)half_data.size() * sizeof(uint16_t));
    }

    // C[M x N] = A[M x K] * B[K x N] where B is a resident fp16 tensor from
    // upload_weight_f16() and A is a per-call activation matrix converted
    // to fp16 here. K and N must be multiples of 16 (transformer hidden
    // sizes always are; throws otherwise). M does NOT need to be a
    // multiple of 16 — it's padded internally with zero rows and only the
    // true M*N outputs are downloaded, so callers can pass a real,
    // possibly-odd prefill batch size directly.
    //
    // THIS IS A GEMM, NOT A GEMV: it exists for prefill (M > 1 — many
    // activation rows at once, enough to fill a 16x16 matrix-core tile).
    // At M == 1 (single-token decode) this computes 16 rows of output to
    // get 1 useful one — use queue_matvec_f32()/queue_matvec_q4_0()/
    // queue_matvec_q8_0() for that case instead. This function does not
    // make that choice for you; see this file's header comment (point 10).
    void queue_matmul_f16(const GpuTensor& Wf16, const float* X, float* Y, uint32_t M, uint32_t K, uint32_t N) {
        if (!batch_open_) throw std::runtime_error("rawllm_vulkan: queue_matmul_f16() called outside begin_batch()/end_batch()");
        if (!coop_matrix_supported_)
            throw std::runtime_error("rawllm_vulkan: queue_matmul_f16() called but this device doesn't support "
                "VK_KHR_cooperative_matrix with a 16x16x16 fp16xfp16->f32 subgroup shape "
                "— check coop_matrix_supported() first");
        if (K % kCoopTile != 0 || N % kCoopTile != 0)
            throw std::runtime_error("rawllm_vulkan: queue_matmul_f16 requires K and N to be multiples of " +
                std::to_string(kCoopTile) + " (got K=" + std::to_string(K) + ", N=" + std::to_string(N) + ")");

        uint32_t Mp = ((M + kCoopTile - 1u) / kCoopTile) * kCoopTile;
        Layout3Slot& slot = next_slot3();

        slot.f16_scratch.resize((size_t)Mp * K);
        for (size_t r = 0; r < M; ++r)
            for (size_t c = 0; c < K; ++c)
                slot.f16_scratch[r * K + c] = float_to_half(X[r * K + c]);
        for (size_t r = M; r < Mp; ++r)
            for (size_t c = 0; c < K; ++c)
                slot.f16_scratch[r * K + c] = 0; // padding rows: zero input, output discarded, never downloaded

        ensure_scratch(slot.x, (VkDeviceSize)Mp * K * sizeof(uint16_t));
        ensure_scratch(slot.y, (VkDeviceSize)Mp * N * sizeof(float));
        upload(slot.x, slot.f16_scratch.data(), (VkDeviceSize)Mp * K * sizeof(uint16_t));

        update_descriptor_set3(slot.set, Wf16.buf, slot.x.buf, slot.y.buf);
        record_dispatch_matmul(pipeline_matmul_f16_, pipeline_layout_coopmat_, slot.set, Mp, K, N, kCoopTile);
        // Row-major C means the true M rows are the first M*N floats,
        // contiguous, regardless of the Mp padding above — downloading
        // exactly M*N floats from the front of the (possibly larger)
        // buffer is correct with no separate trim step.
        pending_downloads_.push_back({&slot.y, Y, (VkDeviceSize)M * N * sizeof(float)});
    }

    // C[M x N] = A[M x K] * B[K x N], all F32, using a plain shared-memory
    // tiled GEMM (matmul_tiled_f32.comp) — no VK_KHR_cooperative_matrix,
    // no device-capability check, works on literally any device this
    // backend's other shaders already run on. Prefer this over
    // queue_matmul_f16() when coop_matrix_supported() is false (most
    // hardware: integrated GPUs, older discrete cards, lavapipe) or when
    // avoiding the fp16 precision drop matters more than matrix-core
    // throughput.
    //
    // Unlike queue_matmul_f16(), M/K/N do NOT need to be multiples of any
    // tile size — matmul_tiled_f32.comp bounds-checks every load/store, so
    // real unpadded prefill dimensions (including M==1, though see the
    // note below) go in directly with no host-side padding step.
    //
    // STILL A GEMM, NOT A GEMV: at M==1 this still launches a full 16x16
    // tile of threads to produce 1 useful row of output — cheaper waste
    // than queue_matmul_f16()'s M==1 case (idle threads, not an
    // underused matrix-core op), but the same lesson applies: use
    // queue_matvec_f32() for single-token decode, this for prefill. This
    // function does not make that choice for you.
    //
    // VALIDATED: unlike every other matmul path in this file, this one
    // has actually been run — matmul_tiled_f32.comp end-to-end against a
    // CPU double-precision reference GEMM on Mesa lavapipe, across 7
    // shapes including deliberately ragged (non-multiple-of-16) M/K/N
    // (37x40x29, 5x33x17, 17x17x17) and the M=1 edge case, all passing
    // within float-accumulation-order tolerance. That test harness isn't
    // part of this file (it built its own minimal Vulkan setup rather
    // than going through this class, specifically to test the shader in
    // isolation); rebuild it against this class's actual plumbing before
    // relying on this for anything real — a standalone harness proves the
    // shader's math, not that queue_matmul_f32() wires it up correctly.
    void queue_matmul_f32(const GpuTensor& W, const float* X, float* Y, uint32_t M, uint32_t K, uint32_t N) {
        if (!batch_open_) throw std::runtime_error("rawllm_vulkan: queue_matmul_f32() called outside begin_batch()/end_batch()");
        Layout3Slot& slot = next_slot3();

        ensure_scratch(slot.x, (VkDeviceSize)M * K * sizeof(float));
        ensure_scratch(slot.y, (VkDeviceSize)M * N * sizeof(float));
        upload(slot.x, X, (VkDeviceSize)M * K * sizeof(float));

        update_descriptor_set3(slot.set, W.buf, slot.x.buf, slot.y.buf);
        record_dispatch_matmul(pipeline_matmul_f32_, pipeline_layout_coopmat_, slot.set, M, K, N, kTileF32);
        pending_downloads_.push_back({&slot.y, Y, (VkDeviceSize)M * N * sizeof(float)});
    }

    // ── quantized batched GEMM (point 12) ──────────────────────────────
    // C[M x N] = A[M x K] * B[K x N] where B is a resident Q4_0/Q8_0
    // tensor from upload_weight_q4_0()/_q8_0() and A is M rows of
    // ALREADY-QUANTIZED activations — xs_q/xs_d/xs_sum are exactly M
    // copies of queue_matvec_q4_0()'s/queue_matvec_q8_0()'s per-row
    // x_q/x_d/x_sum arguments laid out back-to-back (xs_q: M*K int8,
    // row-major; xs_d/xs_sum: M*(K/32), row-major) — this file does not
    // quantize activations itself, same contract as the GEMV path above.
    // K must be a multiple of 32 (one Q4_0/Q8_0 block); N/M have no
    // alignment requirement — matmul_tiled_q4_0.comp/matmul_tiled_q8_0.
    // comp bounds-check the ragged edge the same way matmul_tiled_f32.
    // comp does (see queue_matmul_f32()'s comment).
    //
    // THIS IS A GEMM, NOT A GEMV: exists for prefill (M>1); for M==1
    // (single-token decode) use queue_matvec_q4_0()/queue_matvec_q8_0()
    // instead, same reasoning as queue_matmul_f16()/queue_matmul_f32()
    // above. This function does not make that choice for you.
    //
    // NOT validated end-to-end (new in this pass) — matmul_tiled_f32.comp
    // had a standalone lavapipe harness behind it before its wiring was
    // trusted (see queue_matmul_f32()'s comment); these two shaders have
    // not yet had the equivalent treatment. Treat as compiles-and-should-
    // be-right, not validated, until run against a CPU quantized-GEMM
    // reference the way the F32 path was.
    void queue_matmul_q4_0(const GpuTensor& W, const float* xs_d, const int32_t* xs_sum,
                            const int8_t* xs_q, float* Y, uint32_t M, uint32_t K, uint32_t N) {
        queue_matmul_quantized(pipeline_matmul_q4_0_, W, xs_d, xs_sum, xs_q, Y, M, K, N);
    }

    // Same contract as queue_matmul_q4_0(); xs_sum accepted for API
    // symmetry with the GEMV path (queue_matvec_q8_0()) but ignored —
    // Q8_0 has no zero-point to correct for.
    void queue_matmul_q8_0(const GpuTensor& W, const float* xs_d, const int32_t* xs_sum,
                            const int8_t* xs_q, float* Y, uint32_t M, uint32_t K, uint32_t N) {
        queue_matmul_quantized(pipeline_matmul_q8_0_, W, xs_d, xs_sum, xs_q, Y, M, K, N);
    }

private:
    // Shared plumbing for queue_matmul_q4_0()/queue_matmul_q8_0() — same
    // relationship as queue_matvec_quantized() has to queue_matvec_q4_0()/
    // queue_matvec_q8_0() above, just M rows at a time instead of 1 and
    // recorded through record_dispatch_matmul() instead of
    // record_dispatch(). Reuses Layout5Slot (W/XQ/XD/XSUM/Y — same 5
    // storage buffers as the GEMV quantized path) since the binding
    // shape is identical; only the push-constant range (M,K,N via
    // pipeline_layout5_mm_ instead of cols,rows via pipeline_layout5_)
    // and the dispatch grid differ.
    void queue_matmul_quantized(VkPipeline pipe, const GpuTensor& W, const float* xs_d, const int32_t* xs_sum,
                                 const int8_t* xs_q, float* Y, uint32_t M, uint32_t K, uint32_t N) {
        if (!batch_open_) throw std::runtime_error("rawllm_vulkan: queue_matmul_quantized() called outside begin_batch()/end_batch()");
        if (K % 32 != 0) throw std::runtime_error("rawllm_vulkan: quantized matmul K must be a multiple of 32");
        uint32_t nb = K / 32u;
        Layout5Slot& slot = next_slot5();

        // Same int8->int32 widening as queue_matvec_quantized() (shaders
        // declare X_Q as int[], no 8-bit storage extension assumed) —
        // just M*K elements instead of K.
        slot.xq_widen.resize((size_t)M * K);
        for (size_t i = 0; i < (size_t)M * K; ++i) slot.xq_widen[i] = (int32_t)xs_q[i];

        ensure_scratch(slot.xq, (VkDeviceSize)M * K * sizeof(int32_t));
        ensure_scratch(slot.xd, (VkDeviceSize)M * nb * sizeof(float));
        ensure_scratch(slot.xsum, (VkDeviceSize)M * nb * sizeof(int32_t));
        ensure_scratch(slot.y, (VkDeviceSize)M * N * sizeof(float));

        upload(slot.xq, slot.xq_widen.data(), (VkDeviceSize)M * K * sizeof(int32_t));
        upload(slot.xd, xs_d, (VkDeviceSize)M * nb * sizeof(float));
        upload(slot.xsum, xs_sum, (VkDeviceSize)M * nb * sizeof(int32_t));

        update_descriptor_set5(slot.set, W.buf, slot.xq.buf, slot.xd.buf, slot.xsum.buf, slot.y.buf);
        record_dispatch_matmul(pipe, pipeline_layout5_mm_, slot.set, M, K, N, kTileQuant);
        pending_downloads_.push_back({&slot.y, Y, (VkDeviceSize)M * N * sizeof(float)});
    }

    // One suballocation's bookkeeping: which shared VkDeviceMemory block
    // (by index into mem_blocks_), what offset/size within it. block ==
    // SIZE_MAX means "unallocated" (a fresh GpuBuffer/GpuTensor before
    // its first make_buffer()/upload_weight_bytes() call). See sub_alloc()/
    // sub_free() and this file's header comment (point 8).
    struct SubAlloc { VkDeviceMemory mem = VK_NULL_HANDLE; VkDeviceSize offset = 0; VkDeviceSize size = 0; size_t block = SIZE_MAX; };
    struct GpuBuffer { VkBuffer buf = VK_NULL_HANDLE; SubAlloc alloc; VkDeviceSize size = 0; bool host_visible = true; };

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    uint32_t subgroup_size_ = 0;
    uint32_t rows_per_wg_ = 0;

    // Dedicated transfer-only queue (VK_QUEUE_TRANSFER_BIT set, neither
    // GRAPHICS nor COMPUTE set), if this physical device exposes one —
    // see pick_physical_device()'s find_transfer_queue_family() and this
    // file's header comment (point 7). Falls back to queue_/queue_family_
    // (transfer_queue_family_ == queue_family_, has_dedicated_transfer_
    // queue_ == false) on every device that only exposes one combined
    // queue family, which includes every integrated GPU and lavapipe —
    // i.e. every device this file has actually been tested against.
    VkQueue transfer_queue_ = VK_NULL_HANDLE;
    uint32_t transfer_queue_family_ = 0;
    bool has_dedicated_transfer_queue_ = false;
    VkCommandPool transfer_cmd_pool_ = VK_NULL_HANDLE; // VK_NULL_HANDLE unless has_dedicated_transfer_queue_

    // Signaled by every end_batch() submission to a strictly increasing
    // value (timeline_counter_, pre-incremented); begin_batch()'s
    // defensive sync_slot() and the public wait_batch() both wait on
    // this instead of the old per-batch vkQueueWaitIdle(). See this
    // file's header comment (point 7) and begin_batch()/end_batch().
    VkSemaphore timeline_sem_ = VK_NULL_HANDLE;
    uint64_t timeline_counter_ = 0;

    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    std::string pipeline_cache_path_; // see constructor's doc comment, create_pipeline_cache(), save_pipeline_cache_to_disk()
    VkDescriptorSetLayout desc_set_layout3_ = VK_NULL_HANDLE; // W, X, Y            (f32 pipelines)
    VkDescriptorSetLayout desc_set_layout5_ = VK_NULL_HANDLE; // W, XQ, XD, XSUM, Y (quant pipelines)
    VkPipelineLayout pipeline_layout3_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout5_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_coopmat_ = VK_NULL_HANDLE; // desc_set_layout3_ + a 3-uint32 (M,K,N) push range
    VkPipelineLayout pipeline_layout5_mm_ = VK_NULL_HANDLE; // desc_set_layout5_ + a 3-uint32 (M,K,N) push range — quantized GEMM (point 12)
    VkPipeline pipeline_f32_ = VK_NULL_HANDLE;
    VkPipeline pipeline_f32_scalar_ = VK_NULL_HANDLE;
    VkPipeline pipeline_q4_0_ = VK_NULL_HANDLE;
    VkPipeline pipeline_q8_0_ = VK_NULL_HANDLE;
    VkPipeline pipeline_matmul_f16_ = VK_NULL_HANDLE; // VK_NULL_HANDLE unless coop_matrix_supported_
    VkPipeline pipeline_matmul_f32_ = VK_NULL_HANDLE; // portable tiled GEMM — always built, no extension required
    VkPipeline pipeline_matmul_q4_0_ = VK_NULL_HANDLE; // matmul_tiled_q4_0.spv or _dp4a.spv, see create_pipelines()
    VkPipeline pipeline_matmul_q8_0_ = VK_NULL_HANDLE; // matmul_tiled_q8_0.spv or _dp4a.spv, see create_pipelines()

    // True iff VK_KHR_cooperative_matrix + a 16x16x16 fp16xfp16->f32
    // subgroup shape are both present — see create_device_and_queue() and
    // query_coop_matrix_shape(). Every other member/function in this
    // class works identically regardless of this value.
    bool coop_matrix_supported_ = false;
    static constexpr uint32_t kCoopTile = 16; // the M=N=K this backend targets; see matmul_coopmat_f16.comp
    static constexpr uint32_t kTileF32 = 16;  // matmul_tiled_f32.comp's TILE constant — kept separate from kCoopTile since the two shaders' tile sizes are independent choices that happen to match today
    static constexpr uint32_t kTileQuant = 16; // matmul_tiled_q4_0.comp/matmul_tiled_q8_0.comp's TILE constant — see point 12

    // True iff VK_KHR_shader_integer_dot_product (8-bit signed dot,
    // accumulate) is present — see create_device_and_queue() and this
    // file's header comment, point 12. Selects which of the two
    // matmul_tiled_q4_0/q8_0.spv variants create_pipelines() loads;
    // nothing else in this class branches on it.
    bool dot_product_supported_ = false;

    // True iff VK_EXT_subgroup_size_control is present — lets
    // build_pipeline_specialized() pin matmul_coopmat_f16.comp's
    // subgroup size explicitly via VkPipelineShaderStageRequiredSubgroup
    // SizeCreateInfoEXT instead of trusting the device's default-
    // reported size (see query_coop_matrix_shape()'s doc comment on why
    // that gap existed). Only consulted when coop_matrix_supported_ is
    // also true.
    bool subgroup_size_control_supported_ = false;

    // Max independent ops of one layout family (3-buffer F32, or 5-buffer
    // quantized) that a single begin_batch()/end_batch() span can queue.
    // Used to be a fixed kMaxOpsPerBatch=8 constant sized for one
    // request's Q+K+V+attn_out+gate+up+down (7); now a constructor
    // parameter (see VulkanMatvecBackend's constructor) because point 9's
    // request-batching wrappers (queue_matvec_f32_requests() etc.) want
    // this budget to cover N concurrent requests' worth of ops too, not
    // just one request's independent projections — a caller doing
    // continuous-batching decode across, say, 16 concurrent sequences
    // needs a budget sized for that, not for 7. Queueing past it throws
    // rather than silently corrupting an earlier op's descriptor set
    // (see begin_batch()'s comment).
    uint32_t max_ops_per_batch_ = 8;

    // Two of everything per-batch (command buffer, descriptor-set slots,
    // scratch buffers): begin_batch() alternates between index 0 and 1
    // every call so the batch just submitted (still possibly executing
    // on the GPU, since end_batch() no longer waits) and the batch about
    // to be recorded never share a command buffer or a descriptor set —
    // see this file's header comment (point 7) and begin_batch()'s
    // comment for the wait this still needs when a slot is reused before
    // its previous occupant has actually finished on the GPU.
    static constexpr uint32_t kFramesInFlight = 2;

    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_[kFramesInFlight] = {VK_NULL_HANDLE, VK_NULL_HANDLE}; // double-buffered; reset+re-recorded per begin_batch()
    bool batch_open_ = false; // true between begin_batch() and end_batch()
    uint32_t next_frame_ = 0;  // bump counter; begin_batch() uses next_frame_++ % kFramesInFlight as the slot index
    uint32_t cur_ = 0;         // the slot index (0 or 1) the currently-open batch is recording into

    // One slot = one queued op's private scratch buffers + descriptor set.
    // Distinct per queued op (not shared/reused within a batch) because
    // Vulkan doesn't snapshot descriptor-set contents at bind time — see
    // begin_batch()'s comment for why sharing one set across queued-but-
    // unsubmitted dispatches would be wrong, not just wasteful. Each of
    // the kFramesInFlight double-buffer slots gets its own independent
    // set of these (slots3_[cur_], slots5_[cur_]) for the same reason:
    // the OTHER slot's dispatches may still be reading/writing its
    // scratch buffers on the GPU while this slot is being recorded into.
    struct Layout3Slot { GpuBuffer x, y; VkDescriptorSet set = VK_NULL_HANDLE; std::vector<uint16_t> f16_scratch; };
    struct Layout5Slot { GpuBuffer xq, xd, xsum, y; VkDescriptorSet set = VK_NULL_HANDLE; std::vector<int32_t> xq_widen; };
    std::vector<Layout3Slot> slots3_[kFramesInFlight]; // each sized max_ops_per_batch_ at construction, never resized after
    std::vector<Layout5Slot> slots5_[kFramesInFlight]; // each sized max_ops_per_batch_ at construction, never resized after
    uint32_t batch_slot3_ = 0; // next free index into slots3_[cur_], reset by begin_batch()
    uint32_t batch_slot5_ = 0; // next free index into slots5_[cur_], reset by begin_batch()

    // Staging buffer for the one-time device-local weight uploads. Not
    // double-buffered — weight uploads only happen at model-load time,
    // serially, before any batch exists (see copy_staging_to_weight()'s
    // comment), so there's no in-flight overlap to protect against here.
    GpuBuffer staging_buf_;

    // One pending host-side download per queued op this batch. Used to
    // be executed in end_batch() itself, right after a synchronous
    // submit+wait; now end_batch() moves this list into
    // batch_downloads_[cur_] and returns without waiting, and
    // sync_slot() (called by wait_batch() and defensively by
    // begin_batch()) is what actually waits on the timeline semaphore
    // and runs these downloads — see this file's header comment
    // (point 7).
    struct PendingDownload { GpuBuffer* src; void* dst; VkDeviceSize bytes; };
    std::vector<PendingDownload> pending_downloads_;                    // the batch currently being recorded
    std::vector<PendingDownload> batch_downloads_[kFramesInFlight];     // per-slot, awaiting sync_slot()
    uint64_t last_signal_[kFramesInFlight] = {0, 0}; // timeline value to wait for before this slot's data is valid; 0 == nothing outstanding

    Layout3Slot& next_slot3() {
        if (batch_slot3_ >= max_ops_per_batch_)
            throw std::runtime_error("rawllm_vulkan: batch queued more than max_ops_per_batch_=" +
                std::to_string(max_ops_per_batch_) + " F32 ops — raise the constructor's max_ops_per_batch or split the batch");
        return slots3_[cur_][batch_slot3_++];
    }
    Layout5Slot& next_slot5() {
        if (batch_slot5_ >= max_ops_per_batch_)
            throw std::runtime_error("rawllm_vulkan: batch queued more than max_ops_per_batch_=" +
                std::to_string(max_ops_per_batch_) + " quantized ops — raise the constructor's max_ops_per_batch or split the batch");
        return slots5_[cur_][batch_slot5_++];
    }

    // Waits (if necessary) for slot `f`'s most recent submission to
    // finish on the GPU, then runs every download that submission left
    // pending. A no-op if nothing is outstanding for this slot (either
    // it's never been used, or a previous sync_slot()/wait_batch() call
    // already handled it). Called by wait_batch() (explicit, caller-
    // requested) and by begin_batch() (defensive: about to reuse this
    // slot's command buffer/descriptor sets/scratch buffers, which is
    // only safe once the GPU is actually done with the PREVIOUS
    // occupant — see begin_batch()'s comment).
    void sync_slot(uint32_t f) {
        if (last_signal_[f] == 0) return;
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1; wi.pSemaphores = &timeline_sem_; wi.pValues = &last_signal_[f];
        vk_check(vkWaitSemaphores(device_, &wi, UINT64_MAX), "vkWaitSemaphores(sync_slot)");
        for (auto& d : batch_downloads_[f]) download(*d.src, d.dst, d.bytes);
        batch_downloads_[f].clear();
        last_signal_[f] = 0;
    }

    void create_instance() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "NEON-Vulkan-Matvec";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        vk_check(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");
    }

    // Scores one candidate device for how well-suited it is to this
    // backend's workload. Replaces the previous "any discrete GPU beats
    // any integrated GPU, no further questions asked" rule, which ranked
    // a 4GB budget discrete card with no coop-matrix support above an
    // integrated GPU that might have more VRAM headroom or actually
    // support the fast paths this file cares about (coop-matrix GEMM,
    // integer-dot-product GEMM). Still not real benchmarking — no device
    // is actually run before this decision, it's a static capability/
    // heuristic score from vkGetPhysicalDevice*() queries alone — but it
    // at least looks at more than deviceType before choosing. Device
    // type remains the dominant term (a discrete GPU is still assumed
    // faster than an integrated one of comparable capability, which is
    // usually true), it just no longer overrides everything else
    // unconditionally.
    struct DeviceScore { int64_t value = -1; uint32_t queue_family = 0; };
    DeviceScore score_device(VkPhysicalDevice d) const {
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qcount, qprops.data());
        uint32_t compute_qf = UINT32_MAX;
        for (uint32_t i = 0; i < qcount; ++i) {
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { compute_qf = i; break; }
        }
        if (compute_qf == UINT32_MAX) return DeviceScore{}; // value stays -1: disqualified, no compute queue at all

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);

        // Device type: coarse but still the strongest single signal —
        // weighted heavily but not so heavily that it can never be
        // outweighed by a large VRAM/capability gap in the other
        // direction (see the terms below).
        int64_t type_score = 0;
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   type_score = 3; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: type_score = 2; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    type_score = 1; break;
            default: type_score = 0; break; // CPU (lavapipe) or OTHER — last resort
        }

        // Largest DEVICE_LOCAL memory heap, in GiB (integer-truncated) —
        // a rough stand-in for "how big a model's weights can live
        // resident on this device" (this file's whole point-1 design,
        // see upload_weight_*()), without needing to actually try an
        // allocation.
        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(d, &mem);
        VkDeviceSize local_heap_bytes = 0;
        for (uint32_t h = 0; h < mem.memoryHeapCount; ++h)
            if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                local_heap_bytes = std::max(local_heap_bytes, mem.memoryHeaps[h].size);
        int64_t vram_gib = (int64_t)(local_heap_bytes / (1024ull * 1024ull * 1024ull));

        // Capability bonuses for the two optional fast paths this file
        // actually has code for (points 10 and 12) — a device that can
        // use either is worth preferring over one that can't, all else
        // equal. Cheap to query here (just extension enumeration), no
        // vkCreateDevice needed yet.
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &ext_count, exts.data());
        bool has_coop = false, has_dot = false;
        for (auto& e : exts) {
            if (std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) has_coop = true;
            if (std::strcmp(e.extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0) has_dot = true;
        }
        int64_t capability_bonus = (has_coop ? 2 : 0) + (has_dot ? 1 : 0);

        // type_score dominates (weighted x1000) so device class is still
        // the primary sort key in the common case (e.g. a real discrete
        // GPU vs. lavapipe), but a type_score==3 device with tiny VRAM
        // and no relevant extensions (the motivating "4GB budget card"
        // case from this function's doc comment) can still be edged out
        // by a type_score==2 device with substantially more VRAM and
        // both fast paths, since the vram/capability terms are large
        // enough to close a one-tier type_score gap.
        DeviceScore result;
        result.value = type_score * 1000 + vram_gib * 10 + capability_bonus;
        result.queue_family = compute_qf;
        return result;
    }

    void pick_physical_device() {
        uint32_t count = 0;
        vk_check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
        if (count == 0) throw std::runtime_error("rawllm_vulkan: no Vulkan-capable device found");
        std::vector<VkPhysicalDevice> devices(count);
        vk_check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");

        VkPhysicalDevice best = VK_NULL_HANDLE;
        DeviceScore best_score;
        for (auto d : devices) {
            DeviceScore s = score_device(d);
            if (s.value < 0) continue; // no compute queue family — disqualified
            if (best == VK_NULL_HANDLE || s.value > best_score.value) { best = d; best_score = s; }
        }
        if (best == VK_NULL_HANDLE)
            throw std::runtime_error("rawllm_vulkan: no device with a compute queue family found");
        phys_ = best; queue_family_ = best_score.queue_family;
    }

    // Looks for a transfer-only queue family (VK_QUEUE_TRANSFER_BIT set,
    // neither VK_QUEUE_GRAPHICS_BIT nor VK_QUEUE_COMPUTE_BIT) on the
    // physical device pick_physical_device() just chose. When present,
    // it typically maps to a hardware DMA engine that can run copies
    // concurrently with the compute queue's dispatches. Sets
    // transfer_queue_family_ and has_dedicated_transfer_queue_; on any
    // device without one (integrated GPUs, lavapipe, many discrete GPUs
    // only exposing one general-purpose family), falls back to sharing
    // the compute queue/family, matching the has_dedicated_transfer_
    // queue_ == false path used throughout copy_staging_to_weight().
    void find_transfer_queue_family() {
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, qprops.data());
        for (uint32_t i = 0; i < qcount; ++i) {
            if (i == queue_family_) continue;
            VkQueueFlags flags = qprops[i].queueFlags;
            if ((flags & VK_QUEUE_TRANSFER_BIT) &&
                !(flags & VK_QUEUE_GRAPHICS_BIT) &&
                !(flags & VK_QUEUE_COMPUTE_BIT)) {
                transfer_queue_family_ = i;
                has_dedicated_transfer_queue_ = true;
                return;
            }
        }
        transfer_queue_family_ = queue_family_;
        has_dedicated_transfer_queue_ = false;
    }

    void query_subgroup_properties() {
        VkPhysicalDeviceProperties base;
        vkGetPhysicalDeviceProperties(phys_, &base);
        if (base.apiVersion < VK_API_VERSION_1_1)
            throw std::runtime_error("rawllm_vulkan: device only supports Vulkan < 1.1; "
                "subgroup operations (required by every shader in this backend) need 1.1+");

        VkPhysicalDeviceSubgroupProperties sgprops{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &sgprops;
        vkGetPhysicalDeviceProperties2(phys_, &props2);

        if (!(sgprops.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
            throw std::runtime_error("rawllm_vulkan: device's subgroup operations aren't available in compute shaders");
        if (!(sgprops.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))
            throw std::runtime_error("rawllm_vulkan: device lacks VK_SUBGROUP_FEATURE_ARITHMETIC_BIT "
                "(subgroupAdd), required by every shader in this backend — the previous "
                "shared-memory-tree-reduction shader (no longer shipped) would have run here "
                "instead, but this backend no longer carries that fallback path");

        subgroup_size_ = sgprops.subgroupSize;
        // All four shaders fix local_size_x=256; gl_NumSubgroups at
        // runtime must equal rows_per_wg_ computed here for the host's
        // dispatch-group-count math and the shader's row-indexing math to
        // agree (row = groupID * gl_NumSubgroups + gl_SubgroupID).
        if (subgroup_size_ == 0 || 256u % subgroup_size_ != 0)
            throw std::runtime_error("rawllm_vulkan: device subgroup size (" +
                std::to_string(subgroup_size_) + ") doesn't evenly divide the shaders' "
                "fixed local_size_x=256 — this backend doesn't support that subgroup size yet");
        rows_per_wg_ = 256u / subgroup_size_;
    }

    void create_device_and_queue() {
        // ---- optional: VK_KHR_cooperative_matrix for the fp16 matmul path ----
        // Checked and enabled here but never required — every matvec_*()/
        // queue_matvec_*() path works with zero device extensions, same as
        // before this was added. A device that lacks the extension, or
        // has it but reports no usable 16x16x16 fp16xfp16->f32
        // subgroup-scope shape, just doesn't get queue_matmul_f16()
        // (coop_matrix_supported() reports false) — nothing else changes.
        bool coop_ext_present = false;
        {
            // VkPhysicalDeviceVulkan12Features/Vulkan11Features may only
            // be chained onto vkGetPhysicalDeviceFeatures2()/
            // vkCreateDevice() if the physical device supports that API
            // version — gate the whole coop-matrix feature probe on that
            // first (cooperative matrix drivers are 1.2+ in every real
            // case, but this keeps a 1.1-only device from hitting a
            // validation error over unrelated feature-chain plumbing).
            VkPhysicalDeviceProperties base;
            vkGetPhysicalDeviceProperties(phys_, &base);
            if (base.apiVersion >= VK_API_VERSION_1_2) {
                uint32_t ext_count = 0;
                vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ext_count, nullptr);
                std::vector<VkExtensionProperties> exts(ext_count);
                vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ext_count, exts.data());
                for (auto& e : exts)
                    if (std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) { coop_ext_present = true; break; }
            }
        }

        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceVulkan12Features vk12Feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features vk11Feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        bool coop_features_ok = false;
        if (coop_ext_present) {
            vk11Feat.pNext = &vk12Feat;
            vk12Feat.pNext = &coopFeat;
            VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            feat2.pNext = &vk11Feat;
            vkGetPhysicalDeviceFeatures2(phys_, &feat2);
            // Need all three: the extension's own feature bit, plus the
            // two core-1.1/1.2 bits that let float16_t actually live in
            // an SSBO and be used in shader arithmetic (see
            // matmul_coopmat_f16.comp).
            coop_features_ok = coopFeat.cooperativeMatrix && vk12Feat.shaderFloat16 && vk11Feat.storageBuffer16BitAccess;
        }

        // ---- optional: VK_KHR_shader_integer_dot_product for the Q4_0/Q8_0
        // GEMM shaders' fast path (point 12 above) ----
        // Checked the same way as the coop-matrix extension: present but
        // never required. A device without it just gets the portable
        // matmul_tiled_q4_0.spv/matmul_tiled_q8_0.spv pipeline built with
        // no macro define — identical math, manual unpack-and-multiply
        // loop instead of dotPacked4x8EXT(). Only the two quantized GEMM
        // shaders read this; every GEMV shader and the F32/F16 GEMM
        // shaders are unaffected either way.
        bool dot_product_ext_present = false;
        {
            uint32_t ext_count = 0;
            vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ext_count, nullptr);
            std::vector<VkExtensionProperties> exts(ext_count);
            vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ext_count, exts.data());
            for (auto& e : exts)
                if (std::strcmp(e.extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0) { dot_product_ext_present = true; break; }
        }
        VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dotFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR};
        bool dot_product_ok = false;
        if (dot_product_ext_present) {
            VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            feat2.pNext = &dotFeat;
            vkGetPhysicalDeviceFeatures2(phys_, &feat2);
            dot_product_ok = dotFeat.shaderIntegerDotProduct == VK_TRUE;
        }

        // ---- optional: VK_EXT_subgroup_size_control, to pin the
        // coop-matrix pipeline's subgroup size explicitly instead of
        // trusting the device's *default* reported size (see this file's
        // header comment / query_coop_matrix_shape()'s doc comment on why
        // that gap exists — a device that varies subgroup size by default
        // could answer query_subgroup_properties()'s query at one size
        // and run the pipeline at another). Only relevant when
        // coop_matrix_supported_ ends up true; requested unconditionally
        // here (cheap) so build_pipeline_specialized() can chain the
        // required-subgroup-size struct when both this and coop-matrix
        // support are present.
        bool subgroup_size_control_ext_present = false;
        {
            uint32_t ext_count = 0;
            vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ext_count, nullptr);
            std::vector<VkExtensionProperties> exts(ext_count);
            vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ext_count, exts.data());
            for (auto& e : exts)
                if (std::strcmp(e.extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) == 0) { subgroup_size_control_ext_present = true; break; }
        }
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgSizeFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
        bool subgroup_size_control_ok = false;
        if (subgroup_size_control_ext_present) {
            VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            feat2.pNext = &sgSizeFeat;
            vkGetPhysicalDeviceFeatures2(phys_, &feat2);
            subgroup_size_control_ok = sgSizeFeat.subgroupSizeControl == VK_TRUE;
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qcis[2]{};
        qcis[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qcis[0].queueFamilyIndex = queue_family_;
        qcis[0].queueCount = 1;
        qcis[0].pQueuePriorities = &prio;
        uint32_t qci_count = 1;
        if (has_dedicated_transfer_queue_) {
            qcis[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qcis[1].queueFamilyIndex = transfer_queue_family_;
            qcis[1].queueCount = 1;
            qcis[1].pQueuePriorities = &prio;
            qci_count = 2;
        }
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = qci_count;
        dci.pQueueCreateInfos = qcis;

        // Every extension enabled below is optional and additive — none
        // of them are chained together (each hangs its own feature struct
        // off dci.pNext independently apart from coop-matrix's own
        // 3-struct chain above), so any subset of "present on this
        // device" can end up enabled without the others caring.
        std::vector<const char*> enabled_exts;
        void* feature_chain_tail = nullptr;
        auto push_chain = [&](void* s) {
            if (feature_chain_tail) reinterpret_cast<VkBaseOutStructure*>(feature_chain_tail)->pNext = reinterpret_cast<VkBaseOutStructure*>(s);
            else dci.pNext = s;
            feature_chain_tail = s;
        };
        if (coop_features_ok) {
            // Enable exactly the three bits checked above — the other
            // VkBool32 fields in these structs default to 0/false from
            // the {sType} brace-init above, so nothing else gets
            // silently turned on by requesting this.
            coopFeat.cooperativeMatrix = VK_TRUE;
            vk12Feat.shaderFloat16 = VK_TRUE;
            vk11Feat.storageBuffer16BitAccess = VK_TRUE;
            vk11Feat.pNext = &vk12Feat;
            vk12Feat.pNext = &coopFeat;
            push_chain(&vk11Feat);
            enabled_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        }
        if (dot_product_ok) {
            dotFeat.shaderIntegerDotProduct = VK_TRUE;
            push_chain(&dotFeat);
            enabled_exts.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
        }
        if (subgroup_size_control_ok) {
            sgSizeFeat.subgroupSizeControl = VK_TRUE;
            // computeFullSubgroups isn't needed by anything in this file
            // (matmul_coopmat_f16.comp's specialized local_size_x is
            // already an exact multiple of the pinned subgroup size, see
            // build_pipeline_specialized()) — left VK_FALSE.
            push_chain(&sgSizeFeat);
            enabled_exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        }
        dci.enabledExtensionCount = (uint32_t)enabled_exts.size();
        dci.ppEnabledExtensionNames = enabled_exts.empty() ? nullptr : enabled_exts.data();

        vk_check(vkCreateDevice(phys_, &dci, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
        coop_matrix_supported_ = coop_features_ok; // narrowed further by query_coop_matrix_shape() below
        dot_product_supported_ = dot_product_ok;
        subgroup_size_control_supported_ = subgroup_size_control_ok;

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = queue_family_;
        vk_check(vkCreateCommandPool(device_, &cpci, nullptr, &cmd_pool_), "vkCreateCommandPool");

        if (has_dedicated_transfer_queue_) {
            vkGetDeviceQueue(device_, transfer_queue_family_, 0, &transfer_queue_);
            VkCommandPoolCreateInfo tcpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            tcpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            tcpci.queueFamilyIndex = transfer_queue_family_;
            vk_check(vkCreateCommandPool(device_, &tcpci, nullptr, &transfer_cmd_pool_), "vkCreateCommandPool(transfer)");
        }
    }

    // Enabling VK_KHR_cooperative_matrix doesn't by itself guarantee this
    // backend's specific target shape (16x16x16, fp16 A/B, f32 C/result,
    // Subgroup scope) is one of the combinations the device actually
    // implements — enumerate and check for it explicitly. Narrows (never
    // widens) coop_matrix_supported_ as set by create_device_and_queue().
    //
    // Previously NOT validated (and now addressed, still untested on real
    // hardware): whether this device's *default* reported subgroup size
    // (subgroup_size_, queried by query_subgroup_properties() and reused
    // here to specialize matmul_coopmat_f16's local_size_x) is actually
    // the size this MxNxKxtype combination runs at. A device that varies
    // its subgroup size by default (VK_EXT_subgroup_size_control
    // territory) could in principle answer this query at one size and run
    // the pipeline at another. build_pipeline_specialized() now pins the
    // subgroup size explicitly via VkPipelineShaderStageRequiredSubgroup
    // SizeCreateInfoEXT when subgroup_size_control_supported_ is true
    // (see create_device_and_queue()) — but this remains untested against
    // real hardware, since no coop-matrix-capable device (hardware or
    // software) was available to check it against, and lavapipe supports
    // neither VK_KHR_cooperative_matrix nor VK_EXT_subgroup_size_control
    // to exercise this path at all.
    void query_coop_matrix_shape() {
        if (!coop_matrix_supported_) return;
        // vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR is a KHR
        // EXTENSION function, not one of the core functions the Vulkan
        // loader statically exports from libvulkan.so — linking against
        // it directly (as an earlier version of this function did) fails
        // at link time with "undefined reference" on a standard Vulkan
        // SDK setup (confirmed: g++ + libvulkan-dev 1.3.275). Every
        // extension function must be resolved through
        // vkGetInstanceProcAddr instead; this is exactly what
        // coop_matrix_supported_ being false for lack of the pointer
        // achieves for devices/loaders where it's unavailable for any
        // reason, same as the shape-mismatch case below.
        auto fn = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (!fn) { coop_matrix_supported_ = false; return; }

        uint32_t count = 0;
        fn(phys_, &count, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> props(count, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        fn(phys_, &count, props.data());
        bool found = false;
        for (auto& p : props) {
            if (p.MSize == kCoopTile && p.NSize == kCoopTile && p.KSize == kCoopTile &&
                p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                p.scope == VK_SCOPE_SUBGROUP_KHR) {
                found = true;
                break;
            }
        }
        coop_matrix_supported_ = found;
    }

    // Seeds the pipeline cache from pipeline_cache_path_ on disk if it
    // exists and is non-empty, so create_pipelines()'s five-to-nine
    // vkCreateComputePipelines() calls can skip recompiling SPIR-V to
    // driver-native machine code on every run — the whole point of a
    // pipeline cache. No validation of the file's contents happens here:
    // per the Vulkan spec, VkPipelineCacheCreateInfo::pInitialData may be
    // data from an incompatible driver/device/cache-UUID, in which case
    // the implementation is required to silently ignore it and behave as
    // if an empty cache had been provided (never a validation error, and
    // pipeline creation still succeeds correctly, just without a cache
    // hit) — so a stale file from a previous GPU/driver is harmless, not
    // just a missing one.
    void create_pipeline_cache() {
        std::vector<char> initial_data;
        if (!pipeline_cache_path_.empty()) {
            std::ifstream f(pipeline_cache_path_, std::ios::binary | std::ios::ate);
            if (f) {
                std::streamsize n = f.tellg();
                if (n > 0) {
                    initial_data.resize((size_t)n);
                    f.seekg(0);
                    f.read(initial_data.data(), n);
                    if (!f) initial_data.clear(); // partial/corrupt read — fall back to empty cache rather than feed a truncated blob in
                }
            }
        }
        VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        if (!initial_data.empty()) {
            pcci.initialDataSize = initial_data.size();
            pcci.pInitialData = initial_data.data();
        }
        vk_check(vkCreatePipelineCache(device_, &pcci, nullptr, &pipeline_cache_), "vkCreatePipelineCache");
    }

    // Writes the (possibly now-larger, after this run's
    // vkCreateComputePipelines() calls) cache back to pipeline_cache_path_
    // so the NEXT construction's create_pipeline_cache() can seed from
    // it. Called from the destructor, after vkDeviceWaitIdle() and before
    // pipeline_cache_ is destroyed. Best-effort: any failure (bad path,
    // no write permission, disk full) is silently ignored rather than
    // thrown from a destructor — cold-start latency is what this
    // optimizes, not correctness, so losing a cache write is never worth
    // crashing the process over.
    void save_pipeline_cache_to_disk() {
        if (pipeline_cache_path_.empty() || pipeline_cache_ == VK_NULL_HANDLE) return;
        size_t size = 0;
        if (vkGetPipelineCacheData(device_, pipeline_cache_, &size, nullptr) != VK_SUCCESS || size == 0) return;
        std::vector<char> data(size);
        if (vkGetPipelineCacheData(device_, pipeline_cache_, &size, data.data()) != VK_SUCCESS) return;
        std::ofstream f(pipeline_cache_path_, std::ios::binary | std::ios::trunc);
        if (!f) return; // e.g. parent directory doesn't exist — best-effort, see comment above
        f.write(data.data(), (std::streamsize)size);
    }

    static VkDescriptorSetLayoutBinding storage_binding(uint32_t idx) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = idx;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        return b;
    }

    void create_descriptor_layouts() {
        {
            VkDescriptorSetLayoutBinding bindings[3] = { storage_binding(0), storage_binding(1), storage_binding(2) };
            VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            ci.bindingCount = 3; ci.pBindings = bindings;
            vk_check(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &desc_set_layout3_), "vkCreateDescriptorSetLayout(3)");
        }
        {
            VkDescriptorSetLayoutBinding bindings[5] = {
                storage_binding(0), storage_binding(1), storage_binding(2), storage_binding(3), storage_binding(4)
            };
            VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            ci.bindingCount = 5; ci.pBindings = bindings;
            vk_check(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &desc_set_layout5_), "vkCreateDescriptorSetLayout(5)");
        }

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t)};
        {
            VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            ci.setLayoutCount = 1; ci.pSetLayouts = &desc_set_layout3_;
            ci.pushConstantRangeCount = 1; ci.pPushConstantRanges = &pcr;
            vk_check(vkCreatePipelineLayout(device_, &ci, nullptr, &pipeline_layout3_), "vkCreatePipelineLayout(3)");
        }
        {
            VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            ci.setLayoutCount = 1; ci.pSetLayouts = &desc_set_layout5_;
            ci.pushConstantRangeCount = 1; ci.pPushConstantRanges = &pcr;
            vk_check(vkCreatePipelineLayout(device_, &ci, nullptr, &pipeline_layout5_), "vkCreatePipelineLayout(5)");
        }
        {
            // Reuses desc_set_layout3_ (same 3-storage-buffer shape:
            // weights/B, activations/A, output/C) with a wider
            // push-constant range (M,K,N vs cols,rows) — cheap to create
            // unconditionally even on devices without cooperative-matrix
            // support; only the pipeline object built from it
            // (pipeline_matmul_f16_) is conditional.
            VkPushConstantRange pcr_mm{VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t)};
            VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            ci.setLayoutCount = 1; ci.pSetLayouts = &desc_set_layout3_;
            ci.pushConstantRangeCount = 1; ci.pPushConstantRanges = &pcr_mm;
            vk_check(vkCreatePipelineLayout(device_, &ci, nullptr, &pipeline_layout_coopmat_), "vkCreatePipelineLayout(coopmat)");
        }
        {
            // Same 3-uint32 (M,K,N) push-constant shape as
            // pipeline_layout_coopmat_ above, but built on desc_set_
            // layout5_ (W/XQ/XD/XSUM/Y) instead of desc_set_layout3_ —
            // the quantized GEMM path (point 12) needs the extra two
            // buffers the quantized GEMV path already declared, not the
            // plain W/X/Y shape the F32/F16 GEMM paths use.
            VkPushConstantRange pcr_mm5{VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t)};
            VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            ci.setLayoutCount = 1; ci.pSetLayouts = &desc_set_layout5_;
            ci.pushConstantRangeCount = 1; ci.pPushConstantRanges = &pcr_mm5;
            vk_check(vkCreatePipelineLayout(device_, &ci, nullptr, &pipeline_layout5_mm_), "vkCreatePipelineLayout(quant mm)");
        }
    }

    VkPipeline build_pipeline(const std::string& spirv_path, VkPipelineLayout layout) {
        auto code = read_spirv(spirv_path);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = code.size() * sizeof(uint32_t);
        smci.pCode = code.data();
        VkShaderModule module;
        vk_check(vkCreateShaderModule(device_, &smci, nullptr, &module), "vkCreateShaderModule(" + spirv_path + ")");

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";

        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = layout;
        VkPipeline pipe;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &ci, nullptr, &pipe),
                  "vkCreateComputePipelines(" + spirv_path + ")");
        vkDestroyShaderModule(device_, module, nullptr);
        return pipe;
    }

    // Same as build_pipeline(), but sets local_size_x (specialization
    // constant_id=0 in matmul_coopmat_f16.comp) to spec_local_size_x at
    // pipeline-creation time — see that shader's header comment for why
    // this needs to be a specialization constant rather than a fixed
    // local_size_x like the GEMV shaders use.
    //
    // When subgroup_size_control_supported_ is true, also chains a
    // VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pinning the
    // stage's subgroup size to spec_local_size_x's source value
    // (subgroup_size_, the same size the specialization constant above
    // was set from) — this is the fix for the gap query_coop_matrix_
    // shape()'s doc comment flags: without it, this pipeline runs at
    // whatever subgroup size the device's driver picks at dispatch time,
    // which on a device with VK_EXT_subgroup_size_control's variable-
    // size behavior is not guaranteed to be the default size
    // query_subgroup_properties() queried at construction. On a device
    // without the extension, this is a no-op (pStageInfo stays null) and
    // behavior is unchanged from before — untestable either way on
    // lavapipe, which implements neither VK_KHR_cooperative_matrix nor
    // VK_EXT_subgroup_size_control.
    VkPipeline build_pipeline_specialized(const std::string& spirv_path, VkPipelineLayout layout, uint32_t spec_local_size_x) {
        auto code = read_spirv(spirv_path);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = code.size() * sizeof(uint32_t);
        smci.pCode = code.data();
        VkShaderModule module;
        vk_check(vkCreateShaderModule(device_, &smci, nullptr, &module), "vkCreateShaderModule(" + spirv_path + ")");

        VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};
        VkSpecializationInfo spec_info{};
        spec_info.mapEntryCount = 1;
        spec_info.pMapEntries = &entry;
        spec_info.dataSize = sizeof(uint32_t);
        spec_info.pData = &spec_local_size_x;

        VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT req_sg{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
        req_sg.requiredSubgroupSize = subgroup_size_;

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        stage.pSpecializationInfo = &spec_info;
        if (subgroup_size_control_supported_) stage.pNext = &req_sg;

        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = layout;
        VkPipeline pipe;
        vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &ci, nullptr, &pipe),
                  "vkCreateComputePipelines(" + spirv_path + ")");
        vkDestroyShaderModule(device_, module, nullptr);
        return pipe;
    }

    void create_pipelines(const std::string& shaders_dir) {
        std::string dir = shaders_dir;
        if (!dir.empty() && dir.back() != '/') dir += '/';
        pipeline_f32_        = build_pipeline(dir + "matvec_f32.spv",        pipeline_layout3_);
        pipeline_f32_scalar_ = build_pipeline(dir + "matvec_f32_scalar.spv", pipeline_layout3_);
        pipeline_q4_0_       = build_pipeline(dir + "matvec_q4_0.spv",       pipeline_layout5_);
        pipeline_q8_0_       = build_pipeline(dir + "matvec_q8_0.spv",       pipeline_layout5_);
        // Only attempted when the device actually supports it — a device
        // without VK_KHR_cooperative_matrix (or without the 16x16x16
        // fp16xfp16->f32 shape) never needs matmul_coopmat_f16.spv to
        // exist on disk at all.
        if (coop_matrix_supported_) {
            pipeline_matmul_f16_ = build_pipeline_specialized(dir + "matmul_coopmat_f16.spv", pipeline_layout_coopmat_, subgroup_size_);
        }
        // Portable tiled F32 GEMM: plain shared-memory compute, no
        // extension and no device-capability check needed — built
        // unconditionally, same as the four GEMV pipelines above.
        pipeline_matmul_f32_ = build_pipeline(dir + "matmul_tiled_f32.spv", pipeline_layout_coopmat_);

        // Quantized tiled GEMM (point 12): built unconditionally, same as
        // the F32 GEMM above — no device-capability check gates whether
        // this exists, only which of the two SPIR-V variants gets
        // loaded. dot_product_supported_ picks the dp4a-accelerated
        // build (compiled from the same .comp source with
        // -DHAS_INT8_DOT=1) when VK_KHR_shader_integer_dot_product is
        // present, and falls back to the portable manual-unpack build
        // otherwise — identical math either way, see this file's header
        // comment (point 12) and matmul_tiled_q4_0.comp/matmul_tiled_
        // q8_0.comp's own header comments for what the two variants do
        // differently.
        const std::string q4_0_variant = dot_product_supported_ ? "matmul_tiled_q4_0_dp4a.spv" : "matmul_tiled_q4_0.spv";
        const std::string q8_0_variant = dot_product_supported_ ? "matmul_tiled_q8_0_dp4a.spv" : "matmul_tiled_q8_0.spv";
        pipeline_matmul_q4_0_ = build_pipeline(dir + q4_0_variant, pipeline_layout5_mm_);
        pipeline_matmul_q8_0_ = build_pipeline(dir + q8_0_variant, pipeline_layout5_mm_);
    }

    void create_descriptor_pool_and_sets() {
        // max_ops_per_batch_ sets for layout3 (3 storage buffers each)
        // plus max_ops_per_batch_ sets for layout5 (5 storage buffers
        // each), PER double-buffer slot (kFramesInFlight of everything —
        // see this file's header comment, point 7). Every set is
        // allocated once, up front, for the backend's whole lifetime,
        // then rebound in place (vkUpdateDescriptorSets) before whichever
        // dispatch uses it — no per-call allocate/free, same reasoning as
        // the single-set version this replaces, just sized for
        // max_ops_per_batch_ concurrent in-flight ops per slot instead of
        // a fixed 8 for one slot.
        uint32_t total_sets = max_ops_per_batch_ * 2 * kFramesInFlight;
        uint32_t total_buffers = (max_ops_per_batch_ * 3 + max_ops_per_batch_ * 5) * kFramesInFlight;
        VkDescriptorPoolSize psize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, total_buffers};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = total_sets;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &psize;
        vk_check(vkCreateDescriptorPool(device_, &dpci, nullptr, &desc_pool_), "vkCreateDescriptorPool");

        auto alloc_set = [&](VkDescriptorSetLayout layout) {
            VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dsai.descriptorPool = desc_pool_;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &layout;
            VkDescriptorSet set;
            vk_check(vkAllocateDescriptorSets(device_, &dsai, &set), "vkAllocateDescriptorSets");
            return set;
        };
        for (uint32_t f = 0; f < kFramesInFlight; ++f) {
            slots3_[f].resize(max_ops_per_batch_);
            slots5_[f].resize(max_ops_per_batch_);
            for (auto& s : slots3_[f]) s.set = alloc_set(desc_set_layout3_);
            for (auto& s : slots5_[f]) s.set = alloc_set(desc_set_layout5_);
        }
    }

    void create_command_resources() {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = kFramesInFlight;
        vk_check(vkAllocateCommandBuffers(device_, &cbai, cmd_), "vkAllocateCommandBuffers");

        // The timeline semaphore end_batch()/sync_slot() coordinate
        // through — VK_SEMAPHORE_TYPE_TIMELINE, starting at 0 (matches
        // timeline_counter_'s initial value; the first end_batch() call
        // signals 1). See this file's header comment (point 7).
        VkSemaphoreTypeCreateInfo stci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        stci.initialValue = 0;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &stci;
        vk_check(vkCreateSemaphore(device_, &sci, nullptr, &timeline_sem_), "vkCreateSemaphore(timeline)");
    }

    // Portable F32 -> IEEE-754 binary16 conversion (round-to-nearest-even).
    // No dependency on _Float16/std::float16_t (not guaranteed available
    // in C++17) or F16C intrinsics (portability over speed — this runs on
    // cols/rows-sized host arrays for queue_matmul_f16()/
    // upload_weight_f16(), not an O(rows*cols) hot path). Checked against
    // the compiler's native _Float16 conversion across 2,000,000 random
    // values plus subnormal/overflow/NaN/+-0 edge cases with zero
    // mismatches — see this file's header comment for what that test
    // covered and what it didn't (nothing downstream of this function).
    static uint16_t float_to_half(float f) {
        uint32_t x; std::memcpy(&x, &f, sizeof(x));
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;

        if (((x >> 23) & 0xFFu) == 0xFFu) // inf/NaN
            return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
        if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u); // overflow -> inf
        if (exp <= 0) {
            if (exp < -10) return (uint16_t)sign; // too small -> signed zero
            mant |= 0x800000u; // implicit leading 1
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

    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("rawllm_vulkan: no suitable memory type for requested properties");
    }

    // Prefer a memory type that is device-local AND host-visible+coherent
    // (every integrated GPU and lavapipe expose exactly this as their only
    // type; modern discrete GPUs with resizable BAR expose it too) so
    // weight upload can just memcpy straight into device memory with no
    // staging buffer or copy command at all. Only fall back to a staging
    // buffer + vkCmdCopyBuffer when the device draws a hard line between
    // device-local and host-visible memory (the classic discrete-GPU case
    // without ReBAR).
    struct MemChoice { uint32_t index; bool host_visible; };
    MemChoice find_device_local_memory(uint32_t type_bits) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        VkMemoryPropertyFlags combo = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & combo) == combo)
                return {i, true};
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                return {i, false};
        throw std::runtime_error("rawllm_vulkan: no device-local memory type found");
    }

    // ── suballocator: a handful of large VkDeviceMemory blocks, carved up
    // by a plain offset+size free-list — see this file's header comment
    // (point 8). One MemBlock per (memory-type-index, allocation) pair;
    // sub_alloc() below only ever creates a new block when no existing
    // block of the right type has a free range big enough, so a
    // real-model load (dozens to hundreds of upload_weight_*() calls,
    // most far smaller than kBlockBytes) ends up with a handful of
    // blocks total instead of one vkAllocateMemory per tensor.
    struct MemBlock {
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        uint32_t type_index = 0;
        std::vector<std::pair<VkDeviceSize, VkDeviceSize>> free_ranges; // sorted by offset, coalesced; {offset, size}
    };
    std::vector<MemBlock> mem_blocks_;
    static constexpr VkDeviceSize kBlockBytes = 128ull * 1024 * 1024; // grown on demand if a single request exceeds this

    static VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) { return a == 0 ? v : (v + a - 1) & ~(a - 1); }

    // First-fit: walks blocks of the matching memory type, returns the
    // first free range with room for `size` once aligned to `alignment`.
    // Falls through to allocating a fresh block (sized to fit even a
    // request bigger than kBlockBytes) and retrying exactly once if no
    // existing block has room — the retry is guaranteed to succeed since
    // the freshly-added block is entirely free and sized to fit.
    SubAlloc sub_alloc(VkDeviceSize size, VkDeviceSize alignment, uint32_t type_index) {
        for (size_t bi = 0; bi < mem_blocks_.size(); ++bi) {
            MemBlock& blk = mem_blocks_[bi];
            if (blk.type_index != type_index) continue;
            for (size_t ri = 0; ri < blk.free_ranges.size(); ++ri) {
                VkDeviceSize base = blk.free_ranges[ri].first, len = blk.free_ranges[ri].second;
                VkDeviceSize aligned = align_up(base, alignment);
                VkDeviceSize pad = aligned - base;
                if (len < pad + size) continue;
                VkDeviceSize used_end = aligned + size;
                VkDeviceSize range_end = base + len;
                blk.free_ranges.erase(blk.free_ranges.begin() + (long)ri);
                if (pad > 0) blk.free_ranges.push_back({base, pad});
                if (used_end < range_end) blk.free_ranges.push_back({used_end, range_end - used_end});
                std::sort(blk.free_ranges.begin(), blk.free_ranges.end());
                return SubAlloc{blk.mem, aligned, size, bi};
            }
        }
        VkDeviceSize block_bytes = std::max(kBlockBytes, size + alignment);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = block_bytes;
        mai.memoryTypeIndex = type_index;
        VkDeviceMemory mem;
        vk_check(vkAllocateMemory(device_, &mai, nullptr, &mem), "vkAllocateMemory(pool block)");
        MemBlock blk; blk.mem = mem; blk.size = block_bytes; blk.type_index = type_index;
        blk.free_ranges.push_back({0, block_bytes});
        mem_blocks_.push_back(blk);
        return sub_alloc(size, alignment, type_index);
    }

    // Returns a suballocation's range to its block's free-list and
    // coalesces it with whatever's adjacent, so a backend's lifetime of
    // ensure_scratch() grow/shrink churn or bulk tensor frees (e.g.
    // unloading a model) doesn't fragment a block into permanent slivers.
    // Does NOT call vkFreeMemory — the underlying VkDeviceMemory blocks
    // are only freed in bulk, at backend destruction (destroy_all_blocks()).
    void sub_free(const SubAlloc& a) {
        if (a.block == SIZE_MAX || a.block >= mem_blocks_.size()) return;
        MemBlock& blk = mem_blocks_[a.block];
        blk.free_ranges.push_back({a.offset, a.size});
        std::sort(blk.free_ranges.begin(), blk.free_ranges.end());
        std::vector<std::pair<VkDeviceSize, VkDeviceSize>> merged;
        for (auto& r : blk.free_ranges) {
            if (!merged.empty() && merged.back().first + merged.back().second == r.first)
                merged.back().second += r.second;
            else
                merged.push_back(r);
        }
        blk.free_ranges = std::move(merged);
    }

    // Frees every underlying VkDeviceMemory block. Only safe to call once
    // every GpuBuffer/GpuTensor backed by these blocks has already had
    // its VkBuffer destroyed (see the destructor's ordering) — same
    // "free every tensor before destroying the backend" contract this
    // class already documented on GpuTensor before suballocation existed,
    // just now backed by shared blocks instead of one allocation each.
    void destroy_all_blocks() {
        for (auto& blk : mem_blocks_) if (blk.mem) vkFreeMemory(device_, blk.mem, nullptr);
        mem_blocks_.clear();
    }

    GpuBuffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
        GpuBuffer b; b.size = size; b.host_visible = (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vk_check(vkCreateBuffer(device_, &bci, nullptr, &b.buf), "vkCreateBuffer");

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, b.buf, &req);
        uint32_t type_index = find_memory_type(req.memoryTypeBits, props);
        b.alloc = sub_alloc(req.size, req.alignment, type_index);
        vk_check(vkBindBufferMemory(device_, b.buf, b.alloc.mem, b.alloc.offset), "vkBindBufferMemory");
        return b;
    }

    void destroy_buffer(GpuBuffer& b) {
        if (b.buf) vkDestroyBuffer(device_, b.buf, nullptr);
        if (b.alloc.mem) sub_free(b.alloc);
        b = GpuBuffer{};
    }

    void upload(GpuBuffer& b, const void* src, VkDeviceSize bytes) {
        void* mapped;
        vk_check(vkMapMemory(device_, b.alloc.mem, b.alloc.offset, bytes, 0, &mapped), "vkMapMemory (upload)");
        std::memcpy(mapped, src, (size_t)bytes);
        vkUnmapMemory(device_, b.alloc.mem);
    }

    void download(GpuBuffer& b, void* dst, VkDeviceSize bytes) {
        void* mapped;
        vk_check(vkMapMemory(device_, b.alloc.mem, b.alloc.offset, bytes, 0, &mapped), "vkMapMemory (download)");
        std::memcpy(dst, mapped, (size_t)bytes);
        vkUnmapMemory(device_, b.alloc.mem);
    }

    // Small per-call buffer (activation upload / metadata / output
    // download): grow, never shrink, never reallocate if already big
    // enough. Called every matvec(); the common case (same tensor shapes
    // every token, which is the normal case for a transformer's fixed
    // hidden size) does zero allocation after the first call.
    void ensure_scratch(GpuBuffer& b, VkDeviceSize needed) {
        if (b.size >= needed) return;
        destroy_buffer(b);
        b = make_buffer(needed, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    // One-time (well: once per weight tensor, at load time — not a
    // per-token hot-path cost) device-local upload. Uses the direct-memcpy
    // path when the device's memory exposes a combined device-local +
    // host-visible type, otherwise stages through a host-visible buffer
    // and a one-shot copy command.
    // GpuTensor.size stays the true logical byte count the caller asked
    // for; the underlying VkBuffer is padded up to the next multiple of 4
    // bytes. This matters specifically for Q4_0 (18 bytes/block) and Q8_0
    // (34 bytes/block): whenever a tensor's total block count is odd,
    // rows*nb*18 (or *34) is NOT a multiple of 4, but matvec_q4_0.comp/
    // matvec_q8_0.comp address the buffer as `uint W[]` and are bound with
    // VK_WHOLE_SIZE — Vulkan requires a STORAGE_BUFFER descriptor's range
    // to be a multiple of 4, and even where a loader doesn't reject it,
    // the last row's final readByte() call ends up reading past the
    // buffer's true end. Confirmed on lavapipe: this reproduced as
    // wrong output on exactly the last row whenever rows*(cols/32) was
    // odd (e.g. Q4_0/Q8_0 at cols=32, rows=1 or rows=5) and nowhere else
    // — fixed by over-allocating here; the padding bytes are never read
    // by any shader for a row within [0, rows), so leaving them
    // uninitialized is safe.
    GpuTensor upload_weight_bytes(const void* data, VkDeviceSize bytes) {
        VkDeviceSize alloc_bytes = (bytes + 3u) & ~VkDeviceSize(3u);
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = alloc_bytes; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        GpuTensor t; t.size = bytes;
        vk_check(vkCreateBuffer(device_, &bci, nullptr, &t.buf), "vkCreateBuffer(weight)");
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, t.buf, &req);
        MemChoice choice = find_device_local_memory(req.memoryTypeBits);
        SubAlloc a = sub_alloc(req.size, req.alignment, choice.index);
        t.mem = a.mem; t.offset = a.offset; t.alloc_size = a.size; t.block_index = a.block;
        vk_check(vkBindBufferMemory(device_, t.buf, t.mem, t.offset), "vkBindBufferMemory(weight)");

        if (choice.host_visible) {
            void* mapped;
            vk_check(vkMapMemory(device_, t.mem, t.offset, bytes, 0, &mapped), "vkMapMemory(weight)");
            std::memcpy(mapped, data, (size_t)bytes);
            vkUnmapMemory(device_, t.mem);
            return t;
        }

        // Pure device-local: stage through a host-visible buffer (grown
        // like the other scratch buffers) and copy on the GPU side. Only
        // the true `bytes` are copied; the buffer's extra alloc_bytes-
        // bytes padding tail is left as whatever vkCreateBuffer/driver
        // gives a fresh allocation (never read by any valid dispatch).
        ensure_scratch(staging_buf_, bytes);
        {
            void* mapped;
            vk_check(vkMapMemory(device_, staging_buf_.alloc.mem, staging_buf_.alloc.offset, bytes, 0, &mapped), "vkMapMemory(staging)");
            std::memcpy(mapped, data, (size_t)bytes);
            vkUnmapMemory(device_, staging_buf_.alloc.mem);
        }

        copy_staging_to_weight(staging_buf_.buf, t.buf, bytes);
        return t;
    }

    // One-shot staging->device-local copy for a freshly uploaded weight
    // tensor — see this file's header comment (point 7). Two paths:
    //
    //   - No dedicated transfer queue (has_dedicated_transfer_queue_ ==
    //     false: every integrated GPU and lavapipe, i.e. every device
    //     this file has actually been run against): exactly the old
    //     behavior, a copy recorded and submitted on the compute queue,
    //     synchronous.
    //
    //   - Dedicated transfer queue: the copy runs on transfer_queue_
    //     instead, followed by a queue-family ownership release (on the
    //     transfer queue) and acquire (on the compute queue) barrier
    //     pair — required because `dst` is VK_SHARING_MODE_EXCLUSIVE, so
    //     a queue family other than the one that last wrote it needs an
    //     explicit ownership transfer before it may read it (here: the
    //     compute queue's shader read). Still fully synchronous (both
    //     queues idled before returning) because upload_weight_bytes()
    //     only ever runs at model-load time, serially, before any batch
    //     exists to actually overlap this copy with — see this file's
    //     header comment (point 7) for what would need to change (a
    //     streaming/background weight-load path) for that overlap to
    //     materialize into a real win rather than just correct plumbing.
    //
    // NOT VALIDATED against a device with a genuine dedicated transfer-
    // only queue family — see this file's header comment for what has
    // and hasn't been checked here.
    void copy_staging_to_weight(VkBuffer src, VkBuffer dst, VkDeviceSize bytes) {
        VkBufferCopy region{0, 0, bytes};

        if (!has_dedicated_transfer_queue_) {
            VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cbai.commandPool = cmd_pool_; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
            VkCommandBuffer copy_cmd;
            vk_check(vkAllocateCommandBuffers(device_, &cbai, &copy_cmd), "vkAllocateCommandBuffers(staging copy)");
            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vk_check(vkBeginCommandBuffer(copy_cmd, &cbbi), "vkBeginCommandBuffer(staging copy)");
            vkCmdCopyBuffer(copy_cmd, src, dst, 1, &region);
            vk_check(vkEndCommandBuffer(copy_cmd), "vkEndCommandBuffer(staging copy)");
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1; si.pCommandBuffers = &copy_cmd;
            vk_check(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit(staging copy)");
            vk_check(vkQueueWaitIdle(queue_), "vkQueueWaitIdle(staging copy)");
            vkFreeCommandBuffers(device_, cmd_pool_, 1, &copy_cmd);
            return;
        }

        VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore sem;
        vk_check(vkCreateSemaphore(device_, &semci, nullptr, &sem), "vkCreateSemaphore(staging copy handoff)");

        VkCommandBufferAllocateInfo xcbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        xcbai.commandPool = transfer_cmd_pool_; xcbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; xcbai.commandBufferCount = 1;
        VkCommandBuffer xfer_cmd;
        vk_check(vkAllocateCommandBuffers(device_, &xcbai, &xfer_cmd), "vkAllocateCommandBuffers(transfer copy)");
        VkCommandBufferBeginInfo xcbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        xcbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(xfer_cmd, &xcbbi), "vkBeginCommandBuffer(transfer copy)");
        vkCmdCopyBuffer(xfer_cmd, src, dst, 1, &region);
        // Release: the transfer queue family gives up ownership of `dst`
        // once its write finishes. dstAccessMask is 0 here — the matching
        // access happens on the ACQUIRING side's barrier below, per the
        // Vulkan spec's queue family ownership transfer recipe.
        VkBufferMemoryBarrier release{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        release.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        release.dstAccessMask = 0;
        release.srcQueueFamilyIndex = transfer_queue_family_;
        release.dstQueueFamilyIndex = queue_family_;
        release.buffer = dst; release.offset = 0; release.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(xfer_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              0, 0, nullptr, 1, &release, 0, nullptr);
        vk_check(vkEndCommandBuffer(xfer_cmd), "vkEndCommandBuffer(transfer copy)");

        VkSubmitInfo xsi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        xsi.commandBufferCount = 1; xsi.pCommandBuffers = &xfer_cmd;
        xsi.signalSemaphoreCount = 1; xsi.pSignalSemaphores = &sem;
        vk_check(vkQueueSubmit(transfer_queue_, 1, &xsi, VK_NULL_HANDLE), "vkQueueSubmit(transfer copy)");

        VkCommandBufferAllocateInfo acbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        acbai.commandPool = cmd_pool_; acbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; acbai.commandBufferCount = 1;
        VkCommandBuffer acquire_cmd;
        vk_check(vkAllocateCommandBuffers(device_, &acbai, &acquire_cmd), "vkAllocateCommandBuffers(acquire)");
        VkCommandBufferBeginInfo acbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        acbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(acquire_cmd, &acbbi), "vkBeginCommandBuffer(acquire)");
        // Acquire: the compute queue family takes ownership of `dst`
        // before any shader gets to read it. srcAccessMask is 0 to match
        // the release side above.
        VkBufferMemoryBarrier acquire{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        acquire.srcAccessMask = 0;
        acquire.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        acquire.srcQueueFamilyIndex = transfer_queue_family_;
        acquire.dstQueueFamilyIndex = queue_family_;
        acquire.buffer = dst; acquire.offset = 0; acquire.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(acquire_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 0, nullptr, 1, &acquire, 0, nullptr);
        vk_check(vkEndCommandBuffer(acquire_cmd), "vkEndCommandBuffer(acquire)");

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo asi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        asi.commandBufferCount = 1; asi.pCommandBuffers = &acquire_cmd;
        asi.waitSemaphoreCount = 1; asi.pWaitSemaphores = &sem; asi.pWaitDstStageMask = &wait_stage;
        vk_check(vkQueueSubmit(queue_, 1, &asi, VK_NULL_HANDLE), "vkQueueSubmit(acquire)");

        vk_check(vkQueueWaitIdle(transfer_queue_), "vkQueueWaitIdle(transfer copy)");
        vk_check(vkQueueWaitIdle(queue_), "vkQueueWaitIdle(acquire)");
        vkFreeCommandBuffers(device_, transfer_cmd_pool_, 1, &xfer_cmd);
        vkFreeCommandBuffers(device_, cmd_pool_, 1, &acquire_cmd);
        vkDestroySemaphore(device_, sem, nullptr);
    }

    void update_descriptor_set3(VkDescriptorSet set, VkBuffer w, VkBuffer x, VkBuffer y) {
        VkDescriptorBufferInfo infos[3] = { {w, 0, VK_WHOLE_SIZE}, {x, 0, VK_WHOLE_SIZE}, {y, 0, VK_WHOLE_SIZE} };
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set; writes[i].dstBinding = (uint32_t)i;
            writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
    }

    void update_descriptor_set5(VkDescriptorSet set, VkBuffer w, VkBuffer xq, VkBuffer xd, VkBuffer xsum, VkBuffer y) {
        VkDescriptorBufferInfo infos[5] = {
            {w, 0, VK_WHOLE_SIZE}, {xq, 0, VK_WHOLE_SIZE}, {xd, 0, VK_WHOLE_SIZE}, {xsum, 0, VK_WHOLE_SIZE}, {y, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet writes[5]{};
        for (int i = 0; i < 5; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set; writes[i].dstBinding = (uint32_t)i;
            writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, 5, writes, 0, nullptr);
    }

    // Records ONE dispatch into the currently-open batch's command buffer.
    // Does NOT begin/end/submit/wait — that's begin_batch()'s and
    // end_batch()'s job now, so that several of these can land in the
    // same command buffer and go out in a single submit. (The previous
    // version of this function did begin/end/submit/wait itself, once per
    // matvec call; that per-call round trip is exactly what batching
    // removes for ops queued together.)
    void record_dispatch(VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set, uint32_t cols, uint32_t rows) {
        if (!batch_open_) throw std::runtime_error("rawllm_vulkan: record_dispatch() called outside begin_batch()/end_batch()");
        vkCmdBindPipeline(cmd_[cur_], VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd_[cur_], VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        struct { uint32_t cols, rows; } push{cols, rows};
        vkCmdPushConstants(cmd_[cur_], layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        // gl_NumSubgroups rows finish per workgroup (see
        // query_subgroup_properties()/rows_per_wg_ and every shader's
        // header comment) instead of the old one-row-per-workgroup shape.
        uint32_t groups = (rows + rows_per_wg_ - 1u) / rows_per_wg_;
        vkCmdDispatch(cmd_[cur_], groups, 1, 1);
    }

    // Same job as record_dispatch(), for either matmul pipeline (coop-
    // matrix fp16 or the portable tiled f32 one): 3 push constants instead
    // of 2, and a 2D dispatch grid (one workgroup per output tile) instead
    // of the GEMV shaders' 1D one. Ceil-divides so callers don't have to
    // pre-round M/N to the tile size — matmul_tiled_f32.comp bounds-checks
    // its loads/stores for the ragged edge this produces; queue_matmul_f16()
    // pads M itself before calling this, so it always passes an exact
    // multiple and the ceil-division there is a no-op.
    void record_dispatch_matmul(VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set, uint32_t M, uint32_t K, uint32_t N, uint32_t tile) {
        if (!batch_open_) throw std::runtime_error("rawllm_vulkan: record_dispatch_matmul() called outside begin_batch()/end_batch()");
        vkCmdBindPipeline(cmd_[cur_], VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd_[cur_], VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        struct { uint32_t M, K, N; } push{M, K, N};
        vkCmdPushConstants(cmd_[cur_], layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        uint32_t tiles_m = (M + tile - 1u) / tile;
        uint32_t tiles_n = (N + tile - 1u) / tile;
        vkCmdDispatch(cmd_[cur_], tiles_m, tiles_n, 1);
    }


    void queue_matvec_quantized(VkPipeline pipe, const GpuTensor& W, const float* x_d, const int32_t* x_sum,
                                 const int8_t* x_q, float* y, uint32_t cols, uint32_t rows) {
        if (!batch_open_) throw std::runtime_error("rawllm_vulkan: queue_matvec_quantized() called outside begin_batch()/end_batch()");
        if (cols % 32 != 0) throw std::runtime_error("rawllm_vulkan: quantized matvec cols must be a multiple of 32");
        uint32_t nb = cols / 32u;
        Layout5Slot& slot = next_slot5();

        // Shaders declare X_Q as int[] (no 8-bit storage extension assumed
        // — see shaders/matvec_q4_0.comp's header comment), so widen the
        // int8 activation vector to int32 here. This is a cols-sized
        // (not rows*cols-sized) host-side pass, done once per queued op.
        slot.xq_widen.resize(cols);
        for (uint32_t i = 0; i < cols; ++i) slot.xq_widen[i] = (int32_t)x_q[i];

        ensure_scratch(slot.xq, (VkDeviceSize)cols * sizeof(int32_t));
        ensure_scratch(slot.xd, (VkDeviceSize)nb * sizeof(float));
        ensure_scratch(slot.xsum, (VkDeviceSize)nb * sizeof(int32_t));
        ensure_scratch(slot.y, (VkDeviceSize)rows * sizeof(float));

        upload(slot.xq, slot.xq_widen.data(), (VkDeviceSize)cols * sizeof(int32_t));
        upload(slot.xd, x_d, (VkDeviceSize)nb * sizeof(float));
        upload(slot.xsum, x_sum, (VkDeviceSize)nb * sizeof(int32_t));

        update_descriptor_set5(slot.set, W.buf, slot.xq.buf, slot.xd.buf, slot.xsum.buf, slot.y.buf);
        record_dispatch(pipe, pipeline_layout5_, slot.set, cols, rows);
        pending_downloads_.push_back({&slot.y, y, (VkDeviceSize)rows * sizeof(float)});
    }
};

} // namespace vk_backend

#endif // USE_VULKAN
