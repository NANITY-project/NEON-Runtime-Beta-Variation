NEON — NANITY Runtime (Beta)

A from-scratch C++ inference runtime for **NANITY**, a single fixed
transformer architecture — not a GGUF-guessing nor every-architecture
runtime like llama.cpp. NEON loads exactly one thing: a model that
declares `general.architecture = "nanity"`, either as GGUF or as
[`.nctr`](./NANITY_ARCHITECTURE_SPEC.md), NANITY's own container
format. Everything else — training pipeline, GGUF/.nctr conversion, the
spec itself — is designed around that one fixed shape on purpose. See
[`NANITY_ARCHITECTURE_SPEC.md`](./NANITY_ARCHITECTURE_SPEC.md) for the
full "why" and the exact tensor/metadata contract.

> **Status: beta, source-available ** See
> [`LICENSE`](./LICENSE) for current terms during the prototyping phase.

## What's actually working right now

- **CPU inference**, no ROCm/GPU required to build or run — validated
  end-to-end against real converted models (Llama-3.2-family and
  TinyLlama-family architectures load and generate coherent output).
- **GGUF loading**, including NANITY-converted third-party models via
  [`nanity_convert.py`](./nanity_convert.py) (see its `--detect-only` flag
  to check compatibility before converting anything).
- **`.nctr` parsing** (`rawllm_nctr_loader.hpp` / `NCTRLoader`) — reads and
  validates `.nctr` files standalone. **Not yet wired into the inference
  path** — `ModelWeights`/`transformer_forward`/etc. still take
  `GGUFLoader` concretely; see the integration TODO at the bottom of
  `rawllm_nctr_loader.hpp`. `.nctr` export (`train_nanity_fixed.py`'s
  `export_nctr()`) works; running a `.nctr` file through NEON doesn't, yet.
- **Tokenization**: both byte-level BPE (GPT-2/tiktoken-family — Qwen,
  Llama-3, Phi) and SentencePiece (Llama/TinyLlama-family) vocabularies.
- **Idle-loop / companion-overlay mode** (continuous background generation,
  disk-backed context eviction, Hyprland vision hooks) — optional, off by
  default, gated behind `-DNANITY_ENABLE_IDLE_LOOP` (needs
  `nectar_diskmem.hpp` / `nectar_vision.hpp` / `nectar_splice.hpp`, all
  present in this repo). The core chat/generation path never needs it.

## Known limitations

- **Vulkan backend** (`rawllm_vulkan.hpp`): experimental, F32-matvec-only,
  gated behind `-DUSE_VULKAN`. Verified correct end-to-end (device init,
  descriptor/pipeline setup, the `shaders/matvec_f32.comp` shader itself)
  against a software Vulkan implementation (Mesa lavapipe) across a range of
  matrix sizes including realistic transformer-layer dimensions — but **not
  yet exercised on real GPU hardware**, and **not wired into the actual
  generation hot path**: `--model ... --probe` with a Vulkan-enabled build
  will initialize the backend and log whether a usable device was found,
  but every token is still computed on the CPU path
  (`rawllm_simd_dispatch.hpp`) regardless. Only F32 weights are covered;
  quantized (Q4_0/Q8_0/K-quant) tensors would need a per-row dequant step
  before this shader could touch them, which is real added cost the
  CPU-side fused int8 kernels don't pay — deciding when that trade-off is
  worth it, and actually routing `proj_all_positions()` through the GPU for
  it, is the next piece of work here, not something this header claims to
  do yet.

**NCTR** Nctr is still not wired in and not tested. NCTRloader currently
does not replace NEON's GGUFloader.

- Single-operator, no independent security audit yet (see `LICENSE`).

## Out of scope

Two different kinds of "not here": things this project will never be, and
things that are just sequenced after other work. Mixing those together is
how a focused runtime slowly turns into a second llama.cpp — writing the
line down now is meant to stop that before it starts.

**Permanently out of scope — not planned, not "later":**
- **Continuous batching / paged attention** (vLLM-style block-table KV
  cache, prefix sharing across concurrent requests). This isn't a kernel
  you add, it's request scheduling and memory pooling — architecturally
  most of what makes a multi-tenant serving engine a multi-tenant serving
  engine. NEON is a single-user runtime (one companion, one conversation
  at a time); taking this on would roughly double the codebase's
  conceptual surface for a capability this project doesn't need.
- **Serving many models, or many requests, on one GPU.** Same reasoning —
  NEON is not trying to be an inference server.
- **Mixture-of-Experts.** NANITY is a fixed, single dense architecture on
  purpose (see the spec) — no per-architecture branching, and MoE routing
  is exactly the kind of per-architecture special case this project
  exists to avoid.
- **GPU flash-attention** (finishing the composable_kernel `DeviceMHAFwd`
  stub in `rawllm_rocm.hpp`). Pulling in CK as a real dependency and
  validating a fused attention kernel against real hardware is its own
  project, not an incremental addition to this one. An online-softmax
  CPU-side attempt at part of this was tried and reverted after
  benchmarking showed it regressed at long context (doubled `exp()` calls
  per position outweighed the memory-traffic savings it was meant to
  provide) — see the git history on `rawllm_forward.hpp`'s attention loop
  if you want the specifics before trying this again.
- **A Vulkan quantized (Q4_0/Q8_0/K-quant) compute shader.** Vulkan isn't
  this project's primary GPU target (ROCm is) and int8 shader support
  isn't universal across Vulkan implementations — adding a kernel that
  can't be validated on real hardware here is exactly the kind of surface
  that rots silently.

**Deferred, not ruled out — sequenced behind validating what's already
here:**
- **K-quant (Q4_K/Q5_K/Q6_K) fused int8 kernels.** Only Q4_0/Q8_0 get the
  fast fused path today; K-quants fall through to full dequantize-to-F32.
  Deliberately not adding this yet: Q4_0 itself hasn't been proven to
  produce reliably coherent, instruction-following output end-to-end (see
  Known limitations above — TinyLlama Q4_0 currently produces valid
  English that doesn't follow the prompt, and it isn't confirmed whether
  that's a quantization/model accuracy issue or a runtime bug). Adding
  more quantization formats on top of an unproven one is how you end up
  debugging two things at once instead of one. Once Q4_0/Q8_0 are
  confirmed producing coherent, prompt-following output on a real
  instruct model, K-quants are the natural next step — not before.
- **ROCm and Vulkan, generally.** Both compile and (for ROCm) have run
  real inference on an MI300X — see the notes above for exactly what's
  been verified on real hardware vs. compile-tested only. Neither is
  "done," but both are active, not abandoned.

## CPU SIMD backends

`rawllm_forward.hpp`'s dot-product / fused-quantized-dot kernels are split
into their own headers so a specific ISA backend can be built, read, or
benchmarked in isolation instead of hunting through the forward pass:

- `rawllm_simd_scalar.hpp` — portable fallback, always available, and the
  correctness reference every vectorized backend is checked against.
- `rawllm_simd_avx2.hpp` — AVX2+FMA kernels (auto-selected whenever the
  build already has `__AVX2__`, e.g. via `-mavx2 -mfma`).
- `rawllm_simd_avx512.hpp` — AVX-512 kernels, with an `AVX512-VNNI` fast
  path (`_mm256_dpbusd_epi32`, applied per 32-byte block) when the build
  also has `-mavx512vnni` **and** `-mavx512vnni`'s prerequisite
  `-mavx512vl` (VNNI's 256-bit instruction forms need VL to target
  YMM registers — `-mavx512vnni` alone does not imply it, and previously
  a `-mavx512vnni`-only build failed to compile entirely rather than
  falling back; that's fixed, so that combination now degrades to the
  plain AVX-512F path instead of failing the build). Opt-in via
  `-DUSE_AVX512` (see below) — AVX-512 is never auto-selected, since a
  build box having it doesn't guarantee every machine the binary runs on
  does too.
- `rawllm_simd_dispatch.hpp` — picks AVX-512 > AVX2 > scalar at compile
  time and exposes the winner as `simd::dot_f32` / `simd::axpy_f32` /
  `simd::dot_q4_0_q8_0` / `simd::dot_q8_0_q8_0`; this is the only one of
  the four `rawllm_forward.hpp` actually includes.

Build any translation unit that includes `rawllm_simd_dispatch.hpp` with
`-DRAWLLM_SIMD_SELFTEST` to get `simd::run_selftest()`, which checks
whichever backend got selected against the scalar reference over
randomized Q4_0/Q8_0 blocks and plain dot/axpy inputs — a development/CI
aid, not compiled in by default.

## Build

CPU-only (no GPU needed):
```
g++ -std=c++20 -O2 -pthread NEON-3.cpp -o neon
```
AVX2 (auto-detected from the compiler flag, no extra `-D` needed):
```
g++ -std=c++20 -O2 -pthread -mavx2 -mfma NEON-3.cpp -o neon
```
AVX-512 (opt-in; add `-mavx512bw -mavx512vnni -mavx512vl` for the widened
int8 path — `-mavx512vl` is required alongside `-mavx512vnni`, see above):
```
g++ -std=c++20 -O2 -pthread -mavx512f -mavx512bw -mavx512vnni -mavx512vl -DUSE_AVX512 NEON-3.cpp -o neon
```
With the optional idle-loop features:
```
g++ -std=c++20 -O2 -pthread -DNANITY_ENABLE_IDLE_LOOP NEON-3.cpp -o neon
```
ROCm build (GPU dispatch — see the "ROCm GPU dispatch" note above; not
required for CPU inference). Needs a HIP/ROCm toolchain (`hipcc`) and
linking against `rocblas`/`amdhip64` — not compile-tested in this
environment (no ROCm SDK available here), so the exact flags your ROCm
install needs may differ from a generic example:
```
hipcc -std=c++20 -O2 -pthread -DUSE_ROCBLAS -DUSE_ROCM NEON-3.cpp -lrocblas -lamdhip64 -o neon
```

Vulkan build (experimental, see above — needs the Vulkan SDK loader/headers
and a compiled shader):
```
glslangValidator -V shaders/matvec_f32.comp -o shaders/matvec_f32.spv
g++ -std=c++20 -O2 -pthread -DUSE_VULKAN NEON-3.cpp -lvulkan -o neon
```

## Quick start

```
# 1. Check compatibility before converting anything
python3 nanity_convert.py --detect-only your-model.gguf

# 2. Convert (if the report is clean, or you've reviewed what --drop-*-anyway loses)
python3 nanity_convert.py your-model.gguf your-model-nanity.gguf --quant Q4_0

# 3. Sanity check: loads and passes spec validation, no generation
./neon --model your-model-nanity.gguf --probe

# 4. Generate
./neon --model your-model-nanity.gguf --prompt "Explain photosynthesis in one sentence."
```

`--interactive` runs a persistent stdin/stdout loop (model stays loaded
between prompts, accepts JSON `{"messages": [...]}` for chat-template
formatting and per-request sampling params). `NANITY_DEBUG_LOGITS=1`
dumps the top-10 first-token candidates before sampling — useful for
diagnosing a bad conversion vs. a runtime bug.

## Training your own NANITY model

`train_nanity_fixed.py` trains `modeling_nanity.py` (the reference
PyTorch implementation of the spec) and exports via `export_gguf()` /
`export_nctr()`. `prepare_data.py` and `modeling_nanity.py` are currently
set up for the first model, **Nectar 8** (not yet trained — training
hasn't started). Anyone can reuse the same data pipeline and architecture
to train their own model against the same spec.

This isn't a book you need to read cover to cover — the tutorial content
in this repo is meant to be worked through hands-on, not studied first.

## Repository layout

| File | What it is |
|---|---|
| `NEON-3.cpp` | The runtime itself — CLI, generation loop, tokenizer, model-format dispatch |
| `rawllm_loader.hpp` | GGUF loader |
| `rawllm_nctr_loader.hpp` | `.nctr` loader (parsing works; not yet wired into inference) |
| `rawllm_forward.hpp` | Transformer forward pass — attention, RoPE, GQA, SwiGLU |
| `rawllm_rocm.hpp` | ROCm/HIP backend (optional) |
| `rawllm_common.hpp`, `rawllm_util.hpp`, `rawllm_json.hpp` | Shared utilities |
| `nectar_diskmem.hpp`, `nectar_vision.hpp`, `nectar_splice.hpp` | Idle-loop companion-overlay modules (optional feature) |
| `nanity_convert.py` | Converts third-party GGUF models into NANITY-conformant GGUF |
| `nanity_data_format.py` | `.nctr` container format definitions |
| `train_nanity_fixed.py` | Training pipeline + `export_gguf()` / `export_nctr()` |
| `modeling_nanity.py` | Reference PyTorch implementation of the NANITY architecture |
| `prepare_data.py` | Training data preparation |
| `test_nctr_loader.cpp` | Standalone `.nctr` loader smoke test |
| `NANITY_ARCHITECTURE_SPEC.md` | The spec itself — read this for the full technical contract |
| `nanity.html` | Project website |

## License

Source-available, - see [`LICENSE`](./LICENSE) for exactly what's currently permitted
(download, local compile, testing, bug reports).
