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

- **Bias support (spec-v1.1, opt-in via `nanity.use_bias`).** A file with
  `nanity.use_bias=false` (or the key absent — every existing NANITY v1
  file) is bias-free, byte-identical to before this was added. A file with
  `nanity.use_bias=true` must carry a matching 1-D F32 `.bias` tensor
  alongside every attn Q/K/V/O and FFN gate/up/down projection (plus
  `output.bias` if the output projection is untied) — `validate_config()`
  checks presence, dtype, and shape, and fails loudly (naming the exact
  tensor) if the file's declared flag and its actual tensors disagree in
  either direction.
  Previously, `validate_config()` implemented this check, but
  `rawllm_forward.hpp` never actually read or added a single bias tensor —
  a converted-and-validated bias model would load successfully and then
  silently run with every bias term dropped, which is why Qwen2 conversion
  testing wasn't producing usable output. Fixed and verified end-to-end: a
  synthetic GGUF file with `use_bias=true` and all-zero bias tensors now
  produces output byte-identical to the same weights with no bias tensors
  at all (max logit diff 0), and the same file with nonzero bias produces
  clearly different output (max logit diff ~2, vs a ~1e-4 numerical-noise
  floor) — confirming bias is actually read from the file and flows
  through every projection (CPU dequant path, both fused Q4_0/Q8_0 int8
  paths, and the ROCm GPU dispatch path below), not just validated and
  discarded. `nanity_convert.py --drop-nonzero-bias-anyway` (lossy) is no
  longer the only option for bias-having source models; a converter path
  that preserves bias into `nanity.use_bias=true` output is the natural
  follow-up, not done here.
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
- **ROCm GPU dispatch** (`rawllm_forward.hpp`'s `gpu::` namespace, gated
  behind `-DUSE_ROCBLAS` + `__HIP_PLATFORM_AMD__`/`USE_ROCM`): weight
  tensors are dequantized and uploaded to the device once (keyed by the
  tensor's stable mmap pointer) and reused across every subsequent token,
  with per-projection GEMM dispatch above a size threshold so small
  tensors stay on the CPU int8 fast path. **Not compiled/tested on real
  ROCm hardware in the environment this was written in** — needs an
  on-device run on an actual MI300X-class box (build with `-DUSE_ROCBLAS`
  and diff a few logits against the CPU path) before trusting it for real
  inference.
- Single-operator, no independent security audit yet (see `LICENSE`).

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
