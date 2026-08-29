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

> **Status: beta, source-available (not yet open source).** See
> [`License`](./License) for current terms during the prototyping phase.

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

- **Bias terms aren't supported.** NANITY v1 has no bias tensors anywhere
  in the spec. Models that rely on them (e.g. Qwen2's nonzero attention
  QKV biases) need `nanity_convert.py --drop-nonzero-bias-anyway`, which
  is genuinely lossy — expect degraded output, not a faithful conversion,
  for those specific models. This is a spec limitation, not a runtime bug.
- **Vulkan backend**: not started. CPU and ROCm (`rawllm_rocm.hpp`) exist
  today.
- Single-operator, no independent security audit yet (see `License`).

## Build

CPU-only (no GPU needed):
```
g++ -std=c++20 -O2 -pthread NEON-3.cpp -o neon
```
With the optional idle-loop features:
```
g++ -std=c++20 -O2 -pthread -DNANITY_ENABLE_IDLE_LOOP NEON-3.cpp -o neon
```
ROCm build: see `rawllm_rocm.hpp` (gated behind `__HIP_PLATFORM_AMD__`/
`USE_ROCM`; not required for CPU inference).

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

Source-available, restrictive terms during the current beta/prototyping
phase — see [`License`](./License) for exactly what's currently permitted
(download, local compile, testing, bug reports). Terms are expected to
open up as the project matures.
