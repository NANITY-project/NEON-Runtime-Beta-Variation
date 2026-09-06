# Contributing to NEON

This is a solo-maintained, beta-stage project — contributions are
welcome, but please read this before sending a PR, especially the
"backends I can't personally verify" section below, since that's where
review will be most careful.

Questions before starting on something bigger than a small fix? Email
**nanityofficial@gmail.com** — cheaper than writing a PR that doesn't
fit the project's direction.

## Before you start

- **Check `NANITY_ARCHITECTURE_SPEC.md` first** if your change touches
  anything about the model format or tensor contract. NEON deliberately
  loads exactly one fixed architecture rather than guessing at
  arbitrary GGUF files the way llama.cpp does — changes that would
  require re-introducing per-architecture branching are unlikely to be
  a fit, even if they're otherwise good code.
- **For anything nontrivial, open an issue before a PR.** Saves both of
  us the time of a large PR that doesn't fit, especially for anything
  touching the forward pass, KV cache, or a SIMD backend.

## Build & test

See the README's Build section for the actual compiler invocations per
backend (CPU scalar/AVX2/AVX-512, ROCm, Vulkan). Before sending a PR
that touches `rawllm_forward.hpp` or anything under `rawllm_simd_*.hpp`:

1. **Run the SIMD self-test.** Any translation unit that includes
   `rawllm_simd_dispatch.hpp` can be built with `-DRAWLLM_SIMD_SELFTEST`
   to get `simd::run_selftest()` — it checks every backend's block
   kernels against the plain scalar reference over thousands of
   randomized inputs. This is the fastest way to catch a wrong
   intermediate result before it shows up as subtly-wrong generated
   text three layers into a forward pass.
2. **Build and run `--probe` against a real converted model** if you
   have one handy (`nanity_convert.py --detect-only` on any GGUF file
   will tell you if it's convertible). A file that loads and passes
   validation but was never actually run through generation is a
   different failure mode than one that crashes outright — `--probe`
   only catches the first kind.
3. **State what you tested in the PR description**: which backend(s),
   on what CPU (does it have AVX-512? VNNI?), whether you ran a real
   model through `--probe`/generation or only the self-test. This
   project's own commit history is full of "verified numerically, not
   just compiled" vs. "compile-tested only, not run on real hardware"
   distinctions for exactly this reason — carrying that same honesty
   into PR descriptions makes review much faster than a bare "LGTM,
   works on my machine."

## Backends I can't personally verify

Be aware of this asymmetry before assuming "it compiles" means "it
works":

- **CPU (scalar/AVX2/AVX-512/AVX-512-VNNI)**: fully verified — this is
  what's actually run end-to-end against real converted models.
  Changes here get held to the highest bar, and should come with
  self-test results.
- **ROCm (`rawllm_rocm.hpp`, the `gpu::` dispatch in
  `rawllm_forward.hpp`)**: compile-tested (including against real
  headers where available) but not run on an actual AMD GPU by me.
  **If you have real ROCm hardware, this is one of the most valuable
  places to contribute** — a PR that says "ran this on an MI300X,
  diffed logits against the CPU path, here's the result" carries a lot
  more weight than the code alone can.
- **Vulkan (`rawllm_vulkan.hpp`)**: same situation — compiled against
  real Vulkan headers and the shader compiled to real SPIR-V, but never
  actually dispatched on a GPU. Marked experimental in its own header
  comment for exactly this reason.

If you're fixing a bug in one of these two and can't test it on real
hardware either, say so explicitly in the PR — that's fine, just don't
let it get merged as if it were CPU-verified.

## Code style

Match what's already here rather than a separate style guide:

- **Comments explain *why*, not just *what*** — especially for
  anything performance-related. If you tried an approach and reverted
  it because it measured slower, that's exactly the kind of thing worth
  leaving a comment about (several places in this codebase do this on
  purpose) rather than silently discarding the failed attempt — it
  saves the next person from re-discovering the same dead end.
- **Measure before claiming a performance win.** "Should be faster"
  isn't a substitute for a benchmark, and correctness-only testing
  isn't a substitute for a performance one — this repo has more than
  one comment documenting an optimization that was *reverted* after
  benchmarking showed it was actually a regression. That's the norm to
  match, not an embarrassment to avoid repeating.
- **New quant/backend code paths need the same fused-kernel treatment
  the existing Q4_0/Q8_0 paths got** where it's a reasonable fit — full
  dequant-then-dot is the fallback, not the target, for anything that's
  going to be a common case.

## License

By contributing, you agree your contribution is licensed under this
project's license (AGPLv3 — see `LICENSE`).
