# Security Policy

NEON is beta, source-available, solo-maintained software. This policy sets
expectations accordingly — there's no security team, no SLA, and no bug
bounty, just a best-effort process for getting real issues fixed.

## Reporting a vulnerability

Email **nanityofficial@gmail.com** with a subject line starting
`SECURITY:`. Please include:

- Which file/function and, if it's backend-specific (CPU scalar/AVX2/
  AVX-512, ROCm, Vulkan), which one.
- A minimal repro — a crafted GGUF/`.nctr` file, a specific model
  config, or a command line that triggers it. If the file needed to
  reproduce it is large or sensitive, a description of how to construct
  one is fine.
- What you'd expect to happen vs. what actually happens (crash, memory
  corruption, incorrect output silently accepted as valid, etc).

You'll get an acknowledgment as soon as I see it — realistically that
could be anywhere from same-day to a few days, since this is a solo
project. Please don't open a public issue for anything you believe is
actually exploitable (memory corruption, arbitrary code execution,
anything beyond "produces wrong numbers") until there's been a chance to
land a fix; a plain bug that just produces incorrect output is fine to
file as a normal issue.

## Scope

The realistic attack surface here is **loading untrusted model files**,
not the inference math itself:

- **GGUF/`.nctr` parsing** (`rawllm_loader.hpp`, `rawllm_nctr_loader.hpp`)
  — a hand-rolled binary format parser reading attacker-controlled input
  is exactly the kind of code that tends to have out-of-bounds reads,
  integer overflows in size calculations, or unchecked tensor
  shape/offset math. If you're going to look anywhere first, look here.
- **The `.nctr` audit/signing spec** (`nctr_audit_spec_v1.md`) — if you
  find a way to forge a signature, bypass the transparency log, or make
  an unsigned/tampered file validate as signed, that's squarely in scope.
- **Quantization/dequantization kernels** — mis-sized reads/writes from
  a malformed block layout (bad `nb`/block-size arithmetic, especially
  in the less-exercised paths — K-quants, the ROCm and Vulkan backends)
  rather than anything in the well-fuzzed-by-normal-use Q4_0/Q8_0 fast
  paths.

**Out of scope / not treated as a security issue on its own:**
- A model file producing garbage *output* (wrong tokens, incoherent
  text) without a memory-safety issue — that's a correctness bug, file
  it as a normal issue.
- Anything specific to the ROCm or Vulkan backends being unable to run
  at all in an environment without the matching SDK/driver — that's a
  build/environment issue, not a vulnerability. (Their code paths are
  in scope for the "unsafe if it *did* run" question above; "doesn't
  run without a GPU" itself isn't.)
- Resource exhaustion from deliberately requesting an enormous
  `--ctx-len`/model size on hardware that can't hold it — this is
  inference software you run against models you chose to load, not a
  server accepting arbitrary requests from untrusted parties.

## Disclosure

No formal embargo process — for anything serious, I'd ask for a
reasonable window to land and release a fix before it's made public,
but there's no fixed number of days promised here given this is a solo
effort. Credit (name, handle, or anonymous — your call) in the fix's
commit message and/or README unless you'd rather not be mentioned at
all.
