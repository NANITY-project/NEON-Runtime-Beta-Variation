#!/usr/bin/env python3
"""
nanity_export_mvae.py — convert a Motion-VAE safetensors checkpoint (the format
shipped by electronicarts/character-motion-vaes) into a NEON-loadable GGUF
(general.architecture = "nanity-mvae", read by rawllm_mvae.hpp).

Handles the checkpoint classes whose inference graph NEON implements:
    PoseMixtureVAE            -> kind "pose_mixture_vae"   (MoE decoder + gate)
    PoseVAE                   -> kind "pose_vae"           (plain decoder)
    PoseVAEController/Policy  -> kind "mlp"                (actor: ReLU..., tanh)
Anything else (AutoEncoder, PoseVQVAE, PoseMixtureSpecialistVAE) is refused with
an explicit message instead of being half-converted.

Layout decision (this is what makes the runtime fast): each MoE layer's
(E, in, out) weight is exported as ONE stacked matrix with E*out rows and `in`
contiguous columns, i.e. GGUF shape [in, E*out] — a plain Linear. rawllm_mvae.hpp
evaluates all experts with a single GEMV and blends the outputs (mathematically
identical to blending the weights first; see the header comment there).

Usage:
    python3 nanity_export_mvae.py posevae_c1_e6_l32.safetensors out.gguf [--dtype f32|f16]
    pip install safetensors gguf numpy --break-system-packages
"""
import argparse, json, sys
import numpy as np

try:
    import gguf
    from safetensors import safe_open
except ImportError:
    sys.exit("Missing deps: pip install safetensors gguf numpy --break-system-packages")


def load(path):
    with safe_open(path, "np") as f:
        meta = f.metadata() or {}
        cfg = json.loads(meta.get("config", "{}"))
        tensors = {k: f.get_tensor(k).astype(np.float32) for k in f.keys()}
    return cfg, tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32",
                    help="storage type for weight matrices (biases/norm stats stay F32). "
                         "f16 is lossy and therefore opt-in.")
    a = ap.parse_args()
    cfg, T = load(a.src)
    cls = cfg.get("class")
    wdt = np.float32 if a.dtype == "f32" else np.float16

    w = gguf.GGUFWriter(a.dst, "nanity-mvae")
    out_t = {}   # name -> array

    def mat(name, arr):   # weight matrix [N, K] -> GGUF [K, N]
        out_t[name] = np.ascontiguousarray(arr, dtype=wdt)
    def vec(name, arr):
        out_t[name] = np.ascontiguousarray(arr, dtype=np.float32)

    if cls in ("PoseMixtureVAE", "PoseVAE"):
        fs, ls = int(cfg["frame_size"]), int(cfg["latent_size"])
        nc, nf = int(cfg["num_condition_frames"]), int(cfg["num_future_predictions"])
        mode = cfg.get("normalization_mode", "zscore")
        w.add_string("nanity.mvae.kind", "pose_mixture_vae" if cls == "PoseMixtureVAE" else "pose_vae")
        w.add_uint32("nanity.mvae.frame_size", fs); w.add_uint32("nanity.mvae.latent_size", ls)
        w.add_uint32("nanity.mvae.num_condition_frames", nc); w.add_uint32("nanity.mvae.num_future_predictions", nf)
        w.add_string("nanity.mvae.normalization", mode)
        if mode == "zscore":
            vec("norm.avg", T["data_avg"]); vec("norm.std", T["data_std"])
        elif mode == "minmax":
            vec("norm.min", T["data_min"]); vec("norm.max", T["data_max"])
        if cls == "PoseMixtureVAE":
            E = int(cfg["num_experts"]); w.add_uint32("nanity.mvae.num_experts", E)
            hidden = T["decoder.w0"].shape[2]; w.add_uint32("nanity.mvae.hidden", hidden)
            w.add_uint32("nanity.mvae.gate_hidden", T["decoder.gate.0.weight"].shape[0])
            for i in range(3):
                W = T[f"decoder.w{i}"]            # (E, in, out)
                assert W.shape[0] == E
                mat(f"dec.l{i}.weight", W.transpose(0, 2, 1).reshape(E * W.shape[2], W.shape[1]))
                vec(f"dec.l{i}.bias", T[f"decoder.b{i}"].reshape(-1))   # (E,out) -> E*out, expert-major
            for gi in (0, 2, 4):
                mat(f"dec.gate.{gi}.weight", T[f"decoder.gate.{gi}.weight"])
                vec(f"dec.gate.{gi}.bias", T[f"decoder.gate.{gi}.bias"])
            enc = {"enc.fc1": "encoder.fc1", "enc.fc2": "encoder.fc2", "enc.mu": "encoder.mu", "enc.logvar": "encoder.logvar"}
        else:
            hidden = T["fc4.weight"].shape[0]; w.add_uint32("nanity.mvae.hidden", hidden)
            for dst, src in (("dec.fc4", "fc4"), ("dec.fc5", "fc5"), ("dec.out", "out")):
                mat(dst + ".weight", T[src + ".weight"]); vec(dst + ".bias", T[src + ".bias"])
            enc = {"enc.fc1": "fc1", "enc.fc2": "fc2", "enc.mu": "mu", "enc.logvar": "logvar"}
        for dst, src in enc.items():
            mat(dst + ".weight", T[src + ".weight"]); vec(dst + ".bias", T[src + ".bias"])

    elif cls in ("PoseVAEController", "PoseVAEPolicy"):
        prefix = "actor.actor." if cls == "PoseVAEPolicy" else "actor."
        idx = sorted({int(k[len(prefix):].split(".")[0]) for k in T if k.startswith(prefix)})
        w.add_string("nanity.mvae.kind", "mlp"); w.add_uint32("nanity.mvae.n_layers", len(idx))
        w.add_string("nanity.mvae.act", "relu"); w.add_string("nanity.mvae.final_act", "tanh")
        for n, i in enumerate(idx):
            mat(f"mlp.{n}.weight", T[f"{prefix}{i}.weight"]); vec(f"mlp.{n}.bias", T[f"{prefix}{i}.bias"])
        w.add_uint32("nanity.mvae.observation_dim", int(cfg["observation_dim"]))
        w.add_uint32("nanity.mvae.action_dim", int(cfg["action_dim"]))
        w.add_uint32("nanity.mvae.frame_skip", int(cfg.get("frame_skip", 1)))
    else:
        sys.exit(f"Unsupported checkpoint class {cls!r}. Supported: PoseMixtureVAE, PoseVAE, "
                 f"PoseVAEController, PoseVAEPolicy.")

    for name, arr in out_t.items():
        w.add_tensor(name, arr)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.dst}: class={cls} tensors={len(out_t)} dtype={a.dtype}")


if __name__ == "__main__":
    main()
