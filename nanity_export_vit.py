#!/usr/bin/env python3
"""
nanity_export_vit.py — convert a HuggingFace CLIP / SigLIP style vision tower
(+ optional LLaVA / Idefics3 connector) into a NEON-loadable GGUF
(general.architecture = "nanity-vit", read by rawllm_vit.hpp).

Presets (state-dict naming + the processing conventions that go with them):

  siglip      SiglipVisionModel / SmolVLM / Idefics3 tower. No CLS, gelu_tanh,
              post-LN, mean=std=0.5, resize-squash (bicubic).
  clip        CLIPVisionModel tower alone. CLS, quick_gelu, pre-LN, post-LN,
              CLIP mean/std, shortest-side resize + center crop.
  llava       CLIP tower + LLaVA-1.5 `multi_modal_projector` (linear_1, GELU, linear_2):
              uses hidden_states[-2] (feature_layer=-2, no post-LN), drops CLS,
              pad-to-square preprocessing (mean-colour fill).

--pixel-shuffle S   (siglip preset; Idefics3/SmolVLM): merge SxS neighbouring patch
                    tokens before the connector (fewer tokens for a small LLM).
--connector-prefix  key prefix of the connector weights if present.

Read the tensors from a .safetensors file (--src) — no torch needed. The
exporter VERIFIES shapes against the config it derives and refuses on mismatch
instead of guessing.

    pip install safetensors gguf numpy --break-system-packages
    python3 nanity_export_vit.py --src model.safetensors --preset siglip \\
        --image-size 384 --patch-size 14 --heads 16 out.gguf [--dtype f16] [--pixel-shuffle 3]

Note: `--image-size/--patch-size/--heads` are required because they are not
recoverable from tensor shapes alone (heads, patch) / are config-only (image
size, when position embeddings are interpolated). Layer count, widths and FFN
size ARE derived from the tensors.
"""
import argparse, sys
import numpy as np
try:
    import gguf
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors gguf numpy --break-system-packages")

CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]


def find_prefix(keys, suffix):
    c = [k for k in keys if k.endswith(suffix)]
    if len(c) != 1:
        sys.exit(f"cannot uniquely locate '*{suffix}' in checkpoint (found {len(c)}): {c[:4]}")
    return c[0][: -len(suffix)]


def export(T, a, dst):
    keys = list(T.keys())
    vp = find_prefix(keys, "embeddings.patch_embedding.weight")          # e.g. 'vision_model.' / 'model.vision_model.'
    pw = T[vp + "embeddings.patch_embedding.weight"]                     # (d, 3, p, p)
    d, ch, p, p2 = pw.shape
    assert ch == 3 and p == p2 == a.patch_size, f"patch embedding shape {pw.shape} vs --patch-size {a.patch_size}"
    pos = T[vp + "embeddings.position_embedding.weight"]                 # (n_pos, d)
    use_cls = (vp + "embeddings.class_embedding") in T
    grid = a.image_size // p
    n_pos = grid * grid + (1 if use_cls else 0)
    assert pos.shape == (n_pos, d), f"position_embedding {pos.shape} != ({n_pos},{d}); image-size mismatch or interpolated checkpoint (not supported)"
    nl = 1 + max(int(k[len(vp + "encoder.layers."):].split(".")[0]) for k in keys if k.startswith(vp + "encoder.layers."))
    ff = T[vp + "encoder.layers.0.mlp.fc1.weight"].shape[0]
    assert d % a.heads == 0

    clip_like = a.preset in ("clip", "llava")
    w = gguf.GGUFWriter(dst, "nanity-vit")
    U = w.add_uint32
    U("nanity.vit.image_size", a.image_size); U("nanity.vit.patch_size", p); U("nanity.vit.n_embd", d)
    U("nanity.vit.n_layer", nl); U("nanity.vit.n_head", a.heads); U("nanity.vit.n_ff", ff); U("nanity.vit.n_channels", 3)
    U("nanity.vit.use_cls", int(use_cls))
    pre_ln = (vp + "pre_layrnorm.weight") in T
    post_ln = (vp + "post_layernorm.weight") in T and a.preset != "llava"
    U("nanity.vit.pre_ln", int(pre_ln)); U("nanity.vit.post_ln", int(post_ln))
    w.add_float32("nanity.vit.ln_eps", 1e-5 if clip_like else 1e-6)
    w.add_string("nanity.vit.act", "gelu_quick" if clip_like else "gelu_tanh")   # CLIP/LLaVA towers: quick_gelu; SigLIP: gelu_pytorch_tanh
    U("nanity.vit.drop_cls", 1)
    w.add_int32("nanity.vit.feature_layer", -2 if a.preset == "llava" else -1)
    U("nanity.vit.pixel_shuffle", a.pixel_shuffle)
    w.add_string("nanity.vit.resample", "bicubic")
    w.add_string("nanity.vit.preprocess", {"siglip": "squash", "clip": "shortest_crop", "llava": "pad_square"}[a.preset])

    wdt = np.float16 if a.dtype == "f16" else np.float32
    out = {}
    def mat(n, x): out[n] = np.ascontiguousarray(x, wdt)
    def vec(n, x): out[n] = np.ascontiguousarray(x, np.float32)

    mat("v.patch.weight", pw.reshape(d, -1))          # conv weight flattened (c, py, px) == runtime im2col order
    if (vp + "embeddings.patch_embedding.bias") in T: vec("v.patch.bias", T[vp + "embeddings.patch_embedding.bias"])
    else: vec("v.patch.bias", np.zeros(d))
    if use_cls: vec("v.cls", T[vp + "embeddings.class_embedding"])
    vec("v.pos", pos)
    if pre_ln: vec("v.pre_ln.weight", T[vp + "pre_layrnorm.weight"]); vec("v.pre_ln.bias", T[vp + "pre_layrnorm.bias"])
    if post_ln: vec("v.post_ln.weight", T[vp + "post_layernorm.weight"]); vec("v.post_ln.bias", T[vp + "post_layernorm.bias"])
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_like else ([0.5] * 3, [0.5] * 3)
    vec("v.image_mean", mean); vec("v.image_std", std)
    for i in range(nl):
        b = f"{vp}encoder.layers.{i}."; o = f"v.blk.{i}."
        for src, dstn in (("layer_norm1", "ln1"), ("layer_norm2", "ln2")):
            vec(o + dstn + ".weight", T[b + src + ".weight"]); vec(o + dstn + ".bias", T[b + src + ".bias"])
        for src, dstn in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("out_proj", "attn_output")):
            mat(o + dstn + ".weight", T[b + "self_attn." + src + ".weight"]); vec(o + dstn + ".bias", T[b + "self_attn." + src + ".bias"])
        for src, dstn in (("fc1", "ffn_up"), ("fc2", "ffn_down")):
            mat(o + dstn + ".weight", T[b + "mlp." + src + ".weight"]); vec(o + dstn + ".bias", T[b + "mlp." + src + ".bias"])

    # connector
    proj = "none"
    if a.preset == "llava":
        cp = find_prefix(keys, "linear_1.weight"); proj = "mlp2"
        mat("mm.0.weight", T[cp + "linear_1.weight"]); vec("mm.0.bias", T[cp + "linear_1.bias"])
        mat("mm.2.weight", T[cp + "linear_2.weight"]); vec("mm.2.bias", T[cp + "linear_2.bias"])
        w.add_string("nanity.vit.proj_act", "gelu")
    elif a.connector_prefix:
        k = a.connector_prefix + ".weight"; proj = "linear"
        mat("mm.0.weight", T[k])
        vec("mm.0.bias", T[a.connector_prefix + ".bias"] if (a.connector_prefix + ".bias") in T else np.zeros(T[k].shape[0]))
        assert T[k].shape[1] == d * a.pixel_shuffle ** 2, f"connector in_features {T[k].shape[1]} != d*s^2 = {d * a.pixel_shuffle ** 2}"
    w.add_string("nanity.vit.projector", proj)

    for n, x in out.items(): w.add_tensor(n, x)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {dst}: {a.preset} d={d} layers={nl} heads={a.heads} ff={ff} tokens={n_pos} proj={proj} dtype={a.dtype}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dst"); ap.add_argument("--src", required=True)
    ap.add_argument("--preset", choices=["siglip", "clip", "llava"], required=True)
    ap.add_argument("--image-size", type=int, required=True); ap.add_argument("--patch-size", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f16")
    ap.add_argument("--pixel-shuffle", type=int, default=1)
    ap.add_argument("--connector-prefix", default="", help="e.g. model.connector.modality_projection.proj")
    a = ap.parse_args()
    with safe_open(a.src, "np") as f:
        T = {k: f.get_tensor(k).astype(np.float32) for k in f.keys()}
    export(T, a, a.dst)
