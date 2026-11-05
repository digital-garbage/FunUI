"""Which of a LoRA's weights land in this model, and which cannot.

ComfyUI converts the naming dialects (lora_up / lora_A / PEFT); it does not unwrap a file whose
keys sit under a training wrapper. v4 measured it: an H3 adapter keyed `base_model.model.dit.*`
matched 0 of 532 keys and did nothing, silently. Every known wrapper is tried and the one matching
the most weights wins. A pair that matches by name but not by shape is dropped: ComfyUI reshapes
any delta with the right element count, so a wrong-shaped one merges scrambled (all-NaN mid-render),
and a full-width LoRA on a curve-form H3 checkpoint churned ~25 GB failing per key.
"""

import re

import torch

import comfy.lora
import comfy.lora_convert

# (prefix, replacement). Replacing, not only stripping: ComfyUI's key map has `diffusion_model.<path>`
# and no bare form, so `base_model.model.dit.blocks.0...` must become `diffusion_model.blocks.0...`.
REKEY = (
    ("transformer.", ""), ("diffusion_model.", ""), ("model.diffusion_model.", ""),
    ("base_model.model.", ""), ("base_model.model.dit.", ""), ("base_model.model.diffusion_model.", ""),
    ("lora_model.", ""), ("net.", ""),
    ("base_model.model.dit.", "diffusion_model."), ("base_model.model.", "diffusion_model."),
    ("dit.", "diffusion_model."), ("model.dit.", "diffusion_model."), ("transformer.", "diffusion_model."),
)


_DIFFUSERS = re.compile(r"(^|\.)(transformer_blocks|refiner_blocks)\.\d+\.")
_PART = re.compile(r"^(?P<mod>.+?)(?P<tail>\.(?:lora_[AB]\.weight|lora_[AB]|alpha))$")
_SUFFIX = (("attn.to_out.0", "attn.out_proj"), ("ff.net.0.proj", "mlp.fc1"), ("ff.net.2", "mlp.fc2"))
_QKV = ("to_q", "to_k", "to_v")


def from_diffusers(lora):
    """-> (lora, extra key map). A LoRA written against the Diffusers layout of the H3 transformer, renamed to the one
    ComfyUI loads, or the LoRA unchanged with {} when it is not in that layout.

    Three things differ. The names (`transformer_blocks.N` / `refiner_blocks.N` are `blocks.N` / `token_refiner.blocks.N`;
    `to_out.0`, `ff.net.0.proj`, `ff.net.2` are `out_proj`, `mlp.fc1`, `mlp.fc2`). Q, K and V are three Linears there and
    ONE fused `qkv_proj` here, so each is aimed at its own rows of that weight (an offset, which costs no extra rank).
    And the gated MLP: Diffusers stores the up half before the gate half, ComfyUI the gate half first, so the rows of
    the `fc1` up-matrix swap halves. Without the swap the file loads, matches, and quietly does the wrong thing."""
    if not any(_DIFFUSERS.search(k) for k in lora):
        return lora, {}
    out, extra, sizes = {}, {}, {}
    for k, v in lora.items():
        m = _PART.match(k[len("transformer."):] if k.startswith("transformer.") else k)
        if not m:
            out[k] = v
            continue
        mod, tail = m.group("mod"), m.group("tail")
        mod = mod.replace("transformer_blocks.", "blocks.").replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
        for old, new in _SUFFIX:
            if mod.endswith(old):
                mod = mod[: -len(old)] + new
                break
        if mod.endswith(".mlp.fc1") and tail.replace(".weight", "") == ".lora_B":
            half = v.shape[0] // 2
            v = torch.cat((v[half:], v[:half]), 0)
        part = next((q for q in _QKV if mod.endswith(f".attn.{q}")), None)
        if part is not None:
            base = mod[: -len(part)] + "qkv_proj"
            mod = f"{base}@{part}"
            if tail.endswith("lora_B.weight") or tail == ".lora_B":
                sizes[mod] = int(v.shape[0])
        out[mod + tail] = v
    # Offsets need every part's height: q starts at 0, k after q, v after k.
    for mod in list(sizes):
        base, part = mod.split("@")
        heights = [sizes.get(f"{base}@{q}") for q in _QKV]
        if None in heights:
            continue
        start = sum(heights[: _QKV.index(part)])
        plain = re.sub(r"^(?:base_model\.model\.|model\.diffusion_model\.|diffusion_model\.)+", "", base)       # the file's own wrapper is not part of the model's name
        extra[mod] = (f"diffusion_model.{plain}.weight", (0, start, sizes[mod]))
    return out, extra


def _rekey(lora, prefix, replacement):
    if not any(k.startswith(prefix) for k in lora):
        return None
    return {(replacement + k[len(prefix):] if k.startswith(prefix) else k): v for k, v in lora.items()}


def _misfit(sd, key, adapter):
    """True when a plain LoRA pair cannot make the weight it names (both dimensions, not the count)."""
    w = getattr(adapter, "weights", None)
    window = key[1] if isinstance(key, tuple) else None            # (dim, start, length): a LoRA aimed at some rows of a fused weight (Q, K, V of qkv_proj)
    target = sd.get(key[0] if isinstance(key, tuple) else key)
    if not w or len(w) < 2 or target is None or target.dim() < 2:
        return False
    if (len(w) > 3 and w[3] is not None) or (len(w) > 5 and w[5] is not None):   # locon mid / padded target: not modelled
        return False
    try:
        got = (int(w[0].flatten(start_dim=1).shape[0]), int(w[1].flatten(start_dim=1).shape[-1]))
    except Exception:  # noqa: BLE001 -- not a plain pair (LoKr, LoHa, diff): left to ComfyUI
        return False
    if window is not None:
        dim, start, length = window
        return got != (int(length), int(target.shape[1:].numel())) or start + length > int(target.shape[dim])
    return got != (int(target.shape[0]), int(target.shape[1:].numel()))


def match(model, clip, lora):
    """-> (patches, how, dropped): the best keying's patches, which keying won, and how many were dropped for shape."""
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    extra = {}
    if any(".attn.qkv_proj." in str(v) for v in key_map.values()):         # only the H3 transformer is renamed: other models use `transformer_blocks` as they are
        lora, extra = from_diffusers(lora)
    key_map.update(extra)
    converted = comfy.lora_convert.convert_lora(lora)
    best, how = comfy.lora.load_lora(converted, key_map, log_missing=False), "as-is"
    for prefix, replacement in REKEY:
        variant = _rekey(converted, prefix, replacement)
        if variant is not None:
            got = comfy.lora.load_lora(variant, key_map, log_missing=False)
            if len(got) > len(best):
                best, how = got, f"{prefix} -> {replacement or '(none)'}"
    sd = model.model.state_dict()
    bad = [k for k, a in best.items() if _misfit(sd, k, a)]
    for k in bad:
        best.pop(k)
    return best, how, len(bad)
