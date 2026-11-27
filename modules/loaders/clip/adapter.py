"""Text encoder adapters: lets a MiniMax H3 run on another text encoder (Gemma 4, Qwen 3.5, ...).

An adapter file is a small set of numbers that maps the hidden states of a donor encoder into the space H3's own
encoder (Qwen3-VL-32B, 5120 wide) writes in, so the donor's output can stand in for it. The method is "Plato's Key" by
Alisson Anjos (https://huggingface.co/Alissonerdx/TE-PK-Minimax-H3, https://github.com/alisson-anjos/ComfyUI-H3-Text-Encoders):
`(x - donor_mean) @ R * scale + target_mean`. This is a native port of that transform, not a copy of the node pack.

The loader applies it to the encoder it has just loaded: the encoder's own tokenizer and model objects are shared by every
copy of the CLIP, so the change holds however the CLIP is cloned further down the graph.
"""

import json
import os

import torch

import folder_paths

from ..._core import log

FOLDER = "clip_projections"
_CACHE = {}


def register_folder():
    """`models/clip_projections` is where adapter files live (the same place the node pack's files go)."""
    try:
        folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))
    except Exception:                                    # noqa: BLE001
        pass


def adapter_files():
    try:
        return list(folder_paths.get_filename_list(FOLDER))
    except Exception:                                    # noqa: BLE001
        return []


class Key:
    def __init__(self, tensors, meta, name):
        self.R, self.scale = tensors["R"], tensors["scale"].reshape(1)
        self.donor_mean, self.target_mean = tensors["donor_mean"], tensors["target_mean"]
        self.name = name
        self.d_donor, self.d_target = int(self.R.shape[0]), int(self.R.shape[1])
        if meta.get("donor_layers"):
            self.layers = [int(v) for v in json.loads(meta["donor_layers"])]
        else:
            self.layers = [int(meta.get("donor_layer") or meta.get("tap_layer") or -1)]
        self.donor = meta.get("donor_model", "")


def load_key(name):
    """The adapter file `name` (cached until the file changes)."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    path = folder_paths.get_full_path(FOLDER, name)
    if not path:
        raise RuntimeError(f"The adapter file {name} is no longer where it was listed from.")
    info = os.stat(path)
    stamp = (path, info.st_mtime_ns, info.st_size)
    if stamp not in _CACHE:
        tensors = load_file(path)
        missing = [k for k in ("R", "scale", "donor_mean", "target_mean") if k not in tensors]
        if missing:
            raise RuntimeError(f"{name} is not an adapter file (missing {', '.join(missing)}).")
        with safe_open(path, framework="pt") as f:
            meta = dict(f.metadata() or {})
        _CACHE.clear()
        _CACHE[stamp] = Key(tensors, meta, name)
    return _CACHE[stamp]


def project(x, key, strength=1.0):
    """The donor's hidden states -> H3's space. `x` is [B, T, D] (one layer) or [B, layers, T, D] (every layer)."""
    if x.ndim == 4:
        x = torch.cat([x[:, i] for i in key.layers], dim=-1)
    elif key.layers != [-1]:
        raise RuntimeError(f"{key.name} needs layer {key.layers} of the encoder but this encoder gives only its last one.")
    if x.shape[-1] != key.d_donor:
        raise RuntimeError(f"{key.name} is for an encoder {key.d_donor} wide{' (' + key.donor + ')' if key.donor else ''}, "
                           f"the loaded encoder is {x.shape[-1]} wide. Pick the adapter that matches the encoder.")
    work = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    to = {"device": x.device, "dtype": work}
    out = ((x.to(work) - key.donor_mean.to(**to)) @ key.R.to(**to)) * key.scale.to(**to)
    out = out + key.target_mean.to(**to) if strength == 1.0 else key.target_mean.to(**to) + out * strength
    return out.to(x.dtype)


#: What H3's own tokenizer takes and a donor's does not: the pictures of a request are not shown to the donor.
PICTURES = ("images", "minimax_ref_items", "picture_label")


def attach(clip, key, strength=1.0):
    """Make `clip` write in H3's space: text goes in plain, every encode ends in the adapter."""
    tok = clip.tokenizer
    inner = getattr(tok, getattr(tok, "clip", ""), tok)
    has_start = bool(getattr(inner, "has_start_token", False))       # the start token is kept while encoding (the adapter was fitted so) and dropped after
    said = []
    plain = tok.tokenize_with_weights

    def tokenize(text, return_word_ids=False, **kwargs):
        if any(kwargs.get(k) for k in PICTURES) and not said:
            said.append(1)
            log.info("FunUI CLIP Loader", f"adapter {key.name}: pictures of a request are not read by the adapted encoder, only the words")
        kwargs = {k: v for k, v in kwargs.items() if k not in PICTURES}
        kwargs.setdefault("llama_template", "{}")
        return plain(text, return_word_ids, **kwargs)
    tok.tokenize_with_weights = tokenize

    model = clip.cond_stage_model
    encode = model.encode_token_weights

    def encode_token_weights(pairs):
        out = encode(pairs)
        cond, pooled = project(out[0], key, strength), out[1]
        extra = dict(out[2]) if len(out) > 2 and isinstance(out[2], dict) else {}
        if has_start and cond.shape[-2] > 1:
            cond = cond[..., 1:, :]
            mask = extra.get("attention_mask")
            if isinstance(mask, torch.Tensor) and mask.shape[-1] > 1:
                extra["attention_mask"] = mask[..., 1:]
        extra["minimax_token_tags"] = torch.ones(cond.shape[-2], dtype=torch.long)       # all text: no picture blocks
        return cond, pooled, extra
    model.encode_token_weights = encode_token_weights
    clip.funui_adapter = key.name
