"""Single-file fp8 checkpoints in the ComfyUI text-encoder layout, run by transformers' own model classes.

Such a file holds each large weight as float8 (`name.weight`) with one scale beside it (`name.weight_scale`) and a small
`name.comfy_quant` marker; everything else stays bf16. The model is built from the base model's config with no memory
behind it, then filled from the file: the fp8 weights stay fp8 (about half the memory of bf16) and are widened to the
compute dtype only while a layer runs. The model's code is untouched, so any architecture transformers has works.

Imports torch at the top: only load this module when a model is being built.
"""

import torch
from torch import nn
from torch.nn import functional as F

SKIP_SUFFIXES = (".weight_scale", ".comfy_quant", ".input_scale", "tokenizer_json")


def is_scaled_fp8(path) -> bool:
    """Whether this safetensors file holds scaled fp8 weights (it has `*.weight_scale` entries and float8 tensors)."""
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        keys = list(f.keys())
        return any(k.endswith(".weight_scale") for k in keys) and any(k.endswith(".comfy_quant") for k in keys)


class FP8Linear(nn.Module):
    """A Linear whose weight stays float8; each call widens it to the input's dtype (the last layer keeps its widened copy)."""

    def __init__(self, weight, scale, bias=None, keep=False):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.bias = bias
        self.keep, self._wide = keep, None

    def forward(self, x):
        if self._wide is not None and self._wide.dtype == x.dtype and self._wide.device == x.device:
            w = self._wide
        else:
            w = self.weight.to(x.device).to(x.dtype) * self.scale.to(x.device, x.dtype)
            if self.keep:
                self._wide = w
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


class FP8Embedding(nn.Module):
    """An embedding table kept in float8: only the rows asked for are widened. Keeps the original's own scale factor."""

    def __init__(self, weight, scale, original):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.padding_idx = getattr(original, "padding_idx", None)
        extra = getattr(original, "embed_scale", None)
        self.register_buffer("embed_scale", extra if extra is not None else torch.tensor(1.0), persistent=False)
        self.dtype = torch.bfloat16

    def forward(self, ids):
        rows = self.weight[ids.to(self.weight.device)].to(self.dtype)
        return rows * self.scale.to(rows.device, self.dtype) * self.embed_scale.to(rows.device, self.dtype)


def _slot(model, key):
    """-> (the module that owns `key`, its attribute name)."""
    *path, name = key.split(".")
    owner = model
    for part in path:
        owner = getattr(owner, part)
    return owner, name


def _rename(wanted, found):
    """The file's name for each of the model's tensors: as written, or with a language-model prefix added or dropped."""
    out = {}
    for want in wanted:
        for cand in (want, want.replace("model.", "model.language_model.", 1), want.replace("model.language_model.", "model.", 1), "language_model." + want):
            if cand in found:
                out[want] = cand
                break
    return out


def build(base_dir, weights_file, dtype=torch.bfloat16, device="cpu"):
    """-> the model, from the config in `base_dir` and the weights in `weights_file` (a scaled-fp8 safetensors file)."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    text = AutoConfig.from_pretrained(base_dir).get_text_config()
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(text, dtype=dtype)
    wanted = set(model.state_dict())
    with safe_open(str(weights_file), "pt") as f:
        found = {k for k in f.keys() if not k.endswith(SKIP_SUFFIXES)}
        names = _rename(wanted, found)
        missing = sorted(wanted - {"lm_head.weight"} - set(names))          # an output layer the file lacks is the input table, shared
        if len(missing) > len(wanted) // 20:
            raise ValueError(f"{len(missing)} of the model's {len(wanted)} tensors are not in {weights_file}, for example {missing[:3]}")
        for want, have in names.items():
            tensor = f.get_tensor(have)
            owner, attr = _slot(model, want)
            scale_key = have[:-len("weight")] + "weight_scale" if have.endswith(".weight") else ""
            if tensor.dtype == torch.float8_e4m3fn and scale_key in f.keys() and attr == "weight":
                scale = f.get_tensor(scale_key).float()
                if isinstance(owner, nn.Embedding) or hasattr(owner, "embed_scale"):
                    new = FP8Embedding(tensor, scale, owner)
                    new.dtype = dtype
                else:
                    new = FP8Linear(tensor, scale, owner.bias if getattr(owner, "bias", None) is not None else None)
                parent, name = _slot(model, want.rsplit(".", 1)[0])
                setattr(parent, name, new)
            elif attr in owner._parameters:
                owner._parameters[attr] = nn.Parameter(tensor.to(dtype) if tensor.is_floating_point() else tensor, requires_grad=False)
            else:
                owner._buffers[attr] = tensor.to(dtype) if tensor.is_floating_point() else tensor
        embed = model.get_input_embeddings()
        if "lm_head.weight" in wanted and "lm_head.weight" not in names:           # no output layer of its own in the file: it shares the input table
            if isinstance(embed, FP8Embedding):                                     # keep one widened copy of the fp8 table
                model.lm_head = FP8Linear(embed.weight, embed.scale, keep=True)
            else:
                model.lm_head.weight = embed.weight
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p.device.type == "meta":
            raise ValueError(f"{name} was not filled from {weights_file}")
    try:                                                      # the stop tokens and sampling defaults the model's authors set
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_pretrained(base_dir)
    except Exception:                                         # noqa: BLE001 -- a base without one keeps the config's own
        pass
    return model.to(device).eval()
