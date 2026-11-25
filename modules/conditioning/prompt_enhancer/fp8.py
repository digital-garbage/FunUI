"""Single-file fp8 checkpoints in the ComfyUI text-encoder layout, run by transformers' own model classes.

Such a file holds each large weight as float8 (`name.weight`) with one scale beside it (`name.weight_scale`) and a small
`name.comfy_quant` marker; everything else stays bf16. The INT8 ConvRot files of the same layout hold int8 with a scale per
row, rotated by a block Hadamard matrix along the inputs: they are read the same way and unrotated when widened. The model is built from the base model's config with no memory
behind it, then filled from the file: the fp8 weights stay fp8 (about half the memory of bf16) and are widened to the
compute dtype only while a layer runs. The model's code is untouched, so any architecture transformers has works.

Imports torch at the top: only load this module when a model is being built.
"""

import torch
from torch import nn
from torch.nn import functional as F

SKIP_SUFFIXES = (".weight_scale", ".comfy_quant", ".input_scale", "tokenizer_json")


_HADAMARD = {}


def hadamard(size, device="cpu"):
    """The normalised regular Hadamard matrix ConvRot rotates with (the 4x4 block, repeated by Kronecker products: sizes 4, 16, 64, 256 ...)."""
    key = (size, str(device))
    if key not in _HADAMARD:
        if size < 4 or round(torch.log(torch.tensor(float(size))).item() / torch.log(torch.tensor(4.0)).item(), 6) % 1 != 0:
            raise ValueError(f"a ConvRot group size must be a power of 4, not {size}")
        four = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = four
        while h.shape[0] < size:
            h = torch.kron(h, four)
        _HADAMARD[key] = (h / size ** 0.5).to(device)
    return _HADAMARD[key]


def dequantize(weight, scale, dtype, group=0, chunk=16384):
    """The weights as numbers: the stored values times their scale, and for ConvRot unrotated again (the file holds W @ H^T per group of inputs;
    H is orthogonal and symmetric, so W is the stored matrix times H). A big table goes through in slices, so the float copy never is the whole of it."""
    if weight.shape[0] > chunk:
        scale = scale.to(weight.device)
        return torch.cat([dequantize(weight[i:i + chunk], scale[i:i + chunk] if scale.numel() > 1 else scale, dtype, group, chunk) for i in range(0, weight.shape[0], chunk)])
    wide = weight.to(torch.float32) * scale.to(weight.device, torch.float32)
    if group:
        rows, cols = wide.shape[0], wide.shape[-1]
        wide = (wide.reshape(rows, cols // group, group) @ hadamard(group, weight.device)).reshape(rows, cols)
    return wide.to(dtype).contiguous()


def is_scaled_fp8(path) -> bool:
    """Whether this safetensors file holds scaled fp8 weights (it has `*.weight_scale` entries and float8 tensors)."""
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        keys = list(f.keys())
        return any(k.endswith(".weight_scale") for k in keys) and any(k.endswith(".comfy_quant") for k in keys)


class FP8Linear(nn.Module):
    """A Linear whose weight stays float8; each call widens it to the input's dtype (the last layer keeps its widened copy)."""

    def __init__(self, weight, scale, bias=None, keep=False, group=0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.bias = bias
        self.keep, self._wide, self.plain, self.group = keep, None, False, group

    def widen(self, dtype):
        """Widen the weight once and for all (twice the memory of fp8, but no work per call)."""
        if not self.plain:
            self.weight = dequantize(self.weight, self.scale, dtype, self.group)
            self.plain = True

    def forward(self, x):
        if self.plain:
            return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        if self._wide is not None and self._wide.dtype == x.dtype and self._wide.device == x.device:
            w = self._wide
        else:
            w = dequantize(self.weight.to(x.device), self.scale, x.dtype, self.group)
            if self.keep:
                self._wide = w
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


class FP8Embedding(nn.Module):
    """An embedding table kept in float8: only the rows asked for are widened. Keeps the original's own scale factor."""

    def __init__(self, weight, scale, original, group=0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.group = group
        self.padding_idx = getattr(original, "padding_idx", None)
        extra = getattr(original, "embed_scale", None)
        self.register_buffer("embed_scale", extra if extra is not None else torch.tensor(1.0), persistent=False)
        self.dtype = torch.bfloat16

    def forward(self, ids):
        ids = ids.to(self.weight.device)
        if self.scale.numel() > 1:                                           # a scale per row (INT8): the rows asked for, with their own
            rows = dequantize(self.weight[ids.reshape(-1)], self.scale[ids.reshape(-1)], torch.float32, self.group).reshape(*ids.shape, -1).to(self.dtype)
            return rows * self.embed_scale.to(rows.device, self.dtype)
        rows = self.weight[ids].to(self.dtype)
        return rows * self.scale.to(rows.device, self.dtype) * self.embed_scale.to(rows.device, self.dtype)


def _group(handle, weight_key):
    """The ConvRot group size a tensor was stored with (0: not rotated), from the small JSON marker beside it."""
    import json
    marker = weight_key[:-len("weight")] + "comfy_quant"
    if marker not in handle.keys():
        return 0
    try:
        info = json.loads(bytes(handle.get_tensor(marker).tolist()).decode())
    except (ValueError, UnicodeDecodeError):
        return 0
    return int(info.get("convrot_groupsize", 256)) if info.get("convrot") else 0


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


def fp8_bytes(model) -> int:
    """How many bytes the model's fp8 weights take."""
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, FP8Linear) and not m.plain)


def widen(model, dtype=torch.bfloat16):
    """Widen every fp8 linear layer to `dtype`, one at a time (the output layer shares nothing with the input table afterwards)."""
    for m in model.modules():
        if isinstance(m, FP8Linear):
            m.widen(dtype)
    return model


def build(base_dir, weights_file, dtype=torch.bfloat16, device="cpu", progress=None, precision="auto"):
    """-> the model, from the config in `base_dir` and the weights in `weights_file` (a scaled-fp8 safetensors file).
    `precision`: "fp8" keeps the weights fp8 (smallest); "bf16" widens them once (fastest, twice the memory);
    "auto" widens on a graphics card that has room for it, and keeps fp8 otherwise (on the processor, always)."""
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
        for done, (want, have) in enumerate(names.items()):
            if progress and done % 20 == 0:
                progress(done, len(names))
            tensor = f.get_tensor(have)
            owner, attr = _slot(model, want)
            scale_key = have[:-len("weight")] + "weight_scale" if have.endswith(".weight") else ""
            if tensor.dtype in (torch.float8_e4m3fn, torch.int8) and scale_key in f.keys() and attr == "weight":
                scale = f.get_tensor(scale_key).float()
                group = _group(f, have) if tensor.dtype == torch.int8 else 0
                if isinstance(owner, nn.Embedding) or hasattr(owner, "embed_scale"):
                    new = FP8Embedding(tensor, scale, owner, group)
                    new.dtype = dtype
                else:
                    new = FP8Linear(tensor, scale, owner.bias if getattr(owner, "bias", None) is not None else None, group=group)
                parent, name = _slot(model, want.rsplit(".", 1)[0])
                setattr(parent, name, new)
            elif attr in owner._parameters:
                owner._parameters[attr] = nn.Parameter(tensor.to(dtype) if tensor.is_floating_point() else tensor, requires_grad=False)
            else:
                owner._buffers[attr] = tensor.to(dtype) if tensor.is_floating_point() else tensor
        embed = model.get_input_embeddings()
        if "lm_head.weight" in wanted and "lm_head.weight" not in names:           # no output layer of its own in the file: it shares the input table
            if isinstance(embed, FP8Embedding):                                     # keep one widened copy of the fp8 table
                model.lm_head = FP8Linear(embed.weight, embed.scale, keep=True, group=embed.group)
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
    model = model.to(device).eval()
    model.funui_precision = "fp8"
    if precision == "bf16" or (precision == "auto" and str(device).startswith("cuda") and _room(model, device)):
        widen(model, dtype)
        model.funui_precision = "bf16"
    return model


def _room(model, device) -> bool:
    """Whether the card has about twice the fp8 weights' size free, with some left for the work."""
    try:
        free = torch.cuda.mem_get_info(torch.device(device))[0]
    except Exception:                                         # noqa: BLE001 -- no card to ask: keep fp8
        return False
    return free > 2.2 * fp8_bytes(model) + (1 << 30)
