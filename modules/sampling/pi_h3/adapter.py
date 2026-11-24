"""Reading a pi-H3 adapter file (Lakonik/pi-H3) and putting it on a native MiniMax-H3.

A pi-H3 file is not a plain LoRA: it carries rank-64 LoRA pairs for the attention output and the feed-forward layers of
every block, AND whole replacement tensors (the adaLN projections, the output norm and projections, and a few complete
blocks). The replacements go in as ComfyUI "set" patches, the LoRA pairs as ordinary LoRA patches.

The file layout, the key names and the sampling recipe (2 or 4 steps, shift 12, max_raw_t 0.93 / 0.96, last step 0.6 of the
others) follow Hansheng Chen's pi-H3 and ComfyUI-piFlow (Apache-2.0); see the README Thank you section.
"""

import json
import re

import torch
from safetensors import safe_open

SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0
FINAL_STEP_SCALE = 0.6
MAX_RAW_T = {2: 0.93, 4: 0.96}

_LORA = re.compile(r"^(?P<root>.+)\.lora_(?P<side>A|B)\.weight$")
# diffusers names of the pieces of one block -> ComfyUI's native H3 names
_BLOCK = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
    "adaln_proj.linear": "adaln_proj.linear",
}
_TOP = {
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "norm_out.norm": "final_layer.norm",
    "proj_out": "final_layer.video_out",
    "audio_proj_out": "final_layer.audio_out",
}


class Adapter:
    """What a file holds, as ComfyUI-native names: `full` {weight key: tensor}, `lora` {weight key: (A, B)}."""

    def __init__(self, label, steps, full, lora, dwd):
        self.label, self.steps, self.full, self.lora, self.dwd = label, steps, full, lora, dwd

    @property
    def size(self):
        tensors = list(self.full.values()) + [t for pair in self.lora.values() for t in pair] + list(self.dwd.values())
        return sum(t.numel() * t.element_size() for t in tensors)


def _native_key(name):
    """`transformer_blocks.3.attn.to_out.0.weight` -> (`blocks.3.attn.out_proj.weight`, the swap flag), or None."""
    stem, _, leaf = name.rpartition(".")
    m = re.match(r"^transformer_blocks\.(\d+)\.(.+)$", stem)
    if m and m.group(2) in _BLOCK:
        native = _BLOCK[m.group(2)]
        return f"blocks.{m.group(1)}.{native}.{leaf}", native == "mlp.fc1" and leaf == "weight"
    if stem in _TOP:
        return f"{_TOP[stem]}.{leaf}", False
    return None


def _swap_halves(x):
    """ComfyUI stores the two halves of the feed-forward's first layer in the order opposite to diffusers."""
    half = x.shape[0] // 2
    return torch.cat([x[half:], x[:half]], dim=0)


def parse(path, label=""):
    """Read a pi-H3 file. Raises ValueError, in plain words, for a file that is not one or not supported."""
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    if "policy_config" not in meta:
        raise ValueError(f"{label or path} is not a pi-H3 adapter (it has no policy_config in its header)")
    try:
        policy = json.loads(meta["policy_config"])
    except ValueError:
        policy = {}
    dwd = {k: tensors.pop(k) for k in list(tensors) if k.startswith("dwd_")}
    for part in ("image", "audio"):
        kind = policy.get(part) or {}
        if kind.get("type") != "DX" or kind.get("mode") != "grid":
            raise ValueError(f"{label or path} has a policy this port does not know ({policy}); only DX grid is supported")
    full, lora, pairs = {}, {}, {}
    for name, tensor in tensors.items():
        m = _LORA.match(name)
        if m:
            pairs.setdefault(m.group("root"), {})[m.group("side")] = tensor
            continue
        found = _native_key(name)
        if found is None:
            raise ValueError(f"{label or path}: unexpected tensor {name}")
        key, swap = found
        full[key] = _swap_halves(tensor) if swap else tensor
    for root, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"{label or path}: incomplete LoRA pair for {root}")
        found = _native_key(root + ".weight")
        if found is None:
            raise ValueError(f"{label or path}: unexpected LoRA layer {root}")
        key, swap = found
        lora[key] = (pair["A"], _swap_halves(pair["B"]) if swap else pair["B"])
    steps = 2 if "2step" in str(label or path).lower() else 4 if "4step" in str(label or path).lower() else 0
    return Adapter(label or str(path), steps, full, lora, dwd)


def verify(adapter, native):
    """The file must fit this model exactly; a partial fit would render noise, so a mismatch ends the run."""
    sd = native.state_dict()
    for key, tensor in adapter.full.items():
        if key not in sd:
            raise ValueError(f"{adapter.label} does not fit this model: {key} is not in it (is this the MiniMax H3 FL2VA model?)")
        if tuple(sd[key].shape) != tuple(tensor.shape):
            raise ValueError(f"{adapter.label} does not fit this model: {key} is {tuple(tensor.shape)} there, "
                             f"{tuple(sd[key].shape)} here")
    for key, (a, b) in adapter.lora.items():
        if key not in sd or tuple(sd[key].shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"{adapter.label} does not fit this model at {key}")


def schedule(model_shift, steps):
    """The video sigmas of an n-step pi-Flow run (n + 1 values, the last 0): the raw times fall linearly from max_raw_t to
    the last step's start, and the model's own time shift turns them into noise levels."""
    if steps not in MAX_RAW_T:
        raise ValueError("pi-H3 has a 2-step and a 4-step adapter")
    top = MAX_RAW_T[steps]
    last = top * FINAL_STEP_SCALE / (steps - 1 + FINAL_STEP_SCALE)
    raw = torch.linspace(top, last, steps, dtype=torch.float64)
    sigmas = model_shift * raw / (1 + (model_shift - 1) * raw)
    return torch.cat([sigmas, sigmas.new_zeros(1)]).float()
