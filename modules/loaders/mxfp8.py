"""MXFP8 weights for MiniMax H3, made from an ordinary (bf16/fp16) file at load.

Blackwell's tensor cores run FP8 at twice the bf16 rate, and the MXFP8 format keeps one power-of-two scale
per 32 weights, so no lone outlier spoils a whole row (the failure that gave plain fp8 NaN audio). The
weights are stored in exactly the layout a pre-quantized file has, so ComfyUI's own loader and kernels
(comfy_kitchen scaled_mm_mxfp8) take it from there.

The recipe follows NVIDIA's Sol-H3 (arXiv 2609.35110): the attention and MLP projections of the middle
blocks only. The first two blocks and the last three stay as they are, the timestep modulation (adaLN),
embeddings, the output layer and the text refiner too. The result is an approximation of the file.
"""

import json
import re

import torch

_BIG = re.compile(r"(?:^|\.)(?<!refiner\.)blocks\.(\d+)\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$")
KEEP_FIRST = 2
KEEP_LAST = 3
GROUP = 32
_FLOATS = (torch.bfloat16, torch.float16, torch.float32)
_TAG = torch.tensor(list(json.dumps({"format": "mxfp8"}).encode()), dtype=torch.uint8)


def available(device=None):
    """True when this machine's ComfyUI can run MXFP8 matmuls (Blackwell, torch 2.10+, comfy_kitchen)."""
    try:
        import comfy.model_management as mm
        from comfy.quant_ops import QUANT_ALGOS
        return "mxfp8" in QUANT_ALGOS and bool(mm.supports_mxfp8_compute(device))
    except Exception:  # noqa: BLE001
        return False


def quantize_state_dict(sd, device=None):
    """Rewrite `sd` in place. -> (layers converted, note). Already-quantized layers and weights whose width is
    not a multiple of 32 are left as they are, and the note says so."""
    from comfy.quant_ops import QuantizedTensor
    keys = [(k, int(m.group(1))) for k in sd if (m := _BIG.search(k))]
    if not keys:
        return 0, "mxfp8: no H3 attention/MLP weights found to convert (not an H3 file, or already quantized) -- loaded as is"
    count = max(i for _, i in keys) + 1
    done, odd = 0, 0
    for key, index in keys:
        if index < KEEP_FIRST or index >= count - KEEP_LAST:
            continue
        w = sd[key]
        layer = key[:-len(".weight")]
        if w.dtype not in _FLOATS or w.ndim != 2 or f"{layer}.comfy_quant" in sd:
            continue
        if w.shape[1] % GROUP:
            odd += 1
            continue
        q = QuantizedTensor.from_float(w.to(device) if device is not None else w, "TensorCoreMXFP8Layout")
        for name, tensor in q.state_dict(key).items():
            sd[name] = tensor.cpu()
        sd[f"{layer}.comfy_quant"] = _TAG.clone()
        done += 1
    if not done:
        return 0, "mxfp8: nothing to convert (already quantized, or too few blocks) -- loaded as is"
    return done, (f"mxfp8: {done} layers in blocks {KEEP_FIRST}-{count - KEEP_LAST - 1} stored as MXFP8"
                  + (f"; {odd} left as they were (width not a multiple of {GROUP})" if odd else ""))
