import json

import pytest
import torch

from .. import mxfp8 as m


def _sd(blocks=8):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.bfloat16)
    sd = {"token_refiner.blocks.0.attn.qkv_proj.weight": r(96, 64), "video_patch_proj.weight": r(32, 64)}
    for i in range(blocks):
        sd[f"blocks.{i}.attn.qkv_proj.weight"] = r(96, 64)
        sd[f"blocks.{i}.mlp.fc2.weight"] = r(32, 100)       # 100 is not a multiple of 32
        sd[f"blocks.{i}.adaln_proj.linear.weight"] = r(64, 128)
    return sd


def _quantizable():
    pytest.importorskip("comfy.quant_ops")
    from comfy.quant_ops import QUANT_ALGOS
    if "mxfp8" not in QUANT_ALGOS:
        pytest.skip("this comfy_kitchen has no MXFP8")


def test_edges_and_non_projections_stay_as_they_are():
    _quantizable()
    sd = _sd()
    keep = {k: v.clone() for k, v in sd.items()}
    n, note = m.quantize_state_dict(sd)
    assert n == 3 and "blocks 2-4" in note and "3 left" in note       # 8 blocks: 0,1 and 5,6,7 stay
    for i in (2, 3, 4):
        k = f"blocks.{i}.attn.qkv_proj"
        assert sd[k + ".weight"].dtype == torch.float8_e4m3fn
        assert json.loads(sd[k + ".comfy_quant"].numpy().tobytes()) == {"format": "mxfp8"}
        assert sd[k + ".weight_scale"].dtype == torch.float8_e8m0fnu
    for k, v in keep.items():
        if ".qkv_proj" in k and k.startswith("blocks.") and int(k.split(".")[1]) in (2, 3, 4):
            continue
        assert torch.equal(sd[k], v), k


def test_second_pass_and_foreign_files_are_left_alone():
    _quantizable()
    sd = _sd()
    m.quantize_state_dict(sd)
    again = {k: v.clone() for k, v in sd.items()}
    assert m.quantize_state_dict(sd)[0] == 0 and all(torch.equal(sd[k].view(torch.uint8), again[k].view(torch.uint8)) for k in sd)
    n, note = m.quantize_state_dict({"unet.weight": torch.zeros(4, 4)})
    assert n == 0 and "loaded as is" in note
