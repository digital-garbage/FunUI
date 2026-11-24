"""A pi-H3 adapter file made to fit the tiny native MiniMax-H3 of the PTD tests."""

import json

import torch
from safetensors.torch import save_file

POLICY = json.dumps({"image": {"mode": "grid", "type": "DX"}, "audio": {"mode": "grid", "type": "DX"}})


def adapter_file(path, native, rank=4, full_blocks=(1,), seed=1, dwd=False, policy=POLICY):
    """Diffusers-named tensors for every block's adaLN and LoRA pairs, whole layers for `full_blocks`, and the head."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: (torch.randn(*shape, generator=g) * 0.05).half()      # noqa: E731
    sd = native.state_dict()
    out = {}
    for i in range(len(native.blocks)):
        p = f"transformer_blocks.{i}"
        out[f"{p}.adaln_proj.linear.weight"] = r(*sd[f"blocks.{i}.adaln_proj.linear.weight"].shape)
        for diff, nat in (("attn.to_out.0", "attn.out_proj"), ("ff.net.0.proj", "mlp.fc1"), ("ff.net.2", "mlp.fc2")):
            weight = sd[f"blocks.{i}.{nat}.weight"]
            if i in full_blocks:
                out[f"{p}.{diff}.weight"] = r(*weight.shape)
            else:
                out[f"{p}.{diff}.lora_A.weight"] = r(rank, weight.shape[1])
                out[f"{p}.{diff}.lora_B.weight"] = r(weight.shape[0], rank)
    for diff, nat in (("norm_out.linear", "final_layer.adaln_proj.linear"), ("proj_out", "final_layer.video_out"),
                      ("audio_proj_out", "final_layer.audio_out")):
        out[f"{diff}.weight"] = r(*sd[f"{nat}.weight"].shape)
        out[f"{diff}.bias"] = r(*sd[f"{nat}.bias"].shape)
    out["norm_out.norm.weight"] = r(*sd["final_layer.norm.weight"].shape)
    if dwd:
        out["dwd_norm_out.linear.weight"] = r(4, 4)
    save_file(out, str(path), metadata={"policy_config": policy} if policy else {})
    return out
