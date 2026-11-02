"""A LoRA written for the Diffusers layout of the H3 transformer, loaded onto ComfyUI's own H3 class (tiny, CPU).

The file is built here with random factors, so the expected change to each weight is known exactly and the loader's
answer is compared to it: names, the fused qkv rows, and the swapped halves of the gated MLP."""

import pytest
import torch

RANK = 4


@pytest.fixture(scope="module")
def patcher(comfyui):
    import comfy.model_patcher as mp
    import comfy.supported_models as sm
    config = sm.MiniMaxH3(dict(
        image_model="minimax_h3", hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
        time_embed_dim=32, rope_inv_freq_len=4))
    config.set_inference_dtype(torch.float32, None)
    base = config.get_model({}, device=torch.device("cpu"))
    return mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def _pair(out_dim, in_dim, gen):
    return torch.randn(RANK, in_dim, generator=gen), torch.randn(out_dim, RANK, generator=gen)


def _file(prefix, gen, dtype=torch.float32, device="cpu"):
    """{key: tensor} and the delta each native weight should get."""
    lora, want = {}, {}
    for name, (o, i) in {"attn.to_q": (64, 64), "attn.to_k": (64, 64), "attn.to_v": (64, 64), "attn.to_out.0": (64, 64),
                         "ff.net.0.proj": (128, 64), "ff.net.2": (64, 64)}.items():
        a, b = _pair(o, i, gen)
        lora[f"transformer.{prefix}.{name}.lora_A.weight"] = a.to(device=device, dtype=dtype)
        lora[f"transformer.{prefix}.{name}.lora_B.weight"] = b.to(device=device, dtype=dtype)
        want[name] = b @ a
    return lora, want


def _delta(patcher, key, shape):
    import comfy.lora
    return comfy.lora.calculate_weight(patcher.patches[key], torch.zeros(shape), key)


@pytest.mark.parametrize("prefix,native", [("transformer_blocks.1", "blocks.1"), ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0")])
def test_every_weight_gets_exactly_its_change(patcher, prefix, native):
    from modules.loaders.lora.keys import match
    lora, want = _file(prefix, torch.Generator().manual_seed(0))
    patches, how, dropped = match(patcher, None, lora)
    assert dropped == 0 and len(patches) == 6, (len(patches), how)

    clone = patcher.clone()
    clone.add_patches(patches, 1.0)
    p = f"diffusion_model.{native}."
    assert torch.allclose(_delta(clone, p + "attn.qkv_proj.weight", (192, 64)),
                          torch.cat([want["attn.to_q"], want["attn.to_k"], want["attn.to_v"]], 0), atol=1e-5)
    assert torch.allclose(_delta(clone, p + "attn.out_proj.weight", (64, 64)), want["attn.to_out.0"], atol=1e-5)
    assert torch.allclose(_delta(clone, p + "mlp.fc2.weight", (64, 64)), want["ff.net.2"], atol=1e-5)
    up_first = want["ff.net.0.proj"]
    assert torch.allclose(_delta(clone, p + "mlp.fc1.weight", (128, 64)), torch.cat([up_first[64:], up_first[:64]], 0), atol=1e-5)


def test_the_gated_mlp_swap_makes_the_native_layer_compute_what_diffusers_would(patcher):
    """Not just the bookkeeping: x through the converted fc1 + swiglu equals x through the Diffusers order (up, gate)."""
    from modules.loaders.lora.keys import from_diffusers
    gen = torch.Generator().manual_seed(1)
    a, b = _pair(128, 64, gen)
    out, _ = from_diffusers({"transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight": a,
                             "transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight": b})
    x = torch.randn(3, 64, generator=gen)
    native = x @ (out["blocks.0.mlp.fc1.lora_A.weight"].T @ out["blocks.0.mlp.fc1.lora_B.weight"].T)
    gate, up = native.chunk(2, -1)
    diffusers = x @ (a.T @ b.T)
    d_up, d_gate = diffusers.chunk(2, -1)           # Diffusers' SwiGLU: hidden, gate = proj(x).chunk(2)
    assert torch.allclose(torch.nn.functional.silu(gate) * up, d_up * torch.nn.functional.silu(d_gate), atol=1e-5)


def test_a_native_file_is_left_alone(patcher):
    from modules.loaders.lora.keys import from_diffusers
    lora = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(2, 2)}
    out, extra = from_diffusers(lora)
    assert out is lora and extra == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_conversion_keeps_device_and_dtype_of_every_tensor(dtype):
    from modules.loaders.lora.keys import from_diffusers
    lora, _ = _file("transformer_blocks.0", torch.Generator().manual_seed(2), dtype=dtype)
    meta = {k: v.to("meta") for k, v in lora.items()}      # the stand-in for a tensor that lives on another device
    for src in (lora, meta):
        out, _ = from_diffusers(src)
        assert len(out) == len(src)
        first = next(iter(src.values()))
        assert all(v.dtype == first.dtype and v.device == first.device for v in out.values())
