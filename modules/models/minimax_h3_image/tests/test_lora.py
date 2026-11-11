"""H3 LoRA files on H3-Image: the same loader, the same layouts, the same exact change to each weight (tiny, CPU)."""

import pytest
import torch

RANK = 4


@pytest.fixture(scope="module")
def patcher(comfyui):
    import comfy.model_patcher as mp
    import comfy.supported_models as sm
    config = sm.MiniMaxH3Image(dict(
        image_model="minimax_h3_image", pdd_sigmas=[1.0, 0.5, 0.0], hidden_size=64, num_layers=2, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32,
        time_embed_hidden_size=64, time_embed_dim=32, rope_inv_freq_len=4))
    config.set_inference_dtype(torch.float32, None)
    base = config.get_model({}, device=torch.device("cpu"))
    return mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def _pair(out_dim, in_dim, gen):
    return torch.randn(RANK, in_dim, generator=gen), torch.randn(out_dim, RANK, generator=gen)


def _delta(patcher, key, shape):
    import comfy.lora
    return comfy.lora.calculate_weight(patcher.patches[key], torch.zeros(shape), key)


def test_a_diffusers_layout_h3_lora_lands_exactly_and_audio_weights_are_left_out(patcher):
    from modules.loaders.lora.keys import match
    gen = torch.Generator().manual_seed(0)
    lora, want = {}, {}
    for name, (o, i) in {"attn.to_q": (64, 64), "attn.to_k": (64, 64), "attn.to_v": (64, 64), "attn.to_out.0": (64, 64),
                         "ff.net.0.proj": (128, 64), "ff.net.2": (64, 64)}.items():
        a, b = _pair(o, i, gen)
        lora[f"transformer.transformer_blocks.1.{name}.lora_A.weight"], lora[f"transformer.transformer_blocks.1.{name}.lora_B.weight"] = a, b
        want[name] = b @ a
    # what an H3 video LoRA may also hold: the sound's projections, which this model does not have
    a, b = _pair(32, 64, gen)
    lora["diffusion_model.audio_patch_proj.lora_A.weight"], lora["diffusion_model.audio_patch_proj.lora_B.weight"] = a, b
    patches, how, dropped, _unplaced = match(patcher, None, lora)
    assert dropped == 0 and len(patches) == 6, (len(patches), how)       # all six picture-side weights, nothing for the sound

    clone = patcher.clone()
    clone.add_patches(patches, 1.0)
    p = "diffusion_model.blocks.1."
    assert torch.allclose(_delta(clone, p + "attn.qkv_proj.weight", (192, 64)),
                          torch.cat([want["attn.to_q"], want["attn.to_k"], want["attn.to_v"]], 0), atol=1e-5)
    assert torch.allclose(_delta(clone, p + "attn.out_proj.weight", (64, 64)), want["attn.to_out.0"], atol=1e-5)
    up = want["ff.net.0.proj"]
    assert torch.allclose(_delta(clone, p + "mlp.fc1.weight", (128, 64)), torch.cat([up[64:], up[:64]], 0), atol=1e-5)


def test_a_native_layout_lora_lands_too(patcher):
    from modules.loaders.lora.keys import match
    gen = torch.Generator().manual_seed(1)
    a, b = _pair(64, 64, gen)
    patches, _how, dropped, _unplaced = match(patcher, None, {"diffusion_model.blocks.0.attn.out_proj.lora_A.weight": a,
                                                              "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": b})
    assert dropped == 0 and len(patches) == 1
