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


def test_an_ltx_lora_with_transformer_blocks_names_is_left_as_it_is(tiny_ltx, monkeypatch):
    """`transformer_blocks` is LTX's own name too: only the H3 transformer's LoRAs are renamed to H3's layout."""
    from modules.loaders.lora import keys
    sd = tiny_ltx.patcher.model.state_dict()
    name = "diffusion_model.transformer_blocks.0.attn1.to_q.weight"
    assert name in sd
    out_dim, in_dim = sd[name].shape
    monkeypatch.setattr(keys.comfy.lora, "model_lora_keys_unet",           # the tiny stand-in has no model config to derive Diffusers names from
                        lambda model, key_map: {**key_map, **{k[:-len(".weight")]: k for k in sd if k.endswith(".weight")}})
    lora = {name.replace(".weight", ".lora_A.weight"): torch.randn(RANK, in_dim),
            name.replace(".weight", ".lora_B.weight"): torch.randn(out_dim, RANK)}
    patches, _how, dropped = keys.match(tiny_ltx.patcher, None, lora)
    assert len(patches) == 1 and dropped == 0


@pytest.mark.parametrize("wrapper", ["", "transformer.", "diffusion_model.", "base_model.model.", "model.diffusion_model.",
                                     "base_model.model.diffusion_model.", "base_model.model.dit.", "dit.", "model.dit.", "lora_model.", "net."])
def test_a_wrapped_diffusers_file_gets_every_change_on_a_weight_the_model_has(patcher, wrapper):
    from modules.loaders.lora.keys import match
    lora, want = _file("transformer_blocks.1", torch.Generator().manual_seed(0))
    lora = {wrapper + k[len("transformer."):]: v for k, v in lora.items()}
    patches, how, dropped = match(patcher, None, lora)
    assert dropped == 0 and len(patches) == 6, (wrapper, how, len(patches))
    state = patcher.model.state_dict()
    assert all((k[0] if isinstance(k, tuple) else k) in state for k in patches)       # every patch aims at a weight that exists
    clone = patcher.clone()
    clone.add_patches(patches, 1.0)
    got = _delta(clone, "diffusion_model.blocks.1.attn.qkv_proj.weight", (192, 64))
    assert torch.allclose(got, torch.cat([want["attn.to_q"], want["attn.to_k"], want["attn.to_v"]], 0), atol=1e-5)
    assert torch.allclose(_delta(clone, "diffusion_model.blocks.1.attn.out_proj.weight", (64, 64)), want["attn.to_out.0"], atol=1e-5)


def test_a_qkv_pair_of_the_wrong_width_is_dropped_not_merged_scrambled(patcher):
    from modules.loaders.lora.keys import match
    lora, _ = _file("transformer_blocks.1", torch.Generator().manual_seed(0))
    k = "transformer.transformer_blocks.1.attn.to_q.lora_B.weight"
    lora[k] = torch.randn(96, RANK)                                   # 96 rows where this model has 64 per part
    patches, how, dropped = match(patcher, None, lora)
    assert dropped >= 1 and len(patches) == 5


def test_a_refiner_layout_file_without_a_wrapper_reaches_the_refiner_blocks(patcher):
    from modules.loaders.lora.keys import match
    lora, want = _file("token_refiner.refiner_blocks.0", torch.Generator().manual_seed(0))
    lora = {k[len("transformer."):]: v for k, v in lora.items()}
    patches, how, dropped = match(patcher, None, lora)
    assert dropped == 0 and len(patches) == 6, (how, len(patches))
    state = patcher.model.state_dict()
    assert all((k[0] if isinstance(k, tuple) else k) in state for k in patches)


def test_a_block_the_model_does_not_have_is_dropped_and_counted_not_reported_as_applied(patcher):
    from modules.loaders.lora.keys import match
    lora, _ = _file("transformer_blocks.9", torch.Generator().manual_seed(0))            # the tiny model has 2 blocks
    patches, how, dropped = match(patcher, None, lora)
    assert not patches and dropped >= 1


def test_a_dense_lokr_on_a_native_weight_is_not_mistaken_for_a_misfit_pair(patcher):
    from modules.loaders.lora.keys import match
    gen = torch.Generator().manual_seed(0)
    lora = {"diffusion_model.blocks.1.attn.out_proj.lokr_w1": torch.randn(8, 8, generator=gen),
            "diffusion_model.blocks.1.attn.out_proj.lokr_w2": torch.randn(8, 8, generator=gen)}
    patches, how, dropped = match(patcher, None, lora)
    assert len(patches) == 1 and dropped == 0


@pytest.mark.parametrize("keep", [("to_q",), ("to_k",), ("to_v",), ("to_q", "to_v")])
def test_a_file_that_trains_only_some_of_q_k_v_puts_each_in_its_own_rows(patcher, keep):
    from modules.loaders.lora.keys import match
    lora, want = _file("transformer_blocks.1", torch.Generator().manual_seed(3))
    lora = {k: v for k, v in lora.items() if not any(f"attn.{q}." in k for q in ("to_q", "to_k", "to_v") if q not in keep)}
    patches, _how, dropped = match(patcher, None, lora)
    assert dropped == 0 and len(patches) == 3 + len(keep)
    clone = patcher.clone()
    clone.add_patches(patches, 1.0)
    got = _delta(clone, "diffusion_model.blocks.1.attn.qkv_proj.weight", (192, 64))
    zero = torch.zeros(64, 64)
    assert torch.allclose(got, torch.cat([want[f"attn.{q}"] if q in keep else zero for q in ("to_q", "to_k", "to_v")], 0), atol=1e-5)


def test_the_older_lora_down_up_spelling_is_translated_like_lora_a_b():
    from modules.loaders.lora.keys import from_diffusers
    a, b = torch.randn(2, 8), torch.randn(6, 2)
    new, _ = from_diffusers({"transformer.transformer_blocks.0.ff.net.2.lora_A.weight": a, "transformer.transformer_blocks.0.ff.net.2.lora_B.weight": b})
    old, _ = from_diffusers({"transformer.transformer_blocks.0.ff.net.2.lora_down.weight": a, "transformer.transformer_blocks.0.ff.net.2.lora_up.weight": b})
    assert sorted(old) == sorted(new) == ["blocks.0.mlp.fc2.lora_A.weight", "blocks.0.mlp.fc2.lora_B.weight"]
    assert all(torch.equal(old[k], new[k]) for k in new)
