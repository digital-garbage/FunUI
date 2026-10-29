"""The two-stage hand-off on ComfyUI's real MiniMaxH3 class (a few thousand parameters, CPU): a first stage cut
short at a middle sigma on a small picture, its clean guess resized, a second stage finishing the same
schedule at the full size."""

import pytest
import torch


@pytest.fixture(scope="module")
def model(comfyui):
    import comfy.model_patcher as mp
    import comfy.supported_models as sm
    torch.manual_seed(0)
    config = sm.MiniMaxH3(dict(
        image_model="minimax_h3", hidden_size=64, num_layers=4, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
        time_embed_dim=32, rope_inv_freq_len=4))
    config.set_inference_dtype(torch.float32, None)
    base = config.get_model({}, device=torch.device("cpu"))
    for p in base.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=0.05)
        p.requires_grad_(False)
    return mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def test_a_run_cut_short_hands_over_a_clean_guess_and_a_second_stage_finishes_it_bigger(model):
    from comfy.nested_tensor import NestedTensor
    from modules.latent.second_pass.nodes import FunUILatentResample
    from modules.sampling.sampler.nodes import FunUISampler

    cond = [[torch.randn(1, 5, 48), {}]]
    small = {"samples": NestedTensor((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)))}
    sigmas = torch.tensor([1.0, 0.9, 0.7, 0.3, 0.0])
    first = FunUISampler.execute(model, cond, cond, small, seed=1, steps=4, cfg=1.0, sampler_name="euler",
                                 scheduler="normal", sigmas=sigmas[:3])
    latent, _status, denoised = first.result
    assert len(first.result) == 3
    v_noisy, a_noisy = latent["samples"].unbind()
    v_clean, a_clean = denoised["samples"].unbind()
    assert v_clean.shape == v_noisy.shape and a_clean.shape == a_noisy.shape
    assert all(torch.isfinite(t).all() for t in (v_noisy, a_noisy, v_clean, a_clean))
    assert not torch.equal(v_clean, v_noisy)               # cut at sigma 0.7: the guess is not the noisy state

    big = FunUILatentResample.execute(denoised, None, "upscale", 2.0, cond, cond, 128, 128).result
    assert big[0]["samples"].unbind()[0].shape == (1, 24, 2, 8, 8)

    second = FunUISampler.execute(model, big[1], big[2], big[0], seed=1, steps=4, cfg=1.0, sampler_name="euler",
                                  scheduler="normal", sigmas=sigmas[2:])
    v, a = second.result[0]["samples"].unbind()
    assert v.shape == (1, 24, 2, 8, 8) and a.shape == a_noisy.shape
    assert torch.isfinite(v).all() and torch.isfinite(a).all()
