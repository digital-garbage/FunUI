"""Two stages on ComfyUI's real MiniMaxH3 class (a few thousand parameters, CPU), through ComfyUI's own guider:
the wrapper sits on the model, so no FunUI sampler is needed."""

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


def _run(model, values, sigmas=(1.0, 0.9, 0.7, 0.3, 0.0), latent=None, seen=None):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    from modules.sampling.two_stage import install
    patched = model.clone()
    note = install(patched, values, "funui.two_stage")
    calls = []
    hook = patched.model.diffusion_model.register_forward_pre_hook(
        lambda m, a: calls.append(tuple(a[0][0].shape[-2:]) if isinstance(a[0], (list, tuple)) else tuple(a[0].shape[-2:])))
    if seen is not None:
        patched.model.diffusion_model.register_forward_pre_hook(
            lambda m, a, kw: seen.append(kw["transformer_options"]["sample_sigmas"].tolist()), with_kwargs=True)
    guider = comfy.samplers.CFGGuider(patched)
    cond = [[torch.randn(1, 5, 48), {}]]
    guider.set_conds(cond, cond)
    guider.set_cfg(1.0)
    latent = latent or NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    noise = NestedTensor((torch.randn(1, 24, 2, 8, 8), torch.randn(1, 32, 2, 3)))
    try:
        out = guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), torch.tensor(sigmas),
                            seed=1, disable_pbar=True)
    finally:
        hook.remove()
    return note, out, calls


def test_first_steps_run_small_and_the_rest_full_size(model):
    note, out, calls = _run(model, {"enabled": True, "size": 0.5, "steps": 2})
    assert "first 2 step(s)" in note
    assert calls == [(4, 4), (4, 4), (8, 8), (8, 8)]       # 2 steps at half the latent, then 2 at full
    video, audio = out.unbind()
    assert video.shape == (1, 24, 2, 8, 8) and audio.shape == (1, 32, 2, 3)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()


def test_off_means_the_plain_run(model):
    note, out, calls = _run(model, {"enabled": False})
    assert note is None and calls == [(8, 8)] * 4


def test_a_run_that_cannot_be_cut_is_left_alone(model):
    from comfy.nested_tensor import NestedTensor
    started = NestedTensor((torch.ones(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    _n, _o, calls = _run(model, {"enabled": True, "size": 0.5, "steps": 2}, latent=started)
    assert calls == [(8, 8)] * 4                             # a start from a picture is not cut
    _n, _o, calls = _run(model, {"enabled": True, "size": 0.5, "steps": 4})
    assert calls == [(8, 8)] * 4                             # nothing would be left for the full size


def test_every_stage_sees_the_whole_schedule(model):
    seen = []
    sigmas = (1.0, 0.9, 0.7, 0.3, 0.0)
    _run(model, {"enabled": True, "size": 0.5, "steps": 2}, sigmas=sigmas, seen=seen)
    assert len(seen) == 4 and all(s == pytest.approx(list(sigmas)) for s in seen)


def test_the_whole_schedule_lives_where_the_stage_schedule_does():
    from ..stages import FULL, keep_whole_schedule

    class Guider:
        pass

    guider, extra = Guider(), {"model_options": {}}
    setattr(guider, FULL, torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64))
    stage = torch.zeros(2, device="meta")
    keep_whole_schedule(lambda *a, **k: None, guider, stage, extra)
    got = extra["model_options"]["transformer_options"]["sample_sigmas"]
    assert got.device == stage.device and got.dtype == stage.dtype
