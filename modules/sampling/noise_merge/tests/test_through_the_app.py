"""Noise merging, from the settings the app saves to the finished H3 sampling run.

The app posts its module values to /fun/api/pipeline; the pipeline's settings node carries them as
one JSON string; Load Modifiers installs them on the model; FunUISampler samples with ComfyUI's own
loop. Every hop here is the real one, on ComfyUI's real MiniMaxH3 model class at a few thousand
parameters (lcm / beta (shown as beta57 in the app), 4 steps, Decisiveness and STAS active beside it, as in a real log).
"""

import json

import pytest
import torch

pytest.importorskip("aiohttp")

from core.tests.test_routes_pipeline import _request, registered, server  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _taste_in_tmp(comfyui, tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-1")


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


APP_VALUES = {
    "taste": {"key": "fox"},
    "decisiveness": {"enabled": True, "mode": "manual", "level": 5},
    "stas": {"enabled": True, "block": 1},
    "noise_merge": {"enabled": True, "steps": "1", "order": "b_into_a", "strength": 0.5, "seed": -1},
}


def _saved_by_the_app(server, values):
    status, body = _request(server, "POST", "/fun/api/pipeline", {"values": values})
    assert status == 200, body
    slot = next(s for s in body["slots"] if s["node"] == "FunUIModifierSettings")
    return slot["inputs"]["settings"]


def _sample(model, settings, sampler="lcm", scheduler="beta", steps=4):
    from comfy.nested_tensor import NestedTensor
    from modules.sampling.modifiers.nodes import FunUILoadModifiers, FunUIModifierSettings
    from modules.sampling.sampler.nodes import FunUISampler
    checked = FunUIModifierSettings.execute(settings).result[0]
    patched, status = FunUILoadModifiers.execute(model, checked).result
    forwards = []
    dm = patched.model.diffusion_model
    hook = dm.register_forward_hook(lambda *a: forwards.append(1))
    cond = [[torch.randn(1, 5, 48), {}]]
    latent = {"samples": NestedTensor((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)))}
    try:
        out = FunUISampler.execute(patched, cond, cond, latent, seed=1, steps=steps, cfg=1.0,
                                   sampler_name=sampler, scheduler=scheduler, denoise=1.0)
    finally:
        hook.remove()
    assert all(torch.isfinite(p).all() for p in out.result[0]["samples"].unbind())
    return status, len(forwards)


def test_noise_merging_shows_in_the_manifest_next_to_the_other_sampling_settings(server):
    status, manifest = _request(server, "GET", "/fun/api/modules")
    mods = {m["id"]: m for m in manifest["modules"]}
    assert "noise_merge" in mods and mods["noise_merge"]["mount"] == mods["decisiveness"]["mount"]
    assert {"enabled", "steps", "order", "strength", "seed"} <= set(mods["noise_merge"]["settings"])


def test_what_the_app_saves_reaches_the_h3_sampler_and_adds_one_pass_at_step_one(server, model):
    from core import log
    saved = _saved_by_the_app(server, APP_VALUES)
    assert json.loads(saved)["noise_merge"]["steps"] == "1"
    off = json.loads(saved)
    off["noise_merge"]["enabled"] = False
    _s, plain = _sample(model, json.dumps(off))
    log.new_run()
    status, merged = _sample(model, saved)
    assert "noise_merge" in status and "decisiveness" in status and "stas" in status
    assert merged == plain + 1
    texts = [e["message"] for e in log.history()]
    assert any(t.startswith("on: steps 1") for t in texts)
    assert any(t.startswith("step 1: second answer differs") for t in texts)


def test_a_second_run_on_the_same_cached_model_still_makes_the_extra_pass(server, model):
    """ComfyUI hands the SAME patched model to every run whose Load Modifiers inputs did not change."""
    from comfy.nested_tensor import NestedTensor
    from core import log
    from modules.sampling.modifiers.nodes import FunUILoadModifiers, FunUIModifierSettings
    from modules.sampling.sampler.nodes import FunUISampler
    saved = _saved_by_the_app(server, APP_VALUES)
    patched, _ = FunUILoadModifiers.execute(model, FunUIModifierSettings.execute(saved).result[0]).result
    counts = []
    for seed in (1, 2, 3):
        forwards = []
        hook = patched.model.diffusion_model.register_forward_hook(lambda *a: forwards.append(1))
        cond = [[torch.randn(1, 5, 48), {}]]
        latent = {"samples": NestedTensor((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)))}
        before = sum("second pass took" in e["message"] for e in log.history())
        try:
            FunUISampler.execute(patched, cond, cond, latent, seed=seed, steps=4, cfg=1.0,
                                 sampler_name="lcm", scheduler="beta", denoise=1.0)
        finally:
            hook.remove()
        counts.append((len(forwards), sum("second pass took" in e["message"] for e in log.history()) - before))
    assert counts == [(5, 1)] * 3, counts
