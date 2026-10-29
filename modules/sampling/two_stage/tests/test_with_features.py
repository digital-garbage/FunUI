"""Two stages with the other sampling features on the same model: a feature written for one run must act in both
stages. The wrapper sits on the real MiniMaxH3 class, a few thousand parameters, CPU."""

import pytest
import torch

from .test_two_stage import model  # noqa: F401 -- the module-scoped fixture

SIGMAS = (1.0, 0.9, 0.7, 0.3, 0.0)


def _run(model, install_others, size=0.5, steps=2):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    from modules.sampling.two_stage import install
    patched = model.clone()
    install_others(patched, "before")
    install(patched, {"enabled": True, "size": size, "steps": steps}, "funui.two_stage")
    install_others(patched, "after")
    calls = []
    hook = patched.model.diffusion_model.register_forward_pre_hook(
        lambda m, a: calls.append(tuple(a[0][0].shape[-2:]) if isinstance(a[0], (list, tuple)) else tuple(a[0].shape[-2:])))
    guider = comfy.samplers.CFGGuider(patched)
    cond = [[torch.randn(1, 5, 48), {}]]
    guider.set_conds(cond, cond)
    guider.set_cfg(1.0)
    latent = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    noise = NestedTensor((torch.randn(1, 24, 2, 8, 8), torch.randn(1, 32, 2, 3)))
    try:
        out = guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), torch.tensor(SIGMAS),
                            seed=1, disable_pbar=True)
    finally:
        hook.remove()
    return out, calls


def _noise_merge(values, where="after"):
    from modules.sampling.noise_merge import install

    def go(patched, when):
        if when == where:
            install(patched, dict(values, enabled=True, seed=5), "funui.noise_merge")
    return go


@pytest.fixture
def said(monkeypatch):
    from modules.sampling import noise_merge
    heard = []
    monkeypatch.setattr(noise_merge, "_say", lambda message, *a, **k: heard.append(message))
    return heard


@pytest.mark.parametrize("where", ["before", "after"])
def test_step_answers_merge_in_the_first_stage_and_in_the_second(model, said, where):
    out, calls = _run(model, _noise_merge({"what": "answers", "steps": "1,3"}, where))
    # step 1 and 2 small, step 3 and 4 full; steps 1 and 3 each get a second pass at their own stage's size
    assert calls == [(4, 4), (4, 4), (4, 4), (8, 8), (8, 8), (8, 8)]
    assert not [m for m in said if m.startswith("Inactive")]
    assert torch.isfinite(out.unbind()[0]).all()


def test_the_second_run_follows_the_cut_and_is_not_dropped(model, said):
    out, calls = _run(model, _noise_merge({"what": "parallel", "steps": "1,3"}))
    assert calls == [(4, 4)] * 4 + [(8, 8)] * 3             # a second pass at steps 1, 2 (small) and 3 (full)
    assert not [m for m in said if "dropped" in m or m.startswith("Inactive")]


def test_the_second_run_resizes_its_last_answer_to_the_full_picture():
    from modules.sampling.noise_merge import _resized
    small = torch.randn(1, 24, 2, 4, 4, dtype=torch.float64)
    big = _resized(small, (1, 24, 2, 8, 8))
    assert big.shape == (1, 24, 2, 8, 8) and big.dtype == small.dtype and big.device == small.device
    assert _resized(small, (1, 24, 3, 8, 8)) is None


def test_starting_noise_uses_one_second_seed_for_both_stages(model, monkeypatch):
    import random
    drawn = []
    monkeypatch.setattr(random, "randrange", lambda n: drawn.append(n) or 11)

    def install_it(patched, when):
        from modules.sampling.noise_merge import install
        if when == "after":
            install(patched, {"enabled": True, "what": "noise", "strength": 0.5, "seed": -1}, "funui.noise_merge")
    out, calls = _run(model, install_it)
    assert len(drawn) == 1                                  # not one per stage
    assert calls == [(4, 4), (4, 4), (8, 8), (8, 8)]


def test_a_feature_around_the_run_is_called_once_and_one_inside_each_stage_knows_which_stage(model):
    from comfy.patcher_extension import WrappersMP
    from modules._core import dit_hooks
    outer_seen, stages_seen = [], []

    def install_it(patched, when):
        def outer(executor, noise, *args, **kwargs):
            outer_seen.append((when, tuple(noise.shape), dit_hooks.stage_of(executor.class_obj)))
            return executor(noise, *args, **kwargs)

        def sampler(executor, model_wrap, *args, **kwargs):
            stages_seen.append(dit_hooks.stage_of(model_wrap))
            return executor(model_wrap, *args, **kwargs)
        patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, f"test.outer.{when}", outer)
        if when == "after":
            patched.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, "test.sampler", sampler)

    _run(model, install_it)
    assert [w for w, _s, _g in outer_seen] == ["before", "after"]     # each once, whatever the order
    assert outer_seen[0][1] == outer_seen[1][1]                       # both see the full-size run
    assert all(g is None for _w, _s, g in outer_seen)                 # outside the stages
    assert stages_seen == [1, 2]


def test_a_planted_massive_channel_is_found_and_steered_in_both_stages(model, monkeypatch, tmp_path):
    """STAS reads the channel at a block in the small first stage (4x4 latent) and in the full one (8x8)."""
    from modules._core import dit_hooks
    from modules.sampling import stas
    from modules.system.taste import store
    heard = []
    monkeypatch.setattr(stas.log, "once", lambda key, level, title, message: heard.append(message))
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")

    def plant(args, extra):
        out = extra["original_block"](args)["img"]
        out[..., 3] += 1000.0 * out.abs().mean()          # a channel far over 50x the mean, at every size
        return {"img": out}

    def install_it(patched, when):
        if when != "after":
            return
        dit_hooks.add_block_hook(patched, "test.plant", 0, plant)
        stas.install(patched, {"enabled": True, "mode": "manual", "alpha": 2.0, "block": 1}, "funui.stas")

    sigmas = tuple(1.0 - i / 9 for i in range(10))
    monkeypatch.setattr(__import__("sys").modules[__name__], "SIGMAS", sigmas)
    _run(model, install_it)
    result = [m for m in heard if m.startswith("Active")]
    assert result and "steered 4 call(s)" in result[0]       # steps 1-2 small, steps 3-4 full: 4 of the first 4 steps
    assert not [m for m in heard if m.startswith("Inactive")]
