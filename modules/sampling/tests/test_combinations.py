"""Combination rule (Dex): a feature is tested alone AND with the others, in every pair and all together, on the real
MiniMaxH3 class (tiny, CPU) through ComfyUI's own guider, with and without Two stages. A new feature goes into FEATURES.

Not covered here and said so: triples (the all-on case catches what only shows with many on), and anything that needs a
GPU or real weights."""

import itertools

import pytest
import torch

from modules.sampling.two_stage.tests.test_two_stage import model  # noqa: F401 -- the module-scoped tiny H3

SIGMAS = (1.0, 0.9, 0.7, 0.3, 0.0)

FEATURES = {
    "reins": {"blocks": "2"},
    "attention_bandit": {},
    "q_steer": {"blocks": "2"},
    "attention_temperature": {"temperature": 1.1},
    "stas": {"block": 1},
    "noise_merge": {"seed": 5},
    "sharpness": {},
    "late_branch": {"block": 2},
    "decisiveness": {},
    "shot_memory": {},
    "block_map": {},
}
SUBSETS = [()] + [(f,) for f in FEATURES] + list(itertools.combinations(FEATURES, 2)) + [tuple(FEATURES)]


def _name(subset):
    return "+".join(subset) or "plain"


def _save_a_map(tmp_path, monkeypatch, n):
    """An uneven saved map for the tiny model, so the Block map really reshapes the features that follow it."""
    from modules.sampling.block_map import measure
    monkeypatch.setattr(measure, "FOLDER", tmp_path / "maps")
    monkeypatch.setattr(measure, "SWITCH", tmp_path / "block_map.enabled")
    grid = measure.groups(n)
    res = {"when": "test", "blocks": n, "groups": grid, "points": list(measure.POINTS), "size": 1, "delta": measure.DELTA}
    for kind in measure.KINDS:
        res[kind] = {"leverage": [[0.5 + g + 0.3 * p for p in range(len(measure.POINTS))] for g in range(len(grid))],
                     "fragility": [[0.1 * (g + 1) for p in range(len(measure.POINTS))] for g in range(len(grid))]}
    measure.save(res)


def _run(model, subset, staged):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    settings = {"taste": {"key": "fox"}}
    for f in subset:
        settings[f] = {"enabled": True, **FEATURES[f]}
    if staged:
        settings["two_stage"] = {"enabled": True, "size": 0.5, "steps": 2, "back_up": 0.5}
    patched, status = FunUILoadModifiers.execute(model.clone(), settings).result
    missing = [f for f in subset if f not in status]
    assert not missing, f"never installed (a test that installs nothing proves nothing): {missing}: {status}"
    guider = comfy.samplers.CFGGuider(patched)
    cond = [[torch.randn(1, 5, 48), {}]]
    guider.set_conds(cond, cond)
    guider.set_cfg(1.0)
    latent = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    noise = NestedTensor((torch.randn(1, 24, 2, 8, 8), torch.randn(1, 32, 2, 3)))
    return guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), torch.tensor(SIGMAS),
                         seed=1, disable_pbar=True)


@pytest.mark.parametrize("staged", [False, True], ids=["one-stage", "two-stages"])
@pytest.mark.parametrize("subset", SUBSETS, ids=_name)
def test_every_combination_finishes_with_a_finite_picture_and_sound(model, tmp_path, monkeypatch, subset, staged):
    from core import dit_hooks
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    _save_a_map(tmp_path, monkeypatch, dit_hooks.block_count(model))
    video, audio = _run(model, subset, staged).unbind()
    assert torch.isfinite(video).all() and torch.isfinite(audio).all(), _name(subset)
    assert video.shape == (1, 24, 2, 8, 8)
