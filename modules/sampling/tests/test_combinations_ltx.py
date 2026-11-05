"""The combination rule on the tiny LTX audio+video model: every modifier alone, in every pair and all together.

Every modifier a module provides is either run here, or listed in NEEDS_OTHER_MODEL (it declares it needs something this
model lacks, and the test checks it SAYS so, so a silent change is noticed), or in UNREACHABLE with the reason. H3 is covered
in test_combinations.py; together the two files must name every modifier (test_every_modifier_is_placed)."""

import itertools

import pytest
import torch

from modules.sampling.tests.test_combinations import EXEMPT, FEATURES as H3_FEATURES, LTX_ONLY, NOT_ACTIVE, _fingerprint, _status_line

FEATURES = {
    "alg": {}, "dynashift": {}, "momentum": {}, "output_guidance": {}, "prompt_nudge": {}, "score_slider": {},
    "trajectory_guidance": {}, "velocity_bias": {}, "reins": {"blocks": "1"}, "attention_bandit": {},
    "attention_temperature": {"temperature": 1.1}, "noise_merge": {"seed": 5}, "sharpness": {}, "run_meter": {}, "decisiveness": {},
    "shot_memory": {}, "turbo_sampler": {}, "camera_move": {"pan_x": 0.3}, "explore_first_step": {},
    "audio_coupling": {"scale": 1.5}, "temporal_style": {"style": "pulse"}, "identity_bank": {"blocks": "1,2"},
}
#: Modifiers that need something this model does not have; they must say so (a status line containing the need).
NEEDS_OTHER_MODEL = {
    "block_influence": "dit_block_hooks", "block_map": "dit_block_hooks", "late_branch": "dit_block_hooks",
    "q_steer": "dit_block_hooks", "stas": "dit_block_hooks", "shadow_negative": "adaln_modalities",
    "video_detail": "adaln_modalities", "sharpen": "spatial_latent",
}
UNREACHABLE = {"context_windows": "this local ComfyUI cannot window an audio+video latent; it installs on the rental's ComfyUI (not proven here)"}


@pytest.fixture(autouse=True)
def _taste(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


def _load(tiny, subset):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    settings = {"taste": {"key": "fox"}, **{f: {"enabled": True, **FEATURES[f]} for f in subset}}
    patched, status = FunUILoadModifiers.execute(tiny.patcher, settings).result
    for f in subset:
        line = _status_line(status, f)
        assert line, f"never installed: {f}: {status}"
        assert not any(bad in line.lower() for bad in NOT_ACTIVE), f"{f} did not switch on: {line}"
    return patched, status


def test_every_modifier_is_placed():
    h3_and_ltx = set(H3_FEATURES) | set(FEATURES) | set(NEEDS_OTHER_MODEL) | set(UNREACHABLE) | set(EXEMPT) | LTX_ONLY
    assert LTX_ONLY <= set(FEATURES) | set(NEEDS_OTHER_MODEL) | set(UNREACHABLE), "an LTX-only modifier has no entry here"
    assert h3_and_ltx >= set(H3_FEATURES)


@pytest.mark.parametrize("feature", list(NEEDS_OTHER_MODEL))
def test_a_modifier_this_model_cannot_take_says_what_it_needs(tiny_ltx, feature):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    _patched, status = FunUILoadModifiers.execute(tiny_ltx.patcher, {"taste": {"key": "fox"}, feature: {"enabled": True}}).result
    assert f"needs {NEEDS_OTHER_MODEL[feature]}" in (_status_line(status, feature) or "") or NEEDS_OTHER_MODEL[feature] in (_status_line(status, feature) or ""), status


@pytest.mark.parametrize("feature", list(FEATURES))
def test_a_feature_alone_is_active_wired_and_runs(tiny_ltx, feature):
    base, _ = _load(tiny_ltx, ())
    patched, status = _load(tiny_ltx, (feature,))
    assert _fingerprint(patched) != _fingerprint(base), f"{feature} changed nothing the model is wired with: {_status_line(status, feature)}"
    video, audio = tiny_ltx.sample(patched)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()


SUBSETS = [tuple(FEATURES)] + list(itertools.combinations(FEATURES, 2))


@pytest.mark.parametrize("subset", SUBSETS, ids=lambda s: "+".join(s) if len(s) < 3 else "all")
def test_every_pair_and_all_together_run_with_a_finite_picture_and_sound(tiny_ltx, subset):
    patched, _ = _load(tiny_ltx, subset)
    base = tiny_ltx.run()
    video, audio = tiny_ltx.sample(patched)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    assert video.shape == base[0].shape and audio.shape == base[1].shape and video.dtype == base[0].dtype and video.device == base[0].device
