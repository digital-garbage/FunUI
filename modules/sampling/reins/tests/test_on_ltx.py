"""Taste steering on a (tiny) LTX forward: its blocks hand back (video, audio)."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-1")


def _width(tiny):
    return tiny.patcher.model.diffusion_model.transformer_blocks[0].attn1.to_out[0].weight.shape[0]


def _load(tiny, **values):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    return FunUILoadModifiers.execute(
        tiny.patcher, {"reins": {"enabled": True, **values}, "taste": {"key": "fox"}}).result


def test_it_learns_at_every_block_and_steers_only_the_picture(tiny_ltx):
    from modules.system.taste import store
    liked = torch.zeros(_width(tiny_ltx))
    liked[0] = 1.0
    for i, (vec, rating) in enumerate([(liked, "liked"), (-liked, "disliked")] * 2):
        store.capture("fox", "reins", {1: vec}, prompt_id=f"t{i}")
        store.rate(f"t{i}", rating)
    base_v, base_a = tiny_ltx.run()
    patched, status = _load(tiny_ltx, blocks="1", strength=0.5)
    assert "reins: learning at every block; steers 1" in status
    v, a = tiny_ltx.sample(patched)
    assert not torch.allclose(v, base_v), "the learned direction changed nothing"
    assert v.shape == base_v.shape and a.shape == base_a.shape
