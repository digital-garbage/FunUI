"""The attention explorer on a (tiny) LTX forward: depth groups and clip regions; the prompt focus is H3's."""

import random

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


def _load(tiny, **values):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    settings = {"taste": {"key": "fox"}, "attention_bandit": {"enabled": True, "mode": "manual", "range": 0.5, "blocks": "0-3", **values}}
    return FunUILoadModifiers.execute(tiny.patcher, settings).result


def _outer(patched):
    from comfy.patcher_extension import WrappersMP
    return [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws]


def _run(outer, fn):
    call = fn
    for o in reversed(outer):
        call = (lambda w, inner: (lambda: w(lambda: inner())))(o, call)
    return call()


def _draw(monkeypatch, groups, regions):
    from modules.sampling import attention_bandit as m
    picks = iter([groups, regions])
    monkeypatch.setattr(m, "draw", lambda table, rng=random, **kw: [next(picks)] * len(table))


def test_depth_groups_change_the_picture(tiny_ltx, monkeypatch):
    base = tiny_ltx.run()
    patched, status = _load(tiny_ltx, prompt_focus=False, regions=False)
    assert "4 depth groups over 4 blocks" in status
    monkeypatch.setattr("modules.sampling.attention_bandit.draw", lambda table, rng=random, **kw: [2] * len(table))
    out = _run(_outer(patched), lambda: tiny_ltx.run(patched))
    assert not torch.equal(out[0], base[0]), "the explorer ran and changed nothing"


def test_clip_regions_change_the_picture_and_the_sound_apart(tiny_ltx, monkeypatch):
    from core import log
    base = tiny_ltx.run()
    patched, _ = _load(tiny_ltx, prompt_focus=False)
    _draw(monkeypatch, groups=1, regions=0)          # groups as is, every region sharper
    log.new_run()
    out = _run(_outer(patched), lambda: tiny_ltx.run(patched))
    assert not torch.equal(out[0], base[0]) and not torch.equal(out[1], base[1])
    assert any("clip regions in" in e["message"] for e in log.history())


def test_the_band_helper_leaves_other_calls_alone():
    from modules.sampling.attention_bandit import paired_regions
    q = torch.ones(1, 8, 32)
    other = torch.ones(1, 8, 7)
    assert paired_regions(other, [0.5] * 5, 32, 16) is other
    cut = paired_regions(q, [0.5, 1.0, 1.0, 1.0, 1.0], 32, 16)
    assert torch.equal(cut[:, :2], torch.full((1, 2, 32), 2.0)) and torch.equal(cut[:, 2:], q[:, 2:])
