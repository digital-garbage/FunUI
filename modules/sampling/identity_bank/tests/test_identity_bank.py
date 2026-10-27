"""Identity bank on a real (tiny) LTX forward, learning through a real taste key."""

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
    return FunUILoadModifiers.execute(
        tiny.patcher, {"identity_bank": {"enabled": True, "blocks": "1,2", **values}, "taste": {"key": "fox"}}).result


def _run_and_like(tiny, patched):
    from modules.system.taste import store
    sched = torch.tensor([1.0, 0.5, 0.0])
    tiny.sample(patched, sigma=0.5, sigmas=sched)
    return store.rate("run-1", "liked")


def test_nothing_changes_before_a_clip_is_liked_then_the_bank_pulls_the_picture(tiny_ltx):
    from modules.system.taste import store
    base = tiny_ltx.run()
    patched, status = _load(tiny_ltx, strength=0.3)
    assert "identity_bank: keeps blocks 1,2; leans 0.3" in status
    sched = torch.tensor([1.0, 0.5, 0.0])
    empty = tiny_ltx.sample(patched, sigma=0.5, sigmas=sched)
    assert torch.equal(empty[0], base[0]), "an empty bank must change nothing"
    assert store.rate("run-1", "liked")["recorded"] == [store_kind()]
    row = store.load("fox", store_kind())["rows"][0]["rows"]
    assert sorted(row) == ["k1", "k2", "v1", "v2"]
    pulled = tiny_ltx.sample(patched, sigma=0.5, sigmas=sched)
    assert not torch.allclose(pulled[0], base[0]), "the bank was not used"
    assert pulled[0].shape == base[0].shape and pulled[1].shape == base[1].shape


def store_kind():
    from modules.sampling import identity_bank
    return identity_bank.KIND


def test_a_bank_of_another_size_is_said_and_changes_nothing(tiny_ltx):
    from modules.system.taste import store
    from core import log
    patched, _ = _load(tiny_ltx, strength=0.3)
    _run_and_like(tiny_ltx, patched)
    base = tiny_ltx.run()
    log.new_run()
    other = torch.randn(1, 128, 3, 2, 2)
    out = tiny_ltx.sample(patched, sigma=0.5, sigmas=torch.tensor([1.0, 0.5, 0.0]), video=other)
    assert any("another size" in e["message"] for e in log.history())
    assert torch.equal(tiny_ltx.run(patched, video=other)[0], tiny_ltx.run(video=other)[0])


def test_the_lock_follows_the_runs_own_pace():
    from modules.sampling.identity_bank import lock_scale
    history = []
    assert lock_scale(history, 0.1) == 1.0                      # nothing to compare with yet
    assert lock_scale(history, 0.1) == 1.0                      # steady
    assert lock_scale(history, 0.3) == 2.0                      # three times the pace: held at the ceiling
    assert lock_scale(history, 0.001) == 0.5                    # settled: the floor


def test_off_needs_a_taste_key_and_h3_is_not_offered_it(tiny_ltx):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    _p, status = FunUILoadModifiers.execute(tiny_ltx.patcher, {"identity_bank": {"enabled": True}}).result
    assert "identity_bank:" not in status.splitlines()[0]
