"""The attention explorer: the learner's maths, then the real modifier loader on a real H3 forward."""

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


def _row(reward, arms, sig):
    return {"reward": reward, "rows": {"arms": torch.tensor(arms), "sig": torch.tensor(sig)}}


def test_split_is_by_depth_and_even():
    from modules.sampling.attention_bandit import split
    parts = split(list(range(10)), 4)
    assert [len(p) for p in parts] == [3, 3, 2, 2] and sum(parts, []) == list(range(10))
    assert split([5, 6], 4) == [[5], [6]]


def test_a_fresh_key_mostly_stays_as_is_and_a_liked_arm_takes_over():
    from modules.sampling.attention_bandit import STAY, counts, draw
    rng = random.Random(0)
    table = counts([], [2, 0, 9], 2)
    stay = sum(a == STAY for _ in range(300) for a in draw(table, rng))
    assert stay > 0.5 * 600                     # head start keeps a new key close to the plain model
    rows = [_row(1.0, [0, 2], [2, 0, 9]) for _ in range(12)]
    liked = counts(rows, [2, 0, 9], 2)
    picks = [draw(liked, rng) for _ in range(200)]
    assert sum(p[0] == 0 for p in picks) > 120 and sum(p[1] == 2 for p in picks) > 120


def test_a_dislike_pushes_away_and_other_layouts_and_neutral_ratings_are_ignored():
    from modules.sampling.attention_bandit import counts
    rows = [_row(-1.0, [0, 1], [2, 0, 9]), _row(1.0, [2, 2], [2, 0, 10]), _row(0.0, [2, 2], [2, 0, 9])]
    t = counts(rows, [2, 0, 9], 2)
    assert t[0][0] == [0.0, 1.0] and t[0][2] == [0.0, 0.0] and t[1][1] == [0.0, 1.0]


def _srow(reward, arms, spread, sig=(2, 0, 9)):
    return {"reward": reward, "rows": {"arms": torch.tensor(arms), "sig": torch.tensor(list(sig)),
                                       "spread": torch.tensor(spread)}}


def test_the_size_moves_toward_liked_and_away_from_disliked_and_ignores_unchanged_runs():
    from modules.sampling.attention_bandit import SPREAD_MAX, SPREAD_MIN, START_SPREAD, learned_spread
    sig = [2, 0, 9]
    assert learned_spread([], sig) == START_SPREAD
    assert learned_spread([_srow(1.0, [0, 1], 0.4)], sig) > START_SPREAD
    assert learned_spread([_srow(-1.0, [0, 1], 0.4)], sig) < START_SPREAD
    assert learned_spread([_srow(1.0, [1, 1], 0.4)], sig) == START_SPREAD           # nothing changed
    assert learned_spread([_srow(1.0, [0, 1], 0.4, sig=(3, 0, 9))], sig) == START_SPREAD   # other layout
    assert learned_spread([_srow(1.0, [0, 0], 0.5)] * 30, sig) == pytest.approx(0.5, abs=1e-3)
    assert SPREAD_MIN <= learned_spread([_srow(-1.0, [0, 0], 0.5)] * 30, sig) <= SPREAD_MAX


def test_the_log_names_exact_blocks_and_skips_the_unchanged():
    from modules.sampling.attention_bandit import describe
    groups = [[2, 3, 4], [9, 10, 29, 30], [34, 35], [41, 42]]
    assert describe(groups, [1, 1, 1, 1], 0.2) == "no change this run"
    assert describe(groups, [0, 1, 2, 0], 0.2) == "sharper 0.8x at blocks 2-4,41-42; looser 1.2x at blocks 34-35"


def test_a_learned_run_banks_its_size(tiny_h3, monkeypatch):
    from modules.sampling import attention_bandit as m
    from modules.system.taste import store
    patched, status = _load(tiny_h3, blocks="0-3")
    assert "size learned from ratings" in status
    monkeypatch.setattr(m, "draw", lambda table, rng=random: [0] * len(table))
    _run(_outer(patched), lambda: tiny_h3.run(patched))
    store.rate("run-1", "liked")
    row = store.load("fox", "attention_bandit")["rows"][0]
    assert 0.05 <= float(row["rows"]["spread"]) <= 0.5


def _load(tiny, **values):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    settings = {"taste": {"key": "fox"}, "attention_bandit": {"enabled": True, **values}}
    return FunUILoadModifiers.execute(tiny.patcher, settings).result


def _outer(patched):
    from comfy.patcher_extension import WrappersMP
    return [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws]


def _run(outer, fn):
    call = fn
    for o in reversed(outer):
        call = (lambda w, inner: (lambda: w(lambda: inner())))(o, call)
    return call()


def test_a_run_changes_the_output_banks_its_arms_and_a_rating_teaches_them(tiny_h3, monkeypatch):
    from modules.system.taste import store
    base = tiny_h3.run()
    patched, status = _load(tiny_h3, mode="manual", range=0.4, blocks="0-3")
    assert "4 depth groups over 4 blocks" in status
    from modules.sampling import attention_bandit as m
    monkeypatch.setattr(m, "draw", lambda table, rng=random: [2] * len(table))
    outs = _run(_outer(patched), lambda: tiny_h3.run(patched))
    assert not torch.equal(outs[0], base[0]), "the explorer ran and changed nothing"
    assert store.rate("run-1", "liked")["recorded"] == ["attention_bandit"]
    row = store.load("fox", "attention_bandit")["rows"][0]
    assert row["rows"]["arms"].tolist() == [2, 2, 2, 2] and row["reward"] == 1.0


def test_stay_everywhere_is_the_plain_model(tiny_h3, monkeypatch):
    from modules.sampling import attention_bandit as m
    base = tiny_h3.run()
    patched, _ = _load(tiny_h3, blocks="0-3")
    monkeypatch.setattr(m, "draw", lambda table, rng=random: [m.STAY] * len(table))
    outs = _run(_outer(patched), lambda: tiny_h3.run(patched))
    assert torch.equal(outs[0], base[0])


def test_no_taste_key_installs_nothing_and_says_so(tiny_h3):
    from core import log
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    log.new_run()
    FunUILoadModifiers.execute(tiny_h3.patcher, {"attention_bandit": {"enabled": True}})
    assert any("Taste key" in e["message"] for e in log.history())


def test_a_block_list_with_nothing_usable_is_refused(tiny_h3):
    _p, status = _load(tiny_h3, blocks="40-49")
    assert "attention_bandit: failed to install" in status


def test_a_rating_changes_the_taste_cache_key_so_the_same_seed_runs_again(tiny_h3, monkeypatch):
    from modules.sampling import attention_bandit as m
    from modules.system.taste import cache_key, store
    patched, _ = _load(tiny_h3, blocks="0-3")
    assert cache_key() == ""
    monkeypatch.setattr(m, "draw", lambda table, rng=random: [0] * len(table))
    _run(_outer(patched), lambda: tiny_h3.run(patched))
    assert cache_key() == ""                          # a waiting capture is not a rating
    store.rate("run-1", "liked")
    first = cache_key()
    assert first
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-2")
    _run(_outer(patched), lambda: tiny_h3.run(patched))
    store.rate("run-2", "liked")
    assert cache_key() != first


def test_the_log_counts_rated_runs():
    from modules.sampling.attention_bandit import tally
    rows = [_row(1.0, [1], [1, 0, 0]), _row(-1.0, [1], [1, 0, 0]), _row(-1.0, [1], [1, 0, 0])]
    assert tally(rows) == "3 rated runs (1 liked, 2 disliked)"
    assert tally([]) == "0 rated runs (0 liked, 0 disliked)"
