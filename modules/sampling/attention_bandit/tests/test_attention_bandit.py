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
    from modules.sampling.attention_bandit import LEARN_MAX, SPREAD_MIN, START_SPREAD, learned_spread
    sig = [2, 0, 9]
    assert learned_spread([], sig) == START_SPREAD
    assert learned_spread([_srow(1.0, [0, 1], 0.4)], sig) > START_SPREAD
    assert learned_spread([_srow(-1.0, [0, 1], 0.4)], sig) < START_SPREAD
    assert learned_spread([_srow(1.0, [1, 1], 0.4)], sig) == START_SPREAD           # nothing changed
    assert learned_spread([_srow(1.0, [0, 1], 0.4, sig=(3, 0, 9))], sig) == START_SPREAD   # other layout
    assert learned_spread([_srow(1.0, [0, 0], 0.5)] * 30, sig) == pytest.approx(LEARN_MAX, abs=1e-3)
    assert SPREAD_MIN <= learned_spread([_srow(-1.0, [0, 0], 0.5)] * 30, sig) <= LEARN_MAX


def test_dislikes_of_small_changes_never_grow_the_size():
    from modules.sampling.attention_bandit import START_SPREAD, learned_spread
    assert learned_spread([_srow(-1.0, [0, 1], 0.1)] * 20, [2, 0, 9]) == START_SPREAD
    assert learned_spread([_srow(-1.0, [0, 1], 0.4)] * 20, [2, 0, 9]) < START_SPREAD


def test_the_log_names_exact_blocks_and_skips_the_unchanged():
    from modules.sampling.attention_bandit import describe
    groups = [[2, 3, 4], [9, 10, 29, 30], [34, 35], [41, 42]]
    assert describe(groups, [1, 1, 1, 1], 0.2) == ""
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


def test_the_log_says_how_many_attention_calls_were_changed(tiny_h3, monkeypatch):
    from core import log
    from modules.sampling import attention_bandit as m
    patched, _ = _load(tiny_h3, mode="manual", blocks="0-3")
    monkeypatch.setattr(m, "draw", lambda table, rng=random: [0] * len(table))
    log.new_run()
    _run(_outer(patched), lambda: tiny_h3.run(patched))
    assert any("applied to" in e["message"] and "attention calls" in e["message"] for e in log.history())


# ── prompt weighting, recipes, prompt focus, clip regions ─────────────────────

def _cond(seed, n=16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g)


def test_the_same_prompt_counts_fully_another_prompt_counts_for_the_floor_and_unknown_in_between():
    from modules.sampling.attention_bandit import FLOOR, UNKNOWN, similarity, weights_of
    a, b = _cond(1), _cond(2)
    assert similarity(a, a) == pytest.approx(1.0)
    assert similarity(a, b) < 0.01 and similarity(a, None) is None and similarity(None, a) is None
    rows = [{"reward": 1.0, "rows": {"cond": a.half()}}, {"reward": 1.0, "rows": {"cond": b.half()}},
            {"reward": 1.0, "rows": {}}]
    w, sims = weights_of(rows, a)
    assert w[0] == pytest.approx(1.0, abs=1e-3) and w[1] == pytest.approx(FLOOR, abs=0.02) and w[2] == UNKNOWN


def test_counts_use_the_weights():
    from modules.sampling.attention_bandit import counts
    rows = [_row(1.0, [0], [1, 0, 0]), _row(1.0, [0], [1, 0, 0])]
    assert counts(rows, [1, 0, 0], 1, weights=[1.0, 0.1])[0][0][0] == pytest.approx(1.1)
    rows[0]["rows"]["tarms"] = torch.tensor([2])
    assert counts(rows, [1, 0, 0], 1, "tarms")[0][2] == [1.0, 0.0]          # only the row that has the field


def _combo_row(reward, cond, arms=(0, 1), tarms=(2, 1), rarms=(1, 1, 1, 1, 1), spread=0.3, sig=(2, 0, 9)):
    return {"reward": reward, "rows": {"arms": torch.tensor(list(arms)), "tarms": torch.tensor(list(tarms)),
                                       "rarms": torch.tensor(list(rarms)), "sig": torch.tensor(list(sig)),
                                       "spread": torch.tensor(spread), "cond": cond.half()}}


def test_a_combination_liked_on_this_prompt_is_reused_and_one_liked_on_another_is_not():
    from modules.sampling.attention_bandit import make_plan
    here, other = _cond(1), _cond(2)
    rows = [_combo_row(1.0, here) for _ in range(8)]
    rng = random.Random(0)
    plans = [make_plan(rows, here, [2, 0, 9], 2, True, regions=True, rng=rng) for _ in range(40)]
    reused = [p for p in plans if p["recipe"]]
    assert len(reused) > 30 and all(p["same"] == 8 for p in plans)
    assert sum(p["arms"][0] == 0 for p in reused) > 0.6 * len(reused)         # mostly the recipe, some groups re-drawn
    away = [make_plan(rows, other, [2, 0, 9], 2, True, regions=True, rng=rng) for _ in range(40)]
    assert not any(p["recipe"] for p in away) and all(p["same"] == 0 for p in away)


def test_disliked_combinations_are_not_a_recipe():
    from modules.sampling.attention_bandit import make_plan
    here = _cond(1)
    rows = [_combo_row(-1.0, here) for _ in range(8)]
    assert not any(make_plan(rows, here, [2, 0, 9], 2, True, regions=True, rng=random.Random(i))["recipe"] for i in range(30))


def _force(monkeypatch, arms=1, tarms=1, rarms=1):
    from modules.sampling import attention_bandit as m

    def fixed(table, rng=random):
        n = len(table)
        return [{4: arms, 5: rarms}.get(n, arms)] * n if n == 5 else [arms] * n
    seq = iter([])
    calls = []

    def draw(table, rng=random):
        calls.append(len(table))
        # the first draw is the plain groups, the second the prompt groups, the third the regions
        pick = (arms, tarms, rarms)[min(len(calls) - 1, 2)]
        return [pick] * len(table)
    monkeypatch.setattr(m, "draw", draw)


def test_prompt_focus_alone_changes_the_output_and_says_where(tiny_h3, monkeypatch):
    from core import log
    base = tiny_h3.run()
    patched, status = _load(tiny_h3, mode="manual", range=0.5, blocks="0-3", regions=False)
    assert "prompt focus on" in status
    _force(monkeypatch, arms=1, tarms=2)
    log.new_run()
    outs = _run(_outer(patched), lambda: tiny_h3.run(patched))
    assert not torch.equal(outs[0], base[0])
    messages = [e["message"] for e in log.history()]
    assert any("prompt stronger" in m and "blocks 0-3" in m for m in messages)
    assert any("prompt keys scaled in" in m for m in messages)


def test_with_prompt_focus_off_nothing_about_the_prompt_is_drawn_or_stored(tiny_h3, monkeypatch):
    from modules.system.taste import store
    patched, _ = _load(tiny_h3, blocks="0-3", prompt_focus=False, regions=False)
    _force(monkeypatch, arms=1)
    outs = _run(_outer(patched), lambda: tiny_h3.run(patched))
    assert torch.equal(outs[0], tiny_h3.run()[0])
    store.rate("run-1", "liked")
    assert "tarms" not in store.load("fox", "attention_bandit")["rows"][0]["rows"]


def test_clip_regions_change_only_the_rows_they_name(tiny_h3, monkeypatch):
    from modules.sampling.attention_bandit import REGIONS, region_spans
    captured = {}

    def spy(func, q, k, v, *a, **kw):
        options = kw.get("transformer_options") or {}
        # the prompt refiner also runs with the layout in its options, on the prompt rows alone
        if "minimax_h3_layout" in options and q.shape[2] > captured.get("rows", 0):
            captured["spans"] = region_spans(options)
            captured["rows"] = q.shape[2]
        return func(q, k, v, *a, **kw)

    from core import dit_hooks
    probe = tiny_h3.patcher.clone()
    dit_hooks.add_attention_override(probe, "test.spy", spy)
    tiny_h3.run(probe)
    spans = captured["spans"]
    assert len(spans) == len(REGIONS)
    flat = sorted(spans)
    assert all(a[1] <= b[0] for a, b in zip(flat, flat[1:]))                      # no row in two regions
    assert all(0 <= a < b <= captured["rows"] for a, b in spans)


def test_clip_regions_change_the_output_and_stay_off_when_all_as_is(tiny_h3, monkeypatch):
    from core import log
    base = tiny_h3.run()
    patched, _ = _load(tiny_h3, mode="manual", range=0.5, blocks="0-3", prompt_focus=False)
    _force(monkeypatch, arms=1, rarms=0)
    # prompt focus is off, so the draws are groups then regions
    from modules.sampling import attention_bandit as m
    picks = iter([1, 0])
    monkeypatch.setattr(m, "draw", lambda table, rng=random: [next(picks)] * len(table))
    log.new_run()
    outs = _run(_outer(patched), lambda: tiny_h3.run(patched))
    assert not torch.equal(outs[0], base[0])
    messages = [e["message"] for e in log.history()]
    assert any("clip regions sharper" in m and "sound" in m for m in messages)
    assert any("clip regions in" in m for m in messages)


def test_the_real_sla_override_underneath_still_gets_a_normal_call(tiny_h3, monkeypatch):
    """SLA hands anything off its fast path to dense and records a kernel failure if a call is odd.
    Regions rescale rows of q in place on a copy, so shape, dtype and layout must be what SLA saw before."""
    from core import dit_hooks
    from modules.loaders import sla_attention as sla
    seen = []

    def spy(func, q, k, v, *a, **kw):
        # strides are left out on purpose: the copy of q is compact, which SLA reads as well as the fused-buffer view
        seen.append((tuple(q.shape), q.dtype, tuple(k.shape), kw.get("skip_reshape"), kw.get("mask")))
        return func(q, k, v, *a, **kw)

    def run_with(on):
        probe = tiny_h3.patcher.clone()
        state = sla.new_state()
        to = dit_hooks._transformer_options(probe)
        cfg = {**sla.SLA_DEFAULTS, "engine": "triton", "dense_steps": frozenset(), "min_seq_len": 0}
        to["optimized_attention_override"] = sla.make_override(state, cfg)
        dit_hooks.add_attention_override(probe, "test.spy", spy)       # sits on top: sees exactly what SLA would get
        patched, _ = _load(type("T", (), {"patcher": probe})(), mode="manual", range=0.5, blocks="0-3")
        picks = iter([1, 1, 0 if on else 1])
        from modules.sampling import attention_bandit as m
        monkeypatch.setattr(m, "draw", lambda table, rng=random: [next(picks)] * len(table))
        seen.clear()
        _run(_outer(patched), lambda: tiny_h3.run(patched))
        return list(seen), state

    plain, state_plain = run_with(False)
    scaled, state_scaled = run_with(True)
    assert plain and scaled == plain                                  # the same calls, same shapes, strides and flags
    assert state_scaled["failed"] is None and state_scaled["calls"] == state_plain["calls"]
