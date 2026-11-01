"""The block map: the maths on constructed profiles, and a profile run on the real tiny H3."""

import pytest
import torch

from modules.sampling.block_map import measure


def _data(n=10, lev=None, frg=None):
    grid = measure.groups(n)
    g = len(grid)
    lev = lev or [[1.0] * 3 for _ in range(g)]
    frg = frg or [[0.1] * 3 for _ in range(g)]
    return {"blocks": n, "groups": grid, "points": list(measure.POINTS),
            "attention": {"leverage": lev, "fragility": frg}}


def test_a_flat_profile_changes_nothing():
    view = measure.MapView(_data())
    assert view.flat("attention")
    assert view.multipliers("attention", 0.5, [2, 3, 7]) == {2: 1.0, 3: 1.0, 7: 1.0}


def test_a_missing_kind_or_a_single_block_changes_nothing():
    view = measure.MapView(_data())
    assert view.multipliers("residual", 0.5, [2, 3]) == {2: 1.0, 3: 1.0}
    assert measure.MapView(_data(lev=[[1.0] * 3, [3.0] * 3])).multipliers("attention", 0.5, [4]) == {4: 1.0}


def test_nudges_are_smaller_where_leverage_is_big_and_smaller_where_detail_is_fragile():
    lev = [[1.0] * 3, [4.0] * 3]                   # blocks 5-9 move the picture four times as much
    frg = [[0.1] * 3, [0.1] * 3]
    m = measure.MapView(_data(lev=lev, frg=frg)).multipliers("attention", 0.5, [1, 2, 7, 8])
    assert m[1] > m[7] and m[2] > m[8]
    assert sum(m.values()) / 4 == pytest.approx(1.0, abs=0.35)       # about 1 on average (the clamp may bend it)
    lev = [[1.0] * 3, [1.0] * 3]
    frg = [[0.0] * 3, [0.8] * 3]                   # blocks 5-9 turn their move into fine noise
    m = measure.MapView(_data(lev=lev, frg=frg)).multipliers("attention", 0.5, [1, 2, 7, 8])
    assert m[1] > m[7]


def test_multipliers_stay_inside_their_limits_and_average_one_before_clamping():
    lev = [[1.0] * 3, [1000.0] * 3]
    m = measure.MapView(_data(lev=lev)).multipliers("attention", 0.5, [0, 1, 8, 9])
    assert all(measure.LOW <= v <= measure.HIGH for v in m.values())


def test_response_reads_the_size_of_the_move_and_how_much_of_it_is_fine_detail():
    base = torch.randn(1, 4, 2, 8, 8)
    smooth = torch.ones_like(base) * 0.01
    noisy = torch.randn_like(base) * 0.01
    lev_s, frag_s = measure.response(base, base + smooth)
    lev_n, frag_n = measure.response(base, base + noisy)
    assert lev_s == pytest.approx(lev_n, rel=0.5) and frag_s < 0.05 < 0.5 < frag_n
    assert measure.response(base, base) == (0.0, 0.0)


def test_response_follows_the_picture_dtype_and_device():
    base = torch.randn(1, 2, 2, 6, 6, dtype=torch.bfloat16)
    lev, frag = measure.response(base, base * 1.01)
    assert lev > 0 and 0.0 <= frag <= 1.0


def test_points_are_hit_once_each_and_a_one_step_run_has_none():
    hits = [measure.point_of(i, 8) for i in range(8)]
    assert [h for h in hits if h is not None] == [0, 1, 2]
    assert all(measure.point_of(i, 1) is None for i in range(1))


# --- the profile run, on the real tiny H3 -------------------------------------------------------------

@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(measure, "SWITCH", tmp_path / "switch")
    monkeypatch.setattr(measure, "FOLDER", tmp_path / "maps")


def _load(tiny, settings=None):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    patched, status = FunUILoadModifiers.execute(tiny.patcher, settings or {}).result
    wraps = [w for ws in patched.wrappers.get(WrappersMP.APPLY_MODEL, {}).values() for w in ws]
    outer = [w for ws in patched.wrappers.get(WrappersMP.OUTER_SAMPLE, {}).values() for w in ws]
    return patched, (wraps[-1] if wraps else None), outer


def _run(tiny, patched, wrap, outer, steps=4):
    from conftest import packed_av
    x0, shapes = packed_av(tiny.video, tiny.audio)
    seen = []

    def executor(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kw):
        to = dict(transformer_options or {})
        seen.append(to)
        video, audio = tiny.run(patched, sigma=float(t.max()),
                                options={k: v for k, v in to.items() if k == "funui_probe"})
        return packed_av(video, audio)[0]

    s = torch.linspace(1.0, 0.0, steps + 1)
    call = lambda: [wrap(executor, x0, s[i:i + 1].reshape(1, 1), None, None, None,
                         {"sample_sigmas": s, "sigmas": s[i:i + 1], "cond_or_uncond": [0]},
                         latent_shapes=shapes) for i in range(steps)]
    for o in reversed(outer):
        call = (lambda w, inner: (lambda: w(lambda: inner())))(o, call)
    return call(), seen


def test_off_it_installs_no_hooks_and_publishes_no_map(tiny_h3, root):
    patched, wrap, outer = _load(tiny_h3)
    assert wrap is None and outer == []
    assert patched.model_options["transformer_options"].get("funui_block_map") is None


def test_a_profile_run_writes_a_map_and_returns_the_untouched_answers(tiny_h3, root):
    patched, _w, outer = _load(tiny_h3)
    outs0, _ = _run(tiny_h3, patched, lambda ex, *a, **k: ex(*a, **k), outer)
    measure.set_enabled(True)
    patched, wrap, outer = _load(tiny_h3)
    outs, seen = _run(tiny_h3, patched, wrap, outer)
    assert all(torch.equal(a, b) for a, b in zip(outs, outs0))
    n = len(tiny_h3.patcher.model.diffusion_model.blocks)
    per_point = 2 * len(measure.groups(n))
    assert len(seen) == 4 + 3 * per_point
    assert [bool(t.get("funui_probe")) for t in seen][:1 + per_point] == [False] + [True] * per_point
    assert not measure.enabled()                                      # one profile per switch
    data = measure.load(n)
    assert data["blocks"] == n and set(measure.KINDS) <= set(data)
    for kind in measure.KINDS:
        lev = data[kind]["leverage"]
        assert len(lev) == len(measure.groups(n)) and all(v is not None and v > 0 for row in lev for v in row)


def test_a_later_run_publishes_the_map_and_follow_off_hides_it(tiny_h3, root):
    measure.set_enabled(True)
    patched, wrap, outer = _load(tiny_h3)
    _run(tiny_h3, patched, wrap, outer)
    measure.set_enabled(False)
    patched, _w, _o = _load(tiny_h3)
    view = patched.model_options["transformer_options"]["funui_block_map"]
    assert view is not None and view.n == len(tiny_h3.patcher.model.diffusion_model.blocks)
    patched, _w, _o = _load(tiny_h3, {"block_map": {"follow": False}})
    assert patched.model_options["transformer_options"].get("funui_block_map") is None


def test_a_failing_probe_pass_costs_nothing(tiny_h3, root):
    from conftest import packed_av
    measure.set_enabled(True)
    patched, wrap, outer = _load(tiny_h3)
    x0, shapes = packed_av(tiny_h3.video, tiny_h3.audio)

    def executor(x, t, *a, **kw):
        to = kw.get("transformer_options") or (a[3] if len(a) > 3 else {})
        if (to or {}).get("funui_probe"):
            raise RuntimeError("probe broke")
        return x0
    s = torch.linspace(1.0, 0.0, 5)
    out = wrap(executor, x0, torch.tensor([[1.0]]), None, None, None,
               {"sample_sigmas": s, "sigmas": s[0:1], "cond_or_uncond": [0]}, latent_shapes=shapes)
    assert out.shape == x0.shape


def test_the_helper_every_feature_asks_is_the_value_itself_without_a_map_and_follows_the_step_with_one():
    from modules._core import dit_hooks
    assert dit_hooks.block_map_value({}, "residual", 3, 0.4, [3, 4]) == 0.4
    assert dit_hooks.block_map_value(None, "residual", 3, 0.4, [3, 4]) == 0.4
    data = _data(lev=[[1.0] * 3, [4.0] * 3])
    data["residual"] = data["attention"]
    options = {"funui_block_map": measure.MapView(data), "sample_sigmas": torch.linspace(1, 0, 5), "sigmas": torch.tensor([0.5])}
    assert dit_hooks.block_map_value(options, "residual", 1, 0.4, [1, 8]) > dit_hooks.block_map_value(options, "residual", 8, 0.4, [1, 8])
    assert dit_hooks.block_map_value(options, "attention", 1, 0.4, [1]) == pytest.approx(0.4, rel=0.05)   # one block: nothing to spread, only the soft limit
    broken = {"funui_block_map": object(), "sample_sigmas": None}
    assert dit_hooks.block_map_value(broken, "residual", 1, 0.4, [1, 2]) == 0.4


def _view(lev, frg=None):
    data = _data(lev=lev, frg=frg)
    data["residual"] = data["attention"]
    return measure.MapView(data)


def test_a_huge_setting_is_kept_but_cannot_push_a_block_past_what_it_takes():
    view = _view([[1.0] * 3, [4.0] * 3])
    blocks = [1, 2, 7, 8]
    small = view.adjust("attention", 7, 0.5, blocks, 0.05)
    assert 0.0 < small <= 0.05 * measure.HIGH + 1e-9                  # an ordinary setting passes (spread, not limited)
    big_low, big_high = view.adjust("attention", 1, 0.5, blocks, 10.0), view.adjust("attention", 7, 0.5, blocks, 10.0)
    assert 1.0 < big_low < 10.0 and 0.0 < big_high < big_low          # 10 is respected as far as the block allows
    assert view.adjust("attention", 7, 0.5, blocks, -10.0) == pytest.approx(-big_high)       # the sign is kept


def test_adjust_leaves_neutral_and_flat_settings_alone_and_follows_the_run():
    assert measure.MapView(_data()).adjust("attention", 3, 0.5, [2, 3], 0.3) == 0.3
    view = _view([[1.0] * 3, [4.0] * 3])
    assert view.adjust("attention", 3, 0.5, [2, 3], 0.0) == 0.0
    assert view.adjust("residual", 3, 0.1, [2, 3], 0.3) != view.adjust("attention", 8, 0.1, [2, 3], 0.3)
