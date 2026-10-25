"""Noise merging: the maths and the wrapper on a packed H3 latent."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _sched(steps=8):
    return torch.linspace(1.0, 0.0, steps + 1)


def _to(i, steps=8):
    s = _sched(steps)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1]}


def _load(tiny, **values):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    patched, status = FunUILoadModifiers.execute(
        tiny.patcher, {"noise_merge": {"enabled": True, **values}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    return wrap, status


def _packed():
    from conftest import packed_av
    torch.manual_seed(2)
    return packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))


def test_merge_keeps_the_base_spread_and_strength_zero_is_the_base():
    base, other = torch.randn(2, 4, 3, 4, 4), torch.randn(2, 4, 3, 4, 4) * 3
    assert torch.allclose(merge_(base, other, 0.0), base, atol=1e-5)
    mixed = merge_(base, other, 0.5)
    assert torch.allclose(mixed.flatten(1).std(1), base.flatten(1).std(1), atol=1e-4)
    assert not torch.allclose(mixed, base)


def merge_(*a):
    from modules.sampling.noise_merge import merge
    return merge(*a)


def test_renoise_is_seeded_and_lands_on_the_step_noise_level():
    from modules.sampling.noise_merge import renoise
    x0 = torch.randn(1, 4, 3, 8, 8)
    assert torch.equal(renoise(x0, 0.5, 1, 2), renoise(x0, 0.5, 1, 2))
    assert not torch.equal(renoise(x0, 0.5, 1, 2), renoise(x0, 0.5, 2, 2))
    assert torch.equal(renoise(x0, 0.0, 1, 2), x0)


def test_a_named_step_runs_the_model_twice_and_others_once(tiny_h3):
    wrap, status = _load(tiny_h3, steps="2", strength=0.5)
    assert "steps 2" in status
    x0, shapes = _packed()
    calls = []

    def model(x, *a, **k):
        calls.append(a[4].get("funui_probe"))
        return x0 + 0.1 * x

    for i in range(4):
        wrap(model, x0.clone(), _sched()[i:i + 1], None, None, None, _to(i), latent_shapes=shapes)
    assert len(calls) == 5 and calls.count(True) == 1


def test_the_answer_changes_and_order_matters(tiny_h3):
    from conftest import unpacked
    x0, shapes = _packed()
    model = lambda x, *a, **k: x * 0.5 + 0.3 * x0
    results = {}
    for order in ("b_into_a", "a_into_b"):
        wrap, _ = _load(tiny_h3, steps="1", order=order, strength=0.7)
        results[order] = wrap(model, x0.clone(), _sched()[0:1], None, None, None, _to(0), latent_shapes=shapes)
    plain = model(x0, None)
    a, b = unpacked(results["b_into_a"], shapes), unpacked(results["a_into_b"], shapes)
    p = unpacked(plain, shapes)
    assert not torch.allclose(a[0], p[0]) and not torch.allclose(a[0], b[0])
    assert not torch.allclose(a[1], p[1])             # the sound is merged too


def test_nothing_named_or_strength_zero_installs_nothing(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    for bad in ({"steps": "x"}, {"strength": 0.0}):
        patched, _ = FunUILoadModifiers.execute(tiny_h3.patcher, {"noise_merge": {"enabled": True, **bad}}).result
        assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)


def test_a_negative_seed_is_new_each_run_and_a_number_is_kept(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    x0, shapes = _packed()
    for seed, same in ((-1, False), (5, True)):
        patched, status = FunUILoadModifiers.execute(
            tiny_h3.patcher, {"noise_merge": {"enabled": True, "seed": seed}}).result
        wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
        outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws][-1]
        seen = []
        model = lambda x, *a, **k: (seen.append(x), x0)[1]
        for _ in range(2):
            outer(lambda: wrap(model, x0.clone(), _sched()[0:1], None, None, None, _to(0), latent_shapes=shapes))
        assert torch.equal(seen[1], seen[3]) is same


def test_the_log_says_the_feature_is_on_and_when_nothing_matched(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from core import log
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    x0, shapes = _packed()
    patched, _ = FunUILoadModifiers.execute(
        tiny_h3.patcher, {"noise_merge": {"enabled": True, "steps": "9"}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws][-1]
    log.new_run()
    outer(lambda: wrap(lambda *a, **k: x0, x0.clone(), _sched()[0:1], None, None, None, _to(0), latent_shapes=shapes))
    texts = [e["message"] for e in log.history()]
    assert any(t.startswith("on: steps 9") for t in texts)
    assert any("no step of this run matched" in t for t in texts)


def test_the_sound_is_merged_too(tiny_h3):
    from conftest import unpacked
    x0, shapes = _packed()
    model = lambda x, *a, **k: x * 0.5 + 0.3 * x0
    wrap, _ = _load(tiny_h3, steps="1", strength=0.7)
    out = wrap(model, x0.clone(), _sched()[0:1], None, None, None, _to(0), latent_shapes=shapes)
    plain = unpacked(model(x0, None), shapes)
    got = unpacked(out, shapes)
    assert not torch.allclose(got[0], plain[0]) and not torch.allclose(got[1], plain[1])


def test_sound_off_keeps_the_base_sound_untouched(tiny_h3):
    from conftest import unpacked
    x0, shapes = _packed()
    model = lambda x, *a, **k: x * 0.5 + 0.3 * x0
    wrap, _ = _load(tiny_h3, steps="1", strength=0.7, sound=False)
    out = wrap(model, x0.clone(), _sched()[0:1], None, None, None, _to(0), latent_shapes=shapes)
    plain = unpacked(model(x0, None), shapes)
    got = unpacked(out, shapes)
    assert not torch.allclose(got[0], plain[0]) and torch.allclose(got[1], plain[1])
