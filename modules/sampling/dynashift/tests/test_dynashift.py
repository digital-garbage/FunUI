"""DynaShift: the schedule it makes."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _s(n=4, dtype=torch.float32):
    return torch.linspace(1.0, 0.0, n + 1, dtype=dtype)


def test_ends_kept_order_kept_and_dtype_kept():
    from modules.sampling.dynashift import shifted
    for dt in (torch.float32, torch.float64, torch.bfloat16):
        s = _s(6, dt)
        o = shifted(s, 2.0, 0.5)
        assert o.dtype == dt and o[0] == s[0] and o[-1] == s[-1]
        assert bool((o[1:] < o[:-1]).all())


def test_shift_one_changes_nothing_and_high_shift_lifts_early_levels():
    from modules.sampling.dynashift import shifted
    s = _s(4)
    assert torch.allclose(shifted(s, 1.0, 1.0), s)
    assert float(shifted(s, 3.0, 3.0)[1]) > float(s[1])


def test_long_runs_fade_out():
    from modules.sampling.dynashift import shifted
    s = _s(40)
    assert (shifted(s, 3.0, 3.0) - s).abs().max() < (shifted(s, 3.0, 3.0, adapt=False) - s).abs().max()


def test_two_stages_join_up():
    from modules.sampling.dynashift import shifted
    full = _s(6)
    a, b = full[:4], full[3:]
    oa = shifted(a, 2.0, 1.0, 0, 6)
    ob = shifted(b, 2.0, 1.0, 3, 6)
    assert oa[-1] == ob[0]


def test_it_runs_through_the_loader(tiny_h3):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    patched, status = FunUILoadModifiers.execute(tiny_h3.patcher, {"dynashift": {"enabled": True}}).result
    assert "shift" in str(status)
