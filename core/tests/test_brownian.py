"""Brownian trees built from a schedule never warn about their start (torchsde rounds t0 to 6 decimals)."""

import importlib
import warnings

import pytest
import torch

ksampling = pytest.importorskip("comfy.k_diffusion.sampling")
pytest.importorskip("torchsde")

SIGMAS = [0.9876543, 0.6, 0.4465445876121521, 0.0]       # smallest positive sigma rounds UP at the 7th decimal


def _brownian():
    return importlib.import_module("core.brownian")


def _run(cls, dtype, schedule=SIGMAS):
    sigmas = torch.tensor(schedule, dtype=dtype)
    x = torch.zeros(1, 2, 3, dtype=dtype)
    positive = sigmas[sigmas > 0]
    sampler = cls(x, positive.min(), sigmas.max(), seed=7, cpu=True)
    out = []
    for a, b in zip(positive[:-1], positive[1:]):      # samplers draw no noise on the step to 0
        out.append(sampler(a, b))
    return out


@pytest.fixture
def patched():
    original = ksampling.BrownianTreeNoiseSampler
    mod = _brownian()
    mod._installed = False
    mod.install()
    yield ksampling.BrownianTreeNoiseSampler
    ksampling.BrownianTreeNoiseSampler = original
    mod._installed = False


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stock_class_warns_and_ours_does_not(patched, dtype):
    stock = patched.__mro__[1]
    with pytest.warns(UserWarning, match="ta>=t0"):
        _run(stock, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # any warning fails the test
        out = _run(patched, dtype)
    assert all(o.dtype == dtype and o.device.type == "cpu" for o in out)


def test_install_twice_does_not_stack(patched):
    mod = _brownian()
    before = ksampling.BrownianTreeNoiseSampler
    mod.install()
    assert ksampling.BrownianTreeNoiseSampler is before


def test_widened_follows_dtype_and_device():
    mod = _brownian()
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        lo, hi = mod.widened(torch.tensor(0.4465445876, dtype=dtype), torch.tensor(0.99, dtype=dtype))
        assert lo.dtype == hi.dtype == dtype
        assert float(lo) < 0.4465445876 - 5e-7 and float(hi) > 0.99 or dtype == torch.bfloat16
