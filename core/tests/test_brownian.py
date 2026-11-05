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


def test_the_noise_of_a_seed_is_the_stock_noise(patched):
    """The warning is silenced without moving the tree: same seed, same noise as ComfyUI's own sampler."""
    stock = patched.__mro__[1]
    for schedule in (SIGMAS, [1.0, 0.5, 0.25, 0.0], [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = _run(stock, torch.float32, schedule)
        got = _run(patched, torch.float32, schedule)
        assert all(torch.equal(a, b) for a, b in zip(want, got))


def test_a_query_is_clamped_on_its_own_device_and_never_rounded_out_of_the_span():
    mod = _brownian()
    for dtype in (torch.float32, torch.bfloat16):
        for device in ("cpu", "meta"):
            t = torch.tensor(0.4465445876, dtype=dtype, device="cpu")
            if device == "meta":
                t = t.to("meta")
                continue                                      # float() needs data: the meta stand-in cannot be read
            out = mod.clamped(t, 0.4475, 0.99)               # bf16 holds 0.4473 for it: below the bound, so it is lifted to the bound itself
            assert out.device == t.device and float(out) == 0.4475
    inside = torch.tensor(0.6)
    assert mod.clamped(inside, 0.446545, 0.99) is inside
