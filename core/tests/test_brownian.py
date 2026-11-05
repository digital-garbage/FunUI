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


def test_a_query_is_clamped_in_the_dtype_it_meets_and_stays_inside_after_float32():
    mod = _brownian()
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        t = torch.tensor(0.1, dtype=dtype)
        out = mod.clamped(t, 0.4465455, 0.99)
        assert out.device == t.device and out.dtype == (dtype if dtype != torch.bfloat16 else torch.float32)
        assert float(out.float()) >= 0.4465455                  # the cast a CPU tree applies cannot put it back outside
    inside = torch.tensor(0.6)
    assert mod.clamped(inside, 0.446545, 0.99) is inside
    meta = torch.empty((), device="meta")
    assert meta.device.type == "meta"                          # the helper reads the value, so a meta tensor is refused loudly, not guessed
    with pytest.raises(Exception):
        mod.clamped(meta, 0.1, 0.9)


def test_random_schedules_on_the_cpu_tree_never_warn(patched):
    gen = torch.Generator().manual_seed(3)
    for _ in range(60):
        top = float(torch.rand(1, generator=gen)) * 0.99 + 0.01
        schedule = sorted((top * float(r) for r in torch.rand(5, generator=gen)), reverse=True) + [0.0]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _run(patched, torch.float32, schedule)


def test_the_noise_dtype_is_the_stock_dtype_for_a_reduced_precision_latent(patched):
    stock = patched.__mro__[1]
    x = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    sig = torch.tensor([0.9876543, 0.6, 0.4465445876])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = stock(x, sig.min(), sig.max(), seed=7, cpu=True)(sig[0], sig[1])
    got = patched(x, sig.min(), sig.max(), seed=7, cpu=True)(sig[0], sig[1])
    assert got.dtype == want.dtype
