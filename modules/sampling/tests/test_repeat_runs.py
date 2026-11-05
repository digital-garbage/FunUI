"""Generating again must not change the answer: the Nth run of a model equals the first.

A model stays loaded between runs, and anything a run leaves behind on it (a wrapper that was installed twice, a
buffer that was never reset, a counter that kept counting) makes the next run different from the first, with no
error to say so. LTX was once suspected of this (videos losing quality the more was generated). Each modifier is
switched on alone on the real tiny H3, the same inputs go through it several times, and every run must give the
same numbers; the model's weights must be untouched; and feeding the loader's output back into the loader (what a
cached model does when the next prompt arrives) must install nothing twice.
"""
import hashlib
import random

import pytest
import torch

RUNS = 4


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _registry():
    from core import registry
    return registry.current()


def _cases():
    reg = _registry()
    return [pytest.param(spec.id, id=spec.id) for spec in sorted(reg.specs.values(), key=lambda s: s.id)
            if spec.provides.get("modifier") and "enabled" in spec.settings]


def _apply(model, module_id):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    return FunUILoadModifiers.execute(model.patcher, {"taste": {"key": "sweep"}, module_id: {"enabled": True}}).result


def _weights(patcher):
    h = hashlib.sha256()
    for name, p in sorted(patcher.model.diffusion_model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().float().numpy().tobytes())
    return h.hexdigest()


def _numbers(out):
    return [t.float().clone() for t in (out if isinstance(out, (list, tuple)) else [out]) if torch.is_tensor(t)]


def _wrappers(patcher):
    return sum(len(fns) for by_key in patcher.wrappers.values() for fns in by_key.values()), \
        sum(len(v) for v in patcher.model_options.values() if isinstance(v, list))


@pytest.mark.parametrize("module_id", _cases())
def test_h3_the_same_inputs_give_the_same_answer_every_run(tiny_h3, module_id):
    before = _weights(tiny_h3.patcher)
    patched, status = _apply(tiny_h3, module_id)
    if "failed to install" in status:
        pytest.skip("the four-block stand-in cannot take this module")
    runs = []
    for _ in range(RUNS):
        torch.manual_seed(0)
        random.seed(0)                    # the Attention explorer draws its tries from Python's random on purpose
        runs.append(_numbers(tiny_h3.sample(patched)))
    for n, run in enumerate(runs[1:], start=2):
        for first, now in zip(runs[0], run):
            assert torch.equal(first, now), f"{module_id}: run {n} differs from run 1 by {(first - now).abs().max():.3g}"
    assert _weights(tiny_h3.patcher) == before, f"{module_id} changed the model's weights"


@pytest.mark.parametrize("module_id", _cases())
def test_h3_feeding_the_loaders_output_back_in_installs_nothing_twice(tiny_h3, module_id):
    once, status = _apply(tiny_h3, module_id)
    if "failed to install" in status:
        pytest.skip("the four-block stand-in cannot take this module")
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    settings = {"taste": {"key": "sweep"}, module_id: {"enabled": True}}
    chained = once
    for _ in range(3):
        chained = FunUILoadModifiers.execute(chained, settings).result[0]
    assert _wrappers(chained) == _wrappers(once), f"{module_id}: chaining the loader stacked its hooks"
