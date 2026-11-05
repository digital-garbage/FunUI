"""A model that stays loaded must not get worse (or just different) the more it is used.

ComfyUI keeps the loader's output between runs: the next prompt or seed reaches the SAME patched model, with the same
closures, counters and buffers inside it. The suspicion that started this (LTX seemed to lose quality over many
generations, and in v4 it really did: hooks left on shared blocks stacked up) is checked here for every modifier, alone
and all together, with and without Two stages:

  * load once, sample several times through ComfyUI's own guider, with the seeds going 1, 2, 3 and back to 1: the second
    seed 1 must equal the first, bit for bit (nothing a run left behind may reach the next one);
  * the shared base model (weights, buffers, forward hooks, replaced forwards, patches) must be exactly as it was;
  * a run cut short half way (the interrupt button) must leave the cached model giving the same answer as a clean load.

The explorer-type features draw their recipe at random each run by design, so `random` is seeded with the run, like torch.
Not proven here: anything that needs a GPU, real weights or real kernels (SLA/Veda kernels, MXFP8 matmuls). Those are
Dex's rental runs; their per-run state is covered by the unit tests beside them.
"""
import hashlib
import random

import pytest
import torch

from modules.sampling.tests.test_combinations import (FEATURES, SIGMAS, STAGES, _check, _load, fresh,  # noqa: F401
                                                      _name)
from modules.sampling.two_stage.tests.test_two_stage import model  # noqa: F401 -- the module-scoped tiny H3

SEEDS = (1, 2, 3, 1)
SUBSETS = [(f,) for f in FEATURES] + [tuple(FEATURES)]


def _sample(patched, seed, interrupt_after=None):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    random.seed(seed)
    torch.manual_seed(seed)
    guider = comfy.samplers.CFGGuider(patched)
    cond = [[torch.randn(1, 5, 48), {}]]
    guider.set_conds(cond, cond)
    guider.set_cfg(1.0)
    latent = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    noise = NestedTensor((torch.randn(1, 24, 2, 8, 8), torch.randn(1, 32, 2, 3)))
    hook = None
    if interrupt_after is not None:
        import comfy.model_management as mm
        calls = []

        def cut(module, args):
            calls.append(1)
            if len(calls) > interrupt_after:
                raise mm.InterruptProcessingException()
        hook = patched.model.diffusion_model.register_forward_pre_hook(cut)
    try:
        return guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), torch.tensor(SIGMAS),
                             seed=seed, disable_pbar=True)
    finally:
        if hook is not None:
            hook.remove()


def _residue(base):
    """What a run could leave on the shared base model: its weights and buffers, and how many hooks sit on any
    module. (ComfyUI's own bookkeeping, like `comfy_patched_weights` or the original `forward` it puts back when it
    unpatches an object patch, is not a leftover: what counts is that a plain run still gives the plain answer, below.)"""
    dm = base.model.diffusion_model
    h = hashlib.sha256()
    for name, t in sorted(dm.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().float().numpy().tobytes())
    hooks = [(n, len(m._forward_hooks), len(m._forward_pre_hooks), len(m._backward_hooks)) for n, m in dm.named_modules()]
    return (h.hexdigest(), hooks, _numbers(_sample(_load(base, (), None)[0], 99)))


def _same(before, after, message):
    """Equal, or the message naming exactly what changed."""
    assert before[0] == after[0], f"{message}: weights or buffers changed"
    assert before[1] == after[1], f"{message}: hooks left on {[a[0] for a, b in zip(before[1], after[1]) if a != b]}"
    for a, b in zip(before[2], after[2]):
        assert torch.equal(a, b), f"{message}: a plain run now differs by {(a - b).abs().max():.3g}"


def _numbers(out):
    return [t.clone() for t in out.unbind()]


@pytest.mark.parametrize("variant", STAGES, ids=list(STAGES))
@pytest.mark.parametrize("subset", SUBSETS, ids=_name)
def test_a_cached_model_answers_the_same_after_other_runs(fresh, subset, variant):
    before = _residue(fresh)
    patched, _ = _load(fresh, subset, STAGES[variant])
    outs = {}
    for n, seed in enumerate(SEEDS, 1):
        out = _sample(patched, seed)
        _check(out, f"{_name(subset)} run {n}")
        if seed in outs:
            for first, now in zip(outs[seed], _numbers(out)):
                assert torch.equal(first, now), \
                    f"{_name(subset)} {variant}: run {n} (seed {seed}) differs from the first seed {seed} by {(first - now).abs().max():.3g}"
        else:
            outs[seed] = _numbers(out)
    _same(before, _residue(fresh), f"{_name(subset)}: a run left something on the shared base model")


@pytest.mark.parametrize("variant", STAGES, ids=list(STAGES))
@pytest.mark.parametrize("subset", SUBSETS, ids=_name)
def test_a_run_cut_short_leaves_the_cached_model_as_good_as_a_clean_load(fresh, subset, variant):
    import comfy.model_management as mm
    before = _residue(fresh)
    reference = _numbers(_sample(_load(fresh, subset, STAGES[variant])[0], 7))
    cached, _ = _load(fresh, subset, STAGES[variant])
    for cut in (0, 1, 2):
        with pytest.raises(mm.InterruptProcessingException):
            _sample(cached, 5, interrupt_after=cut)
        again = _numbers(_sample(cached, 7))
        for first, now in zip(reference, again):
            assert torch.equal(first, now), \
                f"{_name(subset)} {variant}: after a run cut at call {cut} the cached model differs by {(first - now).abs().max():.3g}"
    _same(before, _residue(fresh), f"{_name(subset)}: an interrupted run left something on the shared base model")
