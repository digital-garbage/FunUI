"""Alternating prompts: the per-step wrapper and its verdict across the two stages of one run."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _load(tiny, **values):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    patched, status = FunUILoadModifiers.execute(
        tiny.patcher, {"prompt_alternatives": {"separate": True, **values}}).result
    pick = lambda kind: [w for ws in patched.wrappers[kind].values() for w in ws][-1]       # noqa: E731
    return pick(WrappersMP.OUTER_SAMPLE), pick(WrappersMP.APPLY_MODEL), status


def _to(i, options):
    s = torch.linspace(1.0, 0.0, 9)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1], "funui_simultaneous": {"options": options, "cond_len": 5}}


def test_off_without_the_switch_or_without_steps(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    for values in ({"separate": False, "b_steps": "1"}, {"separate": True}):
        patched, _ = FunUILoadModifiers.execute(tiny_h3.patcher, {"prompt_alternatives": values}).result
        assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)


def test_each_assigned_step_reads_its_option_and_the_others_the_prompt_as_it_was(tiny_h3):
    _start, apply, status = _load(tiny_h3, a_steps="1", b_steps="2,1")          # step 1 is in both lists: A wins
    assert "A on steps 1" in status
    ctx = torch.randn(1, 5, 48)
    opts = [torch.full((1, 5, 48), 1.0), torch.full((1, 5, 48), 2.0)]
    seen = []

    def model(x, t, c_concat, c_crossattn, *a, **k):
        seen.append(float(c_crossattn.mean()))
        return x

    for i in range(4):
        apply(model, torch.zeros(1), torch.zeros(1), None, ctx, None, _to(i, opts))
    assert seen[0] == 1.0 and seen[1] == 2.0                                   # step 1 reads A (A wins), step 2 reads B
    assert seen[2] == seen[3] == pytest.approx(float(ctx.mean()))              # the rest: the prompt as it was


def test_the_verdict_counts_both_stages_and_comes_after_the_last(tiny_h3):
    from core import dit_hooks, log
    start, apply, _ = _load(tiny_h3, b_steps="1")
    ctx = torch.randn(1, 5, 48)
    opts = [torch.randn(1, 5, 48), torch.randn(1, 5, 48)]

    class Guider:
        pass

    def stage(n, reads):
        guider = Guider()
        if n:
            setattr(guider, dit_hooks.STAGE, n)

        class Executor:
            class_obj = guider

            def __call__(self, *a, **k):
                if reads:
                    apply(lambda x, *a, **k: x, torch.zeros(1), torch.zeros(1), None, ctx, None, _to(0, opts))

        log._reset()
        start(Executor())
        return [e["message"] for e in log.history() if "Inactive" in e["message"]]

    assert stage(1, True) == [] and stage(2, False) == []          # read in the first stage: never inactive
    assert stage(1, False) == [] and len(stage(2, False)) == 1     # read in neither: said once, after the last
    assert len(stage(None, False)) == 1
