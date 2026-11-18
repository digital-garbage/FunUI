"""An [A|B] typed in a prompt (made-up words) is read per step on the real tiny H3, one stage and Two stages."""

import importlib

import pytest
import torch

from modules.sampling.two_stage.tests.test_two_stage import model  # noqa: F401 -- the module-scoped tiny H3

SIGMAS = (1.0, 0.9, 0.7, 0.3, 0.0)


def _seen(model, typed, staged, **alt):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    from core import prompt_build
    from modules.conditioning.prompt_markup.nodes import FunUIApplyPromptMarkup, FunUIPromptMarkup
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    clip = importlib.import_module("modules.conditioning.prompt_markup.tests.test_prompt_markup").FakeH3Clip()
    settings = {"taste": {"key": "fox"}, "prompt_alternatives": {"separate": True, **alt}}
    if staged:
        settings["two_stage"] = {"enabled": True, "size": 0.5, "steps": 2, "back_up": 0.5}
    patched, _ = FunUILoadModifiers.execute(model.clone(), settings).result
    text = prompt_build.build(typed)                         # what the app sends: the expand step first
    clean, markup, _ = FunUIPromptMarkup.execute(text, settings).result
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(clean))
    shapes = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    patched, positive, status = FunUIApplyPromptMarkup.execute(patched, clip, positive, {"samples": shapes}, markup).result
    assert status == "2 simultaneous option(s)", status
    seen = []
    hook = patched.model.diffusion_model.register_forward_pre_hook(
        lambda m, a, k: seen.append((a[2] if len(a) > 2 else k["context"]).detach().clone()), with_kwargs=True)
    guider = comfy.samplers.CFGGuider(patched)
    guider.set_conds(positive, positive)
    guider.set_cfg(1.0)
    try:
        guider.sample(NestedTensor((torch.randn(1, 24, 2, 8, 8), torch.randn(1, 32, 2, 3))), shapes,
                      comfy.samplers.sampler_object("euler"), torch.tensor(SIGMAS), seed=1, disable_pbar=True)
    finally:
        hook.remove()
    return seen


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("typed", ["a fox [runs|sleeps] in snow", "a fox runs in snow [RUNS|SLEEPS]"])
def test_each_assigned_step_reads_its_own_option(model, staged, typed):
    seen = _seen(model, typed, staged, a_steps="1", b_steps="2")
    assert len(seen) == 4, len(seen)
    assert not torch.allclose(seen[0], seen[1]), "step 1 (A) and step 2 (B) read the same prompt"
    assert torch.allclose(seen[2], seen[3])                  # the rest read the main prompt
