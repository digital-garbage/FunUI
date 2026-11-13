"""DynaShift: a noise-schedule shift that changes from step to step and follows the schedule it is given.

A flow model's noise levels (sigmas) are often bent with one fixed "shift": sigma' = s*sigma / (1 + (s-1)*sigma).
A shift above 1 spends more of the run at high noise (layout and motion), below 1 more at low noise (detail).

This module bends the schedule the run was given (so it works with whatever scheduler is chosen) with a shift that
moves from "Shift at the start" to "Shift at the end" across the whole run, Two stages included. Few-step runs
get the full effect and long runs fade it out (each step of a turbo run matters far more than one of fifty).
The first and last noise level of every stage stay as they are, so the hand-over between stages and the final
clean step are untouched. Video and sound share the noise levels, so both follow.

No extra model pass.
"""

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log

ID = "dynashift"
TITLE = "DynaShift"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["audio_stream"]

FULL_UNTIL = 8        # at or below this many steps the shift is applied in full

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Shift the noise schedule step by step",
        "hint": "Bends the noise levels of the chosen scheduler, built for turbo and few-step runs. No extra model calls.",
    },
    "start": {
        "type": "float", "default": 1.5, "min": 0.25, "max": 4.0, "step": 0.05,
        "label": "Shift at the start", "ui": "slider",
        "hint": "Above 1 spends the early steps on layout and motion, below 1 on detail. 1 = no change.",
        "when": {"enabled": True},
    },
    "end": {
        "type": "float", "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
        "label": "Shift at the end", "ui": "slider",
        "hint": "The same for the last steps; the shift moves from the start value to this one across the run.",
        "when": {"enabled": True},
    },
    "adapt": {
        "type": "bool", "default": True,
        "label": "Fade out on long runs",
        "hint": "Full effect up to 8 steps, weaker as the run gets longer.",
        "when": {"enabled": True},
    },
}


def shifted(sigmas, start, end, offset=0, total=None, adapt=True):
    """`sigmas` (1-D, descending, 1 -> 0) with every interior value bent by a shift running from `start` to `end`
    over `total` steps, this schedule beginning at global step `offset`. First and last values are kept."""
    n = len(sigmas) - 1
    total = total or n
    if n < 2:
        return sigmas
    s = sigmas.detach().double().clone()
    weight = min(1.0, FULL_UNTIL / total) if adapt else 1.0
    prev = s[0]
    for i in range(1, n):
        p = (offset + i) / max(1, total)
        k = 1.0 + (start + (end - start) * p - 1.0) * weight
        v = s[i]
        v = k * v / (1.0 + (k - 1.0) * v)
        v = torch.minimum(v, prev - 1e-4)                  # stay strictly descending
        v = torch.maximum(v, s[n] + 1e-4)
        s[i] = prev = v
    return s.to(device=sigmas.device, dtype=sigmas.dtype)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    start, end = float(values.get("start", 1.5)), float(values.get("end", 1.0))
    adapt = bool(values.get("adapt", True))

    def wrapper(executor, model_wrap, sigmas, *args, **kwargs):
        guider = getattr(model_wrap, "inner_model", None)
        stage = dit_hooks.stage_of(guider)
        full = getattr(guider, dit_hooks.FULL, None)
        steps = len(sigmas) - 1
        total = (len(full) - 1) if full is not None else steps
        offset = total - steps if stage == 2 else 0
        out = shifted(sigmas, start, end, offset, total, adapt)
        log.once(f"{ID}:{stage}:{steps}:{offset}", log.INFO, TITLE,
                 f"{'stage ' + str(stage) if stage else 'run'}: {steps} step(s), noise levels "
                 + ", ".join(f"{float(a):.3f}" for a in out.flatten()[:12].tolist()))
        return executor(model_wrap, out, *args, **kwargs)

    patcher.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, wrapper)
    return f"shift {start:g} -> {end:g}" + (", fading out above 8 steps" if adapt else "")


PROVIDES = {"modifier": install}
