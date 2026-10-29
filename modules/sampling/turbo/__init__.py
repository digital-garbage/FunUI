"""Turbo sampler: a step rule made for few-step turbo runs, aware of Two stages, for any step count.

Replaces the chosen sampler for the run (the model calls are the same: one per step, no extra cost) with a
flow step that has these dials:

* Fresh noise per step. After each model answer the next point is `(1 - s) * picture + s * noise`; the
  noise is the model's own implied noise, blended with fresh noise by `eta` (0 = plain Euler, 1 = a full
  re-draw as in consistency sampling). `eta` runs from "Noise at the start" to "Noise at the end" across the
  WHOLE schedule, so it means the same for 3+2 steps, 4+4 or a run without Two stages.
* Second order late in the run: the picture guesses of the last steps are extrapolated from the one before
  (as in Distilled Flow's `dflow_ord2_ab2`), faded in with the run's progress. History starts fresh in each
  stage, because a Two stages hand-over changes the size.
* Polish step (on by default at 0.25): after the last step, noise the finished picture back to a small sigma and let the
  model answer once more at full size. One extra model call, in the last stage only.

The sound is stepped without fresh noise and without second order. Samplers that call the model twice per
step are replaced, not wrapped; the other sampler wrappers still run.

Ideas and credit: consistency models (Song et al., arXiv 2303.01469) and latent consistency models (Luo et
al., arXiv 2310.04378) for the re-noise-and-answer multistep rule; k-diffusion (Katherine Crowson) for
ancestral-style noise injection; NVIDIA Sol-H3 (arXiv 2609.35110) for the cross-resolution schedule this is
tuned around. See the README Thank you section.
"""

import math

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log

ID = "turbo_sampler"
TITLE = "Turbo sampler"
MOUNT = "generation.sampling"
STAGE = "load"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["audio_stream"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Turbo sampler",
        "hint": "Replaces the chosen sampler with a step rule made for few-step turbo runs. No extra model calls "
                "(except the optional polish step). Works with any step count and with Two stages.",
    },
    "noise_start": {
        "type": "float", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Noise at the start", "ui": "slider",
        "hint": "How much of each early step's noise is drawn fresh. 0 = plain Euler, 1 = a full re-draw.",
        "when": {"enabled": True},
    },
    "noise_end": {
        "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Noise at the end", "ui": "slider",
        "hint": "The same for the last steps; it fades from the start value to this one across the whole run.",
        "when": {"enabled": True},
    },
    "second_order": {
        "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Second order late", "ui": "slider",
        "hint": "How much the late steps lean on the step before. 0 = off. It fades in over the run. Strong values can distort the picture.",
        "when": {"enabled": True},
    },
    "polish": {
        "type": "float", "default": 0.25, "min": 0.0, "max": 0.6, "step": 0.01,
        "label": "Polish step sigma", "ui": "slider",
        "hint": "0 = off. One extra full-size model call after the last step, started from this noise level: a short "
                "re-pass over the finished video. Costs one step. From about 0.4 the sound is redrawn too.",
        "when": {"enabled": True},
    },
}


def _parts(t):
    return list(t.unbind()) if getattr(t, "is_nested", False) else [t]


def _join(like, parts):
    if getattr(like, "is_nested", False):
        from comfy.nested_tensor import NestedTensor
        return NestedTensor(tuple(parts))
    return parts[0]


def _noise(like, generator):
    return torch.randn(like.shape, generator=generator, device=like.device, dtype=like.dtype)


def step_to(x, x0, sigma, to, eta, noise):
    """One flow step from `sigma` to `to` given the picture guess `x0`; `eta` of the next point's noise is fresh `noise`."""
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    to = to.to(device=x.device, dtype=x.dtype)
    implied = (x - (1 - sigma) * x0) / sigma
    if eta > 0.0:
        implied = math.sqrt(max(0.0, 1.0 - eta * eta)) * implied + eta * noise
    return (1 - to) * x0 + to * implied


def make(settings):
    """The sampler function. It reads the stage from the guider behind `model` (set by Two stages), so the same
    function serves the registered `funui_turbo` sampler and the settings module."""
    n_start, n_end = settings["noise_start"], settings["noise_end"]
    order, tail = settings["second_order"], settings["polish"]

    @torch.no_grad()
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **_ignored):
        extra_args = {} if extra_args is None else extra_args
        guider = getattr(model, "inner_model", None)
        stage = dit_hooks.stage_of(guider)
        full = getattr(guider, dit_hooks.FULL, None)
        total = (len(full) - 1) if full is not None else len(sigmas) - 1
        n0 = total - (len(sigmas) - 1) if stage == 2 else 0                  # global index of this run's first step
        last_stage = stage != 1
        steps = len(sigmas) - 1
        log.once(f"turbo:{stage}:{steps}:{n0}", log.INFO, TITLE,
                 f"{'stage ' + str(stage) if stage else 'run'}: {steps} step(s) from step {n0 + 1} of {total}"
                 + (", polish step after the last" if tail > 0 and last_stage else ""))
        base = int(extra_args.get("seed") or 0)
        parts_like = _parts(x)
        gens = []
        for p in parts_like:
            g = torch.Generator(device=p.device)
            g.manual_seed(base + 1000003 * (n0 + 1))
            gens.append(g)
        ones = x.new_ones([x.shape[0]])
        before = before_h = None
        used = x
        for i in range(steps):
            sigma, to = sigmas[i], sigmas[i + 1]
            denoised = model(x, sigma * ones, **extra_args)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
            progress = (n0 + i) / max(1, total - 1)
            eta = n_start + (n_end - n_start) * progress
            h = float((sigma - to).abs())
            d_parts = _parts(denoised)
            used_parts = list(d_parts)
            if order > 0.0 and before is not None and before_h > 1e-7 and h > 1e-7:
                r = min(5.0, max(0.1, h / before_h))
                used_parts[0] = d_parts[0] + order * progress * (d_parts[0] - before[0]) * (r / 2)   # picture only
            before, before_h = d_parts, h
            used = _join(denoised, used_parts)
            if float(to) == 0:
                break
            nxt = []
            for k, (xp, up) in enumerate(zip(_parts(x), used_parts)):
                e = eta if k == 0 else 0.0                                    # the sound gets no fresh noise
                nxt.append(step_to(xp, up, sigma, to, e, _noise(xp, gens[k]) if e > 0.0 else None))
            x = _join(x, nxt)
        else:
            return x
        if tail > 0.0 and last_stage:
            noisy = []
            for k, up in enumerate(_parts(used)):
                t = torch.as_tensor(tail, device=up.device, dtype=up.dtype)
                noisy.append((1 - t) * up + t * _noise(up, gens[k]))
            t = torch.as_tensor(tail, device=parts_like[0].device, dtype=parts_like[0].dtype)
            used = model(_join(used, noisy), t * ones, **extra_args)
        return used
    return sample


def _settings(values):
    return {name: float(values.get(name, spec["default"])) for name, spec in SETTINGS.items() if name != "enabled"}


_CURRENT = {}                                   # the settings of the run in hand, for the registered sampler


def registered(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
    """`funui_turbo` in ComfyUI's sampler list: the settings when the module is on, else the defaults."""
    return make(_CURRENT or _settings({}))(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, **options)


def register():
    """Add the sampler to ComfyUI's own list (in place: its Sampler nodes share the list)."""
    import comfy.k_diffusion.sampling as ksampling
    import comfy.samplers
    if "funui_turbo" not in comfy.samplers.SAMPLER_NAMES:
        ksampling.sample_funui_turbo = registered
        comfy.samplers.SAMPLER_NAMES.append("funui_turbo")


register()


def install(patcher, values, key):
    _CURRENT.clear()
    if not values.get("enabled"):
        return None
    settings = _settings(values)
    _CURRENT.update(settings)

    def sampler_sample(executor, guider, sigmas, *args, **kwargs):
        sampler = executor.class_obj
        inner = getattr(sampler, "sampler_function", None)
        if inner is None:
            log.once("turbo:shape", log.INFO, TITLE, "off this run: the sampler is not a plain sampler function")
            return executor(guider, sigmas, *args, **kwargs)
        sampler.sampler_function = make(settings)       # swapped for this call and passed ON: later wrappers still run
        try:
            return executor(guider, sigmas, *args, **kwargs)
        finally:
            sampler.sampler_function = inner

    patcher.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, sampler_sample)
    return (f"fresh noise {settings['noise_start']:g} to {settings['noise_end']:g}, second order {settings['second_order']:g}"
            + (f", polish at {settings['polish']:g}" if settings["polish"] > 0 else ""))


PROVIDES = {"modifier": install}
