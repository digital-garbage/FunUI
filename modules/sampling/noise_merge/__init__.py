"""Noise merging: at chosen steps, run the step a second time on the same picture with
different noise and blend the two answers.

Step N is answered twice. Picture A is the run's own answer. Picture B is the answer to
the same step after A's picture is re-noised to this step's noise level with a second
seed (and so is the sound, when the model has it). The two are then mixed:

    merged = base + strength * (other - base)

`Merge order` picks the base. "B into A" keeps A's picture and lets B's variation in;
"A into B" starts from B and lets A in. Two independent answers averaged lose contrast
(the independent detail partly cancels), so the mix is rescaled to the base's own spread.

Costs one extra model pass per merged step. The second pass is marked as a probe, so no
learning feature records it.
"""

import random
import time

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, streams

ID = "noise_merge"
TITLE = "Noise merging"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Noise merging",
        "hint": "At the steps below, answers the step twice with different noise and blends the two. Costs one extra pass per merged step.",
    },
    "what": {
        "type": "enum", "default": "answers", "options": [
            {"value": "answers", "label": "Step answers"},
            {"value": "noise", "label": "Starting noise"},
        ],
        "label": "Merge", "ui": "segmented",
        "hint": "Step answers: at chosen steps the model answers twice and the two pictures are blended (one extra pass each). Starting noise: the run starts from your seed's noise mixed with the second seed's, at no extra cost.",
        "when": {"enabled": True},
    },
    "steps": {
        "type": "text", "default": "1",
        "label": "Steps",
        "hint": "Step numbers from 1, e.g. 1 or 2-3 or 1,4. Steps past the schedule's end are ignored.",
        "when": {"enabled": True, "what": "answers"},
    },
    "order": {
        "type": "enum", "default": "b_into_a", "options": [
            {"value": "b_into_a", "label": "B into A"},
            {"value": "a_into_b", "label": "A into B"},
        ],
        "label": "Merge order", "ui": "segmented",
        "hint": "A is the picture your seed makes, B the same step on a second seed. The first named is blended in; the other is the base that keeps its overall look.",
        "when": {"enabled": True, "what": "answers"},
    },
    "strength": {
        "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Merge strength", "ui": "slider",
        "hint": "0 keeps your own result untouched, 1 replaces it with the second seed's. For starting noise, 0.5 is an even mix.",
        "when": {"enabled": True},
    },
    "sound": {
        "type": "bool", "default": True,
        "label": "Merge sound too",
        "hint": "Blends the sound the same way as the picture. Off keeps only the base's own sound.",
        "when": {"enabled": True, "what": "answers"},
    },
    "seed": {
        "type": "int", "default": -1, "min": -1, "max": 2147483647,
        "label": "Second seed",
        "hint": "The noise for picture B (or the second noise). -1 picks a new one every run; a number gives the same B each time.",
        "when": {"enabled": True},
    },
}


def _say(message, level=log.ALERT):
    log.once(f"{ID}:{message}", level, "FunUI Noise merging", message)


def merge(base, other, strength):
    """base + strength * (other - base), rescaled so each clip keeps the base's spread
    (std around its own mean) -- a plain mix of two independent answers is flatter."""
    mixed = base + strength * (other - base)
    dims = tuple(range(1, base.dim()))
    base_std = base.float().std(dim=dims, keepdim=True)
    mixed_std = mixed.float().std(dim=dims, keepdim=True).clamp(min=1e-8)
    mean = mixed.float().mean(dim=dims, keepdim=True)
    return ((mixed.float() - mean) * (base_std / mixed_std) + mean).to(base.dtype)


def renoise(x0, sigma, seed, step):
    """A rectified-flow input at `sigma` for picture x0 with fresh noise: (1 - sigma) x0 + sigma noise."""
    gen = torch.Generator().manual_seed(int(seed) + 7919 * int(step))
    noise = torch.randn(x0.shape, generator=gen).to(device=x0.device, dtype=x0.dtype)
    return (1.0 - sigma) * x0 + sigma * noise


def _sound(packed, video):
    """The part of a packed [B, 1, N] result that follows the picture (H3's sound), or None."""
    if getattr(packed, "dim", None) is None or packed.dim() != 3:
        return None
    n = video[0].numel()
    return packed[..., n:] if packed.shape[-1] > n else None


def mix_noise(noise, other, strength):
    """cos(t) noise + sin(t) other, t = strength * 90 degrees: two independent unit-variance noises
    stay unit variance (a plain average would shrink to 71% at an even mix, and the model reads that as a calmer, blurrier start)."""
    theta = strength * torch.pi / 2
    return (torch.cos(torch.tensor(theta)) * noise + torch.sin(torch.tensor(theta)) * other).to(noise.dtype)


def install_noise(patcher, values, key):
    strength = float(values.get("strength", 0.5))
    if strength <= 0.0:
        _say("off: strength is 0")
        return None
    chosen = int(values.get("seed", -1))

    def sampler_sample(executor, model_wrap, sigmas, extra_args, callback, noise,
                       latent_image=None, denoise_mask=None, disable_pbar=False):
        if not isinstance(noise, torch.Tensor):
            _say("off this run: the starting noise is not a plain tensor")
            return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        seed = chosen if chosen >= 0 else random.randrange(2 ** 31)
        other = torch.randn(noise.shape, generator=torch.Generator().manual_seed(seed)).to(noise)
        log.info("FunUI Noise merging", f"on: starting noise mixed with the second seed's ({seed}) at "
                 f"strength {strength:.2f}, no extra pass")
        return executor(model_wrap, sigmas, extra_args, callback, mix_noise(noise, other, strength),
                        latent_image, denoise_mask, disable_pbar)

    patcher.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, sampler_sample)
    return f"starting noise, strength {strength:.2f}, second seed {'random' if chosen < 0 else chosen}"


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    if values.get("what") == "noise":
        return install_noise(patcher, values, key)
    wanted, problems = dit_hooks.parse_blocks(values.get("steps"), 100000)
    wanted = {i - 1 for i in wanted if i >= 1}
    for p in problems:
        _say(f"ignored step entry: {p}")
    strength = float(values.get("strength", 0.5))
    if not wanted or strength <= 0.0:
        _say("off: " + ("no valid steps named" if not wanted else "strength is 0"))
        return None
    b_first = values.get("order", "b_into_a") == "b_into_a"
    with_sound = bool(values.get("sound", True))
    chosen = int(values.get("seed", -1))
    live = {"seed": max(chosen, 0), "merged": 0}

    def start(executor, *args, **kwargs):
        live["seed"] = chosen if chosen >= 0 else random.randrange(2 ** 31)
        live["merged"] = 0
        log.info("FunUI Noise merging", f"on: steps {values.get('steps')}, strength {strength:.2f}, "
                 f"second seed {live['seed']}")
        try:
            return executor(*args, **kwargs)
        finally:
            if not live["merged"]:
                _say("Inactive | no step of this run matched the steps you named "
                     "(or the model was not called as a plain prompt step)")

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, start)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options") or {}
        out = executor(x, t, *args, **kwargs)
        where = dit_hooks.current_step(to)
        if where is None or dit_hooks.probing(to) or where[0] not in wanted:
            return out
        if any(int(i) != 0 for i in (to.get("cond_or_uncond") or [0])):
            return out                              # a negative-prompt call: guidance is for the prompt's picture
        try:
            split_x, split_a = streams.video_of(x, named), streams.video_of(out, named)
            if split_x is None or split_a is None:
                _say("off this run: could not find the picture in this model's latent")
                return out
            sigma = float(t.max())
            video_a, rebuild_a = split_a
            tail_a = _sound(out, video_a)
            gen_step = where[0]
            x_b = split_x[1](renoise(video_a.detach(), sigma, live["seed"], gen_step).to(split_x[0].dtype))
            if tail_a is not None and with_sound:
                n = x_b.shape[-1] - tail_a.shape[-1]
                x_b = torch.cat([x_b[..., :n], renoise(tail_a.detach(), sigma, live["seed"] + 1, gen_step)], dim=-1)
            args_b, kwargs_b = streams.with_options(args, kwargs, {**to, dit_hooks.PROBE: True})
            began = time.perf_counter()
            out_b = executor(x_b, t, *args_b, **kwargs_b)
            took = time.perf_counter() - began
            split_b = streams.video_of(out_b, named)
            if split_b is None:
                return out
            video_b, rebuild_b = split_b
            tail_b = _sound(out_b, video_b)
            live["merged"] += 1
            diff = float((video_a - video_b).norm() / video_a.norm().clamp(min=1e-8))
            log.info("FunUI Noise merging",
                     f"step {where[0] + 1}: second answer differs from the first by {diff:.1%}, "
                     f"merged at {strength:.2f}{' (picture and sound)' if tail_a is not None and with_sound else ''}; "
                     f"the second pass took {took:.1f}s")
            if b_first:
                base, other, base_tail, other_tail, rebuild = video_a, video_b, tail_a, tail_b, rebuild_a
            else:
                base, other, base_tail, other_tail, rebuild = video_b, video_a, tail_b, tail_a, rebuild_b
            merged = rebuild(merge(base, other, strength).to(base.dtype))
            if with_sound and base_tail is not None and other_tail is not None:
                n = merged.shape[-1] - base_tail.shape[-1]
                merged = torch.cat([merged[..., :n], merge(base_tail, other_tail, strength).to(merged.dtype)], dim=-1)
            return merged
        except Exception as exc:                    # noqa: BLE001
            log.broke("FunUI Noise merging", exc, "merging the second answer")
            return out

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return (f"steps {values.get('steps')}, {'B into A' if b_first else 'A into B'}, "
            f"strength {strength:.2f}, second seed {'random' if chosen < 0 else chosen}")


PROVIDES = {"modifier": install}
