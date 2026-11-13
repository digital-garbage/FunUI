"""Two stages: the first steps of a run on a smaller picture, the last steps at the full size.

The shot's layout and motion are decided in the first steps, when the picture is still mostly noise, and
a smaller picture decides them nearly as well for a fraction of the work. So the schedule is cut in two:
the first steps run on a picture scaled down, the model's clean guess at that point is resized to the full
size, and the remaining steps finish there. References and text are untouched; only the size of what is
sampled differs. This is the cross-resolution schedule of NVIDIA's Sol-H3 (arXiv 2609.35110); see the
README Thank you section.

It installs on the model, so it works with any sampler. Not yet run on a real H3 here.
"""

from comfy.patcher_extension import WrappersMP

import comfy.samplers
import folder_paths

from ..._core import dit_hooks, log

from .stages import METHODS, release_upscaler, keep_whole_schedule, make_wrapper

ID = "two_stage"
TITLE = "Two stages"
MOUNT = "generation.sampling"
STAGE = "load"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["audio_stream", "adaln_modalities"]


def _files():
    try:
        return folder_paths.get_filename_list("latent_upscale_models")
    except Exception:          # noqa: BLE001 -- folder not registered is the same as no files
        return []


def _options():
    return [{"value": f, "label": f} for f in _files()] or [{"value": "", "label": "No file in models/latent_upscale_models"}]


SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Two stages",
        "hint": "Runs the first steps on a smaller picture and the last at full size: cheaper steps, same final size. Skipped when starting from a picture or mask.",
    },
    "size": {
        "type": "float", "default": 0.7, "min": 0.4, "max": 0.95, "step": 0.05,
        "label": "First stage size", "ui": "slider",
        "hint": "Share of the full width and height. 0.7 is about half the work per step.",
        "when": {"enabled": True},
    },
    "steps": {
        "type": "int", "default": 2, "min": 1, "max": 8, "step": 1,
        "label": "Steps in the first stage",
        "hint": "Out of the run's own steps. At least one step is always left for the full size.",
        "when": {"enabled": True},
    },
    "back_up": {
        "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "ui": "slider",
        "label": "Noise added at the hand-over",
        "hint": "How far back the full-size stage starts. 0: where stage one stopped. Higher: more fresh noise, which can redraw soft or doubled outlines.",
        "when": {"enabled": True},
    },
    "resize": {
        "type": "enum", "default": "bicubic",
        "options": [{"value": "bicubic", "label": "Bicubic"}, {"value": "mitchell", "label": "Mitchell"},
                    {"value": "catmull-rom", "label": "Catmull-Rom"}, {"value": "lanczos", "label": "Lanczos"},
                    {"value": "bislerp", "label": "Bislerp"}, {"value": "bilinear", "label": "Bilinear"},
                    {"value": "area", "label": "Area"},
                    {"value": "nearest-exact", "label": "Nearest-exact"},
                    {"value": "upscaler", "label": "Latent upscaler"}],
        "renamed_options": {"trilinear": "bicubic"},   # removed: it was Bilinear here (time is never resampled)
        "label": "How the hand-over is enlarged",
        "hint": "Plain methods are free. Latent upscaler adds learned detail instead of blur, for one small extra pass.",
        "when": {"enabled": True},
    },
    "sampler2": {
        "type": "enum", "default": "", "keep_missing": True,
        "options": [{"value": "", "label": "Same as the first stage"}] + [{"value": n, "label": n} for n in comfy.samplers.SAMPLER_NAMES],
        "label": "Sampler for the full size",
        "hint": "Lets the last steps use another sampler. Leave on the first stage's to change nothing.",
        "when": {"enabled": True},
    },
    "scheduler2": {
        "type": "enum", "default": "", "keep_missing": True,
        "options": [{"value": "", "label": "Same as the first stage"}] + [{"value": n, "label": n} for n in comfy.samplers.SCHEDULER_NAMES],
        "label": "Scheduler for the full size",
        "hint": "Spaces the last steps by another scheduler, joined at the hand-over. Leave on the first stage's to change nothing.",
        "when": {"enabled": True},
    },
    "upscaler": {
        "type": "enum", "keep_missing": True, "default": (_files() or [""])[0],
        "options": _options(),
        "label": "Latent upscaler file",
        "hint": "The H3 latent upscaler file. The list is read again each time Settings opens.",
        "when": {"enabled": True, "resize": "upscaler"},
    },
}


def _known(name, names):
    return name if name in names else None          # "" and a name ComfyUI no longer has both mean "same as the first stage"


def install(patcher, values, key):
    if not values.get("enabled") or values.get("resize") != "upscaler":
        release_upscaler()                      # "unused" means the method was switched away from the upscaler
    if not values.get("enabled"):
        if patcher.get_attachment(dit_hooks.STAGE_PATCHES):
            log.once("two_stage:stage_loras_off", log.WARNING, TITLE,
                     "LoRAs are set for one stage, but Two stages is off, so they are not applied")
        return None
    size, steps = float(values.get("size", 0.7)), int(values.get("steps", 2))
    resize = values.get("upscaler") if values.get("resize") == "upscaler" and values.get("upscaler") else None
    method = values.get("resize") if values.get("resize") in METHODS else "bicubic"
    nofile = values.get("resize") == "upscaler" and not resize
    if nofile:
        log.once("two_stage:no_upscaler_file", log.WARNING, TITLE, "Latent upscaler is chosen but no file is picked (models/latent_upscale_models is empty), so the hand-over is enlarged by Bicubic")
    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, make_wrapper(size, steps, resize, method, float(values.get("back_up", 0.5)),
                                  _known(values.get("sampler2"), comfy.samplers.SAMPLER_NAMES), _known(values.get("scheduler2"), comfy.samplers.SCHEDULER_NAMES)))
    patcher.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, f"{key}.schedule", keep_whole_schedule)
    other = [n for n in (values.get("sampler2"), values.get("scheduler2")) if _known(n, comfy.samplers.SAMPLER_NAMES + comfy.samplers.SCHEDULER_NAMES)]
    return f"first {steps} step(s) at {size:.0%} of the size" + (f", full size by {'/'.join(other)}" if other else "") + (f", enlarged by {resize}" if resize else ", no latent upscaler file found: enlarged by Bicubic" if nofile else f", enlarged by {method}" if method != "bicubic" else "")


PROVIDES = {"modifier": install, "setting_options": lambda: {"upscaler": _options()}}
