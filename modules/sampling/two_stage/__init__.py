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

import folder_paths

from ..._core import log

from .stages import keep_whole_schedule, make_wrapper

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
        "hint": "Runs the first steps on a smaller picture and the last steps at the full size. Cheaper steps "
                "up front, the same final size. Skipped for a run that starts from a picture or a mask.",
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
    "resize": {
        "type": "enum", "default": "bicubic",
        "options": [{"value": "bicubic", "label": "Bicubic"}, {"value": "upscaler", "label": "Latent upscaler"}],
        "label": "How the hand-over is enlarged",
        "hint": "Bicubic is plain and free. Latent upscaler runs H3's learned latent upscaler (models/latent_upscale_models) "
                "on the clean guess, so the added detail is learned instead of blurred; it costs one small network pass "
                "once per run. The first stage size is kept, the upscaler is run at the ratio that reaches the full size.",
        "when": {"enabled": True},
    },
    "upscaler": {
        "type": "enum", "default": (_files() or [""])[0],
        "options": _options(),
        "label": "Latent upscaler file",
        "hint": "The H3 latent upscaler file. The list is read again each time Settings opens.",
        "when": {"enabled": True, "resize": "upscaler"},
    },
}


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    size, steps = float(values.get("size", 0.7)), int(values.get("steps", 2))
    resize = values.get("upscaler") if values.get("resize") == "upscaler" and values.get("upscaler") else None
    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, make_wrapper(size, steps, resize))
    patcher.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, f"{key}.schedule", keep_whole_schedule)
    return f"first {steps} step(s) at {size:.0%} of the size" + (f", enlarged by {resize}" if resize else "")


PROVIDES = {"modifier": install}
