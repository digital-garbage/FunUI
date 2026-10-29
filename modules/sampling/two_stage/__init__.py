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

from ..._core import log

from .stages import make_wrapper

ID = "two_stage"
TITLE = "Two stages"
MOUNT = "generation.sampling"
STAGE = "load"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["audio_stream", "adaln_modalities"]

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
}


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    size, steps = float(values.get("size", 0.7)), int(values.get("steps", 2))
    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, make_wrapper(size, steps))
    return f"first {steps} step(s) at {size:.0%} of the size"


PROVIDES = {"modifier": install}
