"""The interpolation itself: a finished file in the output folder in, a new one beside it out.

It runs beside ComfyUI as a background job of the page, NOT as a ComfyUI prompt: a prompt with no loaders in it makes ComfyUI's
cache let go of the loaders (and the models) of the workflow that ran before, so the next generation would load them again.
"""

import os

from . import framegen
from ..._core import log, video_pipe

SUBFOLDER = "funui_interpolated"


class InterpolateError(Exception):
    """Something the person can act on, in a sentence."""


def interpolate(media, options, interrupted=lambda: None) -> dict:
    """Blocking. `media` is {filename, subfolder, type}. -> {"media": the new file}."""
    try:
        import folder_paths
    except ImportError as exc:
        raise InterpolateError("ComfyUI is not here to say where the output folder is.") from exc
    src = video_pipe.comfy_path(media.get("filename"), media.get("subfolder") or "", media.get("type") or "output")
    if not src or not os.path.isfile(src):
        raise InterpolateError(f"{media.get('filename')} is not in the {media.get('type') or 'output'} folder (anymore).")
    try:
        opts = framegen.options(options)
        fn = framegen.make(opts)
    except (ValueError, RuntimeError) as exc:
        raise InterpolateError(str(exc)) from exc
    out_dir = os.path.join(folder_paths.get_output_directory(), SUBFOLDER)
    os.makedirs(out_dir, exist_ok=True)
    name = video_pipe.free_name(out_dir, media["filename"], f"x{opts['multiplier']}")
    try:
        frames = video_pipe.stream_file(src, os.path.join(out_dir, name), fn, interrupted, opts["multiplier"])
    except video_pipe.ClipError as exc:
        raise InterpolateError(str(exc)) from exc
    log.info("FunUI Interpolate", f"{media['filename']} x{opts['multiplier']} ({opts['mode']}): {frames} frames -> {SUBFOLDER}/{name}")
    return {"media": {"filename": name, "subfolder": SUBFOLDER, "type": "output"}}
