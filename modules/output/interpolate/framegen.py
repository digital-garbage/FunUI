"""NVIDIA RTX Video Frame Generation as an interpolation engine, through the nvidia-vfx add-on (the one RTX Super Resolution uses).

`make(options)` returns a function [B,H,W,3] float 0-1 -> the same frames with the in-between ones added.
It remembers the last frame it saw, so chunks of one video are fed in order and the seams between chunks get their frames too.
"""

import json

NAME = "NVIDIA RTX Frame Generation"        # shown as the engine's name (the UI holds the same string)
MODES = ("LOW", "MEDIUM", "HIGH")
DEFAULTS = {"mode": "MEDIUM", "multiplier": 2, "tenbit": False}
MAX_MULTIPLIER = 4


def options(raw) -> dict:
    """The user's choices, checked. A value out of range is refused by name, never quietly swapped."""
    try:
        given = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
    except ValueError as exc:
        raise ValueError(f"Frame generation options are not valid JSON: {exc}") from exc
    if not isinstance(given, dict):
        raise ValueError("Frame generation options must be an object.")
    o = {**DEFAULTS, **given}
    if o["mode"] not in MODES:
        raise ValueError(f"Frame generation mode {o['mode']!r} is not one of: {', '.join(MODES)}")
    if isinstance(o["multiplier"], bool) or not isinstance(o["multiplier"], int) or not 2 <= o["multiplier"] <= MAX_MULTIPLIER:
        raise ValueError(f"Frame generation multiplier must be a whole number from 2 to {MAX_MULTIPLIER}.")
    if not isinstance(o["tenbit"], bool):
        raise ValueError("Frame generation tenbit must be true or false.")
    return o


def make(o):
    """-> the interpolation function for these (already checked) options. Raises a sentence when the add-on is missing."""
    from ..._core import addons
    from ..._core import nvidia
    ngx_hint, pack10, unpack10 = nvidia.ngx_hint, nvidia.pack10, nvidia.unpack10
    if not addons.enabled("nvidia_framegen"):          # installed but switched off counts too: off means off
        raise RuntimeError(f"{NAME} is not installed or is switched off: switch it on in Settings ▸ Modules and restart ComfyUI.")
    try:
        from nvvfx import VideoFrameGeneration
    except ImportError as exc:
        raise RuntimeError(f"{NAME} is not installed: switch it on in Settings ▸ Modules and restart ComfyUI.") from exc
    if not hasattr(VideoFrameGeneration, "Mode"):
        raise RuntimeError("This nvidia-vfx has no frame generation: update it (needs 0.2 or newer).")
    import torch
    mult = o["multiplier"]
    effects, last = {}, [None]                       # one effect per frame size; the previous frame, on the GPU

    def effect(h, w):
        if (h, w) not in effects:
            kw = {"image_encoding": VideoFrameGeneration.ImageEncoding.RGB10A2} if o["tenbit"] else {}
            e = VideoFrameGeneration(getattr(VideoFrameGeneration.Mode, o["mode"]), **kw)
            e.input_width, e.input_height, e.frame_multiplier = w, h, mult
            try:
                e.load()
            except Exception as exc:                  # noqa: BLE001 -- NvVFXError: say what to do, not just the SDK code
                raise RuntimeError(f"{NAME} could not start ({exc}).{ngx_hint()}") from exc
            effects[(h, w)] = e
        return effects[(h, w)]

    def run(x):
        out = []
        for frame in x:
            h, w = frame.shape[0], frame.shape[1]
            f = frame.to("cuda", torch.float32).permute(2, 0, 1).contiguous()
            cur = pack10(f) if o["tenbit"] else f
            if last[0] is not None:
                for i in range(1, mult):
                    res = effect(h, w).run(last[0], cur, frame_index=i)
                    y = torch.from_dlpack(res.image).clone()      # the effect reuses its buffer: copy before the next call
                    out.append((unpack10(y) if o["tenbit"] else y).permute(1, 2, 0))
            out.append(f.permute(1, 2, 0))
            last[0] = cur
        return torch.stack(out)
    return run
