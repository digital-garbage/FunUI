"""The node: pictures in as LTX guides, nothing in as nothing done."""

import json

import numpy as np
import torch
from comfy_api.latest import io

from ..._core import log, media as media_mod

MAX_FRAMES = 16


def read(media_id):
    """One picture from the media library as a [1, H, W, 3] float tensor, or None when it is gone or unreadable."""
    path = media_mod.path_for(str(media_id))
    if path is None:
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            return torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0)[None]
    except Exception:                                # noqa: BLE001 -- a damaged file raises whatever PIL raises
        return None


def parse(text):
    """-> [(media_id, at 0..1, strength)], or a sentence saying what is wrong with the list."""
    if not (text or "").strip():
        return []
    try:
        raw = json.loads(text)
    except ValueError as exc:
        return f"the frames are not valid JSON ({exc})"
    if not isinstance(raw, list):
        return "the frames must be a list"
    out = []
    for item in raw[:MAX_FRAMES]:
        try:
            out.append((str(item["media_id"]), min(max(float(item.get("at", 0.5)), 0.0), 1.0),
                        min(max(float(item.get("strength", 0.25)), 0.0), 10.0)))
        except (KeyError, TypeError, ValueError, AttributeError):
            return f"{item!r} is not {{media_id, at, strength}}"
    return out


class FunUIGuideFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunUIGuideFrames",
            display_name="FunUI Guide Frames",
            category="FunUI/Conditioning",
            description="Lay pictures from the media library into an LTX clip as guides (ComfyUI's own guide attention), "
                        "each at a fraction of the clip's length. With no list, everything passes through untouched. "
                        "Crop them again after sampling with LTXVCropGuides.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent", tooltip="The picture latent (not the joined picture+sound one)."),
                io.String.Input("frames", default="", optional=True,
                                tooltip='[{"media_id": "...", "at": 0.5, "strength": 0.25}, ...]. The app writes this.'),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Conditioning.Output(display_name="negative"),
                     io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, frames="") -> io.NodeOutput:
        wanted = parse(frames)
        if isinstance(wanted, str):
            raise RuntimeError(f"FunUI Guide Frames: {wanted}.")
        if not wanted:
            return io.NodeOutput(positive, negative, latent)
        from comfy_extras.nodes_lt import LTXVAddGuide
        length = int(latent["samples"].shape[2])
        pixel_frames = (length - 1) * int(vae.downscale_index_formula[0]) + 1
        added = []
        for media_id, at, strength in wanted:
            picture = read(media_id)
            if picture is None:
                log.once(f"guide_frames:gone:{media_id}", log.WARNING, "FunUI Guide Frames",
                         f"a guide picture is gone from the library and was left out ({media_id})")
                continue
            index = int(round(at * (pixel_frames - 1)))
            positive, negative, latent = LTXVAddGuide.execute(
                positive, negative, vae, latent, picture, index, strength).result
            added.append(f"{at:.2f}@{strength:g}")
        if added:
            log.once("guide_frames:ran", log.INFO, "FunUI Guide Frames", f"{len(added)} guide picture(s) at {', '.join(added)}")
        return io.NodeOutput(positive, negative, latent)
