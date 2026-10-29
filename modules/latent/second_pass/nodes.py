"""Latent upscaler loading and between-pass resampling. See the module docstring."""

import comfy.model_management
import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ..._core import log, registry as registry_mod

FOLDER = "latent_upscale_models"


def upscaler_files():
    try:
        return folder_paths.get_filename_list(FOLDER)
    except Exception:                            # noqa: BLE001 -- folder not registered
        return []


class FunUILatentUpscalerLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunUILatentUpscalerLoader",
            display_name="FunUI Load Latent Upscaler",
            category="FunUI/Latent",
            description="Load a latent upscaler, including ones ComfyUI's own loader does not know.",
            inputs=[io.Combo.Input("upscaler_name", options=upscaler_files())],
            outputs=[io.LatentUpscaleModel.Output(display_name="upscaler")],
        )

    @classmethod
    def execute(cls, upscaler_name: str) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise(FOLDER, upscaler_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        for spec, provider in registry_mod.current().providers("latent_upscaler"):
            model = provider(sd)
            if model is not None:
                log.info("FunUI Latent Upscaler", f"{upscaler_name}: {spec.title} upscaler")
                return io.NodeOutput(model)
        newer = ("It looks like the newer attention-based H3 upscaler, which FunUI cannot build yet: "
                 "use an older convolution-based H3 upscaler. ") if any(str(k).startswith("base.") for k in sd) else ""
        raise RuntimeError(
            f"{upscaler_name} is not an upscaler any installed model module can load. {newer}"
            f"If it is Lightricks' or Hunyuan's, ComfyUI's own Load Latent Upscale Model "
            f"loads it.")


def _video_of(samples):
    """(video tensor, rebuild(video) -> samples). Video is the first part of a
    nested latent, or the latent itself."""
    if getattr(samples, "is_nested", False):
        parts = list(samples.unbind())

        def rebuild(video):
            from comfy.nested_tensor import NestedTensor
            return NestedTensor((video, *parts[1:]))
        return parts[0], rebuild
    return samples, lambda video: video


def downscale_to(video, h, w):
    """Antialiased bicubic, frames independent. v4 used area (a 2x2 box at 2x),
    which came back soft AND aliased -- "sharpen leaves smearing" was the box."""
    b, c, f, hh, ww = video.shape
    if (hh, ww) == (h, w):
        return video
    x = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, hh, ww)
    x = F.interpolate(x.float(), size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    return x.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4).to(dtype=video.dtype)


def run_upscaler(upscaler, video, scale):
    module = getattr(upscaler, "model", upscaler)
    upscale = getattr(module, "funui_latent_upscale", None)
    if not callable(upscale):
        raise RuntimeError(
            "this upscaler does not say how to run itself on a latent (no "
            "funui_latent_upscale). ComfyUI's LTXV Latent Upsampler node runs Lightricks' one.")
    device = comfy.model_management.get_torch_device()
    offload = comfy.model_management.unet_offload_device()
    dtype = next(module.parameters()).dtype
    try:
        module.to(device)
        with torch.inference_mode():
            out = upscale(video.to(device=device, dtype=dtype), scale=scale)
        return out.to(device=video.device, dtype=video.dtype)
    finally:
        module.to(offload)


def rescale(conditioning, h, w):
    """(conditioning, resized count), via whichever model module recognises it."""
    if conditioning is None:
        return None, 0
    for _spec, provider in registry_mod.current().providers("rescale_conditioning"):
        got = provider(conditioning, h, w)
        if got is not None:
            return got
    return conditioning, 0


def bicubic_to(video, h, w):
    """The plain way up when there is no upscaler file: bicubic per frame, same as the way down."""
    return downscale_to(video, h, w)


class FunUIStageSize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunUIStageSize",
            display_name="FunUI Stage Size",
            category="FunUI/Latent",
            description="The size of a cheaper first stage: the full size scaled down, kept on the 32 pixel grid.",
            inputs=[
                io.Int.Input("width", default=640, min=32, max=16384, step=32),
                io.Int.Input("height", default=640, min=32, max=16384, step=32),
                io.Float.Input("scale", default=0.7, min=0.25, max=1.0, step=0.05,
                               tooltip="Share of the full width and height the first stage runs at. 0.7 is about half the work per step."),
            ],
            outputs=[io.Int.Output(display_name="width"), io.Int.Output(display_name="height")],
        )

    @classmethod
    def execute(cls, width: int, height: int, scale: float) -> io.NodeOutput:
        def snap(v):
            return max(64, int(round(v * scale / 32.0)) * 32)
        return io.NodeOutput(min(snap(width), int(width)), min(snap(height), int(height)))


class FunUILatentResample(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunUILatentResample",
            display_name="FunUI Latent Resample",
            category="FunUI/Latent",
            description="Between two passes: sharpen the latent, or upscale it and "
                        "keep the conditioning valid at the new size.",
            inputs=[
                io.Latent.Input("latent"),
                io.LatentUpscaleModel.Input("upscaler", optional=True,
                                            tooltip="Without one the latent is resized with bicubic, which is enough to hand a first stage to a second."),
                io.Combo.Input("operation", options=["sharpen", "upscale"], default="sharpen",
                               tooltip="sharpen: up, then back to this size -- pass 2 "
                                       "re-denoises the added detail. upscale: kept larger, "
                                       "pass 2 runs at the new size (much slower)."),
                io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Int.Input("target_width", default=0, min=0, max=16384, step=32, optional=True,
                             tooltip="upscale only: land exactly on this picture width (0 keeps whatever scale gives)."),
                io.Int.Input("target_height", default=0, min=0, max=16384, step=32, optional=True,
                             tooltip="upscale only: land exactly on this picture height."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, latent, upscaler=None, operation: str = "sharpen", scale: float = 2.0,
                positive=None, negative=None, target_width: int = 0, target_height: int = 0) -> io.NodeOutput:
        video, rebuild = _video_of(latent["samples"])
        if getattr(video, "ndim", 0) != 5:
            raise RuntimeError(
                f"expected a video latent [B, C, T, H, W], got {tuple(video.shape)}")
        h, w = int(video.shape[-2]), int(video.shape[-1])
        goal = None
        if operation == "upscale" and target_width and target_height:
            goal = (int(target_height) // 16, int(target_width) // 16)     # H3's picture is 16 pixels per latent cell
            scale = max(1.0, goal[0] / h, goal[1] / w)
        if upscaler is None:
            if operation == "sharpen":
                raise RuntimeError("sharpen needs a latent upscaler: connect one, or use upscale.")
            up = bicubic_to(video, max(2, int(round(h * scale / 2.0)) * 2), max(2, int(round(w * scale / 2.0)) * 2))
        else:
            up = run_upscaler(upscaler, video, float(scale))
        if goal is not None:
            up = downscale_to(up, *goal) if up.shape[-2] >= goal[0] and up.shape[-1] >= goal[1] else bicubic_to(up, *goal)

        # noise_mask is kept: a pinned frame must stay pinned in pass 2, and
        # ComfyUI's prepare_mask resizes a mask to whatever latent it meets.
        out = latent.copy()
        if operation == "sharpen":
            out["samples"] = rebuild(downscale_to(up, h, w))
            status = f"sharpen: {scale:g}x up and back to {h}x{w}"
            return io.NodeOutput(out, positive, negative, status)

        uh, uw = int(up.shape[-2]), int(up.shape[-1])
        out["samples"] = rebuild(up)
        positive, pos_n = rescale(positive, uh, uw)
        negative, neg_n = rescale(negative, uh, uw)
        status = f"upscale: {h}x{w} -> {uh}x{uw} latent ({uh * uw / (h * w):.2g}x the work in pass 2)"
        if pos_n or neg_n:
            status += f"; {pos_n + neg_n} anchor pin(s) resized to match"
        return io.NodeOutput(out, positive, negative, status)
