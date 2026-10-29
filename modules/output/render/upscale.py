"""Post-render upscale: a finished video through a models/upscale_models model.

Frames stream through ffmpeg in small chunks (never the whole clip in RAM), each chunk
goes through ComfyUI's own ImageUpscaleWithModel (tiling, OOM back-off, device handling),
and the result is encoded with the source's frame rate and sound.
"""

from __future__ import annotations

import os

from . import files, rtx
from ..._core import addons, video_pipe

SUBFOLDER = "funui_upscaled"


probe, upscale_file, scaled_fps, free_name = video_pipe.probe, video_pipe.stream_file, video_pipe.scaled_fps, video_pipe.free_name


def models() -> list:
    try:
        import folder_paths
        return list(folder_paths.get_filename_list("upscale_models"))
    except Exception:                                  # noqa: BLE001 -- headless
        return []


def run(media, model, rtx_options="{}", interrupted=lambda: None) -> dict:
    """Blocking. Upscale the output/temp file `media` ({filename, subfolder, type}) with an upscale model or NVIDIA RTX Super Resolution.
    -> {"media": the new file}. The page runs this as a background job of its own: as a ComfyUI prompt it would make ComfyUI's cache
    let go of the workflow's loaders and models (a prompt without them), and the next generation would load them again."""
    import folder_paths

    src = files.comfy_path(media.get("filename"), media.get("subfolder") or "", media.get("type") or "output")
    if not src or not os.path.isfile(src):
        raise RuntimeError(f"{media.get('filename')} is not in the {media.get('type') or 'output'} folder (anymore).")
    if model == rtx.NAME:
        opts = rtx.options(rtx_options)
        fn, tag = rtx.make(opts), "rtx_" + opts["quality"].lower()
    else:
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
        loaded = UpscaleModelLoader.execute(model).result[0]
        fn, tag = (lambda x: ImageUpscaleWithModel.execute(loaded, x).result[0]), model
    out_dir = os.path.join(folder_paths.get_output_directory(), SUBFOLDER)
    os.makedirs(out_dir, exist_ok=True)
    name = free_name(out_dir, media["filename"], tag)
    frames = upscale_file(src, os.path.join(out_dir, name), fn, interrupted)
    from ..._core import log
    log.info("FunUI Upscale", f"{media['filename']} with {tag}: {frames} frames -> {SUBFOLDER}/{name}")
    return {"media": {"filename": name, "subfolder": SUBFOLDER, "type": "output"}}


try:
    from comfy_api.latest import io
except Exception:                                      # noqa: BLE001 -- tests without ComfyUI
    io = None

if io is not None:
    class FunUIUpscaleVideo(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id="FunUIUpscaleVideo",
                display_name="FunUI Upscale Video",
                category="FunUI/Output",
                description="Upscale a finished video file with an upscale model or NVIDIA RTX Super Resolution; keeps its fps and sound.",
                inputs=[
                    io.String.Input("filename", default=""),
                    io.String.Input("subfolder", default=""),
                    io.Combo.Input("type", options=["output", "temp"], default="output"),
                    io.Combo.Input("upscale_model", options=[*([rtx.NAME] if addons.enabled("nvidia_vsr") else []), *(models() or [""])],
                                   tooltip="From ComfyUI/models/upscale_models (use .safetensors), or NVIDIA RTX Super Resolution."),
                    io.String.Input("rtx_options", default="{}", optional=True,
                                    tooltip="NVIDIA RTX Super Resolution options as JSON: quality, resize, scale, width, height, strength, tenbit."),
                ],
                outputs=[],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, filename, subfolder, type, upscale_model, rtx_options="{}") -> io.NodeOutput:
            import comfy.model_management as mm
            done = run({"filename": filename, "subfolder": subfolder, "type": type}, upscale_model, rtx_options, mm.throw_exception_if_processing_interrupted)
            return io.NodeOutput(ui={"videos": [{**done["media"], "format": "video/h264-mp4"}]})
