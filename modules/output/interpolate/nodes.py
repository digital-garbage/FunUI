"""Post-render interpolation: a finished video gets in-between frames (NVIDIA RTX Frame Generation), played at the multiplied frame rate.

Same stream as the upscale (chunks through ffmpeg, the source's sound kept); only the engine and the frame rate differ.
"""

import os

from . import framegen
from ..._core import video_pipe

SUBFOLDER = "funui_interpolated"

try:
    from comfy_api.latest import io
except Exception:                                      # noqa: BLE001 -- tests without ComfyUI
    io = None

if io is not None:
    class FunUIInterpolateVideo(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id="FunUIInterpolateVideo",
                display_name="FunUI Interpolate Video",
                category="FunUI/Output",
                description="Add in-between frames to a finished video file with NVIDIA RTX Frame Generation; the length and sound stay, the frame rate goes up.",
                inputs=[
                    io.String.Input("filename", default=""),
                    io.String.Input("subfolder", default=""),
                    io.Combo.Input("type", options=["output", "temp"], default="output"),
                    io.String.Input("frame_options", default="{}", optional=True,
                                    tooltip="NVIDIA RTX Frame Generation options as JSON: mode (LOW, MEDIUM, HIGH), multiplier (2 to 4), tenbit."),
                ],
                outputs=[],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, filename, subfolder, type, frame_options="{}") -> io.NodeOutput:
            import comfy.model_management as mm
            import folder_paths

            src = video_pipe.comfy_path(filename, subfolder, type)
            if not src or not os.path.isfile(src):
                raise RuntimeError(f"{filename} is not in the {type} folder (anymore).")
            opts = framegen.options(frame_options)
            fn = framegen.make(opts)
            out_dir = os.path.join(folder_paths.get_output_directory(), SUBFOLDER)
            os.makedirs(out_dir, exist_ok=True)
            name = video_pipe.free_name(out_dir, filename, f"x{opts['multiplier']}")
            frames = video_pipe.stream_file(src, os.path.join(out_dir, name), fn, mm.throw_exception_if_processing_interrupted, opts["multiplier"])
            from ..._core import log
            log.info("FunUI Interpolate", f"{filename} x{opts['multiplier']} ({opts['mode']}): {frames} frames -> {SUBFOLDER}/{name}")
            return io.NodeOutput(ui={"videos": [{"filename": name, "subfolder": SUBFOLDER,
                                                 "type": "output", "format": "video/h264-mp4"}]})
