"""Save a video as MP4 into ComfyUI's output folder, fast: H.264 on the GPU's encoder by default.

Core's SaveVideo encodes with libx264 through PyAV, and FFmpeg's library default is ONE thread -- on a
rental's CPU that is where a run's last half-minute went. Here the GPU's own encoder (NVENC) does it when
this PyAV build and the card have one. H.264 by default: it plays in every browser, Firefox included, and
NVENC makes it as fast as H.265. H.265 (smaller files, tagged hvc1) is an option for Chrome/Safari. With no
NVENC it is libx264 on every core: CPU H.265 is several times slower. Same names and metadata as SaveVideo. Takes the frames and sound directly: core's
CreateVideo step in between only bundled them.
"""

import json
import os
import time
from fractions import Fraction

import av
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from comfy.cli_args import args
from comfy_api.latest import io, ui

from ..._core import log, video_pipe

ENCODERS = ("auto", "gpu", "cpu")
CODECS = ("h264", "h265")
# Constant-quality mode. The number is set by `quality` (QUALITY below). NVENC needs a lower number than
# libx264 for the same look, so its table is its own. 23 was the old fixed value: fine for a phone clip,
# but it smears fine texture and shows blocks and mosquito noise on a high-resolution diffusion picture.
QUALITIES = ("standard", "high", "maximum")
QUALITY = {"nvenc": {"standard": "23", "high": "16", "maximum": "10"},
           "x264": {"standard": "23", "high": "16", "maximum": "9"}}
NVENC = {"h265": ("hevc_nvenc", {"preset": "p6", "rc": "vbr", "cq": "16", "b:v": "0"}),
         "h264": ("h264_nvenc", {"preset": "p6", "rc": "vbr", "cq": "16", "b:v": "0"})}
X264 = ("libx264", {"crf": "16", "preset": "medium"})
# Edge smoothing (the FXAA idea): only pixels on a contrasty edge are blended with their 3x3 neighbours, flat areas and
# texture stay as they are. How much of the blend is kept is `smoothing`. Done on the frames' own device before the
# encoder, so NVENC, its CPU fallback and every codec see the same picture; spatial only, so frames never smear into each other.
SMOOTHINGS = ("off", "light", "strong")
_AMOUNT = {"light": 0.5, "strong": 1.0}
_CHUNK = 8                                                    # frames per step: the float working copy stays small on a long 4K clip


def smooth(x, level):
    """x: [T, H, W, 3] floats in 0..1 -> the same shape, dtype and device with edges anti-aliased (`level` "off": x itself)."""
    amount = _AMOUNT.get(level)
    if amount is None:
        return x
    f = x.permute(0, 3, 1, 2).float()
    luma = (f[:, 0:1] * 0.299 + f[:, 1:2] * 0.587 + f[:, 2:3] * 0.114)
    p = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    contrast = F.max_pool2d(p, 3, 1) + F.max_pool2d(-p, 3, 1)          # local max - local min
    edge = ((contrast - 0.04) / 0.16).clamp(0, 1) * amount
    blur = F.avg_pool2d(F.pad(f, (1, 1, 1, 1), mode="replicate"), 3, 1)
    return (f + (blur - f) * edge).permute(0, 2, 3, 1).to(x.dtype)


def to_frames(images, smoothing="off"):
    """-> uint8 [T, H, W, 3] on the CPU. Rounded: a bare cast cuts every value down, a steady half-level darkening."""
    return np.concatenate([(smooth(x, smoothing) * 255).round().clamp(0, 255).byte().cpu().numpy() for x in images.split(_CHUNK)])


def _with_quality(codec, quality):
    """-> the (name, options) pair with its constant-quality number set for `quality` (unknown names: high)."""
    name, options = codec
    table = QUALITY["nvenc" if "nvenc" in name else "x264"]
    number = table.get(quality, table["high"])
    options = dict(options)
    for key in ("cq", "crf"):
        if key in options:
            options[key] = number
    return name, options
# FFmpeg's "minimum required driver" line compares the driver library the PROCESS loaded, not what nvidia-smi reports
_HINT = (" (nvidia-smi can look fine while FFmpeg loads an older or missing libnvidia-encode.so.1; "
         "in a container check it is there and that the NVIDIA 'video' capability is on)")


last_frame_name = video_pipe.last_frame_name


def keep_last_frame(video_file, frame):
    """Save the last decoded picture exactly as the sampler made it: the next shot can continue from this instead of from
    the encoded video (which has already lost some of it). Never worth failing the save over."""
    try:
        from PIL import Image
        folder = folder_paths.get_temp_directory()
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, last_frame_name(video_file))
        if os.path.exists(target):
            os.remove(target)                                  # a failed save below must not leave an older video's picture under this name
        Image.fromarray(frame).save(target, compress_level=1)
    except Exception as exc:                                   # noqa: BLE001 -- the video is saved; the next shot falls back to the video's own last picture
        log.info("FunUI Save Video", f"could not keep the lossless last frame ({exc})")


def _write(path, frames, fps, audio, metadata, codec):
    name, options = codec
    # use_metadata_tags keeps the workflow in the file; faststart puts the index first, which the preview player needs to seek
    with av.open(path, mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as out:
        for key, value in (metadata or {}).items():
            out.metadata[key] = json.dumps(value)
        stream = out.add_stream(name, rate=Fraction(round(fps * 1000), 1000))
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        stream.options = options
        stream.thread_type, stream.codec_context.thread_count = "AUTO", 0       # every core (FFmpeg's default is one)
        if "hevc" in name or "265" in name:
            stream.codec_context.codec_tag = "hvc1"            # hev1 (the default) is refused by Safari and Chrome on a Mac
        sound = None
        if audio is not None:                                  # both tracks before the first write: the header is fixed then
            rate, wave = int(audio["sample_rate"]), audio["waveform"]
            wave = wave[0, :, :int(rate * len(frames) / fps + 0.999)].float().cpu().contiguous().numpy()
            layout = {1: "mono", 2: "stereo"}.get(wave.shape[0], "stereo")
            sound = out.add_stream("aac", rate=rate, layout=layout)
        for frame in frames:
            out.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv420p")))
        out.mux(stream.encode(None))
        if sound is not None:
            chunk = av.AudioFrame.from_ndarray(wave, format="fltp", layout=layout)
            chunk.sample_rate, chunk.pts = rate, 0
            out.mux(sound.encode(chunk))
            out.mux(sound.encode(None))


def encode(path, frames, fps, audio=None, metadata=None, encoder="auto", codec="h264", quality="high"):
    """-> which encoder wrote the file. `frames`: uint8 [T, H, W, 3]. GPU first unless "cpu" is chosen;
    a GPU attempt that fails leaves no half file and falls back to CPU H.264 with a Log alert saying why."""
    if audio is not None and not bool(audio["waveform"].isfinite().all()):      # NaN/Inf sound (typically the model loaded in fp16): AAC refuses it, so the video is kept without it
        audio = None
        log.alert("FunUI Save Video", "The audio contains NaN/Inf values, so you are most likely using the wrong precision for this model (fp16?). The audio has been removed; the video was saved without it.")
    gpu = _with_quality(NVENC[codec], quality)
    why = None
    if encoder != "cpu":
        if gpu[0] not in av.codecs_available:
            why = f"this PyAV build has no NVENC encoder ({gpu[0]} missing)"
        else:
            try:
                _write(path, frames, fps, audio, metadata, gpu)
                return f"{codec.upper()} on NVENC (GPU)"
            except Exception as exc:                           # noqa: BLE001 -- no encoder on this card, or a driver refusal
                if os.path.exists(path):
                    os.remove(path)
                why = f"the GPU encoder (NVENC) could not start: {exc}{_HINT}"
    _write(path, frames, fps, audio, metadata, _with_quality(X264, quality))
    if why:                                                    # the video is saved either way; never lose a generation to the encoder
        log.once("save_nvenc", log.ALERT, "FunUI Save Video", f"{why}. Saved as H.264 on the CPU instead.")
    return "H.264 on libx264 (CPU, all cores)"


class FunUISaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunUISaveVideo",
            display_name="FunUI Save Video",
            category="FunUI/Output",
            description="Save as MP4: on the GPU's encoder (H.264 or H.265) when there is one, otherwise H.264 on every CPU core.",
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio", optional=True),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01),
                io.String.Input("filename_prefix", default="FunUI"),
                io.Combo.Input("encoder", options=list(ENCODERS), default="auto", optional=True,
                               tooltip="auto and gpu: the GPU encoder (NVENC) when it works, otherwise H.264 on the CPU so the video is still saved. cpu: H.264 (libx264) on every core."),
                io.Combo.Input("codec", options=list(CODECS), default="h264", optional=True,
                               tooltip="For the GPU encoder. H.264: plays in every browser. H.265: smaller files, but Firefox may not play it."),
                io.Combo.Input("quality", options=list(QUALITIES), default="high", optional=True,
                               tooltip="How closely the file keeps the picture. standard: small files, but fine texture smears and blocks show on a sharp high-resolution picture. high: visually clean for a bigger file. maximum: near lossless, large."),
                io.Combo.Input("smoothing", options=list(SMOOTHINGS), default="off", optional=True,
                               tooltip="Anti-aliasing for jagged edges (like FXAA in games): only edges with strong contrast are softened, flat areas and texture stay. off: untouched. light: half strength. strong: full strength. A small cost on the GPU, a noticeable one on the CPU for 4K clips."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[],
        )

    @classmethod
    def execute(cls, images, fps=24.0, filename_prefix="FunUI", audio=None, encoder="auto", codec="h264", quality="high", smoothing="off") -> io.NodeOutput:
        frames = to_frames(images, smoothing)
        width, height = frames.shape[2], frames.shape[1]
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        metadata = None
        if not args.disable_metadata:
            metadata = {**(cls.hidden.extra_pnginfo or {}), **({"prompt": cls.hidden.prompt} if cls.hidden.prompt is not None else {})} or None
        file = f"{filename}_{counter:05}_.mp4"
        t = time.perf_counter()
        used = encode(os.path.join(folder, file), frames, float(fps), audio, metadata, encoder, codec, quality)
        log.info("FunUI Save Video", f"{file}: {len(frames)} frames in {time.perf_counter() - t:.1f}s with {used}, quality {quality}" + (f", smoothing {smoothing}" if smoothing != "off" else ""))
        keep_last_frame(file, frames[-1])
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))
