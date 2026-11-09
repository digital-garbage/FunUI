"""Save a video as MP4 into ComfyUI's output folder, fast: H.264 on the GPU's encoder by default.

Core's SaveVideo encodes with libx264 through PyAV, and FFmpeg's library default is ONE thread -- on a
rental's CPU that is where a run's last half-minute went. Here the GPU's own encoder (NVENC) does it when
this PyAV build and the card have one. H.264 by default: it plays in every browser, Firefox included, and
NVENC makes it as fast as H.265. H.265 (smaller files, tagged hvc1) is an option for Chrome/Safari. With no
NVENC it is libx264 on every core: CPU H.265 is several times slower. Same names and metadata as SaveVideo. Takes the frames and sound directly: core's
CreateVideo step in between only bundled them.
"""

import json
import math
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

from ..._core import log, registry as registry_mod, schema as schema_mod, video_pipe
from . import watermark

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
# The finishing look, from Engine > Post (see __init__.SETTINGS), applied to the frames on their own device before encoding,
# so NVENC, its CPU fallback and every codec get the same picture. Order: contrast, brightness, saturation, edge smoothing.
# Contrast and brightness are measured ONCE on a sample of the whole clip and applied as one fixed curve: a per-frame
# auto-level would pump and flicker as the picture changes.
SMOOTHINGS = ("off", "light", "strong")
DEFAULTS = {"smoothing": "off", "auto_contrast": False, "auto_brightness": False, "saturation": 1.0,
            "watermark": "off", "watermark_text": "", "watermark_colour": "#ffffff", "watermark_picture": "", "watermark_corner": "bottom-right",
            "watermark_opacity": 0.6, "watermark_size": 18.0, "watermark_margin": 2.0,
            "watermark_outline": 0.0, "watermark_outline_colour": "#000000", "watermark_outline_opacity": 1.0}
_AMOUNT = {"light": 0.5, "strong": 1.0}
_CHUNK = 8                                                    # frames per step: the float working copy stays small on a long 4K clip
_SAMPLE_FRAMES, _SAMPLE_PIXELS = 16, 1_000_000
_TARGET_MEDIAN, _MAX_STRETCH = 0.45, 0.15                     # brightness aims the median luma here; contrast moves each end by at most this


def _luma(f):
    """f: [T, 3, H, W] -> [T, 1, H, W]."""
    return f[:, 0:1] * 0.299 + f[:, 1:2] * 0.587 + f[:, 2:3] * 0.114


def clip_look(images, values):
    """-> {"lo", "hi", "gamma"} as plain floats (no tensors, so nothing can sit on the wrong device) measured on a sample of the clip."""
    look = {"lo": 0.0, "hi": 1.0, "gamma": 1.0}
    if not (values.get("auto_contrast") or values.get("auto_brightness")) or len(images) == 0:
        return look
    n = len(images)
    pick = sorted({round(i * (n - 1) / max(_SAMPLE_FRAMES - 1, 1)) for i in range(min(_SAMPLE_FRAMES, n))})
    luma = _luma(images[pick].permute(0, 3, 1, 2).float()).flatten()
    luma = luma[::max(1, luma.numel() // _SAMPLE_PIXELS)]
    if values.get("auto_contrast"):
        lo, hi = (float(v) for v in torch.quantile(luma, luma.new_tensor([0.01, 0.99])))
        look["lo"], look["hi"] = min(lo, _MAX_STRETCH), max(hi, 1.0 - _MAX_STRETCH)
        luma = ((luma - look["lo"]) / (look["hi"] - look["lo"])).clamp(0, 1)
    if values.get("auto_brightness"):
        mid = min(max(float(luma.median()), 0.02), 0.98)
        look["gamma"] = min(max(math.log(_TARGET_MEDIAN) / math.log(mid), 0.75), 1.33)
    return look


def smooth(x, level):
    """x: [T, H, W, 3] floats in 0..1 -> the same shape, dtype and device with edges anti-aliased (FXAA idea: only pixels on a
    contrasty edge are blended with their 3x3 neighbours; `level` "off": x itself)."""
    amount = _AMOUNT.get(level)
    if amount is None:
        return x
    f = x.permute(0, 3, 1, 2).float()
    p = F.pad(_luma(f), (1, 1, 1, 1), mode="replicate")
    contrast = F.max_pool2d(p, 3, 1) + F.max_pool2d(-p, 3, 1)          # local max - local min
    edge = ((contrast - 0.04) / 0.16).clamp(0, 1) * amount
    blur = F.avg_pool2d(F.pad(f, (1, 1, 1, 1), mode="replicate"), 3, 1)
    return (f + (blur - f) * edge).permute(0, 2, 3, 1).to(x.dtype)


def finish(x, values, look):
    """One chunk x [T, H, W, 3] through the whole look; the same shape, dtype and device back."""
    sat = float(values.get("saturation", 1.0))
    if look["lo"] == 0.0 and look["hi"] == 1.0 and look["gamma"] == 1.0 and sat == 1.0:
        return smooth(x, values.get("smoothing", "off"))
    f = x.permute(0, 3, 1, 2).float()
    if look["lo"] != 0.0 or look["hi"] != 1.0:
        f = ((f - look["lo"]) / (look["hi"] - look["lo"])).clamp(0, 1)
    if look["gamma"] != 1.0:
        f = f.clamp(0, 1) ** look["gamma"]
    if sat != 1.0:
        grey = _luma(f)
        f = (grey + (f - grey) * sat).clamp(0, 1)
    return smooth(f.permute(0, 2, 3, 1).to(x.dtype), values.get("smoothing", "off"))


def to_frames(images, values=None):
    """-> uint8 [T, H, W, 3] on the CPU, `values` being the Engine > Post settings. Rounded: a bare cast cuts every value down, a steady half-level darkening."""
    values = values or {}
    look = clip_look(images, values)
    mark = watermark.build(values, images.shape[2], images.shape[1])          # last: nothing before it can touch the mark
    out = []
    for x in images.split(_CHUNK):
        f = finish(x, values, look)
        y = (f * 255).round().clamp(0, 255).byte().cpu().numpy()
        if mark is not None:                      # only the mark's corner is blended and pasted: no second copy of the chunk
            part = watermark.corner(f, mark)
            y[:, mark.y:mark.y + part.shape[1], mark.x:mark.x + part.shape[2], :] = (watermark.blend(part, mark) * 255).round().clamp(0, 255).byte().cpu().numpy()
        out.append(y)
    return np.concatenate(out)


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


def _values(settings):
    """The Engine > Post settings from the pipeline's settings object, checked against their declaration (a wrong one: default, said)."""
    spec = registry_mod.current().specs.get("output_save")
    if spec is None:
        return dict(DEFAULTS)
    pick = spec.settings.get("watermark_picture")
    if pick is not None:
        pick["options"] = watermark.picture_options()               # read now: a picture added since Settings was opened still counts
    clean, problems = schema_mod.check_values(spec, (settings or {}).get("output_save"))
    for problem in problems or ():
        log.once(f"save_video:{problem}", log.ALERT, "FunUI Save Video", f"setting ignored, default used: {problem}")
    return clean


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
                               tooltip="How closely the file keeps the picture. standard: small files, but fine texture can smear. high: clean, bigger file. maximum: near lossless, large."),
                io.Custom("FUNUI_SETTINGS").Input("settings", optional=True),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[],
        )

    @classmethod
    def execute(cls, images, fps=24.0, filename_prefix="FunUI", audio=None, encoder="auto", codec="h264", quality="high", settings=None) -> io.NodeOutput:
        look = _values(settings)
        frames = to_frames(images, look)
        width, height = frames.shape[2], frames.shape[1]
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        metadata = None
        if not args.disable_metadata:
            metadata = {**(cls.hidden.extra_pnginfo or {}), **({"prompt": cls.hidden.prompt} if cls.hidden.prompt is not None else {})} or None
        file = f"{filename}_{counter:05}_.mp4"
        t = time.perf_counter()
        used = encode(os.path.join(folder, file), frames, float(fps), audio, metadata, encoder, codec, quality)
        log.info("FunUI Save Video", f"{file}: {len(frames)} frames in {time.perf_counter() - t:.1f}s with {used}, quality {quality}" + "".join(f", {k} {v}" for k, v in look.items() if v != DEFAULTS[k]))
        keep_last_frame(file, to_frames(images[-1:])[-1])             # the sampler's own picture: the Post look belongs to the file, not to the next shot's start
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))
