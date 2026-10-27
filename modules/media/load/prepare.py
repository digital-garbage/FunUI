"""Getting a loaded video ready for a model: the choices ComfyUI's video loaders (VHS Load Video and the like) offer.

Pure tensor functions with no node or file in them, so they are tested on made-up frames. The order is the
loaders': pick the frames at the target rate, drop the first and last ones, take every Nth, cap the count;
and in space: crop the margins, fit to the target size, round to a multiple the model wants.
"""

import torch
import torch.nn.functional as F

FITS = ("crop", "pad", "stretch")

#: The keys a preparation understands, with what each does when left at its default.
DEFAULTS = {
    "fps": 0,            # target frame rate; 0 keeps the video's own
    "skip_first": 0,     # frames dropped from the start (at the target rate)
    "cut_last": 0,       # frames dropped from the end
    "every_nth": 1,      # keep one frame in N
    "frame_cap": 0,      # at most this many frames; 0 = all
    "width": 0,          # target size; 0 keeps the (cropped) video's own
    "height": 0,
    "fit": "crop",       # how a different shape meets the target: crop the overflow, pad the gap, or stretch
    "crop_left": 0,      # percent of the width/height cut off each side before fitting
    "crop_right": 0,
    "crop_top": 0,
    "crop_bottom": 0,
    "multiple_of": 1,    # round the final size down to a multiple of this (models want 16 / 32)
}


def pick_frames(count, src_fps, fps=0, skip_first=0, cut_last=0, every_nth=1, frame_cap=0):
    """Indices into the source frames, and the frame rate they play at.

    -> (indices list, out_fps, start_sec, span_sec): the last two say which stretch of the soundtrack goes with them.
    """
    src_fps = float(src_fps) if src_fps and src_fps > 0 else 24.0
    out_fps = float(fps) if fps and fps > 0 else src_fps
    n = max(1, int(count / src_fps * out_fps + 1e-6)) if out_fps != src_fps else count
    index = [min(count - 1, int(i * src_fps / out_fps + 1e-6)) for i in range(n)] if out_fps != src_fps else list(range(count))
    lo, hi = max(0, int(skip_first)), len(index) - max(0, int(cut_last))
    kept = index[lo:max(lo, hi)]
    if not kept:
        raise ValueError(f"nothing is left: the video has {len(index)} frames at {out_fps:g} fps and {lo} are skipped and {max(0, int(cut_last))} cut")
    start, span = lo / out_fps, len(kept) / out_fps
    nth = max(1, int(every_nth))
    kept = kept[::nth]
    if frame_cap and frame_cap > 0:
        kept = kept[:int(frame_cap)]
        span = min(span, len(kept) * nth / out_fps)
    return kept, out_fps / nth, start, span


def _round(value, multiple):
    return max(multiple, int(value) // multiple * multiple) if multiple > 1 else max(1, int(value))


def shape(images, width=0, height=0, fit="crop", crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, multiple_of=1):
    """Crop the margins, fit to the target size, round to the multiple. images: [N, H, W, C] floats 0..1."""
    n, h, w, c = images.shape
    left, right = int(w * min(max(crop_left, 0), 90) / 100), int(w * min(max(crop_right, 0), 90) / 100)
    top, bottom = int(h * min(max(crop_top, 0), 90) / 100), int(h * min(max(crop_bottom, 0), 90) / 100)
    if left + right >= w or top + bottom >= h:
        raise ValueError("the crop leaves no picture")
    if left or right or top or bottom:
        images = images[:, top:h - bottom, left:w - right]
        h, w = images.shape[1:3]
    tw, th = (width or w), (height or h)
    if bool(width) != bool(height):          # one side given: the other follows the shape
        tw, th = (width, round(width * h / w)) if width else (round(height * w / h), height)
    multiple = max(1, int(multiple_of))
    tw, th = _round(tw, multiple), _round(th, multiple)
    if (tw, th) == (w, h):
        return images
    if fit not in FITS:
        fit = "crop"

    def resize(x, size):
        t = x.permute(0, 3, 1, 2)
        t = F.interpolate(t, size=size, mode="bilinear", align_corners=False, antialias=size[0] < t.shape[2] or size[1] < t.shape[3])
        return t.permute(0, 2, 3, 1).clamp(0, 1)

    if fit == "stretch":
        return resize(images, (th, tw))
    scale = max(tw / w, th / h) if fit == "crop" else min(tw / w, th / h)
    sh, sw = max(1, round(h * scale)), max(1, round(w * scale))
    scaled = resize(images, (sh, sw))
    if fit == "crop":
        y, x = (sh - th) // 2, (sw - tw) // 2
        return scaled[:, y:y + th, x:x + tw]
    out = torch.zeros((n, th, tw, c), dtype=scaled.dtype)
    y, x = (th - sh) // 2, (tw - sw) // 2
    out[:, y:y + sh, x:x + sw] = scaled
    return out


def merged(options):
    """The options with every missing key at its default; unknown keys dropped."""
    return {**DEFAULTS, **{k: v for k, v in (options or {}).items() if k in DEFAULTS}}


def shape_options(o):
    """The part of the options that `shape` takes."""
    return dict(width=o["width"], height=o["height"], fit=o["fit"], crop_left=o["crop_left"], crop_right=o["crop_right"],
                crop_top=o["crop_top"], crop_bottom=o["crop_bottom"], multiple_of=o["multiple_of"])


def prepare(images, src_fps, options=None):
    """The whole job. -> (frames, out_fps, start_sec, span_sec)."""
    o = merged(options)
    index, out_fps, start, span = pick_frames(images.shape[0], src_fps, o["fps"], o["skip_first"], o["cut_last"], o["every_nth"], o["frame_cap"])
    frames = shape(images[torch.tensor(index)], o["width"], o["height"], o["fit"], o["crop_left"], o["crop_right"],
                   o["crop_top"], o["crop_bottom"], o["multiple_of"])
    return frames, out_fps, start, span


def cut_audio(audio, start, span):
    """The soundtrack's matching stretch (audio dict with waveform [1, C, T] and sample_rate)."""
    rate = audio["sample_rate"]
    wave = audio["waveform"]
    if wave.shape[-1] <= 1:
        return audio
    a = min(wave.shape[-1] - 1, int(start * rate))
    return {"waveform": wave[..., a:a + max(1, int(span * rate))], "sample_rate": rate}
