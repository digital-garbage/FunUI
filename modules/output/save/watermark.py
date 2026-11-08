"""The Watermark of Engine > Post: a text or a picture in one corner of the finished video.

`build(values, width, height)` draws it once (a small RGBA layer and where it goes); `apply(frames, layer)` lays it over a chunk of
frames. It runs last, after the picture's own look, so contrast, colour and smoothing never touch it, and the carried last frame
(which never sees Post) never has it. Everything the layer meets is moved to that tensor's device and dtype first.
"""

from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from ..._core import log, media

MODES = ("off", "text", "picture")
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")
MAX_HEIGHT = 0.5                      # the mark never takes more than half the picture's height, whatever its shape
TITLE = "FunUI Save Video"

Layer = namedtuple("Layer", "rgb alpha y x")        # rgb [h, w, 3] and alpha [h, w, 1] floats 0..1 (straight, not premultiplied); where its top-left lands


def picture_options():
    """The pictures in the media bin, for the Picture choice."""
    items = [{"value": "", "label": "None picked"}]
    try:
        items += [{"value": m["id"], "label": m.get("name") or m["id"]} for m in media.listing() if m.get("kind") == "image"]
    except Exception as exc:                                  # noqa: BLE001 -- an unreadable bin is an empty list, said in the log
        log.once("watermark:bin", log.WARNING, TITLE, f"the media bin could not be listed for the Watermark picture ({exc})")
    return items


def _font(px):
    try:
        return ImageFont.load_default(size=px)                # a scalable font (Pillow 10.1+)
    except TypeError:
        return ImageFont.load_default()                       # an older Pillow: a small fixed one, scaled up below


def _text_alpha(text, target_w):
    """-> alpha [h, w] floats of `text` drawn at the size that makes it `target_w` pixels wide."""
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    probe = _font(64)
    l, t, r, b = draw.textbbox((0, 0), text, font=probe)
    px = max(6, round(64 * target_w / max(1, r - l)))
    font = _font(px)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    img = Image.new("L", (max(1, r - l), max(1, b - t)), 0)
    ImageDraw.Draw(img).text((-l, -t), text, font=font, fill=255)
    img = img.crop(img.getbbox() or (0, 0, img.width, img.height))       # the ink itself: the font's own side room is not part of the width asked for
    if abs(img.width - target_w) > 2:                         # the smallest size, or a fixed font from an older Pillow: scaled to the width asked for
        img = img.resize((target_w, max(1, round(img.height * target_w / img.width))), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _grow(alpha, steps):
    """alpha [1, 1, h, w] -> the shape grown by `steps` pixels (a near-round edge: a square step, then a cross step, and so on). Memory: one map, whatever `steps`."""
    a = alpha
    for i in range(steps):
        if i % 2 == 0:
            a = F.max_pool2d(a, 3, 1, 1)
        else:
            a = torch.maximum(F.max_pool2d(a, (1, 3), 1, (0, 1)), F.max_pool2d(a, (3, 1), 1, (1, 0)))
    return a


def _hex(colour):
    c = str(colour or "").lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (0.0, 0.0, 0.0)


def build(values, width, height):
    """-> a Layer for these (checked) Post values and this frame size, or None when there is nothing to draw (said when it should have been)."""
    mode = values.get("watermark", "off")
    if mode not in ("text", "picture"):
        return None
    target_w = max(8, round(width * float(values.get("watermark_size", 18.0)) / 100.0))
    if mode == "text":
        text = str(values.get("watermark_text") or "").strip()
        if not text:
            log.once("watermark:empty", log.ALERT, TITLE, "Watermark is set to Text but the text is empty, so the video has none.")
            return None
        alpha = _text_alpha(text, target_w)
        rgb = np.broadcast_to(np.array(_hex(values.get("watermark_colour", "#ffffff")), dtype=np.float32), alpha.shape + (3,))
    else:
        path = media.path_for(str(values.get("watermark_picture") or "")) if values.get("watermark_picture") else None
        if path is None:
            log.once("watermark:nopic", log.ALERT, TITLE, "Watermark is set to Picture but no picture from the media bin is picked (or it was removed), so the video has none.")
            return None
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as exc:                              # noqa: BLE001 -- say which file and why, never a silent skip
            log.once("watermark:badpic", log.ALERT, TITLE, f"the Watermark picture could not be read ({exc}), so the video has none.")
            return None
        img = img.resize((target_w, max(1, round(img.height * target_w / img.width))), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        rgb, alpha = arr[..., :3], arr[..., 3]
    if alpha.shape[0] > height * MAX_HEIGHT:                  # a tall mark is brought down to fit, keeping its shape
        k = height * MAX_HEIGHT / alpha.shape[0]
        w2, h2 = max(1, round(alpha.shape[1] * k)), max(1, round(alpha.shape[0] * k))
        rgb = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize((w2, h2), Image.LANCZOS), dtype=np.float32) / 255.0
        alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((w2, h2), Image.LANCZOS), dtype=np.float32) / 255.0
    rgb, alpha = torch.from_numpy(np.ascontiguousarray(rgb)).float(), torch.from_numpy(np.ascontiguousarray(alpha)).float()
    mark_a = alpha * float(values.get("watermark_opacity", 0.6))
    ring = int(round(float(values.get("watermark_outline", 0.0)) / 100.0 * alpha.shape[0]))
    out_rgb, out_a = rgb, mark_a
    if ring > 0:
        pad = ring + 1
        a = F.pad(alpha[None, None], (pad, pad, pad, pad))
        rgb = F.pad(rgb.permute(2, 0, 1)[None], (pad, pad, pad, pad))[0].permute(1, 2, 0)
        grown = (_grow(a, ring) > 1e-3).float()
        grown = F.avg_pool2d(F.pad(grown, (1, 1, 1, 1), mode="replicate"), 3, 1)      # a soft edge instead of a staircase
        outline = (grown - a).clamp(0, 1)[0, 0] * float(values.get("watermark_outline_opacity", 1.0))     # only outside the mark: a see-through mark shows no outline behind it
        mark_a = F.pad(mark_a[None, None], (pad, pad, pad, pad))[0, 0]
        total = mark_a + outline * (1 - mark_a)
        col = torch.tensor(_hex(values.get("watermark_outline_colour", "#000000")), dtype=torch.float32)
        out_rgb = (rgb * mark_a[..., None] + col * (outline * (1 - mark_a))[..., None]) / total.clamp(min=1e-6)[..., None]
        out_a = total
    h, w = out_a.shape
    m = round(width * float(values.get("watermark_margin", 2.0)) / 100.0)
    corner = values.get("watermark_corner", "bottom-right")
    x = m if corner.endswith("left") else width - w - m
    y = m if corner.startswith("top") else height - h - m
    x, y = max(0, min(x, width - w)), max(0, min(y, height - h))
    return Layer(out_rgb.clamp(0, 1), out_a.clamp(0, 1)[..., None], y, x)


def blend(region, layer):
    """region [T, h, w, 3] floats (the corner of a chunk, exactly the layer's size) -> the same, with the layer laid over it, in the region's dtype and device."""
    a, c = layer.alpha.to(device=region.device, dtype=torch.float32), layer.rgb.to(device=region.device, dtype=torch.float32)
    return (region.float() * (1 - a) + c * a).to(region.dtype)


def corner(x, layer):
    """The part of x [T, H, W, 3] the layer covers (a view: nothing is copied)."""
    h, w = layer.alpha.shape[0], layer.alpha.shape[1]
    if layer.y + h > x.shape[1] or layer.x + w > x.shape[2]:            # the frame changed size after the layer was drawn: no guess about where it goes
        raise ValueError("the Watermark was drawn for another frame size")
    return x[:, layer.y:layer.y + h, layer.x:layer.x + w, :]


def apply(x, layer):
    """x [T, H, W, 3] floats -> a copy with the layer laid over its corner (same shape, dtype and device). The save path uses corner() and blend() instead: it never copies a whole chunk."""
    if layer is None:
        return x
    out = x.clone()
    view = corner(out, layer)
    view.copy_(blend(view, layer))
    return out
