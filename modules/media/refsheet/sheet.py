"""The sheet itself: pictures in a grid on white, each labelled below, the title in the top left corner.

Pure Pillow, no ComfyUI. Every picture is cut to the slot's shape (centre crop, like CSS `cover`) and resized to the slot, so
all cells are the same size whatever went in. An empty slot keeps its label and shows a plain frame.
"""

import io

from PIL import Image, ImageDraw, ImageFont, ImageOps

FORMATS = {"1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16)}
MAX_SLOTS, MIN_SLOT, MAX_SLOT, MAX_SIDE = 36, 128, 1024, 8192
LABEL_CHARS = 80
INK, PAPER, FRAME = (20, 20, 20), (255, 255, 255), (222, 222, 222)


def _font(px):
    # Pillow's built-in scalable font: no font file to ship. ponytail: one typeface, a font picker if anyone asks.
    return ImageFont.load_default(size=px)


def _fit(draw, text, font_px, width):
    """The text cut with an ellipsis until it fits `width` on one line."""
    font = _font(font_px)
    text = " ".join(text.split())[:LABEL_CHARS]
    if draw.textlength(text, font=font) <= width:
        return text, font
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text.rstrip() + "…", font


def cover(img, size):
    """`img` cut to the shape of `size` from the middle, then resized to it."""
    return ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), size, Image.LANCZOS, centering=(0.5, 0.5))


def check(rows, cols, shape, width, slots):
    """Refuse a bad request by name; -> (rows, cols, (w, h) of one slot)."""
    for name, v, lo, hi in (("rows", rows, 1, 12), ("columns", cols, 1, 12), ("slot width", width, MIN_SLOT, MAX_SLOT)):
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise ValueError(f"{name} must be a whole number from {lo} to {hi}")
    if rows * cols > MAX_SLOTS:
        raise ValueError(f"{rows} x {cols} is {rows * cols} slots; the most is {MAX_SLOTS}")
    if shape not in FORMATS:
        raise ValueError(f"slot format must be one of {', '.join(FORMATS)}")
    if not isinstance(slots, list) or len(slots) > rows * cols:
        raise ValueError(f"send at most {rows * cols} slots")
    a, b = FORMATS[shape]
    return rows, cols, (width, round(width * b / a))


def make(rows, cols, shape, width, title, slots, open_picture):
    """-> PNG bytes. `slots` is a list of {picture, label} in reading order (left to right, top to bottom); `picture` is whatever
    `open_picture` turns into a PIL image (None or missing = an empty slot)."""
    rows, cols, (w, h) = check(rows, cols, shape, width, slots)
    gap, margin = round(w * 0.06), round(w * 0.08)
    label_px, title_px = max(12, round(w * 0.055)), max(16, round(w * 0.09))
    label_h = round(label_px * 1.9)
    title_h = round(title_px * 2.0) if str(title or "").strip() else 0
    W, H = margin * 2 + cols * w + (cols - 1) * gap, margin * 2 + title_h + rows * (h + label_h) + (rows - 1) * gap
    if max(W, H) > MAX_SIDE:
        raise ValueError(f"the sheet would be {W} x {H} pixels; the most is {MAX_SIDE} on a side. Use fewer slots or a smaller slot width")
    sheet = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(sheet)
    if title_h:
        text, font = _fit(draw, str(title), title_px, W - 2 * margin)
        draw.text((margin, margin), text, fill=INK, font=font)
    for i in range(rows * cols):
        slot = slots[i] if i < len(slots) and isinstance(slots[i], dict) else {}
        x = margin + (i % cols) * (w + gap)
        y = margin + title_h + (i // cols) * (h + label_h + gap)
        picture = open_picture(slot.get("picture")) if slot.get("picture") else None
        if picture is not None:
            sheet.paste(cover(picture, (w, h)), (x, y))
        else:
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline=FRAME, width=2)
        label = str(slot.get("label") or "").strip()
        if label:
            text, font = _fit(draw, label, label_px, w)
            draw.text((x, y + h + round(label_px * 0.45)), text, fill=INK, font=font)
    out = io.BytesIO()
    sheet.save(out, "PNG")
    return out.getvalue()
