"""Save the finished video: MP4 (H.264), on the GPU's encoder when there is one, else on every CPU core.

Core's SaveVideo encodes on one CPU thread; on a rental that was most of the time after decoding.
"""

from . import watermark
from .nodes import FunUISaveVideo

ID = "output_save"
TITLE = "Save video"
STAGE = "post"
CATEGORY = "post"
MOUNT = "settings.general"
STATUS = "proven"

NODES = [FunUISaveVideo]

# Engine > Post. Every one is off (or neutral) by default; the node reads them from the pipeline's settings object.
SETTINGS = {
    "smoothing": {
        "type": "enum", "default": "off",
        "options": [{"value": "off", "label": "Off"}, {"value": "light", "label": "Light"}, {"value": "strong", "label": "Strong"}],
        "label": "Smooth jagged edges",
        "hint": "Anti-aliasing like FXAA in games: only edges with strong contrast are softened, flat areas and texture stay. "
                "Cheap on a GPU, noticeably slower on a CPU for 4K clips. "
                "These look settings are in the saved file; a clip trimmed in the timeline starts the next shot from that file's picture, so it carries them too.",
    },
    "auto_contrast": {
        "type": "bool", "default": False,
        "label": "Auto contrast",
        "hint": "Stretches the clip's darkest and brightest levels (by at most 15% each) toward black and white. Measured once from 16 frames spread over the clip and applied as one fixed curve, so it never flickers.",
    },
    "auto_brightness": {
        "type": "bool", "default": False,
        "label": "Auto brightness",
        "hint": "Lifts or lowers the clip's midtones toward a middle level (within a mild range). Measured once from 16 frames spread over the clip, never per frame.",
    },
    "saturation": {
        "type": "float", "default": 1.0, "min": 0.5, "max": 1.5, "step": 0.05, "ui": "slider",
        "label": "Colour strength",
        "hint": "1 leaves the colours as they are; below 1 mutes them, above 1 makes them stronger.",
    },
    "watermark": {
        "type": "enum", "default": "off",
        "options": [{"value": "off", "label": "Off"}, {"value": "text", "label": "Text"}, {"value": "picture", "label": "Picture"}],
        "label": "Watermark",
        "hint": "A text or a picture from the media bin in one corner of the finished video. It is laid on last, so the settings above never change it. "
                "It is in the saved file only: the next shot's start picture and the Preview's own frames do not carry it.",
    },
    "watermark_text": {
        "type": "text", "default": "", "label": "Watermark text", "when": {"watermark": "text"},
        "hint": "The words to show. They are scaled to the width below, whatever their length.",
    },
    "watermark_colour": {
        "type": "color", "default": "#ffffff", "label": "Text colour", "when": {"watermark": "text"},
        "hint": "The colour of the text.",
    },
    "watermark_picture": {
        "type": "enum", "default": "", "options": [{"value": "", "label": "None picked"}], "label": "Watermark picture", "when": {"watermark": "picture"},
        "hint": "A picture from the media bin. A transparent PNG keeps its transparency.",
    },
    "watermark_corner": {
        "type": "enum", "default": "bottom-right", "label": "Corner", "when": {"watermark": ["text", "picture"]},
        "options": [{"value": "top-left", "label": "Top left"}, {"value": "top-right", "label": "Top right"},
                    {"value": "bottom-left", "label": "Bottom left"}, {"value": "bottom-right", "label": "Bottom right"}],
        "hint": "Which corner of the video the mark sits in.",
    },
    "watermark_opacity": {
        "type": "float", "default": 0.6, "min": 0.05, "max": 1.0, "step": 0.05, "ui": "slider", "label": "Opacity", "when": {"watermark": ["text", "picture"]},
        "hint": "1 is solid, lower lets the video show through.",
    },
    "watermark_size": {
        "type": "float", "default": 18.0, "min": 5.0, "max": 60.0, "step": 1.0, "ui": "slider", "label": "Width", "unit": "% of the video's width",
        "when": {"watermark": ["text", "picture"]},
        "hint": "How wide the mark is. It never takes more than half the video's height.",
    },
    "watermark_margin": {
        "type": "float", "default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5, "ui": "slider", "label": "Distance from the edge", "unit": "% of the video's width",
        "when": {"watermark": ["text", "picture"]},
        "hint": "How far the mark stays from the edges. The outline is kept inside it.",
    },
    "watermark_outline": {
        "type": "float", "default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5, "ui": "slider", "label": "Outline size", "unit": "% of the mark's height",
        "when": {"watermark": ["text", "picture"]},
        "hint": "0 draws no outline. For a picture the outline follows its shape, so it suits a transparent PNG.",
    },
    "watermark_outline_colour": {
        "type": "color", "default": "#000000", "label": "Outline colour", "when": {"watermark": ["text", "picture"]},
        "hint": "The colour of the outline.",
    },
    "watermark_outline_opacity": {
        "type": "float", "default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05, "ui": "slider", "label": "Outline opacity", "when": {"watermark": ["text", "picture"]},
        "hint": "1 is solid, lower lets the video show through the outline. Only used when the outline size is above 0.",
    },
}

PROVIDES_OPTIONS = lambda: {"watermark_picture": watermark.picture_options()}


def _link(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def upgrade(slots, outputs):
    """Pipelines saved before this node hold core's CreateVideo -> SaveVideo: one FunUI Save Video instead.
    A SaveVideo left with no feed takes the one picture source nothing else reads (its sound too).
    `outputs(node)` -> that node's output types. -> the slots, changed or not."""
    slots = [dict(s) for s in slots]
    by_id = {s["id"]: s for s in slots}
    read = {v[0] for s in slots for v in (s.get("inputs") or {}).values() if _link(v)}
    drop = set()
    for s in slots:
        if s["node"] != "SaveVideo":
            continue
        feed = (s.get("inputs") or {}).get("video")
        src = by_id.get(feed[0]) if _link(feed) else None
        new = {"filename_prefix": (s.get("inputs") or {}).get("filename_prefix", "FunUI")}
        roles = s.get("roles")
        if src is not None and src["node"] == "CreateVideo":
            new.update({k: v for k, v in (src.get("inputs") or {}).items() if k in ("images", "audio", "fps")})
            roles = src.get("roles") or roles
            if sum(1 for o in slots for v in (o.get("inputs") or {}).values() if _link(v) and v[0] == src["id"]) == 1:
                drop.add(src["id"])
        elif src is None:
            loose = [o for o in slots if o["id"] not in read and "IMAGE" in outputs(o["node"])]
            if len(loose) != 1:
                continue
            kinds = outputs(loose[0]["node"])
            new["images"] = [loose[0]["id"], kinds.index("IMAGE")]
            if "AUDIO" in kinds:
                new["audio"] = [loose[0]["id"], kinds.index("AUDIO")]
        else:
            continue
        s.update(node="FunUISaveVideo", inputs=new)
        if roles:
            s["roles"] = roles
    # Engine > Post settings reach the node through the pipeline's settings object; a pipeline saved before that is linked to it.
    sink = next((s["id"] for s in slots if s["node"] == "FunUIModifierSettings"), None)
    for s in slots:
        if sink and s["node"] == "FunUISaveVideo" and "settings" not in (s.get("inputs") or {}):
            s["inputs"] = {**(s.get("inputs") or {}), "settings": [sink, 0]}
    return [s for s in slots if s["id"] not in drop]


PROVIDES = {"upgrade_slots": upgrade, "setting_options": PROVIDES_OPTIONS}
