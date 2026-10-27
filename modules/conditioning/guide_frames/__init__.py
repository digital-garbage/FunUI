"""Guide frames: pictures from the clips before a shot, laid into it as weak guides (LTX).

Two ways, one mechanism. **Middle frame** (v4's mid-scene guide): the middle picture of the clip before
carries over as a guide, so static things (a side character, props, the room) stay where they were and
the action is free to change. **Memory** (v4's JoyAI-Echo memory): one picture from each of the last few
clips, spread through the shot; it needs the JoyAI-Echo LoRA to mean anything, because the base model was
never trained to read them as memory.

The guides go in through ComfyUI's own LTX guide attention (the LTXVAddGuide logic, applied once per
picture), so they ride the same path as any LTX guide. The app picks the pictures (the editor knows the
timeline) and sends them as a list; with no list, every output is the input, untouched. This module
holds the node and the choices.

From v4's tests: 0.25 is the floor. Higher bleeds the previous composition in; lower leaves the sound as
noise, because the guide is physically in the sequence and the sound attends to it whatever its strength.
"""

from .nodes import FunUIGuideFrames

ID = "guide_frames"
TITLE = "Guide frames"
MOUNT = "settings.general"
STAGE = "conditioning"
CATEGORY = "continuity"
STATUS = "experimental"
REQUIRES = ["ltx_av"]

NODES = [FunUIGuideFrames]

SETTINGS = {
    "mode": {
        "type": "enum", "default": "off", "label": "Guide frames",
        "options": [{"value": "off", "label": "Off"},
                    {"value": "middle", "label": "Middle frame of the clip before"},
                    {"value": "memory", "label": "Memory from the last clips"}],
        "hint": "Lays pictures from the clip before into this shot as weak guides. Middle frame: keeps the room and side characters where they were. "
                "Memory: one picture from each of the last clips (needs the JoyAI-Echo LoRA). Costs roughly half again per step.",
    },
    "strength": {
        "type": "float", "default": 0.25, "min": 0.25, "max": 1.0, "step": 0.05,
        "label": "Strength", "ui": "slider",
        "hint": "0.25 is the sweet spot: the character and action are held, the background can still adapt. Higher bleeds the old composition in.",
        "when": {"mode": ["middle", "memory"]},
    },
    "clips": {
        "type": "int", "default": 3, "min": 1, "max": 8,
        "label": "Clips to remember",
        "hint": "How many earlier clips give one picture each.",
        "when": {"mode": "memory"},
    },
}
