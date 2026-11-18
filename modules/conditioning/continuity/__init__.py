"""Continuity: a shot that follows another starts from where the other ended.

A scene whose Source is "From generated frame" starts from the last picture of the clip before it, so
the shot carries on instead of starting from nothing. The picture is taken from the finished render
(never invented), saved in the media bin, and added to the scene's references for the run, in the next
free reference slot; this module only holds the one choice. The work is the editor's: it asks the render module for the frame.

The end of the clip before can also go on as a video (with its sound) and/or as a sound alone: the last N frames, counted at
the frame rate the clip was generated at and cut from the frames as generated, before any upscale or interpolation.

A clip that ends in a fade to black has no picture to continue from (v4's i2i lesson: 3 of 4 anchors were
solid black), so such a shot always starts without one and says so, rather than starting black. (Not a setting: there is no reason to want a black start.)
"""

ID = "continuity"
TITLE = "Continuity"
MOUNT = "settings.general"
STAGE = "conditioning"
CATEGORY = "continuity"
STATUS = "experimental"
REQUIRES = ["frame_carry"]                # H3 only for now
ROLES = ["assets.reference_1"]        # a pipeline with no reference input has nothing to continue through

SETTINGS = {
    "carry": {
        "type": "bool", "default": True,
        "label": "Pass the last frame of previous scene as reference into next scene",
        "hint": "When active, last frame of your previous scene is saved and passed into the model in the next available reference slot.",
        "home": "project",          # drawn in Project settings, not on the Engine page
    },
    "tail_video": {
        "type": "bool", "default": False,
        "label": "Pass the end of the previous scene as a video reference",
        "hint": "Passes the last frames of the previous scene, with their sound, as a video reference. Taken as generated, before any upscale or interpolation.",
        "home": "project",
    },
    "tail_audio": {
        "type": "bool", "default": False,
        "label": "Pass the sound of the previous scene's end as a sound reference",
        "hint": "Passes only the sound of the same stretch into the next scene's sound reference. Tick both to carry the video and the sound separately.",
        "home": "project",
    },
    "tail_seconds": {
        "type": "float", "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1, "ui": "number",
        "label": "Carry-over length (seconds)",
        "hint": "How much of the end is passed on. Seconds and frames follow each other at the frame rate the clips are generated at.",
        "home": "project", "sync": {"with": "tail_frames", "per_second": True},
    },
    "tail_frames": {
        "type": "int", "default": 50, "min": 1, "max": 1000, "step": 1, "ui": "number",
        "label": "Carry-over length (frames)",
        "hint": "The same length counted in frames at the generation frame rate, not the interpolated one.",
        "home": "project", "sync": {"with": "tail_seconds", "per_second": False},
    },
}
