"""Continuity: a shot that follows another starts from where the other ended.

A scene whose Source is "From generated frame" starts from the last picture of the clip before it, so
the shot carries on instead of starting from nothing. The picture is taken from the finished render
(never invented), saved in the media bin, and added to the scene's references for the run, in the next
free reference slot; this module only holds the one choice. The work is the editor's: it asks the render module for the frame.

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
}
