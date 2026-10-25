"""The modifier nodes. A node-only module."""

from .nodes import FunUILoadModifiers, FunUIModifierSettings

ID = "sampling_modifiers"
TITLE = "Modifiers"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "proven"

NODES = [FunUIModifierSettings, FunUILoadModifiers]
USES = ["modifier"]       # the modifiers it installs act only when both its nodes are in the pipeline


def settings_sink():
    """Where the values a person picked belong in a graph.

    Announced rather than known: core places the app's settings into whatever a
    module says accepts them, so this node is not privileged, it is merely the
    one that volunteered.
    """
    return {"node": "FunUIModifierSettings", "input": "settings"}


PROVIDES = {"settings_sink": settings_sink}
