"""Empty latent. A node-only module."""

from .nodes import FunUIAspectSize, FunUIEmptyLatent

ID = "latent_empty"
TITLE = "Empty latent"
STAGE = "latent"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunUIEmptyLatent, FunUIAspectSize]


BASIS_ROLE = {"at": "project.resolution_basis", "input": "basis"}


def upgrade(slots, _outputs=None):
    """An Aspect Size saved before the project could say what its width and height mean gets the role that carries that, and
    keeps the rule it always had (the pixel count) until the project's own setting says otherwise."""
    if not any(x.get("node") == "FunUIAspectSize" and BASIS_ROLE not in (x.get("roles") or []) for x in slots):
        return slots
    return [{**x, "roles": [*(x.get("roles") or []), BASIS_ROLE], "inputs": {**(x.get("inputs") or {}), "basis": "area"}}
            if x.get("node") == "FunUIAspectSize" and BASIS_ROLE not in (x.get("roles") or []) else x for x in slots]


PROVIDES = {"upgrade_slots": upgrade}
