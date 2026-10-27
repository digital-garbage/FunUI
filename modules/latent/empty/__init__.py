"""Empty latent. A node-only module."""

from .nodes import FunUIAspectSize, FunUIEmptyLatent

ID = "latent_empty"
TITLE = "Empty latent"
STAGE = "latent"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunUIEmptyLatent, FunUIAspectSize]
