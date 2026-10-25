"""Text encoder loader. A node-only module."""

from .nodes import FunUICLIPLoader

ID = "loader_clip"
TITLE = "CLIP loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunUICLIPLoader]
