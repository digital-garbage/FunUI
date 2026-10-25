"""Decode. A node-only module."""

from .nodes import FunUIDecode

ID = "output_decode"
TITLE = "Decode"
STAGE = "post"
CATEGORY = "post"
STATUS = "proven"

NODES = [FunUIDecode]
