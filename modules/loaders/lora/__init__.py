"""LoRA loader. A node-only module."""

from .nodes import FunUILoraLoader

ID = "loader_lora"
TITLE = "LoRA loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunUILoraLoader]
