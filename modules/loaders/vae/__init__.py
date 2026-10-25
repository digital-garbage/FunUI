"""VAE loader.

A node-only module: no settings, no panel, no mount. It exists to contribute one
ComfyUI node, which is why `SETTINGS` is absent rather than empty.
"""

from .nodes import FunUIVAELoader

ID = "loader_vae"
TITLE = "VAE loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunUIVAELoader]
