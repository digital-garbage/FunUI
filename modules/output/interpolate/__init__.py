"""Interpolate: more frames for a finished video, with NVIDIA RTX Frame Generation (the nvidia-vfx add-on).

The length and the sound stay, the frame rate goes up (x2 turns 24 fps into 48 fps). It runs
beside ComfyUI on the file a render wrote, like Upscale, and is a module of its own.

Not yet run on the real SDK (no RTX card in the test machine); the streaming around it is tested against real ffmpeg.
"""

from . import framegen
from . import nodes

try:
    from aiohttp import web
except Exception:                                  # noqa: BLE001 -- tests without aiohttp
    web = None

ID = "interpolate"
TITLE = "Interpolate"
STAGE = "post"
CATEGORY = "post"
STATUS = "experimental"
NODES = [nodes.FunUIInterpolateVideo] if nodes.io is not None else []


def routes(table, base, web):
    @table.get(base + "/frame_generation")
    async def _frame_generation(_req):
        from ..._core import addons
        return web.json_response({
            "rtx": {"name": framegen.NAME, "modes": list(framegen.MODES), "max_multiplier": framegen.MAX_MULTIPLIER} if addons.enabled("nvidia_framegen") else None,
            "rtx_offered": any(a["id"] == "nvidia_framegen" for a in addons.listing())})   # offered = this machine could switch it on in Settings ▸ Modules


PROVIDES = {"routes": routes}
