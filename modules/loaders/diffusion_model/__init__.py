"""Diffusion model loader. A node-only module."""

from .nodes import FunUIDiffusionModelLoader

ID = "loader_diffusion_model"
TITLE = "Diffusion model loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunUIDiffusionModelLoader]


def routes(table, base, web):
    @table.get(base + "/torch")
    async def _torch(_req):
        """Why int8 would run slow on this torch build (None when it would not), for the chip beside Generate."""
        from ..common import slow_int8_build
        return web.json_response({"slow_int8": slow_int8_build()})

    @table.get(base + "/veda")
    async def _veda(_req):
        """The Veda predictor files that are in models/veda, the two official downloads with their state, and the licence."""
        from .. import veda_download
        return web.json_response(veda_download.listing())

    @table.post(base + "/veda/download")
    async def _veda_download(req):
        """Fetch one official predictor straight from Hugging Face: {"id": "t2va" | "r2va"}. Only ever started by a click."""
        from .. import veda_download
        try:
            body = await req.json()
        except ValueError:
            body = None
        started = veda_download.download(str(body.get("id") or "") if isinstance(body, dict) else "")
        return web.json_response({"started": started})


PROVIDES = {"routes": routes}
