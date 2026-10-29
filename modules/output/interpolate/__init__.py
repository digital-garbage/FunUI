"""Interpolate: more frames for a finished video, with NVIDIA RTX Frame Generation (the nvidia-vfx add-on).

The length and the sound stay, the frame rate goes up (x2 turns 24 fps into 48 fps). It runs beside ComfyUI on the file a
render wrote, like Upscale, and is a module of its own. Unlike Upscale it is a background job of the page and not a ComfyUI
prompt, so it never disturbs ComfyUI's cache of the workflow's loaded models.
"""

import asyncio

from . import framegen, nodes
from ..._core import jobs

ID = "interpolate"
TITLE = "Interpolate"
STAGE = "post"
CATEGORY = "post"
STATUS = "experimental"
NODES = []


def routes(table, base, web):
    def bad(detail, status=400):
        return web.json_response({"detail": detail}, status=status)

    @table.get(base + "/frame_generation")
    async def _frame_generation(_req):
        from ..._core import addons
        return web.json_response({
            "rtx": {"name": framegen.NAME, "modes": list(framegen.MODES), "max_multiplier": framegen.MAX_MULTIPLIER} if addons.enabled("nvidia_framegen") else None,
            "rtx_offered": any(a["id"] == "nvidia_framegen" for a in addons.listing())})   # offered = this machine could switch it on in Settings ▸ Modules

    @table.post(base + "/run")
    async def _run(req):
        try:
            body = await req.json()
        except Exception:                          # noqa: BLE001
            body = None
        media = body.get("media") if isinstance(body, dict) else None
        if not isinstance(media, dict) or not isinstance(media.get("filename"), str) or not media["filename"]:
            return bad("Send {media: {filename, subfolder, type}, options}.")
        try:
            framegen.options((body or {}).get("options") or {})            # refuse a bad value now, by name, not after the job was queued
        except ValueError as exc:
            return bad(str(exc))
        job = jobs.new()
        asyncio.create_task(jobs.run(job, nodes.interpolate, media, body.get("options") or {}, error=nodes.InterpolateError))
        return web.json_response({"job_id": job})

    @table.get(base + "/jobs/{job}")
    async def _job(req):
        job = jobs.get(req.match_info["job"])
        return web.json_response(job) if job else bad("Interpolate job not found (ComfyUI restarted?).", 404)


PROVIDES = {"routes": routes}
