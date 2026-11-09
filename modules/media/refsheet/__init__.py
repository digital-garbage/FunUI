"""Reference sheet: pictures from the media bin in a labelled grid on white, saved back to the bin as a picture.

An optional helper for building a reference picture (a character from several angles, outfits, props) that is then used like any
other picture. Remove the folder and nothing else changes.
"""

import asyncio

from . import sheet
from ..._core import media

ID = "refsheet"
TITLE = "Reference sheet"
STAGE = "load"
CATEGORY = "system"
STATUS = "experimental"
NODES = []


def _open(media_id):
    """A bin picture as a PIL image, or None when it is gone or is not a picture."""
    from PIL import Image
    path = media.path_for(media_id) if isinstance(media_id, str) else None
    item = media.get(media_id) if path else None
    if not item or item.get("kind") != "image":
        return None
    try:
        with Image.open(path) as im:
            im.load()
            return im.copy()
    except Exception:                       # noqa: BLE001 -- a file Pillow cannot read is an empty slot, said in "missing"
        return None


def build(body):
    """-> (bin entry, [labels of slots whose picture could not be read]). Raises ValueError naming what to fix."""
    if not isinstance(body, dict):
        raise ValueError("Send {rows, cols, format, width, title, slots: [{picture, label}]}.")
    slots = body.get("slots") or []
    missing = [str(s.get("label") or i + 1) for i, s in enumerate(slots) if isinstance(s, dict) and s.get("picture") and _open(s["picture"]) is None]
    png = sheet.make(body.get("rows", 2), body.get("cols", 3), body.get("format", "1:1"), body.get("width", 512), body.get("title"), slots, _open)
    name = " ".join(str(body.get("title") or "").split())[:60] or "Reference sheet"
    return media.save_upload(f"{name}.png", png, limit=None), missing


def routes(table, base, web):
    @table.post(base + "/make")
    async def _make(req):
        try:
            body = await req.json()
        except Exception:                   # noqa: BLE001
            body = None
        try:
            entry, missing = await asyncio.to_thread(build, body)
        except ValueError as exc:
            return web.json_response({"detail": str(exc)}, status=400)
        return web.json_response({"media": entry, "missing": missing})

    @table.get(base + "/formats")
    async def _formats(_req):
        return web.json_response({"formats": list(sheet.FORMATS), "max_slots": sheet.MAX_SLOTS})


PROVIDES = {"routes": routes}
