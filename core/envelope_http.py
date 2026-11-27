"""The light-privacy envelope on the server (core/envelope.py is the cipher, app/ui/shell/envelope.js the browser half).

The browser sends every call to ONE address, `/fun/x`, as an encrypted message holding the real method, address and body;
this module unwraps it, hands it to the real route as if it had arrived plainly, and wraps the answer (a body, a finished
video, a stream of progress lines) the same way. So the wire shows one address and unreadable data, not "generate" or
"upload picture". The socket's progress and preview frames are wrapped when the browser asks (`?e=1`).

It is light privacy, not security: the key is built into FunUI, the same everywhere. The plain addresses keep working.
"""

import asyncio
import json
import mimetypes
from pathlib import Path

from aiohttp import web
from yarl import URL

from . import config, envelope as env

CIPHER = env.Cipher()
REQUEST, ANSWER, SOCKET = b"q", b"r", b"w"
SKIP = {"content-length", "transfer-encoding", "content-encoding", "connection", "keep-alive", "date", "server"}
OUTER = {"Content-Type": "application/octet-stream", "Cache-Control": "no-store"}


def _headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in SKIP}


def _first(status, headers):
    return CIPHER.frame(json.dumps({"s": status, "h": _headers(headers)}).encode(), ANSWER)


class StreamResponse(web.StreamResponse):
    """web.StreamResponse that is wrapped when its request came through the envelope (for answers written bit by bit)."""

    _wrapped = False

    async def prepare(self, request):
        if not getattr(request, "funui_envelope", False):
            return await super().prepare(request)
        first = _first(self.status, self.headers)
        self._wrapped = True
        self.set_status(200)
        self.headers.clear()
        self.headers.update(OUTER)
        out = await super().prepare(request)
        await super().write(first)
        return out

    async def write(self, data):
        if not self._wrapped:
            return await super().write(data)
        for i in range(0, len(data), env.CHUNK):
            await super().write(CIPHER.frame(bytes(data[i:i + env.CHUNK]), ANSWER))

    async def write_eof(self, data=b""):
        if self._wrapped and data:
            await self.write(data)
            data = b""
        return await super().write_eof(data)


async def _wrap(request, answer):
    """Any aiohttp answer -> the envelope's."""
    if isinstance(answer, web.FileResponse):
        return await _wrap_file(request, answer)
    if isinstance(answer, web.StreamResponse) and not isinstance(answer, web.Response):
        return answer                                    # written through StreamResponse already
    body = answer.body if answer.body is not None else b""
    if not isinstance(body, (bytes, bytearray, memoryview)):
        body = answer.text.encode(answer.charset or "utf-8") if answer.text is not None else bytes(body)
    return web.Response(status=200, body=_first(answer.status, answer.headers) + CIPHER.pack(body, ANSWER), headers=OUTER)


async def _wrap_file(request, answer):
    path = Path(answer._path)
    if not path.is_file():
        return web.Response(status=200, body=_first(404, {}) + CIPHER.pack(b"", ANSWER), headers=OUTER)
    ctype = answer.headers.get("Content-Type") or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    reply = web.StreamResponse(status=200, headers=OUTER)
    await reply.prepare(request)
    await reply.write(_first(200, {**answer.headers, "Content-Type": ctype}))
    loop = asyncio.get_running_loop()
    with open(path, "rb") as f:
        while True:
            piece = await loop.run_in_executor(None, f.read, env.CHUNK)
            if not piece:
                break
            await reply.write(CIPHER.frame(piece, ANSWER))
    await reply.write(CIPHER.frame(b"", ANSWER))
    await reply.write_eof()
    return reply


async def _call(request):
    try:
        parts = CIPHER.unpack(await request.content.read(), REQUEST)      # not request.read(): a request whose body was read cannot be cloned
        meta = json.loads(parts[0])
        method, target = str(meta["m"]).upper(), str(meta["p"])
        if not target.startswith("/") or target.startswith("//"):
            raise ValueError("address")
    except (env.EnvelopeError, ValueError, KeyError, IndexError, TypeError):
        return web.json_response({"error": "this was not a FunUI envelope"}, status=400)
    body = b"".join(parts[1:])
    headers = {k: v for k, v in request.headers.items() if k.lower() not in SKIP | {"content-type"}}
    for k, v in (meta.get("h") or {}).items():
        headers[str(k)] = str(v)
    if body:
        headers["Content-Length"] = str(len(body))
    inner = request.clone(method=method, rel_url=URL(target), headers=headers)
    from aiohttp import streams
    reader = streams.StreamReader(request.protocol, 2 ** 16, loop=asyncio.get_running_loop())
    reader.feed_data(body)
    reader.feed_eof()
    inner._payload, inner._read_bytes = reader, None
    inner.funui_envelope = True                          # its answer is to be wrapped
    try:
        answer = await request.app._handle(inner)
    except web.HTTPException as exc:
        answer = exc
    return await _wrap(inner, answer)


# ---------------------------------------------------------------- the socket (progress, previews)

_patched = False


def patch_websocket():
    """ComfyUI sends progress and preview frames through aiohttp's WebSocketResponse: a socket opened with `?e=1` gets them wrapped."""
    global _patched
    if _patched:
        return
    _patched = True
    plain = web.WebSocketResponse
    prepare, send_str, send_bytes, send_json = plain.prepare, plain.send_str, plain.send_bytes, plain.send_json

    async def _prepare(self, request):
        self._funui_wrapped = request.query.get("e") == "1"
        return await prepare(self, request)

    def _message(kind, data):
        return CIPHER.seal(kind + data, SOCKET)

    async def _str(self, data, *a, **k):
        return await (send_bytes(self, _message(b"\x00", data.encode())) if getattr(self, "_funui_wrapped", False) else send_str(self, data, *a, **k))

    async def _bytes(self, data, *a, **k):
        return await (send_bytes(self, _message(b"\x01", bytes(data))) if getattr(self, "_funui_wrapped", False) else send_bytes(self, data, *a, **k))

    async def _json(self, data, *a, dumps=json.dumps, **k):
        return await (send_bytes(self, _message(b"\x00", dumps(data).encode())) if getattr(self, "_funui_wrapped", False) else send_json(self, data, *a, dumps=dumps, **k))

    plain.prepare, plain.send_str, plain.send_bytes, plain.send_json = _prepare, _str, _bytes, _json


def register(routes, prefix=None):
    P = config.UI_PREFIX if prefix is None else prefix
    patch_websocket()
    routes.post(P + "/x")(_call)
