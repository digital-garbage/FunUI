"""Remote Instance: the local page drives ComfyUI and FunUI on another machine over SSH.

While connected, a middleware on the local server hands every call to the remote machine through an SSH tunnel, except the
ones that belong to this machine: projects, media, shortcuts, the story markers, the shortcut library files, the prompt
expansion (it uses the shortcuts) and the page's own files. So the page shows the remote's modules, models and queue while
projects and shortcuts stay here. Disconnecting (or losing the connection) just switches the middleware off: the next call is
answered here again.

Media a project points to lives here, but the nodes that read it run there: before a prompt or a pipeline call goes out, the
media ids it names are copied across (once each).
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time

from . import config, media, remote_ssh as ssh, update as update_mod

try:
    import aiohttp
    from aiohttp import web
except ImportError:                         # pragma: no cover -- tests of the pure parts run without it
    aiohttp = web = None

P = config.UI_PREFIX
CONFIG_FILE = config.ROOT / "remote.json"

# What stays on this machine while connected (everything else under the server goes to the remote one).
_LOCAL = re.compile(
    "^(" + re.escape(P) + r"(/app/.*|/?|/api/(projects|media|shortcuts|story|files|remote|health|mirror|prompt/expand).*)"
    + "|" + re.escape(config.LEGACY_PREFIX) + "/?.*" + ")$")
_MEDIA_ID = re.compile(r'"([0-9a-f]{12})"')
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding",
        "upgrade", "host", "content-length"}


def is_local(path: str) -> bool:
    return bool(_LOCAL.match(path))


# ---------------------------------------------------------------- this FunUI's identity

_ident = {}


def identity() -> dict:
    """Version and commit of this FunUI: what a remote one is compared with."""
    if not _ident:
        _ident["version"] = update_mod.funui_version()
        try:
            r = update_mod._run_git("rev-parse", "HEAD", timeout=10)
            _ident["commit"] = r.stdout.strip() if r.returncode == 0 else ""
        except Exception:                   # noqa: BLE001 -- not a checkout, or no git
            _ident["commit"] = ""
    return dict(_ident)


def _git_here(*args) -> str:
    try:
        r = update_mod._run_git(*args, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:                       # noqa: BLE001
        return ""


# ---------------------------------------------------------------- the connection

class Conn:
    """One live connection. Module-level state: a server has one remote instance at a time."""
    state = "off"           # off | working | on | failed
    message = ""
    steps: list = []
    login = None
    client = None
    forward = None
    comfy = None
    mirrored: set = set()
    attempt = 0             # which Connect press is current: a cancelled one finishing late must change nothing
    watch = None
    lock = threading.Lock()


def saved_login() -> dict:
    try:
        d = json.loads(CONFIG_FILE.read_text("utf-8"))
        return {k: d.get(k, v) for k, v in {"host": "", "port": 22, "user": "root", "key": ""}.items()}
    except (OSError, ValueError):
        return {"host": "", "port": 22, "user": "root", "key": ""}


def _save_login(login: ssh.Login) -> None:
    CONFIG_FILE.write_text(json.dumps({"host": login.host, "port": login.port, "user": login.user, "key": login.key}), "utf-8")


def snapshot() -> dict:
    return {"state": Conn.state, "message": Conn.message, "steps": list(Conn.steps), "login": saved_login(),
            "paramiko": ssh.paramiko is not None, "target": Conn.login.host if Conn.state == "on" and Conn.login else ""}


def active() -> bool:
    if Conn.state != "on":
        return False
    if not ssh.alive(Conn.client):
        drop("The SSH connection was lost. FunUI is back on this machine.")
        return False
    return True


def drop(message: str = "", state: str = "off") -> None:
    with Conn.lock:
        Conn.attempt += 1                   # whatever was still connecting is now stale
        for thing in (Conn.forward,):
            if thing:
                thing.close()
        if Conn.client:
            try:
                Conn.client.close()
            except Exception:               # noqa: BLE001
                pass
        Conn.client = Conn.forward = Conn.comfy = None
        Conn.mirrored = set()
        Conn.state, Conn.message = state, message


def _say(text: str) -> None:
    Conn.steps.append(text)
    Conn.message = text


class Cancelled(Exception):
    pass


def connect_sync(login: ssh.Login, token: int | None = None) -> None:
    """Everything between pressing Connect and being connected. Raises RemoteError with a sentence."""
    token = Conn.attempt if token is None else token
    here = identity()
    client = ssh.connect(login)
    if token != Conn.attempt:               # cancelled while logging in
        client.close()
        raise Cancelled()
    Conn.client = client
    _say("Looking for ComfyUI on the remote machine")
    ports = ssh.comfy_ports(client)
    if not ports:
        raise ssh.RemoteError("SSH works, but no ComfyUI answers on that machine.")
    port = ports[0]
    fwd = ssh.Forward(client.get_transport(), "127.0.0.1", port)
    Conn.forward = fwd
    stats = ssh.http_get_json(fwd.local_port, "/system_stats")
    comfy = ssh.describe_comfy(client, port, stats)

    _say("Checking FunUI on the remote machine")
    health = ssh.http_get_json(fwd.local_port, P + "/api/health") or {}
    same = (health.get("version") == here["version"]
            and (not here["commit"] or not health.get("commit") or health.get("commit") == here["commit"]))
    if not same:
        branch = _git_here("rev-parse", "--abbrev-ref", "HEAD")
        url = _git_here("config", "--get", "remote.origin.url")
        if not branch or branch == "HEAD" or not url:
            raise ssh.RemoteError("FunUI on the remote machine is missing or a different version, and this FunUI is not a Git checkout, "
                                  "so it cannot copy itself there. Install FunUI there with Git, then connect again.")
        if not comfy.folder:
            raise ssh.RemoteError("ComfyUI answers there, but its folder could not be found, so FunUI cannot be placed in it.")
        funui_dir = ssh.find_funui(client, comfy.folder)
        funui_dir, changed = ssh.bring_funui_here(client, comfy.folder, funui_dir, url, branch, _say)
        if changed:
            ssh.install_requirements(client, funui_dir, comfy.exe, _say)
            ssh.restart_comfy(client, comfy, _say)
            _say("Checking FunUI on the remote machine")
            deadline = time.time() + 60
            health = {}
            while time.time() < deadline and health.get("version") is None:
                health = ssh.http_get_json(fwd.local_port, P + "/api/health") or {}
                time.sleep(2)
        if health.get("version") != here["version"]:
            raise ssh.RemoteError("FunUI on the remote machine still reports version "
                                  f"{health.get('version') or 'none'}, not {here['version']}. Restart ComfyUI there and connect again.")
    held = ssh.http_get_json(fwd.local_port, P + "/api/media") or {}
    Conn.mirrored = {m.get("id") for m in held.get("media", []) if isinstance(m, dict)}
    if token != Conn.attempt:
        raise Cancelled()
    Conn.comfy, Conn.login = comfy, login
    _save_login(login)


async def connect(login: ssh.Login) -> None:
    if Conn.state == "working":
        return
    if Conn.state == "on":
        drop()
    with Conn.lock:
        Conn.attempt += 1
        token = Conn.attempt
        Conn.state, Conn.message, Conn.steps = "working", "Connecting", ["Connecting to " + login.host]

    def job():
        try:
            connect_sync(login, token)
        except Cancelled:
            return                          # the person pressed Cancel: drop() already set the state
        except Exception as exc:            # noqa: BLE001
            if token == Conn.attempt:       # a cancelled attempt's late failure says nothing
                drop(ssh.reason(exc), "failed")
            return
        with Conn.lock:
            if token != Conn.attempt:
                return
            Conn.state, Conn.message = "on", "Remote Mode Active"
        asyncio.run_coroutine_threadsafe(_watch(token), loop)

    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(asyncio.to_thread(job), OVERALL)
    except asyncio.TimeoutError:
        if token == Conn.attempt:
            drop("Connecting took too long and was stopped.", "failed")
    finally:
        if token == Conn.attempt and Conn.state == "working":      # nothing may leave the page waiting
            drop("The connection attempt ended without an answer.", "failed")


OVERALL = 20 * 60           # the longest a connect may take: a clone, requirements and a restart can be slow
WATCH_EVERY, WATCH_MISSES = 6, 2


async def _watch(token: int) -> None:
    """While connected: ask the remote machine every few seconds. A link that stops answering without closing is dropped."""
    misses = 0
    while token == Conn.attempt and Conn.state == "on":
        await asyncio.sleep(WATCH_EVERY)
        if token != Conn.attempt or Conn.state != "on":
            return
        ok = await asyncio.to_thread(ssh.ping, Conn.client)
        misses = 0 if ok else misses + 1
        if misses >= WATCH_MISSES and token == Conn.attempt:
            drop("The remote machine stopped answering. FunUI is back on this machine.")
            return


# ---------------------------------------------------------------- forwarding

def _is_text_call(req) -> bool:
    """A call whose body names media: the prompt ComfyUI is given and FunUI's pipeline compile."""
    return req.method == "POST" and (req.path in ("/prompt", "/api/prompt") or req.path == P + "/api/pipeline")


async def _mirror_media(body: bytes, session, base: str) -> None:
    for mid in set(_MEDIA_ID.findall(body.decode("utf-8", "ignore"))):
        if mid in Conn.mirrored:
            continue
        path = media.path_for(mid)
        item = media.get(mid)
        if path is None or item is None:
            continue
        form = aiohttp.FormData()
        form.add_field("id", mid)
        form.add_field("file", path.read_bytes(), filename=item.get("name") or path.name)
        async with session.post(base + P + "/api/mirror/media", data=form) as r:
            if r.status == 200:
                Conn.mirrored.add(mid)


async def _proxy(req):
    base = f"http://127.0.0.1:{Conn.forward.local_port}"
    url = base + req.rel_url.raw_path_qs
    headers = {k: v for k, v in req.headers.items() if k.lower() not in _HOP}
    if req.headers.get("Upgrade", "").lower() == "websocket":
        return await _proxy_ws(req, url, headers)
    async with aiohttp.ClientSession(auto_decompress=False, timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)) as session:
        body = await req.read() if req.can_read_body else None
        if body and _is_text_call(req):
            try:
                await _mirror_media(body, session, base)
            except Exception as exc:        # noqa: BLE001 -- a failed copy must not hide the real answer
                return web.json_response({"why": f"The remote machine did not take a media file: {ssh.reason(exc)}"}, status=502)
        try:
            async with session.request(req.method, url, headers=headers, data=body, allow_redirects=False) as r:
                out = web.StreamResponse(status=r.status, headers={k: v for k, v in r.headers.items() if k.lower() not in _HOP})
                await out.prepare(req)
                async for chunk in r.content.iter_any():
                    await out.write(chunk)
                await out.write_eof()
                return out
        except (aiohttp.ClientError, OSError) as exc:
            if not ssh.alive(Conn.client):
                drop("The SSH connection was lost. FunUI is back on this machine.")
            return web.json_response({"why": f"The remote machine did not answer: {ssh.reason(exc)}"}, status=502)


async def _proxy_ws(req, url: str, headers: dict):
    ws_url = url.replace("http://", "ws://", 1)
    mine = web.WebSocketResponse(max_msg_size=0)
    await mine.prepare(req)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(ws_url, max_msg_size=0, headers={k: v for k, v in headers.items()
                                                                           if k.lower() in ("cookie", "authorization", "user-agent")}) as theirs:
                async def up():
                    async for m in mine:
                        if m.type == aiohttp.WSMsgType.TEXT:
                            await theirs.send_str(m.data)
                        elif m.type == aiohttp.WSMsgType.BINARY:
                            await theirs.send_bytes(m.data)
                        else:
                            break

                async def down():
                    async for m in theirs:
                        if m.type == aiohttp.WSMsgType.TEXT:
                            await mine.send_str(m.data)
                        elif m.type == aiohttp.WSMsgType.BINARY:
                            await mine.send_bytes(m.data)
                        else:
                            break

                tasks = [asyncio.ensure_future(up()), asyncio.ensure_future(down())]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in tasks:
                    t.cancel()
        except (aiohttp.ClientError, OSError):
            pass
    await mine.close()
    return mine


if web is not None:
    @web.middleware
    async def middleware(request, handler):
        if Conn.state != "on" or is_local(request.path) or not active():
            return await handler(request)
        return await _proxy(request)


def install(app) -> None:
    """Put the forwarding in front of the server's own routes."""
    if web is None or middleware in app.middlewares:
        return
    app.middlewares.insert(0, middleware)


# ---------------------------------------------------------------- the page's routes

def register(routes, prefix=None):
    p = (P if prefix is None else prefix) + "/api/remote"

    @routes.get(p)
    async def _state(_req):
        return web.json_response(snapshot())

    @routes.post(p + "/connect")
    async def _connect(req):
        try:
            b = await req.json()
        except ValueError:
            b = {}
        if not isinstance(b, dict) or not str(b.get("host") or "").strip():
            return web.json_response({"why": "Fill in the host."}, status=400)
        try:
            port = int(b.get("port") or 22)
        except (TypeError, ValueError):
            return web.json_response({"why": "The port is a number."}, status=400)
        login = ssh.Login(host=str(b["host"]), port=port, user=str(b.get("user") or "root"),
                          key=str(b.get("key") or ""), password=str(b.get("password") or ""))
        asyncio.ensure_future(connect(login))                 # the page follows it by asking GET /api/remote
        await asyncio.sleep(0)
        return web.json_response(snapshot())

    @routes.post(p + "/disconnect")
    async def _disconnect(_req):
        drop()
        return web.json_response(snapshot())

    return routes
