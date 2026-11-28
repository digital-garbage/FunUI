"""Remote Instance, end to end in one process: a real SSH server (paramiko, password login, port forwarding, a few commands),
a fake remote ComfyUI behind it, and a local server carrying FunUI's forwarding middleware.

The loop is driven by hand (asyncio.run), like test_routes.py: no pytest plugin needed.
"""
import asyncio
import socket
import threading
import time

import pytest

paramiko = pytest.importorskip("paramiko")
pytest.importorskip("aiohttp")
import aiohttp                                  # noqa: E402
from aiohttp import web                         # noqa: E402
from aiohttp.test_utils import TestServer       # noqa: E402

from core import config, media, remote, remote_ssh as ssh  # noqa: E402

P = config.UI_PREFIX


# ---------------------------------------------------------------- a small SSH server

class SshServer(paramiko.ServerInterface):
    """Password login for root/pw, forwarding to anything, and `answer(command)` for exec requests."""

    def __init__(self, answer):
        self.answer = answer
        self.commands = []
        self.pending = {}
        self.channels = []
        self.key = paramiko.RSAKey.generate(2048)
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.transports = []
        self.closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self.closed:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            t = paramiko.Transport(conn)
            t.add_server_key(self.key)
            self.transports.append(t)
            try:
                t.start_server(server=self)
            except Exception:                   # noqa: BLE001
                continue
            threading.Thread(target=self._channels, args=(t,), daemon=True).start()

    def _channels(self, t):
        while t.is_active():
            chan = t.accept(0.5)
            if chan is None:
                continue
            self.channels.append(chan)                  # kept: a channel nobody holds is closed by the garbage collector
            dest = self.pending.pop(chan.chanid, None)
            if dest:
                threading.Thread(target=self._forward, args=(chan, dest), daemon=True).start()

    def _forward(self, chan, dest):
        try:
            s = socket.create_connection(dest, timeout=5)
        except OSError:
            chan.close()
            return
        threading.Thread(target=ssh._pump, args=(s, chan), daemon=True).start()
        ssh._pump(chan, s)

    def stop(self):
        self.closed = True
        self.sock.close()
        for t in self.transports:
            t.close()

    # paramiko hooks
    def get_allowed_auths(self, username):
        return "password"

    def check_auth_password(self, username, password):
        return paramiko.AUTH_SUCCESSFUL if (username, password) == ("root", "pw") else paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED

    def check_channel_direct_tcpip_request(self, chanid, origin, destination):
        self.pending[chanid] = destination
        return paramiko.OPEN_SUCCEEDED

    def check_channel_exec_request(self, channel, command):
        cmd = command.decode() if isinstance(command, bytes) else command
        self.commands.append(cmd)

        def go():
            time.sleep(0.05)                    # let paramiko answer the request before the channel is used and closed
            out = self.answer(cmd)
            code, text = out if isinstance(out, tuple) else (0, out)
            channel.sendall(text if isinstance(text, bytes) else text.encode())
            channel.send_exit_status(code)
            channel.close()
        threading.Thread(target=go, daemon=True).start()
        return True


# ---------------------------------------------------------------- the fake remote ComfyUI + FunUI

class FakeRemote:
    def __init__(self, version="9.9.9", commit="abc"):
        self.version, self.commit = version, commit
        self.got = []
        self.media = {}

    def app(self):
        r = web.RouteTableDef()

        @r.get("/system_stats")
        async def stats(_):
            return web.json_response({"system": {"comfyui_version": "1", "argv": ["/srv/comfy/main.py"]}, "devices": []})

        @r.get(P + "/api/health")
        async def health(_):
            body = {"ok": True}
            if self.version:
                body.update(version=self.version, commit=self.commit)
            return web.json_response(body)

        @r.get(P + "/api/modules")
        async def modules(_):
            return web.json_response({"where": "remote"})

        @r.get(P + "/api/projects")
        async def projects(_):
            return web.json_response({"where": "remote"})

        @r.get(P + "/api/media")
        async def listing(_):
            return web.json_response({"media": [{"id": k} for k in self.media]})

        @r.post(P + "/api/mirror/media")
        async def mirror(req):
            rd = await req.multipart()
            mid = None
            while (part := await rd.next()) is not None:
                if part.name == "id":
                    mid = (await part.text()).strip()
                else:
                    self.media[mid] = await part.read()
            return web.json_response({"held": True})

        @r.post("/prompt")
        async def prompt(req):
            self.got.append(await req.text())
            return web.json_response({"prompt_id": "p1"})

        @r.get("/ws")
        async def ws(req):
            w = web.WebSocketResponse()
            await w.prepare(req)
            async for m in w:
                await w.send_str("echo:" + m.data)
            return w

        a = web.Application()
        a.add_routes(r)
        return a


def local_app():
    r = web.RouteTableDef()

    @r.get(P + "/api/modules")
    async def modules(_):
        return web.json_response({"where": "local"})

    @r.get(P + "/api/projects")
    async def projects(_):
        return web.json_response({"where": "local"})

    @r.post("/prompt")
    async def prompt(_):
        return web.json_response({"where": "local"})

    remote.register(r, P)
    a = web.Application()
    a.add_routes(r)
    remote.install(a)
    return a


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CONFIG_FILE", tmp_path / "remote.json")
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    monkeypatch.setattr(remote, "identity", lambda: {"version": "9.9.9", "commit": "abc"})
    remote.drop()
    yield
    remote.drop()


def run(coro):
    return asyncio.run(coro)


async def get_json(url, **kw):
    async with aiohttp.ClientSession() as s:
        async with s.request(kw.pop("method", "GET"), url, **kw) as r:
            return r.status, await r.json()


# ---------------------------------------------------------------- tests

def test_sentences_for_the_common_failures():
    assert "refused the login" in ssh.reason(paramiko.AuthenticationException("x"))
    assert "key file was not found" in ssh.reason(FileNotFoundError())
    assert "does not resolve" in ssh.reason(socket.gaierror())
    assert "Nothing answers" in ssh.reason(ConnectionRefusedError())


def test_local_paths():
    for ok in (P, P + "/", P + "/app/x.js", P + "/api/projects", P + "/api/projects/ab", P + "/api/media/x/file",
               P + "/api/shortcuts/export", P + "/api/prompt/expand", P + "/api/remote/connect", P + "/api/story/split", P + "/api/files"):
        assert remote.is_local(ok), ok
    for other in (P + "/api/modules", P + "/api/pipeline", P + "/api/nodes/search", "/prompt", "/ws", "/view", "/object_info", P + "/api/git/status"):
        assert not remote.is_local(other), other


def test_connect_forwards_remote_calls_and_keeps_local_ones_here():
    fake = FakeRemote()

    async def go():
        far = TestServer(fake.app())
        await far.start_server()
        sshd = SshServer(lambda cmd: f"{far.port}\n" if "system_stats" in cmd else "")
        near = TestServer(local_app())
        await near.start_server()
        base = f"http://127.0.0.1:{near.port}"
        try:
            # before connecting: everything is answered here
            assert (await get_json(base + P + "/api/modules"))[1] == {"where": "local"}
            await remote.connect(ssh.Login("127.0.0.1", sshd.port, "root", password="pw"))
            _s, snap = await get_json(base + P + "/api/remote")
            assert snap["state"] == "on", snap
            assert (await get_json(base + P + "/api/modules"))[1] == {"where": "remote"}      # the remote's modules
            assert (await get_json(base + P + "/api/projects"))[1] == {"where": "local"}      # projects stay here
            # a call over the tunnel to ComfyUI's own endpoint
            assert (await get_json(base + "/prompt", method="POST", data=b"{}"))[1] == {"prompt_id": "p1"}
            # the page's live socket
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(base.replace("http", "ws") + "/ws") as w:
                    await w.send_str("hi")
                    assert (await w.receive()).data == "echo:hi"
            # disconnect: back to local
            await get_json(base + P + "/api/remote/disconnect", method="POST")
            assert (await get_json(base + P + "/api/modules"))[1] == {"where": "local"}
            assert (await get_json(base + "/prompt", method="POST", data=b"{}"))[1] == {"where": "local"}
        finally:
            sshd.stop()
            await near.close()
            await far.close()

    run(go())


def test_connection_lost_falls_back_to_local():
    fake = FakeRemote()

    async def go():
        far = TestServer(fake.app())
        await far.start_server()
        sshd = SshServer(lambda cmd: f"{far.port}\n" if "system_stats" in cmd else "")
        near = TestServer(local_app())
        await near.start_server()
        base = f"http://127.0.0.1:{near.port}"
        try:
            await remote.connect(ssh.Login("127.0.0.1", sshd.port, "root", password="pw"))
            assert (await get_json(base + P + "/api/modules"))[1] == {"where": "remote"}
            sshd.stop()
            deadline = time.time() + 10
            while remote.Conn.client.get_transport().is_active() and time.time() < deadline:
                await asyncio.sleep(0.1)
            assert (await get_json(base + P + "/api/modules"))[1] == {"where": "local"}
            assert "lost" in (await get_json(base + P + "/api/remote"))[1]["message"]
        finally:
            await near.close()
            await far.close()

    run(go())


def test_wrong_password_and_no_comfy_say_so():
    async def go():
        sshd = SshServer(lambda cmd: "")
        try:
            await remote.connect(ssh.Login("127.0.0.1", sshd.port, "root", password="nope"))
            assert remote.Conn.state == "failed" and "refused the login" in remote.Conn.message
            await remote.connect(ssh.Login("127.0.0.1", sshd.port, "root", password="pw"))
            assert remote.Conn.state == "failed" and "no ComfyUI answers" in remote.Conn.message
        finally:
            sshd.stop()

    run(go())


def test_media_a_prompt_names_is_copied_across_once(tmp_path):
    fake = FakeRemote()
    saved = media.save_upload("a.png", b"\x89PNG\r\n\x1a\n" + b"x" * 40)

    async def go():
        far = TestServer(fake.app())
        await far.start_server()
        sshd = SshServer(lambda cmd: f"{far.port}\n" if "system_stats" in cmd else "")
        near = TestServer(local_app())
        await near.start_server()
        base = f"http://127.0.0.1:{near.port}"
        try:
            await remote.connect(ssh.Login("127.0.0.1", sshd.port, "root", password="pw"))
            body = ('{"prompt": {"1": {"inputs": {"image": "%s"}}}}' % saved["id"]).encode()
            await get_json(base + "/prompt", method="POST", data=body)
            assert saved["id"] in fake.media and fake.media[saved["id"]].startswith(b"\x89PNG")
            fake.media.clear()
            await get_json(base + "/prompt", method="POST", data=body)
            assert fake.media == {}                                  # already there, not sent twice
        finally:
            sshd.stop()
            await near.close()
            await far.close()

    run(go())


def test_older_remote_is_updated_restarted_then_connected():
    """No FunUI version there: the remote is brought to this branch, ComfyUI there is restarted, then we connect."""
    fake = FakeRemote(version=None)
    state = {"killed": False}

    def answer(cmd):
        if "system_stats" in cmd and "for p in" in cmd:
            return f"{state['port']}\n"
        if cmd.startswith("ss -ltnp"):
            return "4242\n"
        if "readlink /proc/4242/exe" in cmd:
            return "/venv/bin/python\n/srv/comfy\n"
        if cmd.startswith("cat /proc/4242/cmdline"):
            return b"/venv/bin/python\0main.py\0--listen\0"
        if cmd.startswith("cat /proc/4242/environ"):
            return b"PATH=/usr/bin\0HOME=/root\0"
        if "pyproject.toml" in cmd and cmd.startswith("grep"):
            return ""                                               # FunUI is not there
        if cmd.startswith("git clone"):
            return ""
        if "pip install" in cmd:
            return ""
        if cmd.startswith("kill -0"):
            return "down\n" if state["killed"] else "up\n"
        if cmd.startswith("kill "):
            state["killed"] = True
            return ""
        if "setsid nohup" in cmd:
            fake.version, fake.commit = "9.9.9", "abc"             # ComfyUI came back with the new FunUI
            return ""
        return ""

    async def go():
        far = TestServer(fake.app())
        await far.start_server()
        state["port"] = far.port
        sshd = SshServer(answer)
        orig = remote._git_here
        remote._git_here = lambda *a: "dev" if "--abbrev-ref" in a else "https://example.invalid/FunUI"
        try:
            await remote.connect(ssh.Login("127.0.0.1", sshd.port, "root", password="pw"))
            assert remote.Conn.state == "on", remote.Conn.message
            joined = "\n".join(sshd.commands)
            assert "git clone -q --depth 50 -b dev https://example.invalid/FunUI /srv/comfy/custom_nodes/FunUI" in joined
            assert "/venv/bin/python -m pip install" in joined
            assert "cd /srv/comfy" in joined and "setsid nohup /venv/bin/python main.py --listen" in joined
            assert "Downloading FunUI onto the remote machine" in remote.Conn.steps
            assert "Restarting ComfyUI on the remote machine" in remote.Conn.steps
        finally:
            remote._git_here = orig
            sshd.stop()
            await far.close()

    run(go())


def test_cancel_while_connecting_leaves_nothing_stuck_and_a_late_finish_changes_nothing():
    release = threading.Event()

    def slow(cmd):
        release.wait(10)                                    # the scan takes "forever"
        return "1\n"

    async def go():
        sshd = SshServer(slow)
        try:
            task = asyncio.ensure_future(remote.connect(ssh.Login("127.0.0.1", sshd.port, "root", password="pw")))
            deadline = time.time() + 5
            while remote.Conn.state != "working" or "Looking for ComfyUI on the remote machine" not in remote.Conn.steps:
                assert time.time() < deadline
                await asyncio.sleep(0.05)
            remote.drop()                                   # Cancel
            assert remote.Conn.state == "off"
            release.set()
            await asyncio.wait_for(task, 20)
            assert remote.Conn.state == "off" and remote.Conn.client is None      # the late result was ignored
        finally:
            release.set()
            sshd.stop()

    run(go())


def test_a_link_that_stops_answering_is_dropped(monkeypatch):
    fake = FakeRemote()
    monkeypatch.setattr(remote, "WATCH_EVERY", 0.2)
    monkeypatch.setattr(remote, "WATCH_MISSES", 2)

    async def go():
        far = TestServer(fake.app())
        await far.start_server()
        sshd = SshServer(lambda cmd: f"{far.port}\n" if "system_stats" in cmd else "")
        near = TestServer(local_app())
        await near.start_server()
        base = f"http://127.0.0.1:{near.port}"
        try:
            await remote.connect(ssh.Login("127.0.0.1", sshd.port, "root", password="pw"))
            assert remote.Conn.state == "on"
            monkeypatch.setattr(ssh, "ping", lambda client, timeout=8: False)       # silent: the socket is open, nobody answers
            deadline = time.time() + 5
            while remote.Conn.state == "on" and time.time() < deadline:
                await asyncio.sleep(0.1)
            snap = (await get_json(base + P + "/api/remote"))[1]
            assert snap["state"] == "off" and "stopped answering" in snap["message"]
            assert (await get_json(base + P + "/api/modules"))[1] == {"where": "local"}
        finally:
            sshd.stop()
            await near.close()
            await far.close()

    run(go())


def test_remote_comfyui_down_or_hung_gives_a_sentence_not_a_hang():
    """The tunnel is fine but nothing listens behind it: the call answers 502 with why."""
    fake = FakeRemote()

    async def go():
        far = TestServer(fake.app())
        await far.start_server()
        sshd = SshServer(lambda cmd: f"{far.port}\n" if "system_stats" in cmd else "")
        near = TestServer(local_app())
        await near.start_server()
        base = f"http://127.0.0.1:{near.port}"
        try:
            await remote.connect(ssh.Login("127.0.0.1", sshd.port, "root", password="pw"))
            dead = socket.socket()
            dead.bind(("127.0.0.1", 0))
            dead_port = dead.getsockname()[1]
            dead.close()                                    # a port nothing listens on
            remote.Conn.forward = ssh.Forward(remote.Conn.client.get_transport(), "127.0.0.1", dead_port)   # ComfyUI there went away; SSH stays up
            status, body = await get_json(base + P + "/api/modules")
            assert status == 502 and body["why"].startswith("The remote machine did not answer")
            assert remote.Conn.state == "on"                # SSH itself is fine: the page can retry or disconnect
        finally:
            sshd.stop()
            await near.close()
            await far.close()

    run(go())
