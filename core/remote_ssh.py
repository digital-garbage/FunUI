"""The SSH side of Remote Instance: log in, open a tunnel, find ComfyUI and FunUI on the remote machine, bring FunUI there
to the same version as this one, restart ComfyUI there when that changed it.

Everything here is plain functions over a connected `paramiko` client, so the steps can be tested against a small SSH server
in-process. Nothing here knows about aiohttp or the page.
"""
from __future__ import annotations

import json
import shlex
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:                                        # paramiko is in requirements.txt; a machine without it learns that when it connects
    import paramiko
except ImportError:                         # pragma: no cover
    paramiko = None

# Where ComfyUI usually listens, in the order worth asking.
DEFAULT_PORTS = (8188, 8189, 8190, 8187, 8000, 8080, 8888, 7860, 3000, 5000, *range(8180, 8200))


class RemoteError(RuntimeError):
    """A failure whose message is a sentence a person can act on."""


@dataclass
class Login:
    host: str
    port: int = 22
    user: str = "root"
    key: str = ""
    password: str = ""


def reason(exc: BaseException) -> str:
    """What went wrong, as a sentence."""
    if isinstance(exc, RemoteError):
        return str(exc)
    if paramiko is not None:
        if isinstance(exc, paramiko.BadAuthenticationType) or isinstance(exc, paramiko.AuthenticationException):
            return "The server refused the login. Check the user name and the key or password."
        if isinstance(exc, paramiko.PasswordRequiredException):
            return "That key file has a passphrase. Put the passphrase in the password field."
        if isinstance(exc, paramiko.ssh_exception.NoValidConnectionsError):
            return "Nothing answers on that address and port. Is the machine on, and is SSH on that port?"
        if isinstance(exc, paramiko.SSHException) and "not a valid" in str(exc).lower():
            return "The key file could not be read as a private key."
    if isinstance(exc, FileNotFoundError):
        return "The key file was not found."
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "Could not reach that address."
    if isinstance(exc, socket.gaierror):
        return "That address does not resolve."
    if isinstance(exc, ConnectionRefusedError):
        return "Nothing answers on that address and port. Is the machine on, and is SSH on that port?"
    text = str(exc).strip()
    return text.splitlines()[-1] if text else "The SSH connection failed."


def connect(login: Login):
    if paramiko is None:
        raise RemoteError("The SSH library is missing. Install it with: pip install paramiko")
    if not login.host.strip():
        raise RemoteError("Fill in the host.")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())        # first contact is trusted, like ssh's accept-new
    explicit = bool(login.key or login.password)
    try:
        client.connect(
            login.host.strip(), port=int(login.port or 22), username=login.user.strip() or "root",
            password=login.password or None, key_filename=str(Path(login.key).expanduser()) if login.key else None,
            passphrase=login.password or None if login.key else None,
            look_for_keys=not explicit, allow_agent=not explicit, timeout=12, banner_timeout=12, auth_timeout=12,
        )
    except Exception as exc:                # noqa: BLE001 -- turned into a sentence
        client.close()
        raise RemoteError(reason(exc)) from exc
    client.get_transport().set_keepalive(15)
    return client


def alive(client) -> bool:
    t = client.get_transport() if client else None
    return bool(t and t.is_active())


def ping(client, timeout: float = 8) -> bool:
    """Does the remote machine still answer? A link can die without anyone being told."""
    try:
        code, _o, _e = run(client, "true", timeout=timeout)
        return code == 0
    except Exception:                       # noqa: BLE001
        return False


def run(client, command: str, timeout: float = 60, raw: bool = False):
    """(exit code, stdout, stderr). stdout is text unless raw."""
    _in, out, err = client.exec_command(command, timeout=timeout)
    data = out.read()
    code = out.channel.recv_exit_status()
    e = err.read().decode("utf-8", "replace")
    return code, (data if raw else data.decode("utf-8", "replace")), e


class Forward:
    """A listener on 127.0.0.1 whose every connection is carried to `host:port` on the remote machine."""

    def __init__(self, transport, host: str, port: int):
        self.transport, self.host, self.port = transport, host, port
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(32)
        self.local_port = self.sock.getsockname()[1]
        self._open = []
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._closed:
            try:
                conn, addr = self.sock.accept()
            except OSError:
                return
            try:
                chan = self.transport.open_channel("direct-tcpip", (self.host, self.port), addr)
            except Exception:               # noqa: BLE001 -- the far side refused: drop this one connection
                conn.close()
                continue
            self._open += [conn, chan]
            for a, b in ((conn, chan), (chan, conn)):
                threading.Thread(target=_pump, args=(a, b), daemon=True).start()

    def close(self):
        self._closed = True
        for s in [self.sock, *self._open]:
            shut(s)


def _pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:                       # noqa: BLE001 -- either end closing is the normal ending
        pass
    for s in (src, dst):
        shut(s)


def shut(s) -> None:
    """Close a socket or channel so the thread blocked reading it wakes and the far end is told: close() alone leaves a
    blocked recv() holding the connection open."""
    for how in ("shutdown", "close"):
        try:
            getattr(s, how)(socket.SHUT_RDWR) if how == "shutdown" else s.close()
        except Exception:                   # noqa: BLE001
            pass


def http_get_json(port: int, path: str, timeout: float = 6):
    """GET through a local tunnel port; the parsed JSON, or None for anything but a clean answer."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:                       # noqa: BLE001
        return None


# ---------------------------------------------------------------- finding ComfyUI

def scan_script(extra=()) -> str:
    """The shell loop run on the remote machine: prints each port where ComfyUI answers."""
    ports = " ".join(str(p) for p in [*extra, *DEFAULT_PORTS])
    return ("for p in %s; do if (command -v curl >/dev/null && curl -s -m 2 http://127.0.0.1:$p/system_stats "
            "|| wget -q -T 2 -O - http://127.0.0.1:$p/system_stats) 2>/dev/null | grep -q comfyui_version; then echo $p; fi; done" % ports)


def comfy_ports(client, extra=()) -> list[int]:
    _code, out, _err = run(client, scan_script(extra), timeout=90)
    return [int(x) for x in out.split() if x.isdigit()]


@dataclass
class Comfy:
    port: int
    pid: int | None = None
    folder: str = ""            # where main.py is
    cwd: str = ""
    exe: str = ""               # the python running it
    argv: tuple = ()
    env: tuple = ()


def describe_comfy(client, port: int, stats: dict | None) -> Comfy:
    """Folder, python, arguments and environment of the ComfyUI listening on `port`: all of it is needed to start it again."""
    found = Comfy(port=port)
    argv = ((stats or {}).get("system") or {}).get("argv") or []
    if argv:
        found.folder = str(Path(argv[0]).parent) if argv[0].endswith(".py") else ""
    _c, out, _e = run(client, f"ss -ltnp 'sport = :{port}' 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true")
    pid = out.strip()
    if not pid:
        _c, out, _e = run(client, f"lsof -nP -iTCP:{port} -sTCP:LISTEN -t 2>/dev/null | head -1 || true")
        pid = out.strip()
    if pid.isdigit():
        found.pid = int(pid)
        _c, out, _e = run(client, f"readlink /proc/{pid}/exe; readlink /proc/{pid}/cwd")
        lines = out.splitlines()
        found.exe = lines[0] if lines else ""
        found.cwd = lines[1] if len(lines) > 1 else ""
        _c, raw, _e = run(client, f"cat /proc/{pid}/cmdline", raw=True)
        found.argv = tuple(x.decode("utf-8", "replace") for x in raw.split(b"\0") if x)
        _c, raw, _e = run(client, f"cat /proc/{pid}/environ", raw=True)
        found.env = tuple(x.decode("utf-8", "replace") for x in raw.split(b"\0") if b"=" in x)
        if not found.folder and found.argv:
            main = next((a for a in found.argv if a.endswith("main.py")), "")
            if main:
                found.folder = str(Path(main if main.startswith("/") else f"{found.cwd}/{main}").parent)
    return found


# ---------------------------------------------------------------- FunUI on the remote machine

def find_funui(client, comfy_folder: str) -> str:
    """The folder of FunUI inside ComfyUI's custom_nodes on the remote machine, or ''."""
    cmd = f"grep -l 'name = \"ComfyUI-FunPack\"' {shlex.quote(comfy_folder)}/custom_nodes/*/pyproject.toml 2>/dev/null | head -1"
    _c, out, _e = run(client, cmd)
    return str(Path(out.strip()).parent) if out.strip() else ""


def bring_funui_here(client, comfy_folder: str, funui_dir: str, repo_url: str, branch: str, say) -> tuple[str, bool]:
    """Make FunUI on the remote machine a checkout of `branch` from `repo_url`: clone it when missing, switch and pull when not.
    Returns (folder, changed): changed is True when files were added or moved, so ComfyUI there has to start again."""
    q = shlex.quote
    if not funui_dir:
        say("Downloading FunUI onto the remote machine")
        target = f"{comfy_folder}/custom_nodes/FunUI"
        code, _o, err = run(client, f"git clone -q --depth 50 -b {q(branch)} {q(repo_url)} {q(target)}", timeout=600)
        if code:
            raise RemoteError("FunUI could not be downloaded on the remote machine: " + (err.strip().splitlines() or ["git failed"])[-1])
        funui_dir, changed = target, True
    else:
        say("Updating FunUI on the remote machine")
        _c, before, _e = run(client, f"git -C {q(funui_dir)} rev-parse HEAD")
        steps = (f"git -C {q(funui_dir)} fetch -q origin {q(branch)} && "
                 f"(git -C {q(funui_dir)} checkout -q {q(branch)} 2>/dev/null || git -C {q(funui_dir)} checkout -q -b {q(branch)} FETCH_HEAD) && "
                 f"git -C {q(funui_dir)} merge -q --ff-only FETCH_HEAD")
        code, _o, err = run(client, steps, timeout=300)
        if code:
            raise RemoteError("FunUI on the remote machine could not be updated: " + (err.strip().splitlines() or ["git failed"])[-1]
                              + " (local edits there? FunUI stays as it was)")
        _c, after, _e = run(client, f"git -C {q(funui_dir)} rev-parse HEAD")
        changed = before.strip() != after.strip()
    return funui_dir, changed


def install_requirements(client, funui_dir: str, python: str, say):
    say("Installing FunUI's requirements there")
    code, _o, err = run(client, f"{shlex.quote(python or 'python3')} -m pip install -q -r {shlex.quote(funui_dir)}/requirements.txt", timeout=900)
    if code:
        raise RemoteError("FunUI's requirements did not install on the remote machine: " + (err.strip().splitlines() or ["pip failed"])[-1])


def restart_comfy(client, comfy: Comfy, say, wait: float = 120):
    """Stop the ComfyUI process and start it again with the same arguments, folder and environment."""
    if not (comfy.pid and comfy.argv and comfy.cwd):
        raise RemoteError("FunUI is in place on the remote machine, but ComfyUI there could not be restarted from here. Restart it there, then connect again.")
    say("Restarting ComfyUI on the remote machine")
    q = shlex.quote
    env = " ".join(q(e) for e in comfy.env)
    cmd = " ".join(q(a) for a in comfy.argv)
    run(client, f"kill {comfy.pid}")
    end = time.time() + 40
    while time.time() < end:
        _c, out, _e = run(client, f"kill -0 {comfy.pid} 2>/dev/null && echo up || echo down")
        if out.strip() == "down":
            break
        time.sleep(1)
    else:
        run(client, f"kill -9 {comfy.pid}")
    run(client, f"cd {q(comfy.cwd)} && (env -i {env} setsid nohup {cmd} > funui_restart.log 2>&1 < /dev/null &)", timeout=20)
    end = time.time() + wait
    while time.time() < end:
        if comfy.port in comfy_ports(client, [comfy.port]):
            return
        time.sleep(3)
    raise RemoteError("ComfyUI did not come back on the remote machine in time. See funui_restart.log in its folder.")

