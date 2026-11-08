#!/usr/bin/env python3
"""Memory guard: run a command, kill its whole process tree before the Mac runs out of memory.

    python3 tools/memguard.py [--floor-mb 512] [--poll-ms 100] -- <command...>
    python3 tools/memguard.py --selftest

Available memory = free + inactive + speculative pages (what macOS can hand out without swapping
live data). When it drops below the floor the tree is SIGKILLed and the report names the command,
the pytest test (via tools/memguard_pytest.py) or the last output lines (npm / node).
Exit code 137 when the guard killed the command, else the command's own code.
"""
import argparse, collections, ctypes, os, signal, struct, subprocess, sys, threading, time

_lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_lib.mach_host_self.restype = ctypes.c_uint
_HOST = _lib.mach_host_self()
_PAGE = os.sysconf("SC_PAGE_SIZE")


def available_mb():
    buf = (ctypes.c_uint * 38)()
    n = ctypes.c_uint(38)
    if _lib.host_statistics64(_HOST, 4, buf, ctypes.byref(n)):  # HOST_VM_INFO64
        raise OSError("host_statistics64 failed")
    raw = bytes(buf)
    free, _active, inactive = struct.unpack_from("3I", raw, 0)
    spec = struct.unpack_from("I", raw, 92)[0]  # speculative_count
    return (free + inactive + spec) * _PAGE / 2**20


def tree(root):
    out = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True).stdout
    kids = collections.defaultdict(list)
    for line in out.splitlines():
        p, pp = map(int, line.split())
        kids[pp].append(p)
    seen, todo = [], [root]
    while todo:
        p = todo.pop()
        seen.append(p)
        todo += kids[p]
    return seen


def kill_tree(proc):
    pids = tree(proc.pid)  # before the group dies, so reparented children are still found
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(cmd, floor_mb=512, poll_ms=100):
    state = f"/tmp/memguard-{os.getpid()}.state"
    env = dict(os.environ, MEMGUARD_STATE=state, PYTHONUNBUFFERED="1")
    env["PYTEST_PLUGINS"] = ",".join(filter(None, [env.get("PYTEST_PLUGINS"), "memguard_pytest"]))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.dirname(os.path.abspath(__file__)), env.get("PYTHONPATH")]))
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    tail = collections.deque(maxlen=8)

    def pump():
        for line in proc.stdout:
            sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
            tail.append(line.decode(errors="replace").rstrip())

    t = threading.Thread(target=pump, daemon=True); t.start()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: kill_tree(proc))
    killed = None
    while proc.poll() is None:
        a = available_mb()
        if a < floor_mb:
            kill_tree(proc)
            killed = a
            break
        time.sleep(poll_ms / 1000)
    proc.wait(); t.join(2)
    test = ""
    try:
        test = open(state).read().strip()
        os.unlink(state)
    except OSError:
        pass
    if killed is None:
        return proc.returncode, None
    where = f"pytest test: {test}" if test else "last output:\n    " + "\n    ".join(tail)
    print(f"\nMEMGUARD KILLED the process tree: available {killed:.0f} MB < floor {floor_mb} MB (now {available_mb():.0f} MB)\n"
          f"  command: {' '.join(cmd)}\n  {where}", file=sys.stderr)
    return 137, killed


def selftest():
    """Allocate in 64 MB steps until the guard fires at a floor 700 MB under the current free memory."""
    start = available_mb()
    floor = start - 700
    if floor < 600:
        sys.exit(f"selftest skipped: only {start:.0f} MB available, not safe to test")
    code = ("import time\nb=[]\nwhile True:\n b.append(bytearray(64*2**20)); b[-1][::4096]=b'x'*(64*2**20//4096); time.sleep(.025)")
    t0 = time.time()
    rc, hit = run([sys.executable, "-c", code], floor_mb=floor)
    end = available_mb()
    print(f"selftest: start {start:.0f} MB, floor {floor:.0f} MB, killed at {hit and round(hit)} MB after {time.time()-t0:.1f}s, "
          f"rc={rc}, lowest margin to real 512 MB floor: {(hit or 0)-512:.0f} MB, now {end:.0f} MB")
    ok = rc == 137 and hit > 512 and hit > floor - 400
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("--floor-mb", type=int, default=512)
    ap.add_argument("--poll-ms", type=int, default=100)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    sys.exit(run(cmd, a.floor_mb, a.poll_ms)[0])
