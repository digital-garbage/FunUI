"""Background jobs (final render, combined export): ffmpeg can outlast a tunnel's request
timeout, so the request returns an id and the page polls for it."""

import asyncio
import uuid
import weakref
from collections import OrderedDict

MAX_JOBS = 50                         # finished jobs are kept this long for the poll that reads them

_waiting: set = set()                # ids queued behind another job
_jobs: "OrderedDict[str, dict]" = OrderedDict()
_stop: set = set()                    # ids asked to stop; the job notices between chunks
_can_stop: set = set()                # ids whose work checks for it


class Stopped(Exception):
    """Raised inside a job that was asked to stop; its partial output is removed by whoever was writing it."""


def stop(job: str) -> bool:
    """Ask a queued or running job to stop. False when there is no such job or it already ended."""
    entry = _jobs.get(job)
    if job not in _can_stop or not entry or entry.get("state") not in ("queued", "running"):
        return False
    _stop.add(job)
    if entry.get("state") == "queued" and job in _waiting:
        _jobs[job] = {"state": "stopped"}                # still waiting for its turn: it ends now, not when the one ahead does
    return True


def check(job: str) -> None:
    """Raises Stopped once `stop(job)` was called: the hook a long job calls between its chunks."""
    if job in _stop:
        raise Stopped()


def new(stoppable: bool = False) -> str:
    """`stoppable`: the work calls its `interrupted()` hook, so a stop request will end it."""
    job = uuid.uuid4().hex
    _jobs[job] = {"state": "queued"}
    if stoppable:
        _can_stop.add(job)
    for old in [k for k, v in _jobs.items() if v.get("state") in ("done", "error", "stopped")]:
        if len(_jobs) <= MAX_JOBS:
            break
        del _jobs[old]                                 # only finished ones: a running job's poll must keep finding it
    return job


def get(job: str) -> dict | None:
    return _jobs.get(job)


_turns: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()   # one lock per event loop


async def run(job: str, fn, *args, error, stoppable: bool = False, exclusive: bool = False) -> None:
    """Run the blocking `fn(*args)` off the event loop and record how it ended. `error` is the
    exception class whose message is meant for the person; anything else is reported plainly.
    `exclusive`: the work uses a GPU effect that keeps one buffer per kind (Upscale, Interpolate), so such jobs
    run one after the other and a second waits as "queued". Other jobs (the render) never wait for them."""
    turn = _turns.setdefault(asyncio.get_running_loop(), asyncio.Lock()) if exclusive else None
    if turn is not None and turn.locked():
        _waiting.add(job)
    try:
        if turn is not None:
            await turn.acquire()
    finally:
        _waiting.discard(job)
    try:
        if job in _stop:                                    # stopped while it was still queued
            _stop.discard(job)
            _can_stop.discard(job)
            _jobs[job] = {"state": "stopped"}
            return
        _jobs[job] = {"state": "running"}
        try:
            result = await asyncio.to_thread(fn, *args, **({"interrupted": lambda: check(job)} if stoppable else {}))
            _jobs[job] = {"state": "done", **result}
        except Stopped:
            _jobs[job] = {"state": "stopped"}
        except error as exc:
            _jobs[job] = {"state": "error", "detail": str(exc)}
        except Exception as exc:                          # noqa: BLE001
            _jobs[job] = {"state": "error", "detail": f"{type(exc).__name__}: {exc}"}
        finally:
            _stop.discard(job)
            _can_stop.discard(job)
    finally:
        if turn is not None:
            turn.release()
