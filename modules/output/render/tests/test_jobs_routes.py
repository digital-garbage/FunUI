"""Render, export and import as the page calls them: start a job, poll it, read the file it made."""

import http.client
import json
import time

import re

import pytest

pytest.importorskip("aiohttp")

from core import media, projects  # noqa: E402
from core.tests.test_routes_shortcuts import server  # noqa: E402,F401
from core import jobs  # noqa: E402
from modules.output.render import files  # noqa: E402
from modules.output.render.tests.clips import comfy, make_clip, needs_ffmpeg, probe  # noqa: E402,F401

BASE = "/fun/api/m/render"


def call(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(data or b"{}")
    except ValueError:
        return resp.status, data


def finish(port, path, job_id, tries=200):
    for _ in range(tries):
        status, job = call(port, "GET", f"{path}/{job_id}")
        assert status == 200
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("the job never finished")


def clip(name="a.mp4", dur=1.0, **extra):
    return {"filename": name, "subfolder": "", "type": "output", "in": 0, "dur": dur, "fx": {}, "fps": 25,
            "w": 160, "h": 120, "transition": "", "tdur": 0, "volume": 1.0, **extra}


@needs_ffmpeg
def test_render_is_a_job_that_ends_with_a_playable_file(comfy, server):
    make_clip(comfy.output / "a.mp4", 1.0)
    proj = projects.save(projects.Project(width=160, height=120))
    status, queued = call(server, "POST", f"{BASE}/projects/{proj.id}/render", {"clips": [clip(), clip()]})
    assert status == 200
    job = finish(server, f"{BASE}/projects/{proj.id}/render", queued["job_id"])
    assert job["state"] == "done" and job["clips"] == 2
    assert probe(comfy.temp / job["media"]["filename"])[0] == pytest.approx(2.0, abs=0.2)


def test_a_render_that_cannot_be_made_ends_in_an_error_with_a_sentence(comfy, server):
    proj = projects.save(projects.Project())
    status, queued = call(server, "POST", f"{BASE}/projects/{proj.id}/render", {"clips": [clip("gone.mp4")]})
    job = finish(server, f"{BASE}/projects/{proj.id}/render", queued["job_id"])
    assert job["state"] == "error" and "not on disk" in job["detail"]


def test_nothing_to_render_is_refused_before_a_job_starts(comfy, server):
    proj = projects.save(projects.Project())
    status, body = call(server, "POST", f"{BASE}/projects/{proj.id}/render", {"clips": []})
    assert status == 400 and "Nothing to render" in body["detail"]
    assert call(server, "POST", f"{BASE}/projects/{proj.id}/render", {"clips": "x"})[0] == 400
    assert call(server, "POST", f"{BASE}/projects/{proj.id}/render", [])[0] == 400


def test_unknown_project_and_unknown_job_are_404(comfy, server):
    assert call(server, "POST", f"{BASE}/projects/aaaaaaaaaaaa/render", {"clips": [clip()]})[0] == 404
    proj = projects.save(projects.Project())
    assert call(server, "GET", f"{BASE}/projects/{proj.id}/render/nope")[0] == 404
    assert call(server, "GET", f"{BASE}/projects/{proj.id}/export-clips/nope")[0] == 404


@needs_ffmpeg
def test_export_one_clip_answers_directly_and_several_as_a_job(comfy, server):
    make_clip(comfy.output / "a.mp4", 2.0)
    proj = projects.save(projects.Project())
    status, out = call(server, "POST", f"{BASE}/projects/{proj.id}/export-clip",
                       {"clip": {"filename": "a.mp4", "in": 0.5, "dur": 1.0}})
    assert status == 200 and probe(comfy.temp / out["media"]["filename"])[0] == pytest.approx(1.0, abs=0.15)
    assert call(server, "POST", f"{BASE}/projects/{proj.id}/export-clip", {"clip": {}})[0] == 400
    status, queued = call(server, "POST", f"{BASE}/projects/{proj.id}/export-clips",
                          {"clips": [{"filename": "a.mp4", "in": 0, "dur": 1.0}, {"filename": "a.mp4", "in": 1, "dur": 1.0}]})
    job = finish(server, f"{BASE}/projects/{proj.id}/export-clips", queued["job_id"])
    assert job["state"] == "done" and job["clips"] == 2
    assert call(server, "POST", f"{BASE}/projects/{proj.id}/export-clips", {"clips": []})[0] == 400


@needs_ffmpeg
def test_a_clip_imports_into_the_media_library_trimmed_and_named(comfy, server):
    make_clip(comfy.output / "a.mp4", 3.0)
    status, out = call(server, "POST", f"{BASE}/import-clip",
                       {"clip": {"filename": "a.mp4", "in": 1.0, "dur": 1.0}, "name": "my clip"})
    assert status == 200 and out["media"]["name"] == "my clip.mp4" and out["media"]["kind"] == "video"
    assert probe(media.path_for(out["media"]["id"]))[0] == pytest.approx(1.0, abs=0.15)
    status, whole = call(server, "POST", f"{BASE}/import-clip", {"clip": {"filename": "a.mp4"}})
    assert status == 200 and media.path_for(whole["media"]["id"]).read_bytes() == (comfy.output / "a.mp4").read_bytes()
    assert re.fullmatch(r"[0-9a-f]{12}\.mp4", whole["media"]["name"]), "no name given: a hash like every saved render, not the project's name"
    again = call(server, "POST", f"{BASE}/import-clip", {"clip": {"filename": "a.mp4"}})[1]
    assert again["media"]["name"] == whole["media"]["name"]
    assert call(server, "POST", f"{BASE}/import-clip", {"clip": {"filename": "a.mp4", "in": 1.0, "dur": 1.0}})[1]["media"]["name"] != whole["media"]["name"]
    assert call(server, "POST", f"{BASE}/import-clip", {"clip": {}})[0] == 400
    status, body = call(server, "POST", f"{BASE}/import-clip", {"clip": {"filename": "gone.mp4"}})
    assert status == 400 and "not on disk" in body["detail"]


def test_the_effects_and_seams_the_timeline_offers(server):
    status, lib = call(server, "GET", f"{BASE}/library")
    assert status == 200
    assert {"reverse", "crop", "flip_h", "fill_frame", "reset"} <= {e["id"] for e in lib["effects"]}
    assert [t["id"] for t in lib["video_transitions"]] == ["crossfade", "fadeblack", "wipeleft", "wiperight"]


def test_finished_jobs_are_forgotten_past_the_cap(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_JOBS", 3)
    ids = []
    for _ in range(5):
        ids.append(jobs.new())
        jobs._jobs[ids[-1]] = {"state": "done"}
    ids.append(jobs.new())
    assert jobs.get(ids[0]) is None and jobs.get(ids[5]) is not None


def test_a_running_job_is_never_forgotten_to_make_room(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_JOBS", 3)
    first = jobs.new()
    jobs._jobs[first] = {"state": "running"}
    for _ in range(6):
        jobs._jobs[jobs.new()] = {"state": "done"}
    assert jobs.get(first) == {"state": "running"}


@needs_ffmpeg
def test_render_keeps_the_real_frame_rate_of_interpolated_clips_only_when_asked(comfy, server):
    make_clip(comfy.output / "fast.mp4", 1.0, rate=48)
    proj = projects.save(projects.Project(width=160, height=120))
    for body, expected in (({}, "24/1"), ({"real_fps": True}, "48/1")):     # the setting says 24, the file runs at 48
        status, queued = call(server, "POST", f"{BASE}/projects/{proj.id}/render", {"clips": [clip("fast.mp4", fps=24)], **body})
        assert status == 200
        job = finish(server, f"{BASE}/projects/{proj.id}/render", queued["job_id"])
        assert job["state"] == "done"
        assert files.frame_rate(str(comfy.temp / job["media"]["filename"])) == expected


def test_a_running_job_asked_to_stop_ends_stopped_and_stays_pollable(monkeypatch):
    import asyncio
    import threading
    monkeypatch.setattr(jobs, "MAX_JOBS", 3)
    started, seen = threading.Event(), []

    def long_job(interrupted):
        started.set()
        while True:
            interrupted()                                   # between chunks, as stream_file does
            seen.append(1)
            threading.Event().wait(0.01)

    async def go():
        job = jobs.new(stoppable=True)
        task = asyncio.create_task(jobs.run(job, long_job, error=RuntimeError, stoppable=True))
        while not started.is_set():
            await asyncio.sleep(0.01)
        assert jobs.stop(job) is True
        await task
        return job

    job = asyncio.run(go())
    assert jobs.get(job) == {"state": "stopped"}              # a poll finds it: no 404, so no "ComfyUI restarted"
    assert jobs.stop(job) is False and jobs.stop("nope") is False


def test_a_job_stopped_before_it_started_never_runs():
    import asyncio
    ran = []
    job = jobs.new(stoppable=True)
    assert jobs.stop(job) is True
    asyncio.run(jobs.run(job, lambda interrupted: ran.append(1), error=RuntimeError, stoppable=True))
    assert not ran and jobs.get(job) == {"state": "stopped"}


@needs_ffmpeg
def test_a_stopped_stream_leaves_no_half_written_file(comfy):
    from core import video_pipe
    src = comfy.output / "a.mp4"
    make_clip(src, seconds=1)
    dst = comfy.output / "out.mp4"
    calls = []

    def stop_after_first_chunk():
        calls.append(1)
        if len(calls) > 1:
            raise jobs.Stopped()
    with pytest.raises(jobs.Stopped):
        video_pipe.stream_file(str(src), str(dst), lambda x: x, stop_after_first_chunk)
    assert not dst.exists()


def test_the_stop_route_answers_whether_it_stopped_anything(comfy, server):
    job = jobs.new(stoppable=True)
    status, body = call(server, "POST", f"{BASE}/jobs/{job}/stop")
    assert status == 200 and body == {"stopped": True}
    assert call(server, "POST", f"{BASE}/jobs/nope/stop")[1] == {"stopped": False}


def test_a_job_whose_work_never_checks_cannot_be_stopped():
    job = jobs.new()                                              # a final render or an export: it runs to its end
    assert jobs.stop(job) is False


@needs_ffmpeg
def test_a_stop_that_lands_after_the_last_chunk_still_removes_the_file(comfy):
    from core import video_pipe
    src = comfy.output / "a.mp4"
    make_clip(src, seconds=1)
    counted = []
    video_pipe.stream_file(str(src), str(comfy.output / "count.mp4"), lambda x: x, lambda: counted.append(1))
    dst = comfy.output / "out.mp4"
    calls = []

    def stop_at_the_last_check():
        calls.append(1)
        if len(calls) >= len(counted) - 1:                 # the checks after the loop: before closing, and after the encoder flushed
            raise jobs.Stopped()
    with pytest.raises(jobs.Stopped):
        video_pipe.stream_file(str(src), str(dst), lambda x: x, stop_at_the_last_check)
    assert not dst.exists()


def test_exclusive_jobs_run_one_after_the_other_and_the_second_waits_as_queued():
    import asyncio
    import threading
    gate, order = threading.Event(), []

    def first():
        order.append("first start")
        gate.wait(5)
        order.append("first end")
        return {}

    def second():
        order.append("second start")
        return {}

    async def go():
        a, b = jobs.new(), jobs.new()
        ta = asyncio.create_task(jobs.run(a, first, error=RuntimeError, exclusive=True))
        while "first start" not in order:
            await asyncio.sleep(0.01)
        tb = asyncio.create_task(jobs.run(b, second, error=RuntimeError, exclusive=True))
        await asyncio.sleep(0.1)
        waiting = (jobs.get(a)["state"], jobs.get(b)["state"], list(order))
        gate.set()
        await asyncio.gather(ta, tb)
        return waiting, order, jobs.get(b)["state"]

    waiting, order, last = asyncio.run(go())
    assert waiting == ("running", "queued", ["first start"])
    assert order == ["first start", "first end", "second start"] and last == "done"


def test_a_render_never_waits_for_an_exclusive_job_and_a_queued_one_can_be_stopped():
    import asyncio
    import threading
    gate, ran = threading.Event(), []

    def long_effect(interrupted):
        gate.wait(5)
        return {}

    def render():
        ran.append("render")
        return {}

    def effect(interrupted):
        ran.append("effect")
        return {}

    async def go():
        a, b, r = jobs.new(stoppable=True), jobs.new(stoppable=True), jobs.new()
        ta = asyncio.create_task(jobs.run(a, long_effect, error=RuntimeError, stoppable=True, exclusive=True))
        await asyncio.sleep(0.05)
        tb = asyncio.create_task(jobs.run(b, effect, error=RuntimeError, stoppable=True, exclusive=True))
        await asyncio.sleep(0.05)
        await jobs.run(r, render, error=RuntimeError)                       # not held up by the effect running
        seen = (jobs.get(r)["state"], jobs.get(b)["state"])
        assert jobs.stop(b) is True
        stopped_now = jobs.get(b)["state"]                                  # ends at once, not when the one ahead ends
        gate.set()
        await asyncio.gather(ta, tb)
        return seen, stopped_now, jobs.get(b)["state"]

    seen, stopped_now, last = asyncio.run(go())
    assert seen == ("done", "queued") and stopped_now == "stopped" and last == "stopped"
    assert ran == ["render"]                                                # the stopped one never started
