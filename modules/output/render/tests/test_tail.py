"""Carrying the end of a clip into the next shot: the frames as generated, counted at the generation rate, as video and/or sound."""
import http.client
import json
import subprocess

import pytest

pytest.importorskip("aiohttp")

from core import media, projects, video_pipe  # noqa: E402
from core.tests.test_routes_shortcuts import server  # noqa: E402,F401
from modules.output.render import files, serving  # noqa: E402
from modules.output.render.tests.clips import comfy, make_clip, needs_ffmpeg, probe  # noqa: E402,F401

BASE = "/fun/api/m/render"


@pytest.fixture(autouse=True)
def _temp_for_the_pipe(comfy, monkeypatch):
    monkeypatch.setattr(video_pipe, "comfy_dir", files.comfy_dir)


def _project(render_file="a.mp4", **scene):
    proj = projects.Project(frame_rate=25, num_frames_per_scene=100, scenes=[projects.Scene(id="s1", **scene)],
                            scene_renders={"s1": {"media": {"filename": render_file, "subfolder": "", "type": "output"}, "inSec": 0}})
    projects.save(proj)
    return proj


def _post(port, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = (resp.status, json.loads(resp.read() or b"{}"))
    conn.close()
    return out


def _frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                          "-of", "default=nw=1:nk=1", str(path)], check=True, capture_output=True, text=True).stdout
    return int(out.strip())


@needs_ffmpeg
def test_the_last_frames_come_out_as_a_video_with_its_sound_and_a_sound_alone(comfy):
    make_clip(comfy.output / "a.mp4", 4.0)
    proj = _project()
    got = serving.tail(proj, "s1", frames=25, video=True, audio=True)
    clip, wave = media.path_for(got["video_id"]), media.path_for(got["audio_id"])
    assert (got["frames"], got["fps"], got["raw"], got["sound"]) == (25, 25.0, True, True)
    assert _frames(clip) == 25 and probe(clip)[3] is True
    assert media.get(got["video_id"])["kind"] == "video" and media.get(got["video_id"])["sound"] is True
    assert media.get(got["audio_id"])["kind"] == "audio" and wave.suffix == ".wav"
    assert wave.stat().st_size > 0
    assert clip.parent == comfy.temp and got["video_id"] not in [m["id"] for m in media.listing()]


@needs_ffmpeg
def test_sound_alone_makes_no_video_and_a_silent_clip_says_so(comfy):
    make_clip(comfy.output / "a.mp4", 2.0, audio=False)
    got = serving.tail(_project(), "s1", frames=10, video=False, audio=True)
    assert got["video_id"] is None and got["audio_id"] is None and any("no sound" in n for n in got["notes"])


@needs_ffmpeg
def test_length_is_counted_at_the_generation_rate_from_the_file_as_generated_not_the_interpolated_one(comfy):
    make_clip(comfy.output / "raw.mp4", 4.0, rate=25)
    make_clip(comfy.output / "smooth.mp4", 4.0, rate=50)            # what interpolation made from it: twice the frames, same length
    video_pipe.inherit_raw({"filename": "raw.mp4", "subfolder": "", "type": "output"}, "smooth.mp4")
    proj = _project("smooth.mp4")
    got = serving.tail(proj, "s1", frames=25, video=True)
    assert got["raw"] is True and got["fps"] == 25.0 and _frames(media.path_for(got["video_id"])) == 25      # 1 s, not half a second
    assert abs(probe(media.path_for(got["video_id"]))[0] - 1.0) < 0.1


@needs_ffmpeg
def test_a_chain_of_steps_still_points_at_the_original_and_a_lost_original_is_said(comfy):
    make_clip(comfy.output / "raw.mp4", 2.0)
    make_clip(comfy.output / "up.mp4", 2.0)
    make_clip(comfy.output / "up_x2.mp4", 2.0, rate=50)
    raw = {"filename": "raw.mp4", "subfolder": "", "type": "output"}
    video_pipe.inherit_raw(raw, "up.mp4")
    video_pipe.inherit_raw({"filename": "up.mp4", "subfolder": "", "type": "output"}, "up_x2.mp4")
    assert video_pipe.raw_of({"filename": "up_x2.mp4"}) == raw
    (comfy.output / "raw.mp4").unlink()
    got = serving.tail(_project("up_x2.mp4"), "s1", frames=10, video=True)
    assert got["raw"] is False and got["fps"] == 25.0 and any("as generated is gone" in n for n in got["notes"])      # the project's rate stands in for the lost file's


@needs_ffmpeg
def test_a_reversed_clip_ends_on_the_renders_first_pictures_and_the_window_and_length_are_kept_inside_the_clip(comfy):
    make_clip(comfy.output / "a.mp4", 4.0)
    proj = _project()
    got = serving.tail(proj, "s1", frames=2000, video=True)
    assert got["frames"] <= serving.MAX_TAIL_FRAMES and _frames(media.path_for(got["video_id"])) == got["frames"]
    short = serving.tail(proj, "s1", {"media": {"filename": "a.mp4", "subfolder": "", "type": "output"}, "inSec": 0}, {"dur": 0.5, "src_in": 1.0}, True, 100, True)
    assert short["frames"] == 12 or short["frames"] == 13        # only what the window holds
    with pytest.raises(files.ClipError):
        serving.tail(proj, "s1", None, {"dur": 1.0, "src_in": 9.0}, False, 10, True)      # a window past the end of the render


@needs_ffmpeg
def test_the_route_validates_and_answers(comfy, server):
    make_clip(comfy.output / "a.mp4", 2.0)
    proj = _project()
    url = f"{BASE}/projects/{proj.id}/tail"
    assert _post(server, url, {})[0] == 400
    assert _post(server, url, {"scene_id": "s1", "frames": 0, "video": True})[0] == 400
    assert _post(server, url, {"scene_id": "s1", "frames": 10})[0] == 400
    assert _post(server, url, {"scene_id": "nope", "frames": 10, "video": True})[0] == 404
    status, got = _post(server, url, {"scene_id": "s1", "frames": 10, "video": True, "audio": True})
    assert status == 200 and media.path_for(got["video_id"]) and media.path_for(got["audio_id"])
