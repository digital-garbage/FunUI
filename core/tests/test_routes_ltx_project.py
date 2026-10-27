"""An LTX project as Dex runs it, over HTTP on the real route table: the direct video-to-video preset (also as saved with
the old wiring), a project with a video in References and Video input switched on, and every call the app makes at start-up
and for a render. Each must answer JSON: a page that gets anything else fails on 'r is null'."""

import pytest

from core.tests.test_routes_pipeline import _request, registered, server  # noqa: F401


def _preset(pid):
    from modules.models.ltx.pipeline import presets
    return next(p["slots"] for p in presets() if p["id"] == pid)


@pytest.mark.parametrize("pid", ["ltx23_video_to_video_direct", "ltx23_video_to_video", "ltx23_image_to_video", "ltx23_anchor_guide"])
def test_every_startup_and_render_call_answers_json_on_an_ltx_project(server, pid):
    slots = _preset(pid)
    old = [dict(s, inputs={k: (["source_image", 0] if v == ["source_image", 1] else v) for k, v in s["inputs"].items()}) for s in slots]
    for saved in (slots, old):
        status, body = _request(server, "POST", "/fun/api/pipeline", {"slots": saved})
        assert status == 200 and "raw" not in body, (status, body)
        users = [(s["id"], k) for s in body["slots"] for k, v in s["inputs"].items() if isinstance(v, list) and v[0] == "source_image"]
        assert all(next(s for s in body["slots"] if s["id"] == i)["inputs"][k][1] == 1 for i, k in users), "the old wiring was not brought up to date"
        status, body = _request(server, "GET", "/fun/api/pipeline")
        assert status == 200 and "slots" in body, body

    traits = "audio_stream,ltx_av"
    for path in ("/fun/api/story/markers", f"/fun/api/modules?traits={traits}", "/fun/api/modules", "/fun/api/shortcuts",
                 "/fun/api/readiness", "/fun/api/system", "/fun/api/addons", "/fun/api/m/render/library", "/fun/api/pipeline/presets"):
        status, body = _request(server, "GET", path)
        assert status < 500 and "raw" not in body, (path, status, body)

    status, body = _request(server, "GET", f"/fun/api/modules?traits={traits}")
    ids = [m["id"] for m in body.get("modules", [])]
    assert "conditioning_shot_camera" not in ids, "shot camera must be hidden for LTX"
    assert "conditioning_shot_camera" in [m["id"] for m in body.get("incompatible", [])]
