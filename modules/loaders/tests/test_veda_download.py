"""The optional predictor download: one downloader (core/fetch), only on a click, from the official address, and every way it can fail."""
import asyncio
import hashlib
import http.server
import threading
import time

import pytest

from core import fetch
from modules.loaders import veda_download as vd

BODY = b"x" * 3_000_000


@pytest.fixture
def dl(tmp_path):
    (tmp_path / "dl").mkdir()
    return tmp_path / "dl"


@pytest.fixture
def server():
    """A local server with a few misbehaving files: /ok, /short (cut off after half), /gone (404)."""
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/gone":
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY if self.path == "/ok" else BODY[:len(BODY) // 2])      # /short: the connection closes early

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def _left(folder):
    return sorted(p.name for p in folder.iterdir())


def test_a_good_download_is_renamed_only_when_complete_and_reports_progress(server, dl):
    seen = []
    fetch.to_file(f"{server}/ok", dl / "m.safetensors", lambda d, t: seen.append((d, t)), size=len(BODY), sha256=hashlib.sha256(BODY).hexdigest())
    assert _left(dl) == ["m.safetensors"] and (dl / "m.safetensors").read_bytes() == BODY
    assert seen[-1] == (len(BODY), len(BODY)) and len(seen) >= 3


@pytest.mark.parametrize("path,kw,words", [
    ("/short", {}, "stopped at"),                                       # a cut connection
    ("/ok", {"size": 1234}, "expected 1234"),                          # the wrong file size
    ("/ok", {"sha256": "0" * 64}, "checksum"),                         # the right size, the wrong bytes
    ("/gone", {}, "404"),                                              # the file is not there
])
def test_a_bad_download_leaves_nothing_behind_and_says_why(server, dl, path, kw, words):
    with pytest.raises(OSError, match=words):
        fetch.to_file(server + path, dl / "m.safetensors", **kw)
    assert _left(dl) == []                                         # no file, and no .part that could be mistaken for one


def test_no_network_is_an_oserror_and_leaves_nothing(dl):
    with pytest.raises(OSError):
        fetch.to_file("http://127.0.0.1:9/none", dl / "m.safetensors", timeout=2)
    assert _left(dl) == []


@pytest.fixture
def veda_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(vd, "folder", lambda: tmp_path / "models" / "veda")
    vd._state.clear()
    return tmp_path / "models" / "veda"


def _wait(model_id):
    for _ in range(200):
        if vd._state[model_id]["state"] != "downloading":
            return vd._state[model_id]
        time.sleep(0.01)
    raise AssertionError("the download never finished")


def test_the_listing_names_the_official_files_sizes_and_the_licence(veda_dir):
    rows = {r["id"]: r for r in vd.listing()["models"]}
    assert rows["t2va"]["file"] == "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors" and rows["t2va"]["size"] == 275415648
    assert rows["r2va"]["file"] == "minimax_h3_r2va_veda_preview_fp8.safetensors" and rows["r2va"]["size"] == 275415752
    assert rows["t2va"]["url"] == ("https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/resolve/main/"
                                   "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors")
    assert rows["r2va"]["url"].startswith("https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/")
    assert vd.listing()["license"]["name"] == "MiniMax H3 Community License" and vd.listing()["license"]["url"].endswith("/blob/main/LICENSE")
    assert {r["state"] for r in rows.values()} == {"absent"}


def test_download_goes_to_models_veda_with_the_pinned_size_and_checksum(veda_dir):
    calls = []

    def get(url, dest, progress, size=None, sha256=None):
        calls.append((url, dest, size, sha256))
        progress(10, size)
        dest.write_bytes(b"model")
    assert vd.download("t2va", get=get) is True
    assert _wait("t2va")["state"] == "ready"
    url, dest, size, sha = calls[0]
    assert url.startswith("https://huggingface.co/") and dest == veda_dir / "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"
    assert size == 275415648 and len(sha) == 64
    assert vd.files() == [dest.name] and next(r for r in vd.listing()["models"] if r["id"] == "t2va")["state"] == "ready"
    assert vd.download("t2va", get=get) is False                          # already there: nothing is fetched twice


def test_unknown_ids_are_refused_and_a_failure_is_shown_beside_the_button(veda_dir):
    assert vd.download("../../etc", get=lambda *a, **k: None) is False
    assert vd.download("", get=lambda *a, **k: None) is False

    def boom(url, dest, progress, size=None, sha256=None):
        raise OSError("the download stopped at 5 of 10 bytes")
    assert vd.download("r2va", get=boom) is True
    state = _wait("r2va")
    assert state["state"] == "error" and "stopped at 5 of 10" in state["error"]
    row = next(r for r in vd.listing()["models"] if r["id"] == "r2va")
    assert row["state"] == "error" and "stopped at" in row["error"] and vd.files() == []
    assert vd.download("r2va", get=lambda url, dest, p, size=None, sha256=None: dest.write_bytes(b"x")) is True       # retry is allowed


def test_a_part_file_is_not_a_predictor(veda_dir):
    veda_dir.mkdir(parents=True)
    (veda_dir / "a.safetensors.part").write_bytes(b"half")
    (veda_dir / "b.safetensors").write_bytes(b"x")
    assert vd.files() == ["b.safetensors"]


def test_the_routes_list_and_start_but_never_fetch_on_their_own(veda_dir, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from modules.loaders import diffusion_model
    started = []
    monkeypatch.setattr(vd, "download", lambda model_id, get=None: started.append(model_id) or True)

    async def go():
        app, table = web.Application(), web.RouteTableDef()
        diffusion_model.routes(table, "/m", web)
        app.add_routes(table)
        async with TestClient(TestServer(app)) as c:
            listing = await (await c.get("/m/veda")).json()
            assert started == []                                           # looking never downloads
            go_ = await (await c.post("/m/veda/download", json={"id": "t2va"})).json()
            junk = await c.post("/m/veda/download", data=b"not json")
            return listing, go_, junk.status
    listing, go_, junk = asyncio.run(go())
    assert [m["id"] for m in listing["models"]] == ["t2va", "r2va"] and go_ == {"started": True} and started == ["t2va", ""] and junk == 200
