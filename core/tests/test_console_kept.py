"""ComfyUI's console survives a restart: written as it prints, the run before this one can be read back."""

from core import backend_log, config


def test_the_console_is_kept_as_it_prints_and_the_next_start_can_show_it(tmp_path, monkeypatch):
    now, before = tmp_path / "console.log", tmp_path / "console.prev.log"
    monkeypatch.setattr(config, "CONSOLE_FILE", now)
    monkeypatch.setattr(config, "CONSOLE_PREV_FILE", before)

    assert backend_log.start_session() == now
    logs = backend_log.Kept(maxlen=50, path=now)
    for text in ("loading the model\n", "\x1b[31mCUDA out of memory\x1b[0m\n", "\r 40%|####", "step 3 of 8\n"):
        logs.append({"t": 0, "m": text})
    assert now.read_text() == "loading the model\nCUDA out of memory\nstep 3 of 8\n"          # colour and progress redraws left out
    assert len(logs) == 4                                                                      # the in-memory copy ComfyUI reads is unchanged

    # the process dies here (nothing closes the file); the next start moves the console aside and begins a fresh one
    assert backend_log.start_session() == now
    assert now.read_text() == ""
    got = backend_log.recent(previous=True)
    assert got["lines"] == ["loading the model", "CUDA out of memory", "step 3 of 8"]
    assert got["tones"][1] == "error"


def test_no_earlier_run_is_said_not_shown_as_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONSOLE_PREV_FILE", tmp_path / "none.log")
    got = backend_log.recent(previous=True)
    assert got["lines"] == [] and "No earlier run" in got["detail"]


def test_a_long_console_is_cut_back_and_an_unwritable_one_never_stops_printing(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_log, "KEEP_BYTES", 2000)
    path = tmp_path / "c.log"
    logs = backend_log.Kept(maxlen=10, path=path)
    for i in range(1200):
        logs.append({"m": f"line {i:04d} " + "x" * 20 + "\n"})
    assert path.stat().st_size < 2000 + 500 * 40 and path.read_text().rstrip().endswith("line 1199 " + "x" * 20)
    bad = backend_log.Kept(maxlen=10, path=tmp_path / "missing_dir" / "c.log")
    bad.append({"m": "still printed\n"})                                                         # no exception
    assert bad.path is None and len(bad) == 1
