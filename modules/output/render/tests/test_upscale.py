import pytest

from modules.output.render.tests.clips import comfy, make_clip, needs_ffmpeg  # noqa: F401
from modules.output.render import files, upscale

torch = pytest.importorskip("torch")


@needs_ffmpeg
def test_upscale_doubles_size_keeps_fps_frames_and_sound(comfy):
    src, dst = str(comfy.output / "a.mp4"), str(comfy.output / "up.mp4")
    make_clip(src, seconds=1.0, size="64x48", rate=10, audio=True)
    seen = []

    def twice(x):
        seen.append(x.shape[0])
        return x.repeat_interleave(2, 1).repeat_interleave(2, 2)

    assert upscale.upscale_file(src, dst, twice) == 10
    assert seen == [8, 2]                                   # streamed in chunks, not all at once
    w, h, fps, sound = upscale.probe(dst)
    assert (w, h, fps, sound) == (128, 96, "10/1", True)
    assert files.moov_position(dst) == "front"


@needs_ffmpeg
def test_interrupt_stops_and_leaves_no_file(comfy):
    src, dst = str(comfy.output / "a.mp4"), str(comfy.output / "up.mp4")
    make_clip(src, seconds=1.0, size="64x48", rate=10, audio=False)
    calls = []

    def stop():
        calls.append(1)
        if len(calls) > 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        upscale.upscale_file(src, dst, lambda x: x, stop)
    import os
    assert not os.path.exists(dst)


def test_free_name_never_overwrites(tmp_path):
    (tmp_path / "a_m.mp4").write_bytes(b"x")
    assert upscale.free_name(str(tmp_path), "a.mp4", "sub/m.safetensors") == "a_m_1.mp4"


from modules.output.render import rtx  # noqa: E402


def test_rtx_options_are_checked_by_name():
    assert rtx.options("{}")["quality"] == "ULTRA"
    assert rtx.options({"scale": 3, "quality": "DEBLUR_HIGH"})["scale"] == 3
    for bad in ({"quality": "ULTRAA"}, {"scale": 5}, {"strength": -1}, {"resize": "huge"}, {"width": 8}, {"tenbit": "false"}):
        with pytest.raises(ValueError):
            rtx.options(bad)
    with pytest.raises(ValueError):
        rtx.options("not json")


def test_rtx_output_size():
    o = rtx.options({"scale": 2})
    assert rtx.out_size(640, 360, o) == (1280, 720)
    assert rtx.out_size(361, 460, o) == (720, 920)                     # snapped to a multiple of 8
    assert rtx.out_size(640, 360, rtx.options({"resize": "size", "width": 1000, "height": 510})) == (1000, 512)
    assert rtx.out_size(641, 361, rtx.options({"quality": "DENOISE_HIGH", "scale": 4})) == (641, 361)   # cleans at its own size


def test_rtx_ten_bit_packing_round_trips():
    x = torch.rand(3, 4, 5)
    back = rtx.unpack10(rtx.pack10(x))
    assert back.shape == x.shape and (back - x).abs().max() < 1 / 1023
    assert rtx.pack10(torch.ones(3, 1, 1)).item() == 0xFFFFFFFF        # all ten-bit channels full, alpha 3


def test_rtx_without_the_add_on_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "nvvfx", None)
    with pytest.raises(RuntimeError, match="Settings ▸ Modules"):
        rtx.make(rtx.options("{}"))


def test_rtx_refuses_when_switched_off_even_if_installed(monkeypatch):
    from core import addons
    monkeypatch.setitem(__import__("sys").modules, "nvvfx", type("M", (), {"VideoSuperRes": object}))   # the package "is installed"
    monkeypatch.setattr(addons, "enabled", lambda _id: False)
    with pytest.raises(RuntimeError, match="switched off"):
        rtx.make(rtx.options("{}"))



@needs_ffmpeg
def test_run_writes_the_upscaled_file_beside_the_original_without_a_comfy_prompt(comfy, monkeypatch):
    import sys
    import types
    make_clip(str(comfy.output / "a.mp4"), seconds=0.5, size="64x48", rate=10, audio=True)
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_output_directory=lambda: str(comfy.output)))
    monkeypatch.setattr(rtx, "make", lambda o: (lambda x: x.repeat_interleave(2, 1).repeat_interleave(2, 2)))
    got = upscale.run({"filename": "a.mp4", "subfolder": "", "type": "output"}, rtx.NAME, {"quality": "HIGH"})
    assert got == {"media": {"filename": "a_rtx_high.mp4", "subfolder": "funui_upscaled", "type": "output"}}
    assert upscale.probe(str(comfy.output / "funui_upscaled" / "a_rtx_high.mp4"))[:2] == (128, 96)
    with pytest.raises(RuntimeError, match="not in the output folder"):
        upscale.run({"filename": "gone.mp4"}, rtx.NAME)


@needs_ffmpeg
def test_an_upscaled_or_interpolated_file_inherits_the_lossless_last_frame_of_its_source(comfy, monkeypatch):
    import sys
    import types
    from core import video_pipe
    from modules.output.interpolate import framegen, nodes as interp
    monkeypatch.setattr(video_pipe, "comfy_dir", lambda kind: str(comfy.output if kind == "output" else comfy.temp))
    make_clip(str(comfy.output / "a.mp4"), seconds=0.5, size="64x48", rate=10, audio=True)
    (comfy.temp / "funui_last_a.png").write_bytes(b"png-as-the-sampler-made-it")
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_output_directory=lambda: str(comfy.output)))
    monkeypatch.setattr(framegen, "make", lambda o: (lambda prev, cur, i: []))
    monkeypatch.setattr(video_pipe, "stream_file", lambda src, dst, fn, interrupted, mult: open(dst, "wb").close() or 1)
    got = interp.interpolate({"filename": "a.mp4", "subfolder": "", "type": "output"}, {"multiplier": 2})["media"]["filename"]
    assert (comfy.temp / video_pipe.last_frame_name(got)).read_bytes() == b"png-as-the-sampler-made-it"       # the picture from before the step
    video_pipe.inherit_last_frame("nothing.mp4", "other.mp4")           # no picture to inherit: quietly nothing
    assert not (comfy.temp / "funui_last_other.png").exists()


@needs_ffmpeg
def test_an_encoder_that_cannot_write_says_why_not_a_broken_pipe(comfy):
    from core import video_pipe
    src = comfy.output / "a.mp4"
    make_clip(src, seconds=1)
    with pytest.raises(video_pipe.ClipError) as err:
        video_pipe.stream_file(str(src), str(comfy.output / "no_such_folder" / "b.mp4"), lambda x: x)
    assert "Broken pipe" not in str(err.value) and "ffmpeg could not encode" in str(err.value)


def test_a_file_ffprobe_cannot_read_is_a_clip_error(tmp_path):
    from core import video_pipe
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(video_pipe.ClipError):
        video_pipe.probe(str(bad))
