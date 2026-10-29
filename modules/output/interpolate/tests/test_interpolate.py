import pytest

from modules.output.interpolate import framegen
from modules.output.render import upscale
from modules.output.render.tests.clips import comfy, make_clip, needs_ffmpeg  # noqa: F401

torch = pytest.importorskip("torch")


def test_scaled_fps_keeps_the_fraction():
    assert upscale.scaled_fps("24/1", 2) == "48/1"
    assert upscale.scaled_fps("30000/1001", 2) == "60000/1001"
    assert upscale.scaled_fps("25", 3) == "75/1"


@needs_ffmpeg
def test_interpolation_stream_adds_frames_and_keeps_length_and_sound(comfy):
    src, dst = str(comfy.output / "a.mp4"), str(comfy.output / "fi.mp4")
    make_clip(src, seconds=1.0, size="64x48", rate=10, audio=True)
    last = [None]

    def between(x):                                   # stands in for the effect: an average frame, carried across chunks
        out = []
        for f in x:
            if last[0] is not None:
                out.append((last[0] + f) / 2)
            out.append(f)
            last[0] = f
        return torch.stack(out)

    assert upscale.upscale_file(src, dst, between, fps_factor=2) == 19      # 10 frames -> 10 + 9 between, none lost at the chunk seam
    w, h, fps, sound = upscale.probe(dst)
    assert (w, h, fps, sound) == (64, 48, "20/1", True)


def test_frame_generation_options_are_checked_by_name():
    assert framegen.options("{}") == framegen.DEFAULTS
    assert framegen.options({"multiplier": 4, "mode": "HIGH"})["multiplier"] == 4
    for bad in ({"mode": "ULTRA"}, {"multiplier": 1}, {"multiplier": 5}, {"multiplier": 2.5}, {"multiplier": True}, {"tenbit": 1}):
        with pytest.raises(ValueError):
            framegen.options(bad)
    with pytest.raises(ValueError):
        framegen.options("not json")


def test_frame_generation_without_the_add_on_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "nvvfx", None)
    with pytest.raises(RuntimeError, match="Settings ▸ Modules"):
        framegen.make(framegen.options("{}"))


def test_frame_generation_feeds_pairs_in_order_across_chunks(monkeypatch):
    """A stand-in nvvfx (CPU) records the calls: every pair once, in order, indexes 1..N-1, the seam between chunks included."""
    import sys
    import types
    calls, loads = [], []

    class Effect:
        class Mode: LOW = 0; MEDIUM = 1; HIGH = 2
        class ImageEncoding: RGB8 = 0; RGB10A2 = 1
        def __init__(self, mode, **kw): self.mode = mode
        def load(self): loads.append(1)
        def run(self, a, b, frame_index=1):
            calls.append((float(a.mean()), float(b.mean()), frame_index))
            return types.SimpleNamespace(image=(a + b) / 2)

    from core import nvidia
    monkeypatch.setattr(nvidia, "_held", {})
    monkeypatch.setitem(sys.modules, "nvvfx", types.SimpleNamespace(VideoFrameGeneration=Effect))
    from core import addons
    monkeypatch.setattr(addons, "enabled", lambda _id: True)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)            # no GPU here: "cuda" is this device
    monkeypatch.setattr(torch, "from_dlpack", lambda t: t)
    fn = framegen.make(framegen.options({"multiplier": 3}))
    frames = torch.stack([torch.full((2, 2, 3), v) for v in (0.0, 0.3, 0.6, 0.9)])
    a, b = fn(frames[:2]), fn(frames[2:])
    assert a.shape[0] == 4 and b.shape[0] == 6                  # 1 + 3, then 3 + 3: the seam pair got its frames too
    assert [round(c[1], 2) for c in calls[::2]] == [0.3, 0.6, 0.9] and [c[2] for c in calls] == [1, 2] * 3
    assert calls[2][0] == pytest.approx(0.3) and calls[2][1] == pytest.approx(0.6)   # the seam pair was interpolated
    again = framegen.make(framegen.options({"multiplier": 3}))             # a second job: NVIDIA's model stays loaded
    again(frames[:2])
    assert len(loads) == 1


@needs_ffmpeg
def test_the_job_writes_a_doubled_file_beside_the_original(comfy, monkeypatch):
    import sys
    import types
    from modules.output.interpolate import nodes
    make_clip(str(comfy.output / "a.mp4"), seconds=1.0, size="64x48", rate=10, audio=True)
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_output_directory=lambda: str(comfy.output)))
    monkeypatch.setattr(nodes.video_pipe, "comfy_dir", lambda kind: str(comfy.output))
    last = [None]

    def between(x):
        out = []
        for f in x:
            if last[0] is not None:
                out.append((last[0] + f) / 2)
            out.append(f)
            last[0] = f
        return torch.stack(out)
    monkeypatch.setattr(nodes.framegen, "make", lambda o: between)
    got = nodes.interpolate({"filename": "a.mp4", "subfolder": "", "type": "output"}, {})["media"]
    assert got == {"filename": "a_x2.mp4", "subfolder": "funui_interpolated", "type": "output"}
    assert upscale.probe(str(comfy.output / "funui_interpolated" / "a_x2.mp4")) == (64, 48, "20/1", True)
    with pytest.raises(nodes.InterpolateError, match="not in the output folder"):
        nodes.interpolate({"filename": "../x.mp4"}, {})
