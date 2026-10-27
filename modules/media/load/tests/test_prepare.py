"""Video preparation: the loader choices (rate, skip, cut, every Nth, cap, crop, size), on made-up frames and a real file."""

import json

import pytest
import torch

from modules.media.load import prepare as prep


def _clip(n=48, h=40, w=80):
    """Frame i is filled with the value i/255, so which frames survived can be read off the pixels."""
    return torch.stack([torch.full((h, w, 3), i / 255.0) for i in range(n)])


def _ids(frames):
    return [round(float(f[0, 0, 0]) * 255) for f in frames]


def test_frames_are_picked_in_loader_order():
    frames, fps, start, span = prep.prepare(_clip(), 24, {"skip_first": 4, "cut_last": 4, "every_nth": 2, "frame_cap": 10})
    assert _ids(frames) == [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    assert fps == 12 and start == pytest.approx(4 / 24) and span == pytest.approx(20 / 24)


def test_target_rate_picks_the_frame_showing_at_each_tick():
    frames, fps, *_ = prep.prepare(_clip(48), 48, {"fps": 24})
    assert fps == 24 and _ids(frames) == list(range(0, 48, 2))


def test_nothing_left_is_refused():
    with pytest.raises(ValueError, match="nothing is left"):
        prep.prepare(_clip(10), 24, {"skip_first": 6, "cut_last": 4})


def test_crop_fit_fills_the_target_and_loses_the_overflow():
    out = prep.shape(_clip(2, 40, 80), 32, 32, "crop")
    assert out.shape == (2, 32, 32, 3)


def test_pad_fit_keeps_everything_with_black_bars():
    out = prep.shape(_clip(1, 40, 80) + 0.5, 64, 64, "pad")
    assert out.shape == (1, 64, 64, 3)
    assert float(out[0, 0, 0, 0]) == 0.0 and float(out[0, 32, 32, 0]) > 0.4


def test_stretch_and_one_sided_target_and_multiple():
    assert prep.shape(_clip(1, 40, 80), 64, 64, "stretch").shape[1:3] == (64, 64)
    assert prep.shape(_clip(1, 40, 80), 64, 0).shape[1:3] == (32, 64)          # height follows the shape
    assert prep.shape(_clip(1, 50, 90), 0, 0, multiple_of=32).shape[1:3] == (32, 64)


def test_margin_crop_and_a_crop_that_removes_everything():
    assert prep.shape(_clip(1, 40, 80), crop_left=25, crop_right=25).shape[1:3] == (40, 40)
    with pytest.raises(ValueError, match="no picture"):
        prep.shape(_clip(1, 4, 4), crop_left=50, crop_right=50)


def test_sound_follows_the_picked_stretch():
    wave = torch.arange(48000, dtype=torch.float32).reshape(1, 1, -1)
    cut = prep.cut_audio({"waveform": wave, "sample_rate": 48000}, 0.25, 0.5)
    assert cut["waveform"].shape[-1] == 24000 and float(cut["waveform"][0, 0, 0]) == 12000


# --- the node, on a real encoded file --------------------------------------------------------

@pytest.fixture
def stored(comfyui, tmp_path, monkeypatch):
    from core import config, media
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    import av
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        for i in range(48):
            frame = av.VideoFrame.from_ndarray(torch.full((32, 64, 3), i * 5, dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return media.save_upload("clip.mp4", path.read_bytes())


def test_the_node_prepares_a_real_video(stored):
    from modules.media.load.nodes import FunUILoadMedia
    options = {"skip_first": 8, "cut_last": 8, "every_nth": 2, "width": 32, "height": 32, "fit": "crop", "multiple_of": 16}
    first, frames, _audio = FunUILoadMedia.execute(stored["id"], full=True, prepare=json.dumps(options)).result
    assert frames.shape == (16, 32, 32, 3) and first.shape[0] == 1


def test_no_options_keeps_the_old_behaviour(stored):
    from modules.media.load.nodes import FunUILoadMedia
    _first, frames, _audio = FunUILoadMedia.execute(stored["id"], full=True).result
    assert frames.shape[0] == 48 and frames.shape[1:3] == (32, 64)


def test_a_cut_that_leaves_nothing_is_said_in_plain_words(stored):
    from modules.media.load.nodes import FunUILoadMedia
    with pytest.raises(RuntimeError, match="cannot be prepared"):
        FunUILoadMedia.execute(stored["id"], full=True, prepare=json.dumps({"skip_first": 40, "cut_last": 20})).result
    with pytest.raises(RuntimeError, match="cannot be prepared"):
        FunUILoadMedia.execute(stored["id"], full=True, prepare="{not json").result
