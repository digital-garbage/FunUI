import os

import av
import numpy as np
import pytest

from modules.output.save import nodes


def _clip(n=30):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (n, 64, 96, 3), dtype=np.uint8)


def test_cpu_encode_writes_a_playable_h264_mp4_with_sound_and_metadata(tmp_path):
    import torch
    path = str(tmp_path / "a.mp4")
    audio = {"waveform": torch.zeros(1, 2, 44100 * 2), "sample_rate": 44100}
    used = nodes.encode(path, _clip(), 24.0, audio, {"prompt": {"1": "x"}}, encoder="cpu")
    assert "libx264" in used
    with av.open(path) as f:
        v = f.streams.video[0]
        assert v.codec_context.name == "h264" and v.frames == 30 and f.streams.audio
        assert "prompt" in f.metadata
    head = open(path, "rb").read(4096)
    assert head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat")), "index first: seekable while streaming"


def test_auto_falls_back_to_the_cpu_when_there_is_no_gpu_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "NVENC", {"h264": ("no_such_encoder_x", {})})
    monkeypatch.setattr(av, "codecs_available", av.codecs_available | {"no_such_encoder_x"})
    path = str(tmp_path / "b.mp4")
    assert "libx264" in nodes.encode(path, _clip(10), 24.0, encoder="auto")


def test_gpu_choice_still_saves_on_the_cpu_when_the_encoder_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "NVENC", {"h264": ("no_such_encoder_x", {})})
    path = str(tmp_path / "c.mp4")
    assert "libx264" in nodes.encode(path, _clip(5), 24.0, encoder="gpu")
    assert os.path.getsize(path) > 0


def test_a_refused_gpu_open_leaves_no_half_file_and_the_cpu_saves_it(tmp_path, monkeypatch):
    # a codec that exists but refuses to open with these options, as NVENC does on a driver mismatch
    monkeypatch.setattr(nodes, "NVENC", {"h264": ("libx264", {"preset": "no-such-preset-x", "crf": "abc"})})
    alerts = []
    monkeypatch.setattr(nodes.log, "once", lambda key, level, who, msg: alerts.append(msg))
    path = str(tmp_path / "e.mp4")
    assert "libx264" in nodes.encode(path, _clip(5), 24.0, encoder="gpu")
    assert os.path.getsize(path) > 0
    assert alerts and "Saved as H.264 on the CPU" in alerts[0]


def test_h265_is_tagged_hvc1_so_a_mac_browser_plays_it(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "NVENC", {"h265": ("libx265", {"crf": "23"})})       # the software HEVC encoder stands in for NVENC here
    if "libx265" not in av.codecs_available:
        pytest.skip("no libx265 in this PyAV build")
    path = str(tmp_path / "d.mp4")
    assert "H265" in nodes.encode(path, _clip(10), 24.0, encoder="gpu", codec="h265")
    with av.open(path) as f:
        assert f.streams.video[0].codec_context.name == "hevc" and f.streams.video[0].codec_tag == "hvc1"


def test_nan_sound_is_dropped_and_the_video_still_saves(tmp_path):
    import torch
    path = str(tmp_path / "n.mp4")
    wave = torch.zeros(1, 2, 44100 * 2)
    wave[0, 0, 5] = float("nan")
    nodes.encode(path, _clip(), 24.0, {"waveform": wave, "sample_rate": 44100}, None, encoder="cpu")
    with av.open(path) as f:
        assert f.streams.video[0].frames == 30 and not f.streams.audio


def test_quality_sets_the_constant_quality_number_and_high_is_the_default():
    for name, key in (("libx264", "crf"), ("h264_nvenc", "cq")):
        base = (name, {key: "23"})
        assert nodes._with_quality(base, "standard")[1][key] == "23"
        assert int(nodes._with_quality(base, "high")[1][key]) < 23
        assert int(nodes._with_quality(base, "maximum")[1][key]) < int(nodes._with_quality(base, "high")[1][key])
        assert nodes._with_quality(base, "nonsense")[1][key] == nodes._with_quality(base, "high")[1][key]
    assert nodes.X264[1]["crf"] == "16"


def test_higher_quality_keeps_more_of_the_picture(tmp_path):
    sizes = {}
    for q in ("standard", "high"):
        path = str(tmp_path / f"{q}.mp4")
        nodes.encode(path, _clip(20), 24.0, encoder="cpu", quality=q)
        sizes[q] = os.path.getsize(path)
    assert sizes["high"] > sizes["standard"]


def test_the_last_decoded_picture_is_kept_exactly_for_the_next_shot(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(nodes.folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    frame = np.random.default_rng(1).integers(0, 256, (12, 16, 3), dtype=np.uint8)
    nodes.keep_last_frame("FunUI_00007_.mp4", frame)
    name = tmp_path / "temp" / nodes.last_frame_name("FunUI_00007_.mp4")
    assert name.name == "funui_last_FunUI_00007_.png"
    assert np.array_equal(np.array(Image.open(name).convert("RGB")), frame)           # lossless: every value as the sampler made it
    monkeypatch.setattr(nodes.folder_paths, "get_temp_directory", lambda: (_ for _ in ()).throw(OSError("no temp")))
    nodes.keep_last_frame("x.mp4", frame)           # a failure here never fails the save


def test_a_failed_keep_never_leaves_an_older_videos_picture_under_the_name(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    old = tmp_path / nodes.last_frame_name("FunUI_00001_.mp4")
    old.write_bytes(b"picture of a deleted video")
    monkeypatch.setattr("PIL.Image.Image.save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    nodes.keep_last_frame("FunUI_00001_.mp4", np.zeros((4, 4, 3), dtype=np.uint8))
    assert not old.exists()


# --- smoothing (edge anti-aliasing) ----------------------------------------------------------------------------------

def _staircase():
    """A hard diagonal edge (black / white), the picture a jagged edge makes: [T, H, W, 3] floats."""
    import torch
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    return (xx > yy).float()[None, :, :, None].repeat(2, 1, 1, 3)


def test_smoothing_off_is_the_old_path_and_the_edge_is_softened_only_where_it_is():
    import torch
    x = _staircase()
    assert nodes.smooth(x, "off") is x
    light, strong = nodes.smooth(x, "light"), nodes.smooth(x, "strong")
    assert ((light - x).abs() > 0.05).any() and ((strong - x).abs() > (light - x).abs()).any()
    assert torch.equal(light[:, 0, -1], x[:, 0, -1]) and torch.equal(light[:, -1, 0], x[:, -1, 0])     # flat corners untouched
    flat = torch.full((1, 16, 16, 3), 0.4)
    assert torch.allclose(nodes.smooth(flat, "strong"), flat)
    assert torch.equal(light[0], light[1])                          # frames never mix: spatial only


def test_smoothing_follows_the_device_and_dtype_of_its_input():
    import torch
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = _staircase().to(dtype)
        assert nodes.smooth(x, "strong").dtype == dtype
    meta = torch.empty(2, 32, 32, 3, device="meta", dtype=torch.float16)      # the cloud stand-in for "some other device"
    out = nodes.smooth(meta, "light")
    assert out.device.type == "meta" and out.dtype == torch.float16 and out.shape == meta.shape


def test_chunked_conversion_equals_one_go_and_off_equals_the_plain_cast():
    import torch
    x = torch.rand(nodes._CHUNK * 2 + 3, 16, 16, 3)
    plain = (x * 255).round().clamp(0, 255).byte().numpy()
    assert (nodes.to_frames(x, "off") == plain).all()
    whole = (nodes.smooth(x, "strong") * 255).round().clamp(0, 255).byte().numpy()
    assert (nodes.to_frames(x, "strong") == whole).all()


def test_every_smoothing_saves_with_every_encoder_codec_and_quality(tmp_path):
    # registry-driven: the cross product of what the node offers; no NVENC here, so "auto"/"gpu" are the NVENC -> CPU fallback
    import itertools
    x = _staircase()
    for smoothing, encoder, codec, quality in itertools.product(nodes.SMOOTHINGS, nodes.ENCODERS, nodes.CODECS, nodes.QUALITIES):
        path = str(tmp_path / f"{smoothing}_{encoder}_{codec}_{quality}.mp4")
        frames = nodes.to_frames(x, smoothing)
        used = nodes.encode(path, frames, 24.0, encoder=encoder, codec=codec, quality=quality)
        assert used and os.path.getsize(path) > 0
        with av.open(path) as f:
            assert f.streams.video[0].width == 32


def test_smoothing_is_a_node_input_defaulting_to_off():
    schema = nodes.FunUISaveVideo.define_schema()
    combo = next(i for i in schema.inputs if i.id == "smoothing")
    assert combo.default == "off" and list(combo.options) == list(nodes.SMOOTHINGS)
