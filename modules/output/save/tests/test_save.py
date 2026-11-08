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


def test_the_carried_picture_is_the_samplers_not_the_post_look(tmp_path, monkeypatch):
    import torch
    images = torch.rand(3, 16, 16, 3)
    seen = []
    monkeypatch.setattr(nodes, "keep_last_frame", lambda file, frame: seen.append(frame))
    monkeypatch.setattr(nodes, "encode", lambda *a, **k: "libx264")
    monkeypatch.setattr(nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nodes.folder_paths, "get_save_image_path", lambda *a: (str(tmp_path), "FunUI", 1, "", "FunUI"))
    monkeypatch.setattr(nodes.FunUISaveVideo, "hidden", type("H", (), {"extra_pnginfo": None, "prompt": None})(), raising=False)
    nodes.FunUISaveVideo.execute(images, settings={"output_save": {"saturation": 0.2, "smoothing": "strong"}})
    assert np.array_equal(seen[0], nodes.to_frames(images[-1:])[-1])           # not the desaturated, smoothed one the file got


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
    assert (nodes.to_frames(x) == plain).all()
    whole = (nodes.smooth(x, "strong") * 255).round().clamp(0, 255).byte().numpy()
    assert (nodes.to_frames(x, {"smoothing": "strong"}) == whole).all()


def _combos():
    """Every on/off combination of the declared Post settings (a slider: its two ends and its neutral value)."""
    import itertools
    from modules.output import save
    choices = []
    for key, spec in save.SETTINGS.items():
        choices.append([(key, v) for v in ([o["value"] for o in spec["options"]] if spec["type"] == "enum" else [False, True] if spec["type"] == "bool" else [spec["min"], spec["default"], spec["max"]])])
    return [dict(c) for c in itertools.product(*choices)]


def test_every_post_combination_saves_and_every_encoder_codec_quality_takes_the_all_on_look(tmp_path):
    # registry-driven: no NVENC here, so "auto"/"gpu" are the NVENC -> CPU fallback
    import itertools
    x = _dull()
    for n, values in enumerate(_combos()):
        path = str(tmp_path / f"c{n}.mp4")
        assert nodes.encode(path, nodes.to_frames(x, values), 24.0, encoder="auto") and os.path.getsize(path) > 0
    everything = {"smoothing": "strong", "auto_contrast": True, "auto_brightness": True, "saturation": 1.5}
    frames = nodes.to_frames(x, everything)
    for encoder, codec, quality in itertools.product(nodes.ENCODERS, nodes.CODECS, nodes.QUALITIES):
        path = str(tmp_path / f"{encoder}_{codec}_{quality}.mp4")
        assert nodes.encode(path, frames, 24.0, encoder=encoder, codec=codec, quality=quality)
        with av.open(path) as f:
            assert f.streams.video[0].width == 16


def test_the_node_reads_its_settings_from_the_pipelines_settings_object():
    assert nodes._values(None) == nodes.DEFAULTS
    assert nodes._values({"output_save": {"smoothing": "light", "saturation": 1.2}})["smoothing"] == "light"
    assert nodes._values({"output_save": {"smoothing": "bogus"}})["smoothing"] == "off"          # wrong value: default, said


def test_post_settings_are_declared_off_and_never_become_node_sockets():
    from modules.output import save
    assert {k: v["default"] for k, v in save.SETTINGS.items()} == nodes.DEFAULTS
    assert "smoothing" not in [i.id for i in nodes.FunUISaveVideo.define_schema().inputs]
    assert "settings" in [i.id for i in nodes.FunUISaveVideo.define_schema().inputs]


def _dull():
    import torch
    torch.manual_seed(0)
    return (0.3 + 0.2 * torch.rand(20, 16, 16, 3))               # a flat, grey, dark clip: levels 0.3..0.5


def test_auto_contrast_stretches_one_fixed_curve_for_the_whole_clip():
    import torch
    x = _dull()
    look = nodes.clip_look(x, {"auto_contrast": True})
    assert look["lo"] == 0.15 and look["hi"] == 0.85               # capped: a dull clip is stretched only so far
    out = nodes.finish(x, {"auto_contrast": True}, look)
    assert out.std() > x.std()
    same = nodes.finish(x[:3], {"auto_contrast": True}, look)       # a chunk gets the same curve as the whole: nothing is per frame
    assert torch.equal(same, out[:3])
    assert nodes.clip_look(x, {}) == {"lo": 0.0, "hi": 1.0, "gamma": 1.0}


def test_auto_brightness_moves_the_midtones_toward_the_middle_within_limits():
    import torch
    dark, bright = _dull() * 0.3, 0.7 + 0.3 * _dull()
    gd, gb = nodes.clip_look(dark, {"auto_brightness": True})["gamma"], nodes.clip_look(bright, {"auto_brightness": True})["gamma"]
    assert gd == 0.75 and gb > 1.0 and gb <= 1.33
    assert nodes.finish(dark, {"auto_brightness": True}, {"lo": 0.0, "hi": 1.0, "gamma": gd}).mean() > dark.mean()
    assert nodes.clip_look(torch.zeros(0, 4, 4, 3), {"auto_contrast": True})["gamma"] == 1.0      # an empty clip is not measured


def test_saturation_mutes_and_boosts_and_neutral_changes_nothing():
    import torch
    x = torch.rand(2, 8, 8, 3)
    look = {"lo": 0.0, "hi": 1.0, "gamma": 1.0}
    grey = nodes.finish(x, {"saturation": 0.5}, look)
    assert (grey.max(-1).values - grey.min(-1).values).mean() < (x.max(-1).values - x.min(-1).values).mean()
    assert torch.equal(nodes.finish(x, {"saturation": 1.0}, look), x)


def test_every_look_follows_the_device_and_dtype_of_its_input():
    import torch
    values = {"smoothing": "strong", "auto_contrast": True, "auto_brightness": True, "saturation": 1.3}
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = _dull().to(dtype)
        out = nodes.finish(x, values, nodes.clip_look(x, values))
        assert out.dtype == dtype and out.shape == x.shape
    meta = torch.empty(2, 16, 16, 3, device="meta", dtype=torch.float16)
    out = nodes.finish(meta, {**values}, {"lo": 0.1, "hi": 0.9, "gamma": 0.9})
    assert out.device.type == "meta" and out.dtype == torch.float16


def test_saved_pipelines_get_the_settings_link_and_a_saved_pipeline_keeps_its_own():
    from modules.output import save
    slots = [{"id": "s", "node": "FunUIModifierSettings", "inputs": {}}, {"id": "v", "node": "FunUISaveVideo", "inputs": {"images": ["x", 0]}}]
    assert save.upgrade(slots, lambda n: [])[1]["inputs"]["settings"] == ["s", 0]
    own = [slots[0], {"id": "v", "node": "FunUISaveVideo", "inputs": {"settings": ["other", 0]}}]
    assert save.upgrade(own, lambda n: [])[1]["inputs"]["settings"] == ["other", 0]
    assert "settings" not in save.upgrade(slots[1:], lambda n: [])[0]["inputs"]               # nothing to link to
