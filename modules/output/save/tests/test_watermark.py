"""The Watermark of Engine > Post: where it lands, what it changes, and that a missing picture is said and never silent."""
import numpy as np
import pytest
import torch
from PIL import Image

from .. import nodes, watermark


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """nodes imports comfy."""


GREY = torch.full((2, 90, 160, 3), 0.5)


def _changed(values, x=GREY):
    out = nodes.to_frames(x, values).astype(np.int16)
    base = nodes.to_frames(x).astype(np.int16)
    mask = (out != base).any(axis=(0, 3))
    ys, xs = np.nonzero(mask)
    return out, mask, (ys.min(), ys.max(), xs.min(), xs.max()) if len(ys) else None


TEXT = {"watermark": "text", "watermark_text": "FunUI", "watermark_opacity": 1.0, "watermark_size": 20.0}


@pytest.mark.parametrize("corner", watermark.CORNERS)
def test_the_mark_sits_in_its_corner_and_nothing_else_changes(corner):
    _, mask, box = _changed({**TEXT, "watermark_corner": corner})
    y0, y1, x0, x1 = box
    assert (y1 < 45) == corner.startswith("top") and (y0 >= 45) == corner.startswith("bottom")
    assert (x1 < 80) == corner.endswith("left") and (x0 >= 80) == corner.endswith("right")
    assert mask.sum() == mask[y0:y1 + 1, x0:x1 + 1].sum()                        # nothing outside the box moved


def test_the_text_is_as_wide_as_asked_whatever_its_length():
    for text in ("Hi", "a much longer line of words"):
        _, _, (y0, y1, x0, x1) = _changed({**TEXT, "watermark_text": text, "watermark_size": 30.0})
        assert abs((x1 - x0 + 1) - 48) <= 3


def test_opacity_blends_with_the_picture_and_zero_outline_draws_none():
    solid, _, _ = _changed({**TEXT, "watermark_colour": "#ffffff"})
    half, _, _ = _changed({**TEXT, "watermark_colour": "#ffffff", "watermark_opacity": 0.5})
    assert solid.max() == 255 and 128 + 40 < half.max() < 255 - 40                # 0.5 grey + half of the way to white
    _, _, box = _changed({**TEXT, "watermark_outline": 0.0})
    _, _, boxed = _changed({**TEXT, "watermark_outline": 10.0})
    assert boxed[1] - boxed[0] > box[1] - box[0]                                  # an outline makes the mark's footprint larger


def test_the_outline_has_its_own_colour_and_opacity():
    values = {**TEXT, "watermark_outline": 10.0, "watermark_outline_colour": "#ff0000", "watermark_outline_opacity": 1.0, "watermark_colour": "#ffffff"}
    out, mask, _ = _changed(values)
    reds = ((out[0, ..., 0] > 200) & (out[0, ..., 1] < 60) & (out[0, ..., 2] < 60)).sum()
    assert reds > 20
    faint, _, _ = _changed({**values, "watermark_outline_opacity": 0.2})
    assert (faint[0, ..., 0].max() < out[0, ..., 0].max()) or (faint[0, ..., 1] != out[0, ..., 1]).any()


def test_a_picture_from_the_media_bin_keeps_its_transparency(tmp_path, monkeypatch):
    png = tmp_path / "logo.png"
    img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    for x in range(10, 30):
        for y in range(10, 30):
            img.putpixel((x, y), (255, 0, 0, 255))
    img.save(png)
    monkeypatch.setattr(watermark.media, "path_for", lambda mid: png if mid == "abc" else None)
    out, mask, box = _changed({"watermark": "picture", "watermark_picture": "abc", "watermark_opacity": 1.0, "watermark_size": 20.0, "watermark_corner": "top-left"})
    assert out[0, box[0]:box[1] + 1, box[2]:box[3] + 1, 0].max() == 255
    assert mask.sum() < (box[1] - box[0] + 1) * (box[3] - box[2] + 1) * 1.0 + 1


def test_a_picture_that_is_not_there_is_said_and_the_video_goes_out_without(monkeypatch):
    said = []
    monkeypatch.setattr(watermark.log, "once", lambda key, level, title, text: said.append(text))
    monkeypatch.setattr(watermark.media, "path_for", lambda mid: None)
    for values in ({"watermark": "picture", "watermark_picture": "gone"}, {"watermark": "picture", "watermark_picture": ""}, {"watermark": "text", "watermark_text": "  "}):
        assert (nodes.to_frames(GREY, values) == nodes.to_frames(GREY)).all()
    assert len(said) == 3 and all("has none" in t for t in said)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_the_layer_follows_the_frames_dtype(dtype):
    layer = watermark.build({**TEXT, "watermark_outline": 5.0}, 160, 90)
    x = GREY.to(dtype)
    out = watermark.apply(x, layer)
    assert out.dtype == dtype and out.device == x.device and out.shape == x.shape
    assert (out != x).any()


def test_a_layer_drawn_for_another_size_is_refused_not_misplaced():
    layer = watermark.build(TEXT, 160, 90)
    with pytest.raises(ValueError, match="another frame size"):
        watermark.apply(torch.zeros(1, 20, 20, 3), layer)


def test_a_tall_mark_is_brought_down_to_half_the_height():
    layer = watermark.build({**TEXT, "watermark_text": "x", "watermark_size": 60.0}, 160, 40)
    assert layer.alpha.shape[0] <= 20 + 1


def test_the_carried_last_frame_never_has_the_mark(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(nodes, "keep_last_frame", lambda file, frame: seen.append(frame))
    monkeypatch.setattr(nodes, "encode", lambda *a, **k: "libx264")
    monkeypatch.setattr(nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nodes.folder_paths, "get_save_image_path", lambda *a: (str(tmp_path), "FunUI", 1, "", "FunUI"))
    monkeypatch.setattr(nodes.FunUISaveVideo, "hidden", type("H", (), {"extra_pnginfo": None, "prompt": None})(), raising=False)
    nodes.FunUISaveVideo.execute(GREY, settings={"output_save": TEXT})
    assert (seen[0] == nodes.to_frames(GREY[-1:])[-1]).all()


def test_the_biggest_outline_on_a_4k_frame_stays_cheap():
    """The outline grows by small steps on one map: a kernel as large as the mark once used all the memory of a small machine."""
    import resource
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    layer = watermark.build({**TEXT, "watermark_text": "FunUI studio", "watermark_size": 60.0, "watermark_outline": 20.0}, 3840, 2160)
    grown = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / (1024 * 1024 if __import__("sys").platform == "darwin" else 1024)
    assert layer.alpha.shape[0] > 100 and grown < 400            # MB over what the process already held
