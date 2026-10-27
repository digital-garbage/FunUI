"""Guide frames: the real LTX guide logic with a stand-in picture VAE, and a crop that undoes it."""

import json

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


class _Vae:
    downscale_index_formula = (8, 32, 32)

    def encode(self, pixels):          # [frames, H, W, 3] -> [1, 128, (frames-1)//8+1, H/32, W/32]
        frames, h, w = pixels.shape[0], pixels.shape[1], pixels.shape[2]
        return torch.ones(1, 128, (frames - 1) // 8 + 1, h // 32, w // 32) * 0.5


def _inputs(length=4):
    cond = [[torch.zeros(1, 3, 8), {}]]
    return cond, [[torch.zeros(1, 3, 8), {}]], {"samples": torch.zeros(1, 128, length, 2, 2)}


def _picture(monkeypatch, got=None):
    from modules.conditioning.guide_frames import nodes
    monkeypatch.setattr(nodes, "read", lambda mid: None if mid == "gone" else torch.rand(1, 64, 64, 3))
    return nodes


def test_no_list_changes_nothing(monkeypatch):
    nodes = _picture(monkeypatch)
    pos, neg, lat = _inputs()
    out = nodes.FunUIGuideFrames.execute(pos, neg, _Vae(), lat, "").result
    assert out[0] is pos and out[1] is neg and out[2] is lat


def test_each_picture_adds_a_guide_frame_and_the_crop_takes_them_off_again(monkeypatch):
    from comfy_extras.nodes_lt import LTXVCropGuides
    nodes = _picture(monkeypatch)
    pos, neg, lat = _inputs(length=5)
    frames = json.dumps([{"media_id": "a", "at": 0.5, "strength": 0.25}, {"media_id": "b", "at": 0.0, "strength": 0.4}])
    p, n, latent = nodes.FunUIGuideFrames.execute(pos, neg, _Vae(), lat, frames).result
    assert latent["samples"].shape[2] == 5 + 2                       # one latent frame per picture, at the tail
    assert p[0][1].get("keyframe_idxs") is not None
    mask = latent["noise_mask"]
    assert abs(float(mask[0, 0, 5, 0, 0]) - 0.75) < 1e-6 and abs(float(mask[0, 0, 6, 0, 0]) - 0.6) < 1e-6      # 1 - strength
    cropped = LTXVCropGuides.execute(p, n, latent).result
    assert cropped[2]["samples"].shape == lat["samples"].shape
    assert cropped[0][0][1].get("keyframe_idxs") is None


def test_a_picture_gone_from_the_library_is_left_out_and_bad_lists_are_refused(monkeypatch):
    nodes = _picture(monkeypatch)
    pos, neg, lat = _inputs()
    p, n, latent = nodes.FunUIGuideFrames.execute(
        pos, neg, _Vae(), lat, json.dumps([{"media_id": "gone"}, {"media_id": "a"}])).result
    assert latent["samples"].shape[2] == 4 + 1
    for bad in ("not json", '{"a": 1}', '[{"nothing": 1}]'):
        with pytest.raises(RuntimeError):
            nodes.FunUIGuideFrames.execute(pos, neg, _Vae(), lat, bad)


def test_a_nan_position_is_refused_not_crashed():
    from modules.conditioning.guide_frames import nodes
    assert isinstance(nodes.parse('[{"media_id": "a", "at": NaN}]'), str)
