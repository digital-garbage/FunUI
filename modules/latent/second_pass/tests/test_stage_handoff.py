import asyncio

import torch


def _run(coro_or_value):
    return coro_or_value


def test_stage_size_snaps_to_the_grid_and_never_grows():
    from ..nodes import FunUIStageSize
    w, h = FunUIStageSize.execute(640, 352, 0.7)
    assert (w, h) == (448, 256)
    assert all(v % 32 == 0 for v in (w, h))
    assert tuple(FunUIStageSize.execute(96, 96, 1.0)) == (96, 96)
    assert tuple(FunUIStageSize.execute(64, 64, 0.25)) == (64, 64)        # never below 64


def test_resample_lands_on_the_target_grid_without_an_upscaler_file():
    from ..nodes import FunUILatentResample
    video = torch.randn(1, 24, 5, 16, 28)            # 256x448 pixels
    out = FunUILatentResample.execute({"samples": video}, None, "upscale", 1.5, None, None, 640, 352)
    assert out.result[0]["samples"].shape == (1, 24, 5, 22, 40)           # 352x640
    assert "upscale" in out.result[3]


def test_resample_hands_the_audio_part_through_untouched():
    from comfy.nested_tensor import NestedTensor
    from ..nodes import FunUILatentResample
    video, audio = torch.randn(1, 24, 5, 8, 8), torch.randn(1, 32, 2, 10)
    out = FunUILatentResample.execute({"samples": NestedTensor((video, audio))}, None, "upscale", 2.0, None, None, 256, 256)
    v, a = out.result[0]["samples"].unbind()
    assert v.shape == (1, 24, 5, 16, 16) and torch.equal(a, audio)
