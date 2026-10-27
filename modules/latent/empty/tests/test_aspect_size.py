"""FunUIAspectSize: the picture's shape at the project's pixel budget."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy_api."""


def _size(picture_w, picture_h, budget_w, budget_h):
    from modules.latent.empty.nodes import FunUIAspectSize
    return tuple(FunUIAspectSize.execute(torch.zeros(1, picture_h, picture_w, 3), budget_w, budget_h).result)


def test_the_picture_gives_the_shape_and_the_budget_the_scale():
    w, h = _size(1000, 2000, 768, 512)                     # a portrait photo against a landscape budget
    assert h > w and (w % 32, h % 32) == (0, 0)
    assert abs(w / h - 0.5) < 0.05 and abs(w * h - 768 * 512) / (768 * 512) < 0.1


def test_a_picture_with_the_budgets_own_shape_gives_the_budget():
    assert _size(512, 341, 768, 512) == (768, 512)         # the plain grey stand-in is the same shape as the budget
    assert _size(768, 512, 768, 512) == (768, 512)


def test_a_very_wide_picture_never_gives_a_side_of_zero():
    w, h = _size(4000, 40, 768, 512)
    assert w >= 32 and h >= 32
