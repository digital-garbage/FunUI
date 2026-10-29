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


def test_the_basis_says_what_the_budget_means():
    from modules.latent.empty.nodes import FunUIAspectSize
    wide = torch.zeros(1, 1080, 1920, 3)
    run = lambda w, h, basis: tuple(FunUIAspectSize.execute(wide, w, h, 32, basis).result)
    assert run(1024, 1024, "area") == (1376, 768)          # the original rule: the pixel count stays
    assert run(1024, 1024, "short") == (1824, 1024)        # the shorter side is the number typed
    assert run(1024, 1024, "long") == (1024, 576)          # the longer side is the number typed
    assert run(1344, 768, "short") == run(768, 1344, "short") == (1376, 768)      # the smaller number, whichever box it is in
    assert run(1344, 768, "long") == run(768, 1344, "long") == (1344, 768)
    portrait = tuple(FunUIAspectSize.execute(torch.zeros(1, 1920, 1080, 3), 1024, 1024, 32, "short").result)
    assert portrait == (1024, 1824)                       # short side is the width of a portrait picture
    assert tuple(FunUIAspectSize.execute(wide, 1024, 1024).result) == (1376, 768)  # an old graph that never heard of basis


def test_an_unknown_basis_keeps_the_original_rule():
    from modules.latent.empty.nodes import fit_size
    assert fit_size(1920, 1080, 1024, 1024, "diagonal") == fit_size(1920, 1080, 1024, 1024, "area")


def test_old_aspect_slots_get_the_basis_role_and_keep_their_rule():
    from modules.latent.empty import upgrade
    old = [{"id": "aspect", "node": "FunUIAspectSize", "roles": [{"at": "project.video", "input": "width"}], "inputs": {"width": 768}}]
    new = upgrade(old)
    assert {"at": "project.resolution_basis", "input": "basis"} in new[0]["roles"] and new[0]["inputs"]["basis"] == "area"
    assert upgrade(new) == new and upgrade([{"id": "x", "node": "Other"}]) == [{"id": "x", "node": "Other"}]
