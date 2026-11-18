"""These tests build ComfyUI's own H3-Image model and node, which only newer ComfyUI releases have: without them the tests are skipped, not failed."""
import pytest


def _has_h3_image():
    try:
        from comfy.ldm.minimax.model import MiniMaxH3ImageModel  # noqa: F401
        import comfy_extras.nodes_minimax_h3 as n
    except Exception:  # noqa: BLE001
        return False
    return hasattr(n, "MiniMaxH3ImageConditioning")


#: the tests that build ComfyUI's H3-Image model or node (the other tests in this folder need neither)
NEEDS = {"test_features.py", "test_lora.py", "test_h3_image.py::test_the_preset_builds_a_real_graph_and_is_offered"}


def pytest_collection_modifyitems(config, items):
    if _has_h3_image():
        return
    skip = pytest.mark.skip(reason="this ComfyUI has no MiniMax H3-Image model or node")
    for item in items:
        if "minimax_h3_image" in str(item.fspath) and (item.fspath.basename in NEEDS or f"{item.fspath.basename}::{item.name}" in NEEDS):
            item.add_marker(skip)
