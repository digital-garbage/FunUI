"""The MiniMax H3-Image module: recognising the file, and a preset that builds into a real graph."""

import asyncio
from pathlib import Path

import pytest

IMAGE = {"video_patch_proj.weight", "condition_proj.weight", "blocks.0.attn.qkv_proj.weight"}


@pytest.fixture(scope="module")
def registered(comfyui):
    import nodes as comfy_nodes

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(str(Path(__file__).resolve().parents[4]), module_parent="custom_nodes")

    asyncio.run(load())
    return comfy_nodes


def test_the_image_file_is_recognised_and_the_video_model_is_not(comfyui):
    from modules.models import minimax_h3, minimax_h3_image as m

    assert m.detect(IMAGE)
    assert m.detect({f"model.diffusion_model.{k}" for k in IMAGE})                 # a full checkpoint's prefix
    assert not m.detect(IMAGE | {"audio_patch_proj.weight"})                       # H3 video has an audio stream
    assert not m.detect({"video_patch_proj.weight"})
    assert not m.detect({"input_blocks.0.0.weight"})
    video = IMAGE | {"audio_patch_proj.weight"}
    assert minimax_h3.detect(video) and not minimax_h3.detect(IMAGE)               # and the video module leaves the image file alone


def test_traits_before_loading(comfyui):
    from modules.models import minimax_h3_image as m

    assert "temporal_latent" in m.probe_traits(IMAGE)
    assert "audio_stream" not in m.probe_traits(IMAGE)
    assert m.probe_traits({"video_patch_proj.weight"}) == []


def test_the_preset_builds_a_real_graph_and_is_offered(registered):
    from core import graph, routes
    from modules.models.minimax_h3_image import MAX_REFERENCES, text_to_image

    slots = text_to_image()
    _prompt, problems = graph.build(slots)
    unset_files = ("needs 'model_name'", "needs 'clip_name", "needs 'vae_name'")
    assert [p for p in problems if not any(x in p for x in unset_files)] == []
    assert "minimax_h3_image" in {p["id"] for _s, make in routes.modules().providers("pipeline_presets") for p in make()}
    refs = [s for s in slots if s["id"].startswith("ref_source_")]
    assert len(refs) == MAX_REFERENCES
    assert refs[0]["roles"][0]["wireTo"] == {"slot": "cond", "input": "images.image_1"}     # image_1 is the picture to edit
    sampler = next(s for s in slots if s["id"] == "sampler")["inputs"]
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == (6, 1.0, "euler", "simple")


def test_it_saves_a_picture_and_offers_no_length_or_fps(registered):
    from core import graph, routes
    from modules.models.minimax_h3_image import PROJECT, presets, text_to_image

    slots = {s["id"]: s for s in text_to_image()}
    assert not [r for s in slots.values() for r in s.get("roles") or [] if r.get("drives") in ("frames", "fps")]
    assert slots["save"]["node"] == "SaveImage" and "hold" not in slots          # a PNG, not a clip
    assert PROJECT == {"num_frames_per_scene": 1, "frame_rate": 1} and presets()[0]["project"] == PROJECT
    prompt, _problems = graph.build(list(slots.values()))
    assert [n["class_type"] for n in prompt.values()].count("SaveImage") == 1
    served = [p for _s, make in routes.modules().providers("pipeline_presets") for p in make() if p["id"] == "minimax_h3_image"]
    assert served and served[0]["project"] == PROJECT
