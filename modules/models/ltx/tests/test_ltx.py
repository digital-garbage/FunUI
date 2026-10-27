"""The LTX module: recognising the model, the joint latent, the decode, and its pipelines."""

import asyncio
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def registered(comfyui):
    import nodes as comfy_nodes

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(str(Path(__file__).resolve().parents[4]), module_parent="custom_nodes")

    asyncio.run(load())
    return comfy_nodes


def _ltx():
    from modules.models import ltx
    return ltx


def test_the_audio_video_architecture_is_recognised_by_its_key_names_alone(comfyui):
    m = _ltx()
    assert m.detect({"audio_adaln_single.linear.weight"})
    assert m.detect({"model.diffusion_model.audio_adaln_single.linear.weight"})
    assert not m.detect({"adaln_single.linear.weight"})                      # video-only LTX
    assert not m.detect({"diffusion_model.audio_adaln_single.linear.weight"})  # not a prefix ComfyUI reads
    assert not m.detect({"video_patch_proj.weight", "audio_patch_proj.weight"})  # that is H3
    assert "audio_stream" in m.probe_traits({"audio_adaln_single.linear.weight"})
    assert m.probe_traits({"x"}) == []


def test_it_does_not_claim_block_hooks_it_has_no_modules_for(comfyui, monkeypatch):
    m = _ltx()
    monkeypatch.setattr(m, "has_block", lambda model, cls: True)
    assert "dit_block_hooks" not in m.traits(object())


@pytest.mark.parametrize("asked, got", [(1, 1), (9, 9), (10, 17), (121, 121), (122, 129)])
def test_a_length_off_the_8k_plus_1_grid_is_rounded_up(comfyui, asked, got):
    assert _ltx().frames_for(asked) == got


class _AudioVae:
    latent_channels = 8
    first_stage_model = types.SimpleNamespace(
        latent_frequency_bins=16, output_sample_rate=24000,
        num_of_latents_from_frames=lambda frames, rate: int(frames / rate * 25))

    def decode(self, latent):
        import torch
        return torch.zeros(latent.shape[0], 100, 2)


def test_the_latent_is_video_and_audio_sized_from_the_audio_vae(comfyui, monkeypatch):
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)
    out = m.empty_latent(object(), width=768, height=512, length=121, batch_size=1,
                         audio_vae=_AudioVae(), frame_rate=25.0)
    video, audio = out["samples"].unbind()
    assert tuple(video.shape) == (1, 128, 16, 16, 24)
    assert tuple(audio.shape) == (1, 8, 121, 16)
    assert out["downscale_ratio_spacial"] == 32 and out["downscale_ratio_temporal"] == 8


def test_no_audio_vae_is_refused_not_made_silent(comfyui, monkeypatch):
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)
    with pytest.raises(RuntimeError, match="audio VAE"):
        m.empty_latent(object(), 768, 512, 121, 1)


def test_another_model_is_not_claimed(comfyui, monkeypatch):
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: False)
    assert m.empty_latent(object(), 768, 512, 121, 1) is None
    assert m.decode(object(), model=object()) is None


def test_decode_gives_a_picture_and_a_waveform(comfyui, monkeypatch):
    import torch
    from comfy.nested_tensor import NestedTensor
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)

    class Vae:
        def decode(self, latent):
            return torch.zeros(1, 9, 64, 64, 3)

    latent = NestedTensor((torch.zeros(1, 128, 2, 2, 2), torch.zeros(1, 8, 5, 16)))
    images, audio = m.decode(latent, model=object(), vae=Vae(), audio_vae=_AudioVae())
    assert tuple(images.shape) == (9, 64, 64, 3)
    assert audio["sample_rate"] == 24000 and audio["waveform"].shape[0] == 1
    with pytest.raises(RuntimeError, match="audio VAE"):
        m.decode(latent, model=object(), vae=Vae(), audio_vae=None)


def test_the_empty_latent_node_passes_the_audio_vae_to_a_provider_that_asks_and_not_to_one_that_does_not(registered):
    from core import registry
    from modules.latent.empty.nodes import FunUIEmptyLatent
    seen = {}

    def wants(model, width, height, length, batch_size, audio_vae=None, frame_rate=25.0):
        seen["wants"] = (audio_vae, frame_rate)
        return {"samples": 1}

    def plain(model, width, height, length, batch_size):
        seen["plain"] = True
        return None

    class Reg:
        def providers(self, cap):
            return [(types.SimpleNamespace(id="x"), plain), (types.SimpleNamespace(id="y"), wants)]
    import unittest.mock as mock
    with mock.patch.object(registry, "current", lambda rescan=False: Reg()):
        FunUIEmptyLatent.execute(object(), 64, 64, 1, 1, audio_vae="A", frame_rate=30.0)
    assert seen == {"plain": True, "wants": ("A", 30.0)}


@pytest.mark.parametrize("pid", ["ltx23_video_to_video_direct", "ltx23_image_to_video_references"])
def test_the_pipelines_build_a_real_graph(registered, pid):
    from core import graph
    from modules.models.ltx.pipeline import presets

    slots = next(p["slots"] for p in presets() if p["id"] == pid)
    ids = [s["id"] for s in slots]
    assert len(ids) == len(set(ids))
    for slot in slots:
        for value in slot["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in ids
    _prompt, problems = graph.build(slots)
    unset = ("needs 'ckpt_name'", "needs 'text_encoder'", "needs 'model_name'", "needs 'vae_name'", "needs 'clip_name1'", "is '', which is not one of")   # a file nobody has picked yet
    structural = [p for p in problems if not any(u in p for u in unset)]
    assert structural == [], structural


def test_the_presets_are_offered_through_the_registry(registered):
    from core import routes
    found = {preset["id"] for _spec, make in routes.modules().providers("pipeline_presets") for preset in make()}
    assert {p["id"] for p in _ltx().pipeline.presets()} == {"ltx23_image_to_video_references", "ltx23_video_to_video_direct"} <= found


def test_the_picture_is_found_in_both_shapes_a_sampling_latent_takes(comfyui):
    import torch
    from comfy.nested_tensor import NestedTensor
    m = _ltx()
    video, audio = torch.randn(1, 128, 2, 2, 3), torch.randn(1, 8, 5, 16)
    found, rebuild = m.PROVIDES["video_stream"](NestedTensor((video, audio)))
    assert torch.equal(found, video)
    packed = torch.cat([video.reshape(1, 1, -1), audio.reshape(1, 1, -1)], dim=-1)
    found, rebuild = m.PROVIDES["video_stream"](packed, {"latent_shapes": [video.shape, audio.shape]})
    assert torch.equal(found, video)
    assert torch.equal(rebuild(found * 2)[..., video.numel():], packed[..., video.numel():])
    assert m.PROVIDES["video_stream"](torch.randn(1, 4, 8, 8)) is None


def test_a_tiled_decode_is_asked_the_way_comfy_asks_it(comfyui, monkeypatch):
    import torch
    from comfy.nested_tensor import NestedTensor
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)
    seen = {}

    class Vae:
        def temporal_compression_decode(self): return 8
        def spacial_compression_decode(self): return 32
        def decode_tiled(self, latent, **kw):
            seen.update(kw)
            return torch.zeros(1, 9, 64, 64, 3)

    latent = NestedTensor((torch.zeros(1, 128, 2, 2, 2), torch.zeros(1, 8, 5, 16)))
    m.decode(latent, model=object(), vae=Vae(), audio_vae=_AudioVae(), tile_size=512)
    assert seen == {"tile_x": 16, "tile_y": 16, "overlap": 4, "tile_t": 8, "overlap_t": 1}


def test_a_size_that_is_not_a_multiple_of_32_is_said(comfyui, monkeypatch):
    import torch
    from core import log
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)
    said = []
    monkeypatch.setattr(log, "once", lambda key, level, source, msg: said.append(msg))
    m.empty_latent(object(), 816, 512, 9, 1, audio_vae=_AudioVae(), frame_rate=25.0)
    assert any("800x512" in m_ for m_ in said)
    assert not any("tested" in m_ for m_ in said)                        # a working model needs no notice


def test_ltx_presets_are_the_two_that_were_tested(registered):
    from core import routes
    titles = [p["title"] for _s, make in routes.modules().providers("pipeline_presets") for p in make() if p["id"].startswith("ltx")]
    assert len(titles) == 2 and not any("UNTESTED" in t for t in titles)


@pytest.mark.parametrize("pid", ["ltx23_image_to_video_references", "ltx23_video_to_video_direct"])
def test_what_a_run_puts_on_the_start_slot_is_accepted(registered, pid):
    """The app sets the picked video, its whole-video flag and its preparation per run: none of them may be a wired input,
    and what takes the picture must be wired to the frames output (a wired input cannot be overridden)."""
    from core import graph
    from modules.models.ltx.pipeline import presets

    slots = next(p["slots"] for p in presets() if p["id"] == pid)
    _slots, problems = graph.override(slots, {"source_image": {"media_id": "abc", "full": True, "prepare": "{}", "blank_width": 768, "blank_height": 512, "blank_frames": 97}})
    assert problems == []
    users = [(s["id"], k) for s in slots for k, v in s["inputs"].items() if isinstance(v, list) and v[0] == "source_image"]
    assert users and all(slots_v == 1 for slots_v in [next(s for s in slots if s["id"] == i)["inputs"][k][1] for i, k in users])


def test_a_pipeline_saved_with_the_old_first_frame_wiring_is_brought_up_to_date():
    from modules.models.ltx.pipeline import presets, upgrade
    slots = next(p["slots"] for p in presets() if p["id"] == "ltx23_video_to_video_direct")
    old = [dict(s, inputs={k: (["source_image", 0] if v == ["source_image", 1] else v) for k, v in s["inputs"].items()}) for s in slots]
    assert old != slots
    assert upgrade(old) == slots
    plain = [{"id": "source_image", "node": "FunUILoadMedia", "roles": [{"at": "assets.source_image"}], "inputs": {}}, {"id": "a", "node": "X", "inputs": {"image": ["source_image", 0]}}]
    assert upgrade(plain) == plain


def test_h3_only_features_are_not_offered_on_ltx(comfyui):
    """Shot camera writes [Shot N] prompts and SLA is H3's sparse attention: the traits say so, and LTX has neither."""
    from core import registry, traits
    from modules.models import ltx
    from modules.models.minimax_h3 import H3_ONLY
    spec = registry.scan().specs["conditioning_shot_camera"]
    assert "shot_prompts" in spec.requires and set(H3_ONLY) == {"shot_prompts", "sparse_attention"}
    have = ltx.probe_traits({"audio_adaln_single.linear.weight"})
    assert traits.missing_for(spec, have) == ["shot_prompts"]
    assert not set(H3_ONLY) & set(have)


def test_a_direct_v2v_pipeline_saved_before_video_only_still_ignores_pictures_and_the_prompt_switch():
    from modules.models.ltx.pipeline import presets, upgrade
    slots = next(p["slots"] for p in presets() if p["id"] == "ltx23_video_to_video_direct")
    old = [dict(s, roles=[{k: v for k, v in r.items() if k != "video_only"} for r in s.get("roles") or []]) if s["id"] == "source_image" else s for s in slots]
    assert not any(r.get("video_only") for s in old for r in s.get("roles") or [])
    assert upgrade(old) == slots
    i2v = next(p["slots"] for p in presets() if p["id"] == "ltx23_image_to_video_references")
    assert not any(r.get("video_only") for s in upgrade(i2v) for r in s.get("roles") or [])          # i2v keeps taking pictures


def test_image_to_video_from_references_takes_its_start_from_the_first_reference_and_has_no_other_picture_slot():
    from modules.models.ltx.pipeline import presets
    slots = next(p["slots"] for p in presets() if p["id"] == "ltx23_image_to_video_references")
    roles = [r["at"] for s in slots for r in s.get("roles") or []]
    assert "assets.reference_1" in roles and "assets.source_image" in roles      # the start from References; the timeline picture gives the shape
    aspect = next(s for s in slots if s["id"] == "aspect")
    assert aspect["inputs"]["image"] == ["aspect_source", 0]
    latent = next(s for s in slots if s["id"] == "latent")
    assert latent["inputs"]["width"] == ["aspect", 0] and latent["inputs"]["height"] == ["aspect", 1]
    assert not any(s["node"] == "FunUIIdentityTransfer" for s in slots)
    fallback = next(r["text_fallback"] for s in slots for r in s.get("roles") or [] if r.get("text_fallback"))
    assert {"slot": "anchor", "input": "bypass", "value": True} in fallback["set"]      # nothing in References: the anchor is bypassed
    anchor = next(s for s in slots if s["id"] == "anchor")
    assert anchor["inputs"]["image"] == ["source_image", 1]


def test_an_image_to_video_pipeline_saved_before_the_aspect_source_gets_it():
    from core import graph
    from modules.models.ltx.pipeline import presets, upgrade
    now = next(p["slots"] for p in presets() if p["id"] == "ltx23_image_to_video_references")
    ids = [x["id"] for x in now]
    old = [x for x in now if x["id"] not in ("aspect_source", "aspect")]
    latent = next(x for x in old if x["id"] == "latent")
    latent["inputs"] = {**latent["inputs"], "width": 1024, "height": 576}
    latent["roles"] = [{"at": "project.video", "input": "width", "label": "Width"}, {"at": "project.video", "input": "height", "label": "Height"}, *latent["roles"]]
    fixed = upgrade(old)
    assert [x["id"] for x in fixed] == ids
    aspect = next(x for x in fixed if x["id"] == "aspect")
    assert (aspect["inputs"]["width"], aspect["inputs"]["height"]) == (1024, 576)           # what was typed stays the budget
    latent = next(x for x in fixed if x["id"] == "latent")
    assert latent["inputs"]["width"] == ["aspect", 0] and not any(r["input"] in ("width", "height") for r in latent["roles"])
    assert upgrade(fixed) == fixed                                                           # and it is done once
    assert upgrade(now) == now


def test_the_block_count_is_read_off_the_header(comfyui):
    m = _ltx()
    keys = {f"model.diffusion_model.transformer_blocks.{i}.attn1.to_q.weight" for i in range(48)} | {"model.diffusion_model.audio_adaln_single.linear.weight"}
    assert m.probe_blocks(keys) == 48
    assert m.probe_blocks({"audio_adaln_single.linear.weight"}) is None
    assert m.probe_blocks({"transformer_blocks.0.x", "transformer_blocks.9.x"}) == 2          # counts distinct numbers, as the file names them


def test_the_h3_block_count_ignores_its_token_refiner(comfyui):
    from importlib import import_module
    h3 = import_module("modules.models.minimax_h3")
    keys = {f"blocks.{i}.attn.q.weight" for i in range(50)} | {f"token_refiner.blocks.{i}.x" for i in range(2)}
    assert h3.probe_blocks(keys) == 50
