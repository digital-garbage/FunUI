"""Direct LTX video-to-video, run for real on CPU: a multi-frame mp4 in the media library, a pipeline saved with the old
first-frame wiring, the per-run values the app sets, the graph FunUI builds from them, then the nodes that make the
latent run in order (Load Media, size, VAE encode, empty sound, concat) and the sampler on a real tiny LTX model.

Only the two VAEs are stand-ins (their weights are not in the repo): they have the real shapes (8x time, 32x space)."""

import json

import pytest
import torch


@pytest.fixture
def clip_id(comfyui, tmp_path, monkeypatch):
    import av
    from core import config, media
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 128, 64, "yuv420p"
        for i in range(41):
            frame = av.VideoFrame.from_ndarray(torch.full((64, 128, 3), i * 6, dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return media.save_upload("clip.mp4", path.read_bytes())["id"]


class StubVae:
    """The LTX video VAE's shape contract: 8x in time (first frame alone), 32x in space, 128 channels."""
    def encode(self, pixels):
        b, f, h, w, _ = pixels.shape if pixels.ndim == 5 else (1, *pixels.shape)
        return torch.randn(1, 128, (f - 1) // 8 + 1, h // 32, w // 32)


class StubAudioVae:
    latent_channels = 8

    class first_stage_model:
        latent_frequency_bins = 16

        @staticmethod
        def num_of_latents_from_frames(frames, rate):
            return max(1, round(frames / rate * 25))


@pytest.fixture
def real_tiny_ltx(comfyui):
    """ComfyUI's own model_base.LTXAV at a few thousand parameters: the real sampling object and packing."""
    import comfy.model_patcher as mp
    import comfy.supported_models as sm
    torch.manual_seed(0)
    config = sm.LTXAV(dict(
        image_model="ltxav", in_channels=128, audio_in_channels=128, cross_attention_dim=32, audio_cross_attention_dim=16,
        attention_head_dim=16, audio_attention_head_dim=8, num_attention_heads=2, audio_num_attention_heads=2,
        caption_channels=48, num_layers=2, caption_proj_before_connector=True))
    config.set_inference_dtype(torch.float32, None)
    base = config.get_model({}, device=torch.device("cpu"))
    for p in base.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=0.05)
        p.requires_grad_(False)
    return mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def _run(prompt, stubs):
    """Run the graph's nodes in order. Loaders are replaced by `stubs` {slot id: outputs tuple}."""
    import nodes as comfy_nodes
    done = {}

    def value(v):
        return done[v[0]][v[1]] if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in done else v

    def run(node_id):
        if node_id in done:
            return
        spec = prompt[node_id]
        for v in spec["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and v[0] in prompt:
                run(v[0])
        if node_id in stubs:
            done[node_id] = stubs[node_id]
            return
        cls = comfy_nodes.NODE_CLASS_MAPPINGS[spec["class_type"]]
        kwargs = {k: value(v) for k, v in spec["inputs"].items()}
        if hasattr(cls, "GET_SCHEMA"):
            out = cls.execute(**kwargs)
            done[node_id] = tuple(out.result)
        else:
            done[node_id] = getattr(cls(), cls.FUNCTION)(**kwargs)
    return run, done


def test_direct_video_to_video_end_to_end(clip_id, real_tiny_ltx, monkeypatch):
    import asyncio
    from pathlib import Path
    import nodes as comfy_nodes
    from core import graph
    from modules.models.ltx.pipeline import presets, upgrade

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(str(Path(__file__).resolve().parents[4]), module_parent="custom_nodes")
    asyncio.run(load())

    import types
    monkeypatch.setattr(comfy_nodes.NODE_CLASS_MAPPINGS["GetImageSize"], "hidden", types.SimpleNamespace(unique_id=None), raising=False)   # no executor here
    slots = next(p["slots"] for p in presets() if p["id"] == "ltx23_video_to_video_direct")
    # saved by the build that wired the first-frame output: brought up to date as a run does
    old = [dict(s, inputs={k: (["source_image", 0] if v == ["source_image", 1] else v) for k, v in s["inputs"].items()}) for s in slots]
    # what the old wiring did, which is what a run showed as "1 frames": the whole-video flag cannot reach output 0
    old_slots, _ = graph.override(old, {"source_image": {"media_id": clip_id, "full": True}})
    old_prompt, _ = graph.build(old_slots)
    run_old, done_old = _run(old_prompt, {"vae": (StubVae(), ""), "audio_vae": (StubAudioVae(), "")})
    run_old("size")
    assert done_old["size"][2] == 1
    slots = upgrade(old)

    prepare = json.dumps({"skip_first": 1, "cut_last": 0, "every_nth": 1, "multiple_of": 32, "fps": 25})
    slots, problems = graph.override(slots, {"source_image": {"media_id": clip_id, "full": True, "prepare": prepare},
                                              "positive": {"text": "a red fox runs"}})
    assert problems == []
    prompt, problems = graph.build(slots)
    unset = ("needs 'ckpt_name'", "needs 'model_name'", "needs 'vae_name'", "needs 'clip_name1'", "is '', which is not one of")
    assert [p for p in problems if not any(u in p for u in unset)] == []

    # the typed prompt is what the sampler's positive input comes from, through the conditioning node and nothing else
    node = prompt["sampler"]["inputs"]["positive"]
    while prompt[node[0]]["class_type"] != "CLIPTextEncode":
        node = next(v for v in prompt[node[0]]["inputs"].values() if isinstance(v, list) and prompt[v[0]]["class_type"] in ("LTXVConditioning", "CLIPTextEncode"))
    assert prompt[node[0]]["inputs"]["text"] == "a red fox runs"
    assert prompt["sampler"]["inputs"]["latent"][0] == "latent" and prompt["sampler"]["inputs"]["denoise"] == 1.0

    vae, audio_vae = StubVae(), StubAudioVae()
    run, done = _run(prompt, {"model": (real_tiny_ltx, ""), "vae": (vae, ""), "audio_vae": (audio_vae, ""), "clip": (None, "")})
    run("latent")

    frames = done["source_image"][1]
    assert frames.shape[0] == 40 and frames.shape[1:3] == (64, 128)                  # the whole video, first frame skipped
    assert done["size"][2] == 40                                                      # the sound is sized from the real count
    video, audio = done["latent"][0]["samples"].unbind()
    assert video.shape == (1, 128, 5, 2, 4)                                           # 40 frames -> 1 + (40-1)//8 latent frames
    assert audio.shape[-1] == 16

    # the sampler runs on it from part-way down the schedule, with the real tiny model
    ctx = torch.randn(1, 5, 48)
    from modules.sampling.sampler.nodes import FunUISampler
    run("sigmas")
    sig = done["sigmas"][0]
    assert [round(float(x), 3) for x in sig] == [0.909, 0.725, 0.422, 0.0]                    # the tail of the distilled schedule
    start = done["latent"][0]
    out = FunUISampler.execute(real_tiny_ltx, [[ctx, {}]], [[ctx, {}]], start, 1, 2, 1.0, "euler", "normal", sigmas=sig).result[0]
    # the prompt reaches the sampler: a different text embedding, same seed and latent, gives a different video AND sound
    other = FunUISampler.execute(real_tiny_ltx, [[torch.randn(1, 5, 48), {}]], [[ctx, {}]], start, 1, 2, 1.0, "euler", "normal", sigmas=sig).result[0]
    assert not torch.allclose(other["samples"].unbind()[0], out["samples"].unbind()[0])
    assert not torch.allclose(other["samples"].unbind()[1], out["samples"].unbind()[1])
    got_video, got_audio = out["samples"].unbind()
    assert got_video.shape == video.shape and got_audio.shape == audio.shape
    assert torch.isfinite(got_video).all() and not torch.allclose(got_video, video)    # it was denoised, not passed through


def test_direct_video_to_video_with_no_video_runs_the_whole_schedule_over_a_grey_clip(comfyui, real_tiny_ltx, monkeypatch):
    """What the app sends when nothing is fed in (text to video): a stand-in clip of the project's size and the whole schedule."""
    import asyncio
    import types
    from pathlib import Path
    import nodes as comfy_nodes
    from core import graph
    from modules.models.ltx.pipeline import DISTILLED_SIGMAS, presets

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(str(Path(__file__).resolve().parents[4]), module_parent="custom_nodes")
    asyncio.run(load())
    monkeypatch.setattr(comfy_nodes.NODE_CLASS_MAPPINGS["GetImageSize"], "hidden", types.SimpleNamespace(unique_id=None), raising=False)
    slots = next(p["slots"] for p in presets() if p["id"] == "ltx23_video_to_video_direct")
    slots, problems = graph.override(slots, {"source_image": {"media_id": "", "blank_width": 128, "blank_height": 64, "blank_frames": 17},
                                             "sigmas": {"sigmas": DISTILLED_SIGMAS}})
    assert problems == []
    prompt, _ = graph.build(slots)
    run, done = _run(prompt, {"model": (real_tiny_ltx, ""), "vae": (StubVae(), ""), "audio_vae": (StubAudioVae(), ""), "clip": (None, "")})
    run("latent")
    assert done["source_image"][1].shape == (17, 64, 128, 3) and done["size"][2] == 17
    video, audio = done["latent"][0]["samples"].unbind()
    assert video.shape == (1, 128, 3, 2, 4)                                           # 17 frames -> 1 + 16//8 latent frames
    run("sigmas")
    assert round(float(done["sigmas"][0][0]), 3) == 1.0 and len(done["sigmas"][0]) == 9
