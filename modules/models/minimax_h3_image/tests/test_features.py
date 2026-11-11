"""REINS and the Attention explorer on a real (tiny, random-weight) H3-Image forward: they run, and they do not touch a sound that is not there."""

import random

import pytest
import torch


@pytest.fixture(autouse=True)
def root(comfyui, tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-1")


class TinyImage:
    """ComfyUI's own MiniMaxH3ImageModel, four blocks wide, on the CPU."""

    def __init__(self, blocks=4):
        import comfy.ops
        from comfy.latent_formats import MiniMaxH3Video
        from comfy.ldm.minimax.model import MiniMaxH3ImageModel
        from comfy.model_patcher import ModelPatcher

        torch.manual_seed(0)
        dm = MiniMaxH3ImageModel(
            pdd_sigmas=[1.0, 0.5, 0.0], hidden_size=64, num_layers=blocks, token_refiner_num_layers=1,
            num_attention_heads=2, attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32,
            time_embed_hidden_size=64, time_embed_dim=32, rope_inv_freq_len=4, dtype=torch.float32,
            operations=comfy.ops.manual_cast)
        for p in dm.parameters():
            torch.nn.init.normal_(p, std=0.05)
            p.requires_grad_(False)
        dm.rope.inv_freq.copy_(torch.rand(4))

        class Config:
            latent_format = MiniMaxH3Video

        class Base(torch.nn.Module):
            pass

        base = Base()
        base.diffusion_model, base.model_config = dm, Config()
        self.patcher = ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        self.latent, self.context = torch.randn(1, 24, 1, 4, 4), torch.randn(1, 5, 48)

    def run(self, patcher=None, sigma=0.5):
        import comfy.patcher_extension as ext
        patcher = patcher or self.patcher
        to = dict(patcher.model_options.get("transformer_options", {}))
        to["wrappers"] = ext.copy_nested_dicts(to.get("wrappers", {}))
        ext.merge_nested_dicts(to["wrappers"], patcher.wrappers, copy_dict1=False)
        to.setdefault("sample_sigmas", torch.tensor([1.0, sigma, 0.0]))
        to["sigmas"] = torch.tensor([sigma])
        with torch.inference_mode():
            return patcher.model.diffusion_model(self.latent, torch.tensor([sigma * 1000.0]), self.context, transformer_options=to)


def _load(tiny, **settings):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    return FunUILoadModifiers.execute(tiny.patcher, {"taste": {"key": "fox"}, **settings}).result


def _through(patched, fn):
    from comfy.patcher_extension import WrappersMP
    call = fn
    for ws in reversed([w for ws in patched.wrappers.get(WrappersMP.OUTER_SAMPLE, {}).values() for w in ws]):
        call = (lambda w, inner: (lambda: w(lambda: inner())))(ws, call)
    return call()


def test_the_loaded_model_announces_what_the_two_features_need():
    from core import traits
    from modules.models import minimax_h3, minimax_h3_image as m

    tiny = TinyImage()
    assert m.is_image_model(tiny.patcher) and not minimax_h3.is_h3(tiny.patcher)
    found = set(traits.announced(tiny.patcher)) if hasattr(traits, "announced") else set(m.traits(tiny.patcher))
    assert {"block_rows", "block_scope"} <= found and "audio_stream" not in found


def test_reins_captures_the_picture_rows_and_a_learned_direction_moves_the_picture():
    from modules.system.taste import store
    tiny = TinyImage()
    base = tiny.run()
    patched, status = _load(tiny, reins={"enabled": True, "blocks": "3", "strength": 0.5})
    assert "reins" in status.splitlines()[0]
    _through(patched, lambda: tiny.run(patched, sigma=0.5))        # the last step: captured
    assert store.rate("run-1", "liked")["recorded"] == ["reins"]
    hidden = tiny.patcher.model.diffusion_model.blocks[0].attn.out_proj.weight.shape[0]
    liked = torch.zeros(hidden)
    liked[0] = 1.0
    for i, (vec, rating) in enumerate([(liked, "liked"), (-liked, "disliked")] * 2):
        store.capture("fox", "reins", {3: vec}, prompt_id=f"t{i}")
        store.rate(f"t{i}", rating)
    moved, _ = _load(tiny, reins={"enabled": True, "blocks": "3", "strength": 0.5})
    assert not torch.allclose(_through(moved, lambda: tiny.run(moved)), base)
    quiet, _ = _load(tiny, reins={"enabled": True, "blocks": "3", "strength": 0.0})
    assert torch.equal(_through(quiet, lambda: tiny.run(quiet)), base)


def test_the_attention_explorer_changes_the_picture_with_no_sound_and_no_clip_regions(monkeypatch):
    from core import log
    from modules.sampling import attention_bandit as m
    tiny = TinyImage()
    base = tiny.run()
    patched, _ = _load(tiny, attention_bandit={"enabled": True, "mode": "manual", "range": 0.4, "blocks": "0-3", "sound": False})
    monkeypatch.setattr(m, "draw", lambda table, rng=random, **kw: [2] * len(table))
    log.new_run()
    out = _through(patched, lambda: tiny.run(patched))
    assert not torch.equal(out, base), "the explorer ran and changed nothing"
    said = " ".join(e["message"] for e in log.history())
    assert "clip regions" not in said and "sound could not be told apart" not in said and "changed nothing" not in said
    monkeypatch.setattr(m, "draw", lambda table, rng=random, **kw: [m.STAY] * len(table))
    plain, _ = _load(tiny, attention_bandit={"enabled": True, "blocks": "0-3"})
    assert torch.equal(_through(plain, lambda: tiny.run(plain)), base)


def test_what_was_rated_with_a_sound_stream_does_not_teach_a_model_without_one():
    from modules.sampling import attention_bandit as m
    tiny = TinyImage()
    sig = []
    orig = m.make_plan
    m.make_plan = lambda rows, cond, signature, *a, **k: (sig.append(list(signature)), orig(rows, cond, signature, *a, **k))[1]
    try:
        patched, _ = _load(tiny, attention_bandit={"enabled": True, "blocks": "0-3"})
        _through(patched, lambda: tiny.run(patched))
    finally:
        m.make_plan = orig
    assert sig and sig[0][-1] == 2
