"""Text encoder adapters: a donor encoder (Gemma 4 here, made of plain objects) writes in the space of H3's own."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """The loader module imports comfy."""


D, T, OUT, LAYERS = 6, 5, 8, 4


def _file(tmp_path, monkeypatch, meta=None, d=D):
    from safetensors.torch import save_file
    import folder_paths
    g = torch.Generator().manual_seed(1)
    tensors = {"R": torch.randn(d, OUT, generator=g), "scale": torch.tensor([0.5]), "donor_mean": torch.randn(d, generator=g), "target_mean": torch.randn(OUT, generator=g)}
    save_file(tensors, str(tmp_path / "key.safetensors"), metadata=meta or {"donor_layer": "-2"})
    monkeypatch.setattr(folder_paths, "get_full_path", lambda kind, name: str(tmp_path / name))
    return tensors


class _Tok:
    has_start_token = True


class _Tokenizer:
    clip = "gemma4"
    gemma4 = _Tok()

    def __init__(self):
        self.seen = []

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        self.seen.append(kwargs)
        return {"gemma4": [[(1, 1.0)]]}


class _Model:
    def __init__(self, x):
        self.x = x

    def encode_token_weights(self, pairs):
        return self.x, None, {"attention_mask": torch.ones(1, self.x.shape[-2], dtype=torch.long)}


class _Clip:
    def __init__(self, x):
        self.tokenizer, self.cond_stage_model = _Tokenizer(), _Model(x)


def test_the_donor_layer_is_mapped_into_the_target_space_and_its_start_token_dropped(tmp_path, monkeypatch):
    from modules.loaders.clip import adapter
    t = _file(tmp_path, monkeypatch)
    x = torch.randn(1, LAYERS, T, D)
    clip = _Clip(x)
    adapter.attach(clip, adapter.load_key("key.safetensors"))
    cond, pooled, extra = clip.cond_stage_model.encode_token_weights([[(1, 1.0)]])
    want = ((x[:, -2] - t["donor_mean"]) @ t["R"]) * 0.5 + t["target_mean"]
    assert cond.shape == (1, T - 1, OUT) and torch.allclose(cond, want[:, 1:], atol=1e-5)
    assert extra["attention_mask"].shape[-1] == T - 1 and extra["minimax_token_tags"].tolist() == [1] * (T - 1)
    assert pooled is None


def test_strength_moves_the_output_from_the_target_mean(tmp_path, monkeypatch):
    from modules.loaders.clip import adapter
    t = _file(tmp_path, monkeypatch)
    x = torch.randn(1, LAYERS, T, D)
    key = adapter.load_key("key.safetensors")
    assert torch.allclose(adapter.project(x, key, 0.0), t["target_mean"].expand(1, T, OUT))
    one, two = adapter.project(x, key, 1.0), adapter.project(x, key, 2.0)
    assert torch.allclose(two - t["target_mean"], (one - t["target_mean"]) * 2, atol=1e-5)


def test_several_layers_are_joined_and_a_single_layer_encoder_needs_a_last_layer_key(tmp_path, monkeypatch):
    from modules.loaders.clip import adapter
    _file(tmp_path, monkeypatch, {"donor_layers": "[-1, -2]"}, d=2 * D)
    key = adapter.load_key("key.safetensors")
    x = torch.randn(1, LAYERS, T, D)
    assert adapter.project(x, key).shape == (1, T, OUT)
    with pytest.raises(RuntimeError, match="only its last one"):
        adapter.project(torch.randn(1, T, 2 * D), key)
    _file(tmp_path, monkeypatch, {"donor_layer": "-1"})
    assert adapter.project(torch.randn(1, T, D), adapter.load_key("key.safetensors")).shape == (1, T, OUT)


def test_a_key_for_another_encoder_is_refused_by_name(tmp_path, monkeypatch):
    from modules.loaders.clip import adapter
    _file(tmp_path, monkeypatch, {"donor_layer": "-1", "donor_model": "Gemma-4-12B"})
    with pytest.raises(RuntimeError, match="6 wide .*Gemma-4-12B.*is 9 wide"):
        adapter.project(torch.randn(1, T, 9), adapter.load_key("key.safetensors"))


def test_the_words_go_in_plain_and_the_pictures_of_the_request_stay_out(tmp_path, monkeypatch):
    from modules.loaders.clip import adapter
    _file(tmp_path, monkeypatch)
    clip = _Clip(torch.randn(1, LAYERS, T, D))
    adapter.attach(clip, adapter.load_key("key.safetensors"))
    clip.tokenizer.tokenize_with_weights("a dog", False, images=[object()], minimax_ref_items=[1], picture_label="Picture %d: ")
    assert clip.tokenizer.seen == [{"llama_template": "{}"}]


def test_the_loader_applies_the_adapter_it_is_given(tmp_path, monkeypatch):
    import comfy.sd
    import folder_paths
    from modules.loaders.clip import nodes
    _file(tmp_path, monkeypatch)
    clip = _Clip(torch.randn(1, LAYERS, T, D))
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda kind: [])
    monkeypatch.setattr(comfy.sd, "load_clip", lambda **kw: clip)
    out = nodes.FunUICLIPLoader.execute(clip_name1="e.safetensors", type="minimax", adapter="key.safetensors", adapter_strength=1.0)
    assert out.args[0].funui_adapter == "key.safetensors" and "adapter key.safetensors" in out.args[1]
    plain = _Clip(torch.randn(1, LAYERS, T, D))
    monkeypatch.setattr(comfy.sd, "load_clip", lambda **kw: plain)
    nodes.FunUICLIPLoader.execute(clip_name1="e.safetensors", type="minimax")
    assert not hasattr(plain, "funui_adapter")
