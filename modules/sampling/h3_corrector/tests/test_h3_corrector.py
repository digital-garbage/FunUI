import pytest
import torch

KW = dict(video_channels=4, text_dim=8, width=32, depth=2, heads=2, head_dim=16, time_embed_dim=8)


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def tiny(seed=0):
    from modules.sampling.h3_corrector.network import Config, Corrector
    torch.manual_seed(seed)
    net = Corrector(Config(**KW)).eval()
    for p in net.parameters():
        torch.nn.init.normal_(p, 0, 0.1)
    return net


def test_the_network_returns_a_residual_of_the_pictures_shape_and_refuses_a_picture_it_was_not_made_for():
    net = tiny()
    z = torch.randn(2, 4, 2, 4, 6)
    out = net(z, torch.randn_like(z), 0.8, 0.6, torch.randn(2, 3, 8))
    assert out.shape == z.shape and out.abs().sum() > 0
    with pytest.raises(ValueError):
        net(torch.randn(1, 5, 2, 4, 6), torch.randn(1, 5, 2, 4, 6), 0.8, 0.6, torch.randn(1, 3, 8))       # wrong channels
    with pytest.raises(ValueError):
        net(torch.randn(1, 4, 2, 3, 6), torch.randn(1, 4, 2, 3, 6), 0.8, 0.6, torch.randn(1, 3, 8))       # odd height


def test_the_residual_is_capped_at_a_share_of_the_prediction_and_small_ones_pass_nearly_unchanged():
    from modules.sampling.h3_corrector.network import cap
    v = torch.randn(2, 4, 2, 4, 6)
    big, share = cap(100 * torch.randn_like(v), v, 0.5, 0.001)
    assert (share <= 0.5 + 1e-4).all() and (share > 0.4).all()
    small = 1e-3 * torch.randn_like(v)
    kept, _ = cap(small, v, 0.5, 0.001)
    assert torch.allclose(kept, small, rtol=1e-2, atol=1e-6)


def test_the_first_token_does_not_turn_and_later_ones_do():
    from modules.sampling.h3_corrector.network import rotation
    cos, sin = rotation(2, 3, 4, 16, "cpu")
    assert cos.shape == (24, 8) and torch.allclose(cos[0], torch.ones(8)) and torch.allclose(sin[0], torch.zeros(8))
    assert not torch.allclose(cos[1], cos[0])


def test_a_file_without_the_architecture_in_its_metadata_is_not_a_corrector():
    from modules.sampling.h3_corrector.network import Config
    with pytest.raises(ValueError, match="not an H3 corrector"):
        Config.from_metadata({})
    with pytest.raises(ValueError, match="not an H3 corrector"):
        Config.from_metadata(None)


def _save(folder, name, net):
    import dataclasses
    import json
    from safetensors.torch import save_file
    cfg = dataclasses.asdict(net.config)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(folder / name), metadata={"h3_corrector": json.dumps(cfg)})


def test_a_corrector_file_loads_once_and_is_kept_for_the_next_run(tmp_path, monkeypatch):
    import folder_paths
    from modules.sampling import h3_corrector as m
    net = tiny()
    _save(tmp_path, "t.safetensors", net)
    monkeypatch.setitem(folder_paths.folder_names_and_paths, m.FOLDER, ([str(tmp_path)], {".safetensors"}))
    m._LOADED.clear()
    assert m._files() == ["t.safetensors"]
    first = m.load("t.safetensors")
    assert m.load("t.safetensors") is first
    z = torch.randn(1, 4, 2, 4, 4)
    t = torch.randn(1, 3, 8)
    assert torch.allclose(first.model(z, z, 0.5, 0.4, t), net(z, z, 0.5, 0.4, t))
    with pytest.raises(FileNotFoundError):
        m.load("nope.safetensors")
    m.release()
    assert not m._LOADED


def _args(net, sigma=0.8, schedule=(1.0, 0.8, 0.5, 0.0), text=True):
    import comfy.utils
    video, sound = torch.randn(1, 4, 2, 4, 4), torch.randn(1, 3, 5)
    shapes = [tuple(video.shape), tuple(sound.shape)]
    packed = comfy.utils.pack_latents([video, sound])[0]
    out = comfy.utils.pack_latents([torch.randn_like(video), torch.randn_like(sound)])[0]

    class Model:
        latent_shapes = shapes
    return {"denoised": out, "input": packed, "sigma": torch.tensor([sigma]), "model": Model(),
            "model_options": {"transformer_options": {"sample_sigmas": torch.tensor(schedule)}},
            "cond": [{"cross_attn": torch.randn(1, 3, 8)}] if text else [{}]}, shapes


def test_the_fix_goes_into_the_picture_part_only_and_follows_the_schedule():
    import comfy.utils
    from modules.sampling.h3_corrector import make_hook
    net = tiny()
    args, shapes = _args(net)
    hook = make_hook(net, 1.0, 0.5)
    fixed = hook(args)
    before, after = comfy.utils.unpack_latents(args["denoised"], shapes), comfy.utils.unpack_latents(fixed, shapes)
    assert not torch.allclose(before[0], after[0])
    assert torch.equal(before[1], after[1])                                  # the sound is untouched
    assert fixed.shape == args["denoised"].shape
    zero = make_hook(net, 0.0, 0.5)(args)
    assert torch.allclose(zero, args["denoised"], atol=1e-6)                 # strength 0 is no change
    double = make_hook(net, 2.0, 0.5)(args)
    d1, d2 = (comfy.utils.unpack_latents(t, shapes)[0] - before[0] for t in (fixed, double))
    assert torch.allclose(d2, 2 * d1, atol=1e-4)


def test_a_missing_schedule_or_text_is_said_not_guessed():
    from modules.sampling.h3_corrector import make_hook
    net = tiny()
    args, _ = _args(net)
    args["model_options"] = {}
    assert make_hook(net, 1.0, 0.5)(args) is args["denoised"]               # no schedule: the prediction as it was
    args, _ = _args(net, text=False)
    with pytest.raises(RuntimeError, match="text states"):
        make_hook(net, 1.0, 0.5)(args)


def test_install_does_nothing_when_off_or_without_a_file_and_hooks_the_model_when_on(tmp_path, monkeypatch):
    import folder_paths
    from modules.sampling import h3_corrector as m
    net = tiny()
    _save(tmp_path, "t.safetensors", net)
    monkeypatch.setitem(folder_paths.folder_names_and_paths, m.FOLDER, ([str(tmp_path)], {".safetensors"}))
    m._LOADED.clear()

    class Patcher:
        def __init__(self):
            self.hooks, self.extra = [], {}

        def set_additional_models(self, key, models):
            self.extra[key] = models

        def set_model_sampler_post_cfg_function(self, fn, *a):
            self.hooks.append(fn)
    off = Patcher()
    assert m.install(off, {"enabled": False}, "k") is None and not off.hooks
    nofile = Patcher()
    assert m.install(nofile, {"enabled": True, "file": ""}, "k") is None and not nofile.hooks
    zero = Patcher()
    assert "nothing to do" in m.install(zero, {"enabled": True, "file": "t.safetensors", "strength": 0}, "k") and not zero.hooks
    on = Patcher()
    status = m.install(on, {"enabled": True, "file": "t.safetensors", "strength": 1.0, "budget": 0.5}, "k")
    assert "t.safetensors" in status and len(on.hooks) == 1 and getattr(on.hooks[0], "_funui_key", None) == "k"
    assert m.ID in on.extra and on.extra[m.ID][0].model.config.depth == 2
