"""pi-H3 on a tiny real MiniMax-H3: the file's two kinds of tensors, the shifts, the schedule and clean replacement.

They prove the wiring and the maths of our side on random weights, nothing about picture quality on the real model.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptd" / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tiny  # noqa: E402
from _pifile import adapter_file  # noqa: E402


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture
def pih3():
    import importlib
    return importlib.import_module("modules.sampling.pi_h3")


@pytest.fixture
def folder(tmp_path):
    import folder_paths
    folder_paths.add_model_folder_path("pi_h3", str(tmp_path), is_default=True)
    return tmp_path


def _install(pih3, patcher, name, steps="auto", strength=1.0):
    from modules._core import patching
    guarded = patching.GuardedPatcher(patcher, "funui.pi_h3", patching.Dropped())
    return pih3.install(guarded, {"enabled": True, "adapter": name, "steps": steps, "strength": strength}, "funui.pi_h3")


def test_the_schedule_is_the_published_one():
    from modules.sampling.pi_h3 import adapter as ad
    two = ad.schedule(12.0, 2)
    raw = [0.93, 0.93 * 0.6 / 1.6]
    assert two.tolist()[-1] == 0.0 and len(two) == 3
    for got, t in zip(two.tolist(), raw):
        assert got == pytest.approx(12 * t / (1 + 11 * t), rel=1e-5)
    four = ad.schedule(12.0, 4)
    assert len(four) == 5 and four[0] == pytest.approx(12 * 0.96 / (1 + 11 * 0.96), rel=1e-5)
    assert all(a > b for a, b in zip(four.tolist(), four.tolist()[1:]))
    with pytest.raises(ValueError):
        ad.schedule(12.0, 3)


def test_whole_tensors_replace_and_lora_pairs_add(pih3, folder):
    base = _tiny.native_model()
    native = base.get_model_object("diffusion_model")
    file = adapter_file(folder / "pi_h3_2step_adapter.safetensors", native)
    before = {k: v.clone() for k, v in native.state_dict().items()}
    model = base.clone()
    note = _install(pih3, model, "pi_h3_2step_adapter.safetensors")
    assert "2 steps" in note
    model.patch_model(device_to=torch.device("cpu"), force_patch_weights=True)
    try:
        now = native.state_dict()
        assert torch.allclose(now["final_layer.video_out.weight"], file["proj_out.weight"].float(), atol=1e-3)
        assert torch.allclose(now["final_layer.adaln_proj.linear.weight"], file["norm_out.linear.weight"].float(), atol=1e-3)
        assert torch.allclose(now["blocks.0.adaln_proj.linear.weight"], file["transformer_blocks.0.adaln_proj.linear.weight"].float(), atol=1e-3)
        # block 1 is a whole block in this file: its out projection is replaced
        assert torch.allclose(now["blocks.1.attn.out_proj.weight"], file["transformer_blocks.1.attn.to_out.0.weight"].float(), atol=1e-3)
        # block 0 carries LoRA pairs: base + B @ A
        want = before["blocks.0.attn.out_proj.weight"] + (file["transformer_blocks.0.attn.to_out.0.lora_B.weight"].float()
                                                          @ file["transformer_blocks.0.attn.to_out.0.lora_A.weight"].float())
        assert torch.allclose(now["blocks.0.attn.out_proj.weight"], want, atol=1e-3)
        # the feed-forward's first layer keeps ComfyUI's half order: the diffusers halves swap
        first = file["transformer_blocks.0.ff.net.0.proj.lora_B.weight"].float() @ file["transformer_blocks.0.ff.net.0.proj.lora_A.weight"].float()
        half = first.shape[0] // 2
        swapped = torch.cat([first[half:], first[:half]])
        assert torch.allclose(now["blocks.0.mlp.fc1.weight"], before["blocks.0.mlp.fc1.weight"] + swapped, atol=1e-3)
    finally:
        model.unpatch_model(torch.device("cpu"))
    assert torch.equal(native.state_dict()["final_layer.video_out.weight"], before["final_layer.video_out.weight"])


def test_the_shifts_and_the_schedule_are_the_files(pih3, folder):
    import comfy.samplers
    from comfy.patcher_extension import WrappersMP
    base = _tiny.native_model()
    adapter_file(folder / "pi_h3_4step_adapter.safetensors", base.get_model_object("diffusion_model"))
    model = base.clone()
    assert "4 steps" in _install(pih3, model, "pi_h3_4step_adapter.safetensors")
    sampling = model.get_model_object("model_sampling")
    assert (sampling.shift, sampling.audio_shift) == (12.0, 3.0)
    assert model.model_options["transformer_options"]["minimax_h3_sigma_shift_audio"] == 3.0

    class Guider:
        cfg = 5.0
        model_patcher = None

    guider, given = Guider(), {}

    class Executor:
        class_obj = guider

        def __call__(self, noise, latent_image, sampler, sigmas, *rest, **kwargs):
            given.update(sampler=sampler, sigmas=sigmas, rest=rest)
            return "done"

    run = model.wrappers[WrappersMP.OUTER_SAMPLE]["funui.pi_h3"][0]
    sigmas = torch.linspace(1.0, 0.0, 21, device="meta")
    assert run(Executor(), "noise", "latent", "the chosen sampler", sigmas, None, None, True, 7) == "done"
    assert guider.cfg == 1.0                                                              # guidance is not applied
    assert len(given["sigmas"]) == 5 and given["sigmas"].device == sigmas.device          # the file's schedule, where the run's is
    assert given["sampler"].__class__ is comfy.samplers.KSAMPLER and given["sampler"] != "the chosen sampler"
    assert given["rest"][-1] == 7
    # the base model is not left changed
    assert not base.patches and not base.object_patches and "funui_pi_h3" not in base.model_options


def test_two_stages_cuts_the_files_own_schedule(pih3, folder):
    """The schedule is replaced around the whole run before Two stages sees it, so it cuts the file's 4 steps, not the chosen 20."""
    assert "two_stage" in pih3.BEFORE


def test_it_samples_on_the_tiny_model_and_differs_from_the_base(pih3, folder):
    import comfy.nested_tensor
    import comfy.samplers
    base = _tiny.native_model()
    adapter_file(folder / "pi_h3_2step_adapter.safetensors", base.get_model_object("diffusion_model"))

    def sample(model):
        guider = comfy.samplers.CFGGuider(model)
        context = torch.randn(1, 6, _tiny.DIMS["text_dim"], generator=torch.Generator().manual_seed(3))
        guider.set_conds([[context, {}]], [[context, {}]])
        guider.set_cfg(1.0)
        video, audio = torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)
        noise = comfy.nested_tensor.NestedTensor([torch.randn(video.shape, generator=torch.Generator().manual_seed(4)),
                                                  torch.randn(audio.shape, generator=torch.Generator().manual_seed(5))])
        try:
            out = guider.sample(noise, comfy.nested_tensor.NestedTensor([video, audio]), comfy.samplers.sampler_object("euler"),
                                torch.linspace(1.0, 0.0, 21), disable_pbar=True, seed=7)
        finally:
            from comfy import model_management
            model_management.unload_all_models()
        return [t.clone() for t in out.unbind()]

    plain = sample(base.clone())
    model = base.clone()
    _install(pih3, model, "pi_h3_2step_adapter.safetensors")
    got = sample(model)
    assert all(torch.isfinite(t).all() for t in got)
    assert not torch.allclose(plain[0], got[0])


def test_a_second_pass_replaces_the_first_install_and_off_removes_it(pih3, folder):
    from modules._core import patching
    base = _tiny.native_model()
    adapter_file(folder / "pi_h3_2step_adapter.safetensors", base.get_model_object("diffusion_model"))
    model = base.clone()
    _install(pih3, model, "pi_h3_2step_adapter.safetensors")
    count = {k: len(v) for k, v in model.patches.items()}
    again = model.clone()
    assert patching.strip(again, "funui.") >= 1
    _install(pih3, again, "pi_h3_2step_adapter.safetensors")
    assert {k: len(v) for k, v in again.patches.items()} == count                       # replaced, not stacked
    assert {k: len(v) for k, v in model.patches.items()} == count                       # the first pass's model is untouched
    off = again.clone()
    patching.strip(off, "funui.")
    assert pih3.install(patching.GuardedPatcher(off, "funui.pi_h3", patching.Dropped()), {"enabled": False}, "funui.pi_h3") is None
    assert not off.patches


def test_files_that_do_not_fit_are_refused_in_plain_words(pih3, folder):
    base = _tiny.native_model()
    native = base.get_model_object("diffusion_model")
    adapter_file(folder / "plain.safetensors", native, policy=None)
    adapter_file(folder / "pi_h3_adapter.safetensors", native)
    with pytest.raises(ValueError, match="not a pi-H3 adapter"):
        _install(pih3, base.clone(), "plain.safetensors")
    from modules._core import control
    with pytest.raises(control.Unavailable, match="choose 2 or 4"):
        _install(pih3, base.clone(), "pi_h3_adapter.safetensors")
    assert "4 steps" in _install(pih3, base.clone(), "pi_h3_adapter.safetensors", steps="4")
    bigger = _tiny.native_model(hidden_size=96, attention_head_dim=48)
    with pytest.raises(ValueError, match="does not fit"):
        _install(pih3, bigger.clone(), "pi_h3_adapter.safetensors", steps="2")


def test_no_file_chosen_is_said(pih3, folder):
    from modules._core import control, patching
    with pytest.raises(control.Unavailable, match="models/pi_h3"):
        pih3.install(patching.GuardedPatcher(_tiny.native_model().clone(), "funui.pi_h3", patching.Dropped()),
                     {"enabled": True, "adapter": "", "steps": "auto"}, "funui.pi_h3")


def _sample_nested(model, steps=20):
    import comfy.nested_tensor
    import comfy.samplers
    guider = comfy.samplers.CFGGuider(model)
    context = torch.randn(1, 6, _tiny.DIMS["text_dim"], generator=torch.Generator().manual_seed(3))
    guider.set_conds([[context, {}]], [[context, {}]])
    guider.set_cfg(1.0)
    video, audio = torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)
    noise = comfy.nested_tensor.NestedTensor([torch.randn(video.shape, generator=torch.Generator().manual_seed(4)),
                                              torch.randn(audio.shape, generator=torch.Generator().manual_seed(5))])
    try:
        return guider.sample(noise, comfy.nested_tensor.NestedTensor([video, audio]), comfy.samplers.sampler_object("euler"),
                             torch.linspace(1.0, 0.0, steps + 1), disable_pbar=True, seed=7)
    finally:
        from comfy import model_management
        model_management.unload_all_models()


def test_a_dwd_file_writes_pixels_at_the_last_step_and_the_decode_takes_them(pih3, folder):
    from modules.sampling.pi_h3 import dwd
    base = _tiny.native_model()
    adapter_file(folder / "pi_h3_2step_dwd_adapter.safetensors", base.get_model_object("diffusion_model"), dwd=True)
    model = base.clone()
    assert "DWD pixels" in _install(pih3, model, "pi_h3_2step_dwd_adapter.safetensors")
    assert dwd.carries(model) and not dwd.carries(base)
    out = _sample_nested(model)
    video = out.unbind()[0]
    pixels = dwd.take(video)
    assert pixels is not None and pixels.shape[-1] == 3 and pixels.shape[1:3] == (64, 64)
    assert torch.isfinite(pixels.float()).all() and 0.0 <= float(pixels.min()) and float(pixels.max()) <= 1.0
    # another run's latent has no pixels: it must not decode as if it had
    assert dwd.take(video + 1) is None
    from modules.models import minimax_h3
    images = minimax_h3._dwd_images(model, video)
    assert images.shape == pixels.shape
    with pytest.raises(RuntimeError, match="cannot be decoded by the VAE"):
        minimax_h3._dwd_images(model, video + 1)
    assert minimax_h3._dwd_images(base, video) is None


def test_a_second_pass_does_not_wrap_the_dwd_hook_twice(pih3, folder):
    from modules._core import patching
    base = _tiny.native_model()
    adapter_file(folder / "pi_h3_2step_dwd_adapter.safetensors", base.get_model_object("diffusion_model"), dwd=True)
    model = base.clone()
    _install(pih3, model, "pi_h3_2step_dwd_adapter.safetensors")
    again = model.clone()
    patching.strip(again, "funui.")
    _install(pih3, again, "pi_h3_2step_dwd_adapter.safetensors")
    out = _sample_nested(again)
    from modules.sampling.pi_h3 import dwd
    assert dwd.take(out.unbind()[0]) is not None


def test_two_stages_runs_the_files_four_steps_small_then_full(pih3, folder):
    """A run asked for 20 steps: pi-H3 makes it 4, and Two stages (installed after it) cuts those 4 into 2 small and 2 full."""
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    from modules.sampling.two_stage import install as two_stage
    base = _tiny.native_model()
    adapter_file(folder / "pi_h3_4step_adapter.safetensors", base.get_model_object("diffusion_model"))
    model = base.clone()
    _install(pih3, model, "pi_h3_4step_adapter.safetensors")
    two_stage(model, {"enabled": True, "size": 0.5, "steps": 2}, "funui.two_stage")
    calls = []
    model.model.diffusion_model.register_forward_pre_hook(
        lambda m, a: calls.append(tuple(a[0][0].shape[-2:]) if isinstance(a[0], (list, tuple)) else tuple(a[0].shape[-2:])))
    guider = comfy.samplers.CFGGuider(model)
    context = torch.randn(1, 6, _tiny.DIMS["text_dim"], generator=torch.Generator().manual_seed(3))
    guider.set_conds([[context, {}]], [[context, {}]])
    guider.set_cfg(7.0)
    noise = NestedTensor((torch.randn(1, 24, 2, 8, 8), torch.randn(1, 32, 2, 3)))
    latent = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 3)))
    try:
        out = guider.sample(noise, latent, comfy.samplers.sampler_object("dpmpp_2m"), torch.linspace(1.0, 0.0, 21), seed=1, disable_pbar=True)
    finally:
        from comfy import model_management
        model_management.unload_all_models()
    assert calls == [(4, 4), (4, 4), (8, 8), (8, 8)]
    assert all(torch.isfinite(t).all() for t in out.unbind())


def test_quantized_layers_that_carry_the_adapter_are_reported(pih3, folder, monkeypatch):
    from modules._core import log
    base = _tiny.native_model()
    adapter_file(folder / "pi_h3_2step_adapter.safetensors", base.get_model_object("diffusion_model"))
    said = []
    monkeypatch.setattr(log, "once", lambda key, level, source, message: said.append((level, message)))
    monkeypatch.setattr(pih3, "_is_quantized", lambda w: w is not None and w.dim() == 2 and w.shape[0] > 100)
    model = base.clone()
    _install(pih3, model, "pi_h3_2step_adapter.safetensors")
    quant = [m for lvl, m in said if "quantized layers carry the adapter" in m]
    assert quant and "merged into them once at load" in quant[0]               # a plain patcher merges at load
    said.clear()
    monkeypatch.setattr(type(model), "is_dynamic", lambda self: True)
    _install(pih3, base.clone(), "pi_h3_2step_adapter.safetensors")
    assert any(lvl == log.ALERT and "--disable-dynamic-vram" in m for lvl, m in said)


def test_a_lower_strength_blends_toward_the_models_own_weights(pih3, folder):
    base = _tiny.native_model()
    native = base.get_model_object("diffusion_model")
    file = adapter_file(folder / "pi_h3_2step_adapter.safetensors", native)
    before = {k: v.clone() for k, v in native.state_dict().items()}
    model = base.clone()
    assert "strength 0.5" in _install(pih3, model, "pi_h3_2step_adapter.safetensors", strength=0.5)
    model.patch_model(device_to=torch.device("cpu"), force_patch_weights=True)
    try:
        now = native.state_dict()
        theirs = file["proj_out.weight"].float()
        assert torch.allclose(now["final_layer.video_out.weight"], before["final_layer.video_out.weight"] + 0.5 * (theirs - before["final_layer.video_out.weight"]), atol=1e-3)
        want = before["blocks.0.attn.out_proj.weight"] + 0.5 * (file["transformer_blocks.0.attn.to_out.0.lora_B.weight"].float()
                                                                @ file["transformer_blocks.0.attn.to_out.0.lora_A.weight"].float())
        assert torch.allclose(now["blocks.0.attn.out_proj.weight"], want, atol=1e-3)
    finally:
        model.unpatch_model(torch.device("cpu"))
    off = base.clone()
    _install(pih3, off, "pi_h3_2step_adapter.safetensors", strength=0.0)
    off.patch_model(device_to=torch.device("cpu"), force_patch_weights=True)
    try:
        assert torch.allclose(native.state_dict()["final_layer.video_out.weight"], before["final_layer.video_out.weight"], atol=1e-3)
    finally:
        off.unpatch_model(torch.device("cpu"))


def test_the_sound_layer_has_its_own_strength(pih3, folder):
    from modules._core import patching
    base = _tiny.native_model()
    native = base.get_model_object("diffusion_model")
    file = adapter_file(folder / "pi_h3_2step_adapter.safetensors", native)
    before = {k: v.clone() for k, v in native.state_dict().items()}
    model = base.clone()
    guarded = patching.GuardedPatcher(model, "funui.pi_h3", patching.Dropped())
    note = pih3.install(guarded, {"enabled": True, "adapter": "pi_h3_2step_adapter.safetensors", "steps": "auto", "strength": 1.0, "sound_strength": 0.0}, "funui.pi_h3")
    assert "sound 0" in note
    model.patch_model(device_to=torch.device("cpu"), force_patch_weights=True)
    try:
        now = native.state_dict()
        assert torch.allclose(now["final_layer.video_out.weight"], file["proj_out.weight"].float(), atol=1e-3)          # the picture's layer: the file
        assert torch.allclose(now["final_layer.audio_out.weight"], before["final_layer.audio_out.weight"], atol=1e-3)   # the sound's: the model's own
    finally:
        model.unpatch_model(torch.device("cpu"))
