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


def _install(pih3, patcher, name, steps="auto"):
    from modules._core import patching
    guarded = patching.GuardedPatcher(patcher, "funui.pi_h3", patching.Dropped())
    return pih3.install(guarded, {"enabled": True, "adapter": name, "steps": steps}, "funui.pi_h3")


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


def test_the_shifts_and_the_sampler_are_the_files(pih3, folder, monkeypatch):
    import comfy.samplers
    from comfy.patcher_extension import WrappersMP
    base = _tiny.native_model()
    adapter_file(folder / "pi_h3_4step_adapter.safetensors", base.get_model_object("diffusion_model"))
    model = base.clone()
    assert "4 steps" in _install(pih3, model, "pi_h3_4step_adapter.safetensors")
    sampling = model.get_model_object("model_sampling")
    assert (sampling.shift, sampling.audio_shift) == (12.0, 3.0)
    assert model.model_options["transformer_options"]["minimax_h3_sigma_shift_audio"] == 3.0
    given = {}
    monkeypatch.setattr(comfy.samplers.KSAMPLER, "sample",
                        lambda self, wrap, sigmas, extra_args, *a, **k: given.update(sigmas=sigmas) or "done")

    class Guider:
        cfg = 5.0
        model_patcher = None

    guider = Guider()
    extra = {"model_options": {"transformer_options": {}}}
    run = model.wrappers[WrappersMP.SAMPLER_SAMPLE]["funui.pi_h3"][0]
    sigmas = torch.linspace(1.0, 0.0, 21, device="meta")
    assert run(None, guider, sigmas, extra, None, None, None, None, True) == "done"
    assert guider.cfg == 1.0                                                              # guidance is not applied
    assert len(given["sigmas"]) == 5 and given["sigmas"].device == sigmas.device
    assert extra["model_options"]["transformer_options"]["sample_sigmas"] is given["sigmas"]
    # the base model is not left changed
    assert not base.patches and not base.object_patches and "funui_pi_h3" not in base.model_options


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
    adapter_file(folder / "pi_h3_2step_dwd_adapter.safetensors", native, dwd=True)
    adapter_file(folder / "plain.safetensors", native, policy=None)
    adapter_file(folder / "pi_h3_adapter.safetensors", native)
    with pytest.raises(ValueError, match="DWD"):
        _install(pih3, base.clone(), "pi_h3_2step_dwd_adapter.safetensors")
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
