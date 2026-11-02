"""A few-step H3 file that stacks several output heads per stream loads at every weight dtype,
even when its metadata does not say how many heads there are.

The fixture is a tiny real H3 saved the way such a file looks: head weights and biases stacked
32 times, config metadata with explicit head sizes but no `pdd_steps`.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sampling" / "ptd" / "tests"))

STEPS = 32


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """These import comfy directly."""


def _file(steps=STEPS):
    from _tiny import native_model
    sd = {k: v.clone() for k, v in native_model(pdd_steps=steps).model.diffusion_model.state_dict().items()}
    return sd


def _meta(**transformer):
    return {"config": json.dumps({"transformer": transformer})}


def test_metadata_without_the_head_count_is_completed_from_the_weights():
    from modules.loaders.common import head_bank_metadata
    fixed, note = head_bank_metadata(_file(), _meta(latents_dim=24, audio_latents_dim=32))
    assert json.loads(fixed["config"])["transformer"]["pdd_steps"] == STEPS
    assert json.loads(fixed["config"])["transformer"]["latents_dim"] == 24
    assert note and str(STEPS) in note


def test_a_wrong_head_count_is_corrected_and_a_right_one_left_alone():
    from modules.loaders.common import head_bank_metadata
    fixed, note = head_bank_metadata(_file(), _meta(pdd_steps=1))
    assert json.loads(fixed["config"])["transformer"]["pdd_steps"] == STEPS and note
    same = _meta(pdd_steps=STEPS)
    kept, note = head_bank_metadata(_file(), same)
    assert kept is same and note is None


def test_plain_and_foreign_files_are_untouched():
    from modules.loaders.common import head_bank_metadata
    plain = _file(1)
    meta = _meta()
    assert head_bank_metadata(plain, meta) == (meta, None)
    assert head_bank_metadata({"x": torch.zeros(2)}, None) == (None, None)


@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float16, torch.float32])
def test_loads_at_every_weight_dtype(monkeypatch, dtype):
    import comfy.sd
    import comfy.utils
    import folder_paths
    from modules.loaders.diffusion_model import nodes

    sd = _file()
    meta = _meta(latents_dim=24, audio_latents_dim=32)
    # Stock ComfyUI fails on this exact input; this is the failure being fixed.
    with pytest.raises(RuntimeError, match="size mismatch"):
        comfy.sd.load_diffusion_model_state_dict(
            {k: v.clone() for k, v in sd.items()}, model_options={} if dtype is None else {"dtype": dtype}, metadata=meta)

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda p, **kw: ({k: v.clone() for k, v in sd.items()}, meta))
    names = {None: "default", torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    out = nodes.FunUIDiffusionModelLoader.execute(
        model_name="m.safetensors", weight_dtype=names[dtype], compute_dtype="default", attention="default")
    head = out[0].model.diffusion_model.final_layer.video_out
    assert head.weight.shape[0] == 96 * STEPS and head.bias.shape[0] == 96 * STEPS


def test_the_head_count_reads_shapes_only_so_any_device_or_dtype_works():
    from modules.loaders.common import head_bank_metadata
    sd = {k: v.to(device="meta", dtype=torch.float16) for k, v in _file().items()}
    fixed, _ = head_bank_metadata(sd, _meta(latents_dim=24))
    assert json.loads(fixed["config"])["transformer"]["pdd_steps"] == STEPS
