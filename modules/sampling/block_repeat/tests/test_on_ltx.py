"""Block repeat on a (tiny) LTX forward: picture and sound go through each block as a pair."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _load(tiny, **values):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    return FunUILoadModifiers.execute(tiny.patcher, {"block_repeat": {"enabled": True, **values}}).result


def _calls(tiny, patched):
    counts, handles = {}, []
    for i, block in enumerate(tiny.patcher.model.diffusion_model.transformer_blocks):
        handles.append(block.register_forward_pre_hook(lambda m, a, kw=None, i=i: counts.__setitem__(i, counts.get(i, 0) + 1)))
    try:
        tiny.run(patched)
    finally:
        for h in handles:
            h.remove()
    return counts


def test_each_named_block_runs_times_plus_one(tiny_ltx):
    patched, status = _load(tiny_ltx, blocks="1", times=2)
    assert "block_repeat: blocks 1 run 3x each" in status
    counts = _calls(tiny_ltx, patched)
    assert counts[1] == 3 and counts[0] == 1


def test_it_changes_the_picture_and_video_only_leaves_the_sound_single_pass(tiny_ltx):
    base_v, base_a = tiny_ltx.run()
    both, _ = _load(tiny_ltx, blocks="1")
    v, a = tiny_ltx.run(both)
    assert not torch.allclose(v, base_v) and not torch.allclose(a, base_a)
    only, _ = _load(tiny_ltx, blocks="1", video_only=True)
    v2, a2 = tiny_ltx.run(only)
    assert not torch.allclose(v2, base_v)
    assert v2.shape == base_v.shape and a2.shape == base_a.shape


def test_whole_span_runs_as_each_block_here(tiny_ltx):
    patched, status = _load(tiny_ltx, blocks="1-2", shape="span")
    assert "run 2x each" in status
