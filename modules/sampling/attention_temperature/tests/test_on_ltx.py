"""The randomizer on a real (tiny) LTX forward: its blocks take (video, audio) as a pair."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def test_it_is_offered_on_ltx_and_changes_the_output(tiny_ltx):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    base_v, base_a = tiny_ltx.run()
    patched, status = FunUILoadModifiers.execute(
        tiny_ltx.patcher, {"attention_temperature": {"enabled": True, "amount": -0.5, "blocks": "0-1"}}).result
    assert "attention_temperature: 0.5x (sharper) at blocks 0-1" in status
    v, a = tiny_ltx.run(patched)
    assert not torch.allclose(v, base_v), "temperature ran and changed nothing"
    assert v.shape == base_v.shape and a.shape == base_a.shape


def test_blocks_past_the_models_last_are_said_not_silently_dropped(tiny_ltx):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    from core import dit_hooks
    n = dit_hooks.block_count(tiny_ltx.patcher)
    assert n > 0, "LTX's blocks were not counted"
    _p, status = FunUILoadModifiers.execute(
        tiny_ltx.patcher, {"attention_temperature": {"enabled": True, "amount": -0.5, "blocks": f"0-{n + 3}"}}).result
    assert f"0-{n - 1}" in status


def test_row_editing_modules_stay_off_ltx(tiny_ltx):
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    _p, status = FunUILoadModifiers.execute(tiny_ltx.patcher, {"reins": {"enabled": True}}).result
    assert "reins: needs dit_block_hooks" in status
