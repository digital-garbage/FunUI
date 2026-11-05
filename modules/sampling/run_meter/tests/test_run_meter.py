import json

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def _folder(tmp_path, monkeypatch):
    from modules.sampling import run_meter
    monkeypatch.setattr(run_meter, "FOLDER", tmp_path / "run_meter")


def _load(tiny, **v):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    patched, status = FunUILoadModifiers.execute(tiny.patcher, {"run_meter": {"enabled": True, **v}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws][-1]
    return wrap, outer, status


def _to(i, steps=4):
    s = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1]}


def _run(wrap, outer, answers, seed=7):
    ts = torch.linspace(1.0, 0.0, len(answers) + 1)
    x = torch.zeros_like(answers[0])

    class Run:                                              # what OUTER_SAMPLE wraps: the whole sampling call
        def __call__(self, *a, **k):
            for i, o in enumerate(answers):
                got = wrap(lambda *aa, o=o, **kk: o, x, ts[i:i + 1], None, None, None, _to(i, len(answers)))
                assert got is o                             # the answer is never changed
            return "done"

    return outer(Run(), None, None, None, None, None, None, None, seed)


def test_off_installs_nothing(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunUILoadModifiers
    patched, _ = FunUILoadModifiers.execute(tiny_h3.patcher, {"run_meter": {"enabled": False}}).result
    assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)


def test_noise_reads_grainier_than_a_smooth_picture_and_settling_falls_to_zero(tiny_h3):
    from modules.sampling.run_meter import measure
    smooth = torch.linspace(0, 1, 8).expand(1, 4, 3, 8, 8).contiguous()
    noisy = torch.randn(1, 4, 3, 8, 8)
    assert measure(smooth)[0] < 0.3 * measure(noisy)[0]
    across, over, settle = measure(noisy, noisy)
    assert float(settle) == 0.0 and float(over) > 0.0


def test_a_run_is_logged_saved_and_compared_with_the_previous_run_of_the_same_seed(tiny_h3, tmp_path):
    from modules.sampling import run_meter
    from modules._core import log
    wrap, outer, status = _load(tiny_h3)
    assert "run_meter" in status
    base = torch.randn(1, 4, 3, 8, 8)
    answers = [base * f for f in (0.9, 0.5, 0.2, 0.1)]
    assert _run(wrap, outer, answers) == "done"
    saved = json.loads((tmp_path / "run_meter" / "last.json").read_text())
    assert saved["seed"] == 7 and saved["steps"] == 4 and saved["final"]["settle"] is not None
    assert not (tmp_path / "run_meter" / "last.json.tmp").exists()
    assert _run(wrap, outer, answers) == "done"
    said = [r["message"] for r in log._records if r["source"] == run_meter.TITLE]
    assert "nothing saved to compare with yet" in said[-2]
    assert "against the previous run of this seed" in said[-1] and "same" in said[-1]
    assert _run(wrap, outer, answers, seed=8) == "done"
    assert "another seed" in [r["message"] for r in log._records if r["source"] == run_meter.TITLE][-1]


def test_a_bigger_picture_after_the_hand_over_has_no_settling_to_compare(tiny_h3):
    from modules.sampling import run_meter
    from modules._core import log
    wrap, outer, _ = _load(tiny_h3)
    small, big = torch.randn(1, 4, 3, 4, 4), torch.randn(1, 4, 3, 8, 8)
    assert _run(wrap, outer, [small, small * 0.5, big, big * 0.5]) == "done"
    text = [r["message"] for r in log._records if r["source"] == run_meter.TITLE][-1]
    assert "8x8" in text and "n/a (first call at this size)" in text


def test_a_stopped_run_saves_nothing(tiny_h3, tmp_path):
    wrap, outer, _ = _load(tiny_h3)

    def stops(*a, **k):
        wrap(lambda *aa, **kk: torch.randn(1, 4, 3, 8, 8), torch.zeros(1, 4, 3, 8, 8), torch.ones(1), None, None, None, _to(0))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        outer(stops, None, None, None, None, None, None, None, 1)
    assert not (tmp_path / "run_meter").exists()


def test_a_second_call_in_a_step_does_not_replace_the_first(tiny_h3):
    from modules.sampling import run_meter
    from modules._core import log
    wrap, outer, _ = _load(tiny_h3)
    a, b = torch.randn(1, 4, 3, 8, 8), torch.randn(1, 4, 3, 8, 8)
    ts = torch.linspace(1.0, 0.0, 3)

    def run(*args, **k):
        for i, o in ((0, a), (0, b), (1, a * 0.5)):                     # step 0 is asked twice, then step 1
            wrap(lambda *aa, o=o, **kk: o, torch.zeros_like(a), ts[i:i + 1], None, None, None, _to(i, 2))

    outer(run, None, None, None, None, None, None, None, 1)
    text = [r["message"] for r in log._records if r["source"] == run_meter.TITLE][-1]
    assert text.count("step 1/2") == 1 and text.count("step 2/2") == 1


def test_tensors_follow_the_input_device_and_dtype():
    """Device safety: nothing here is created on a fixed device, so a meta tensor passes through and a half one stays half."""
    from modules.sampling.run_meter import measure
    for dtype in (torch.float16, torch.bfloat16):
        v = torch.randn(1, 4, 3, 8, 8).to(dtype)
        across, over, settle = measure(v, v.float())             # the previous answer may sit in another dtype
        assert across.device == v.device and across.dtype == torch.float32
        assert float(settle) == 0.0 or float(settle) < 1e-2
    meta = torch.empty(1, 4, 3, 8, 8, device="meta")
    across, over, settle = measure(meta, torch.empty(1, 4, 3, 8, 8, device="cpu"))
    assert across.device.type == "meta" and settle.device.type == "meta"
