"""The leak report names the holder of a model whose patcher is gone."""
import gc

import pytest
import torch

from core import leak_report


@pytest.fixture(autouse=True)
def _unwrapped():
    """Another test may already have installed the report on the shared ComfyUI module; each test here starts from the plain functions."""
    try:
        import comfy.model_management as mm
    except Exception:           # noqa: BLE001 -- ComfyUI absent: the tests that need it skip themselves
        yield
        return
    cleanup, switch = mm.cleanup_models_gc, mm.LoadedModel._switch_parent
    mm.cleanup_models_gc = lambda: None if getattr(cleanup, leak_report.TAG, False) else cleanup()
    mm.LoadedModel._switch_parent = (lambda self: None) if getattr(switch, leak_report.TAG, False) else switch
    yield
    mm.cleanup_models_gc, mm.LoadedModel._switch_parent = cleanup, switch


class Holder:
    """Stands for the closure or cache that keeps a model alive."""
    def __init__(self, model):
        self.keep = model


def test_a_model_kept_by_something_else_is_named_before_and_after_gc(comfyui, capsys):
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    base = torch.nn.Linear(2, 2)
    base.model_config = type("C", (), {})()
    patcher = mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    loaded = mm.LoadedModel(patcher)
    import weakref
    loaded.real_model = weakref.ref(base)
    holder = Holder(base)
    mm.current_loaded_models.append(loaded)
    try:
        del patcher
        gc.collect()
        assert loaded.is_dead()
        assert leak_report.report(mm, "test") == 1
        shown = capsys.readouterr().err
        assert "Linear outlived its patcher" in shown and "Holder" in shown
    finally:
        mm.current_loaded_models.remove(loaded)
        del holder


def test_install_wraps_once_and_a_clean_state_logs_nothing(comfyui, capsys):
    import comfy.model_management as mm
    original, switch = mm.cleanup_models_gc, mm.LoadedModel._switch_parent
    try:
        assert leak_report.install(mm) and not leak_report.install(mm)
        mm.cleanup_models_gc()
        assert "FunUI Leak" not in capsys.readouterr().err
    finally:
        mm.cleanup_models_gc, mm.LoadedModel._switch_parent = original, switch
        if leak_report._gc_flag in gc.callbacks:
            gc.callbacks.remove(leak_report._gc_flag)


def test_a_hand_over_that_finds_no_parent_is_said(comfyui, capsys):
    import weakref
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    original = mm.LoadedModel._switch_parent
    try:
        assert leak_report.watch_switch(mm) and not leak_report.watch_switch(mm)
        base = torch.nn.Linear(2, 2)
        base.model_config = type("C", (), {})()
        parent = mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        child = parent.clone()
        loaded = mm.LoadedModel(child)
        loaded.real_model = weakref.ref(base)
        keep = Holder(base)
        del parent
        gc.collect()                    # the parent is held by the child: still there
        assert loaded._parent_model() is not None
        loaded._parent_model = lambda: None          # what a gc that cleared the weak link leaves behind
        loaded._switch_parent()
        shown = capsys.readouterr().err
        assert "its parent is gone" in shown and "Python's gc running: False" in shown
        del keep
    finally:
        mm.LoadedModel._switch_parent = original
        if leak_report._gc_flag in gc.callbacks:
            gc.callbacks.remove(leak_report._gc_flag)


def test_an_entry_whose_patcher_died_without_a_hand_over_is_tracked_again_through_a_live_one(comfyui, capsys):
    """The state the rental showed: the cache keeps the loader, LoRA and modifier patchers (one chain), the sampling clone died and
    ComfyUI could not hand over to its parent. The entry must be tracked again, and ComfyUI's own check must stay quiet."""
    import weakref
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    base = torch.nn.Linear(2, 2)
    base.model_config = type("C", (), {})()
    root = mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    chain = [root]
    for _ in range(3):
        chain.append(chain[-1].clone())
    sampling = chain[-1].clone()
    loaded = mm.LoadedModel(sampling)
    loaded.real_model = weakref.ref(base)
    mm.current_loaded_models.append(loaded)
    try:
        loaded._parent_model = lambda: None          # the hand-over finds no parent
        del sampling
        gc.collect()
        assert loaded.is_dead()
        assert leak_report.relink(mm) == 1 and not loaded.is_dead() and loaded.model is chain[-1]
        assert "tracks it again" in capsys.readouterr().err
        assert leak_report.relink(mm) == 0
    finally:
        mm.current_loaded_models.remove(loaded)


def test_an_entry_with_no_live_patcher_is_left_for_the_report(comfyui):
    import weakref
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    base = torch.nn.Linear(2, 2)
    base.model_config = type("C", (), {})()
    patcher = mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    loaded = mm.LoadedModel(patcher)
    loaded.real_model = weakref.ref(base)
    keep = Holder(base)
    mm.current_loaded_models.append(loaded)
    try:
        del patcher
        gc.collect()
        assert loaded.is_dead() and leak_report.relink(mm) == 0 and loaded.is_dead()
        del keep
    finally:
        mm.current_loaded_models.remove(loaded)


def test_a_hand_over_with_no_parent_is_logged_after_the_first_forty(monkeypatch):
    from core import leak_report
    monkeypatch.setattr(leak_report, "_seen", [500])
    lines = []
    monkeypatch.setattr(leak_report.log, "info", lambda *a, **k: lines.append(a))
    mm = type("MM", (), {})()

    class LM:
        def __init__(self): self._parent_model = lambda: None; self.real_model = lambda: None
        def _switch_parent(self): pass
    mm.LoadedModel = LM
    leak_report.watch_switch(mm)
    LM()._switch_parent()
    assert lines and "hand-over" in lines[0][1] and "gone" in lines[0][1]


def test_install_runs_relink_before_comfyuis_own_cleanup(monkeypatch):
    order = []
    mm = type("MM", (), {})()
    mm.cleanup_models_gc = lambda: order.append("comfy")
    monkeypatch.setattr(leak_report, "watch_switch", lambda m: True)
    monkeypatch.setattr(leak_report, "relink", lambda m: order.append("relink") or 0)
    monkeypatch.setattr(leak_report, "report", lambda m, when: order.append(when) or 0)
    assert leak_report.install(mm)
    mm.cleanup_models_gc()
    assert order == ["relink", "before ComfyUI's gc", "comfy", "after ComfyUI's gc"]


def test_the_first_forty_hand_overs_with_a_parent_are_logged_and_the_forty_first_is_not(monkeypatch):
    monkeypatch.setattr(leak_report, "_seen", [0])
    lines = []
    monkeypatch.setattr(leak_report.log, "info", lambda *a, **k: lines.append(a))
    mm = type("MM", (), {})()

    class LM:
        def __init__(self): self._parent_model = lambda: object(); self.real_model = lambda: None
        def _switch_parent(self): pass
    mm.LoadedModel = LM
    leak_report.watch_switch(mm)
    for _ in range(45):
        LM()._switch_parent()
    assert len(lines) == 40
