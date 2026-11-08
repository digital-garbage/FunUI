"""The leak report names the holder of a model whose patcher is gone."""
import gc

import torch

from core import leak_report


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
    original = mm.cleanup_models_gc
    try:
        assert leak_report.install(mm) and not leak_report.install(mm)
        mm.cleanup_models_gc()
        assert "FunUI Leak" not in capsys.readouterr().err
    finally:
        mm.cleanup_models_gc = original
