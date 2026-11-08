"""Names what still holds a model when ComfyUI warns "memory leak with model X".

ComfyUI only says the model is still referenced. Right before its own check runs a full gc, this walks the referrers
of every model it tracks as dead (its patcher is gone, the model is not) and logs the chain: types, function names,
dict KEY names and source lines, never a value, so no prompt or setting is ever printed.
"""

import gc
import os
import sys

from . import log

TAG = "_funui_leak_report"
PATHS, DEPTH, NODES = 8, 8, 600


def _name(obj) -> str:
    if isinstance(obj, dict):
        keys = [k for k in list(obj)[:6] if isinstance(k, str)]
        return f"dict{keys}" + ("..." if len(obj) > 6 else "")
    if isinstance(obj, (list, tuple, set)):
        return f"{type(obj).__name__}[{len(obj)}]"
    if hasattr(obj, "__func__"):
        obj = obj.__func__
    if hasattr(obj, "__qualname__"):
        return f"{type(obj).__name__} {getattr(obj, '__module__', '?')}.{obj.__qualname__}"
    if type(obj).__name__ == "frame":
        return f"frame {os.path.basename(obj.f_code.co_filename)}:{obj.f_lineno} {obj.f_code.co_name}"
    if type(obj).__name__ == "cell":
        return "closure cell"
    return type(obj).__name__


def chains(target, depth=DEPTH, nodes=NODES, paths=PATHS) -> list:
    """Up to `paths` strings 'model <- holder <- holder ...', each ending where nothing (or a module) holds the last one."""
    mine = {id(sys._getframe())}
    out, seen, queue = [], {id(target)}, [[target]]
    while queue and len(out) < paths and len(seen) < nodes:
        path = queue.pop(0)
        holders = [r for r in gc.get_referrers(path[-1]) if id(r) not in seen and id(r) not in mine and r is not queue
                   and r is not path and not any(r is q for q in queue)]
        if not holders or len(path) >= depth or type(path[-1]).__name__ == "module":
            out.append(" <- ".join(_name(o) for o in path))
            continue
        for r in holders[:6]:
            seen.add(id(r))
            queue.append(path + [r])
    return out


def _owners(model):
    """The live patchers whose model this is, deepest clone first."""
    held = [r for r in gc.get_referrers(model) if type(r).__name__ == "dict" and "patches" in r]
    found = {id(o): o for h in held for o in gc.get_referrers(h) if type(o).__name__.startswith("ModelPatcher") and getattr(o, "model", None) is model}

    def depth(p):
        n = 0
        while getattr(p, "parent", None) is not None and n < 50:
            p, n = p.parent, n + 1
        return n
    return sorted(found.values(), key=depth, reverse=True)


def relink(mm) -> int:
    """ComfyUI follows a loaded model through its patcher, and when that patcher dies it hands over to the patcher's direct parent
    only. A parent that is gone leaves the entry dead although other patchers of the same model are alive (the loader's, the
    LoRA's, the modifiers' outputs, all cached by ComfyUI): it then warns "memory leak" on every load. Such an entry is pointed
    at the deepest live patcher of its model, so the model is tracked again. An entry with no live patcher stays dead and is
    reported. -> how many were re-linked."""
    n = 0
    for m in list(mm.current_loaded_models):
        if not m.is_dead():
            continue
        model = m.real_model()
        owners = _owners(model) if model is not None else []
        if not owners:
            continue
        m._set_model(owners[0])
        m.device = owners[0].load_device
        n += 1
        log.info("FunUI Leak", f"{type(model).__name__} had lost its patcher; ComfyUI tracks it again through one of the {len(owners)} patchers still holding it")
    return n


def report(mm, when) -> int:
    dead = [m for m in mm.current_loaded_models if m.is_dead()]
    if dead:
        log.warning("FunUI Leak", f"{when}: ComfyUI tracks " + ", ".join(
            f"{type(l.real_model()).__name__ if callable(l.real_model) and l.real_model() is not None else '?'}"
            f"({'DEAD' if l.is_dead() else 'patcher alive' if l.model is not None else 'gone'}"
            f"{', has parent link' if getattr(l, '_parent_model', None) is not None else ''})" for l in mm.current_loaded_models))
    for m in dead:
        model = m.real_model()
        if model is None:
            continue
        held = [r for r in gc.get_referrers(model) if type(r).__name__ == "dict" and "patches" in r]
        owners = [o for h in held for o in gc.get_referrers(h) if type(o).__name__.startswith("ModelPatcher")]

        def depth(p):
            n = 0
            while getattr(p, "parent", None) is not None and n < 50:
                p, n = p.parent, n + 1
            return n
        lineage = ", ".join(f"{type(o).__name__}(parents {depth(o)}, tracked {any(l.model is o for l in mm.current_loaded_models)})" for o in owners)
        fin = getattr(m, "_patcher_finalizer", None)
        state = f"its hand-over {'registered' if fin is not None else 'was never registered'}{' and still pending' if fin is not None and fin.alive else ''}"
        log.warning("FunUI Leak", f"{when}: {type(model).__name__} outlived its patcher ({state}). Patchers holding it: {lineage or 'none'}. What holds it:\n" + "\n".join(chains(model)))
    return len(dead)


_in_gc = [False]
_seen = [0]


def _gc_flag(phase, info):
    _in_gc[0] = phase == "start"


def watch_switch(mm) -> bool:
    """ComfyUI follows a model through its patcher's parent when the patcher dies. Say when that hand-over finds no parent, and whether
    Python's own gc was running at that moment (it clears weak links to every patcher in a cycle before the hand-over can use them)."""
    original = getattr(mm.LoadedModel, "_switch_parent", None)
    if original is None or getattr(original, TAG, False):
        return False
    if _gc_flag not in gc.callbacks:
        gc.callbacks.append(_gc_flag)

    def _switch_parent(self):
        try:
            parent = self._parent_model()
            real = self.real_model() if callable(self.real_model) else None
            _seen[0] += 1
            if _seen[0] <= 40:          # every hand-over, so a missing one shows too
                depth, p = 0, parent
                while p is not None and getattr(p, "parent", None) is not None and depth < 50:
                    p, depth = p.parent, depth + 1
                log.info("FunUI Leak", f"hand-over {_seen[0]}: a {type(real).__name__ if real is not None else 'dead'} patcher died; its parent is "
                         f"{'gone' if parent is None else type(parent).__name__ + f' with {depth} more above it'}; Python's gc running: {_in_gc[0]}")
        except Exception as exc:            # noqa: BLE001 -- a diagnostic must never stop a run
            log.failed("leak report", exc)
        return original(self)

    setattr(_switch_parent, TAG, True)
    mm.LoadedModel._switch_parent = _switch_parent
    return True


def install(mm=None) -> bool:
    if mm is None:
        import comfy.model_management as mm
    original = getattr(mm, "cleanup_models_gc", None)
    if original is None or getattr(original, TAG, False):
        return False
    watch_switch(mm)

    def cleanup_models_gc():
        try:
            relink(mm)
            report(mm, "before ComfyUI's gc")
        except Exception as exc:            # noqa: BLE001 -- a diagnostic must never stop a run
            log.failed("leak report", exc)
        original()
        try:
            report(mm, "after ComfyUI's gc")
        except Exception as exc:            # noqa: BLE001
            log.failed("leak report", exc)

    setattr(cleanup_models_gc, TAG, True)
    mm.cleanup_models_gc = cleanup_models_gc
    return True
