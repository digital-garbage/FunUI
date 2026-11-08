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


def report(mm, when) -> int:
    dead = [m for m in mm.current_loaded_models if m.is_dead()]
    for m in dead:
        model = m.real_model()
        if model is None:
            continue
        log.warning("FunUI Leak", f"{when}: {type(model).__name__} outlived its patcher. What holds it:\n" + "\n".join(chains(model)))
    return len(dead)


def install(mm=None) -> bool:
    if mm is None:
        import comfy.model_management as mm
    original = getattr(mm, "cleanup_models_gc", None)
    if original is None or getattr(original, TAG, False):
        return False

    def cleanup_models_gc():
        try:
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
