"""The Chat tab's history and the model's conversation memory, kept on disk so a restart keeps both.

One small JSON file next to the pack's other user data. It is read once when the module loads and
rewritten after every run; a file that cannot be read or written costs the history, never a run.
"""

import collections
import json
import os

from ..._core import config, log

FILE = config.ROOT / "enhancer_history.json"
MAX_RUNS = 1000
TEXT = 20000            # the longest text kept per field of a run


def _small(run):
    return {k: (v[:TEXT] if isinstance(v, str) else v) for k, v in run.items()}


def load(runs, convos, keep):
    """Fill `runs` (a deque) and `convos` (an OrderedDict) from the file. `keep`: how many turns a conversation holds."""
    try:
        data = json.loads(FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        log.warning("FunUI Prompt enhancer", f"the chat history could not be read ({exc}); it starts empty")
        return
    runs.extend(r for r in data.get("runs", []) if isinstance(r, dict))
    for key, state in (data.get("convos") or {}).items():
        if isinstance(state, dict):
            convos[key] = {"summary": str(state.get("summary") or ""), "turns": collections.deque((t for t in state.get("turns", []) if isinstance(t, dict)), maxlen=keep),
                           "seen": set(state.get("seen") or [])}


def save(runs, convos):
    """Write both, atomically."""
    data = {"runs": [_small(r) for r in runs],
            "convos": {k: {"summary": v["summary"], "turns": list(v["turns"]), "seen": sorted(v["seen"], key=str)} for k, v in convos.items()}}
    tmp = FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, FILE)
    except OSError as exc:
        log.once("enhancer:history", log.WARNING, "FunUI Prompt enhancer", f"the chat history could not be saved ({exc})")
