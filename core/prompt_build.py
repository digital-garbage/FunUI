"""The text actually sent as a scene's prompt at generation time.

`Scene.text` (core/projects.py) stays the user's literal typed words -- this
module never writes back to it. It only ever runs at the moment a prompt
value is about to leave the app for a real run, so editing a scene and
looking at what it produced always shows what was actually typed, not an
expanded copy nothing can trace back.

Order matches v4's: anchor and postfix are joined to the scene text FIRST,
then shortcuts expand across the whole combined string (a shortcut trigger
sitting right at the anchor/scene boundary still matches), then `$variables`
resolve last. Resolving last means a variable's OWN text is never re-scanned
for shortcut triggers -- only the anchor/scene/postfix text is -- but a
variable may still reference another variable, since resolve_variables()
recurses through the variable map itself (see its own docstring for the
cycle-safety that recursion needs).
"""

from __future__ import annotations

import re

from . import shortcuts as shortcuts_mod


#: A typed [A|B], found in the raw text (before any shortcut or variable has put brackets of its own in).
_TYPED_ALTERNATIVES = re.compile(r"(?<!\\)\[([^\[\]|]+(?:\|[^\[\]|]+)+)\]")


def _protect(m):
    opts = [o.strip() for o in m.group(1).split("|") if o.strip()]
    if len(opts) < 2:
        return m.group(0)
    return shortcuts_mod.ALT_OPEN + shortcuts_mod.ALT_SEP.join(opts) + shortcuts_mod.ALT_CLOSE


def build(scene_text: str, *, anchor: str = "", postfix: str = "",
          postfix_enabled: bool = True, variables=None, shortcuts=None, seed: int = 0,
          commit: bool = False) -> str:
    anchor = (anchor or "").strip()
    postfix = (postfix or "").strip() if postfix_enabled else ""
    combined = " ".join(p for p in (anchor, str(scene_text or "").strip(), postfix) if p)
    combined = _TYPED_ALTERNATIVES.sub(_protect, combined)
    expanded = shortcuts_mod.expand(combined, shortcuts=shortcuts, seed=seed, commit=commit, alternatives=True)
    return shortcuts_mod.resolve_variables(expanded, variables)


_HEAD_MARK, _TAIL_MARK = "", ""


def build_parts(scene_text: str, **kw) -> dict | None:
    """-> {"head", "tail"}: the expanded anchor and postfix exactly as they sit in build()'s result, so a caller can
    work on the text between them and put them back untouched. None when they cannot be told apart (a shortcut
    that reaches across the boundary, a random pick that falls differently): the caller then treats the prompt as one piece."""
    full = build(scene_text, **kw)
    marked = build(str(scene_text or ""), **{**kw, "anchor": (kw.get("anchor") or "").strip() + _HEAD_MARK,
                                             "postfix": _TAIL_MARK + (kw.get("postfix") or "").strip()})
    head, sep, rest = marked.partition(_HEAD_MARK)
    if not sep:
        return None
    mid, _, tail = rest.partition(_TAIL_MARK)
    head, tail = head.strip(), tail.strip()
    if " ".join(" ".join(p for p in (head, mid.strip(), tail) if p).split()) != " ".join(full.split()):
        return None
    if (head and not full.startswith(head)) or (tail and not full.endswith(tail)) or len(head) + len(tail) > len(full):
        return None
    return {"head": head, "tail": tail}
