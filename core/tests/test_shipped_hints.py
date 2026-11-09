"""Every setting a module offers says what it does, and the words match the control.

A setting with no hint leaves the person guessing; a hint that names a control that is not
there ("set it in the inspector") sends them looking for something that does not exist.
"""

import pytest


@pytest.fixture(scope="module")
def shipped(comfyui):
    from core import registry as registry_mod
    return registry_mod.scan()


def _settings(registry):
    for spec in registry.specs.values():
        for name, setting in (spec.settings or {}).items():
            yield spec.id, name, setting


def test_every_setting_has_a_hint(shipped):
    bare = [f"{mod}.{name}" for mod, name, s in _settings(shipped) if not str(s.get("hint") or "").strip()]
    assert not bare, f"settings with no hint: {bare}"


def test_no_two_rows_of_one_module_share_a_label(shipped):
    from collections import Counter
    twins = []
    for spec in shipped.specs.values():
        count = Counter(s.get("label") or n for n, s in (spec.settings or {}).items())
        twins += [f"{spec.id}: {label}" for label, n in count.items() if n > 1]
    assert not twins, twins


def test_no_hint_points_at_the_inspector_for_a_control_it_does_not_have():
    from modules.output.render import library
    text = " ".join(str(e.get("description", "")) for e in library.EFFECTS)
    assert "inspector" not in text.lower()


def test_no_setting_hint_points_at_the_inspector(shipped):
    assert not [f"{m}.{n}" for m, n, s in _settings(shipped) if "inspector" in str(s.get("hint", "")).lower()]


def test_no_hint_or_tooltip_is_an_essay():
    """Hints are one or two plain sentences. Details belong in the docs."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    limit, long = 200, []
    for path in sorted(root.glob("modules/**/*.py")) + sorted(root.glob("core/*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            pairs = []
            if isinstance(n, ast.keyword) and n.arg in ("tooltip", "hint"):
                pairs = [n.value]
            elif isinstance(n, ast.Dict):
                pairs = [v for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value in ("tooltip", "hint")]
            for v in pairs:
                try:
                    text = ast.literal_eval(v)
                except Exception:   # noqa: BLE001 -- built at run time, not checked here
                    continue
                if isinstance(text, str) and len(text) > limit:
                    long.append(f"{path.relative_to(root)}:{v.lineno} ({len(text)})")
    assert not long, f"hints over {limit} characters: {long}"
