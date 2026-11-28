def _base():
    return [
        {"id": "clip", "node": "FunUICLIPLoader", "inputs": {}},
        {"id": "positive", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "generation.prompt", "input": "text", "label": "Prompt"}], "inputs": {"clip": ["clip", 0], "text": "hi"}},
    ]


def test_an_enhancer_added_by_hand_is_wired_in_front_of_the_prompt_and_keeps_its_settings():
    from modules.conditioning.prompt_enhancer import upgrade_slots
    slots = _base() + [{"id": "funuienhanceprompt", "node": "FunUIEnhancePrompt", "inputs": {"temperature": 0.9, "engine_model": "x"}}]
    fixed = upgrade_slots(slots)
    e = next(s for s in fixed if s["node"] == "FunUIEnhancePrompt"); p = next(s for s in fixed if s["id"] == "positive")
    assert e["inputs"]["temperature"] == 0.9 and e["inputs"]["engine_model"] == "x"
    assert e["inputs"]["clip"] == ["clip", 0] and e["inputs"]["text"] == "hi"
    assert p["inputs"]["text"] == ["funuienhanceprompt", 0] and p["roles"] == []
    assert {r["input"] for r in e["roles"]} == {"text", "enabled"}
    assert fixed.index(e) < fixed.index(p)
    assert upgrade_slots(fixed) == fixed                                # done once


def test_a_wired_enhancer_and_a_pipeline_without_one_are_left_alone():
    from modules.conditioning.prompt_enhancer import upgrade_slots
    assert upgrade_slots(_base()) == _base()
    wired = _base()
    wired[1] = {**wired[1], "inputs": {"clip": ["clip", 0], "text": ["enhance", 0]}}
    wired.insert(1, {"id": "enhance", "node": "FunUIEnhancePrompt", "inputs": {}})
    assert upgrade_slots(wired) == wired


def test_an_enhancer_is_linked_to_the_pipeline_settings_so_prompt_steps_run_before_it():
    from modules.conditioning.prompt_enhancer import upgrade_slots
    slots = _base() + [{"id": "hub", "node": "FunUIModifierSettings", "inputs": {}},
                       {"id": "enhance", "node": "FunUIEnhancePrompt", "inputs": {}}]
    fixed = upgrade_slots(slots)
    assert next(s for s in fixed if s["id"] == "enhance")["inputs"]["settings"] == ["hub", 0]
    assert upgrade_slots(fixed) == fixed


def test_prompt_steps_that_are_on_run_before_the_model_and_a_broken_one_is_skipped(monkeypatch):
    import types
    from modules.conditioning.prompt_enhancer import nodes
    seen = []
    steps = [(types.SimpleNamespace(id="a"), lambda text, settings, seconds: (seen.append((text, settings, seconds)), text + " [A]")[1]),
             (types.SimpleNamespace(id="b"), lambda text, settings, seconds: 1 / 0),
             (types.SimpleNamespace(id="c"), lambda text, settings, seconds: None)]
    monkeypatch.setattr(nodes.registry_mod, "current", lambda: types.SimpleNamespace(providers=lambda cap: steps if cap == "prompt_step" else []))
    assert nodes.prepare("go", {"x": 1}, "5.0 seconds (121 frames at 24 fps)") == "go [A]"
    assert seen == [("go", {"x": 1}, 5.0)]
