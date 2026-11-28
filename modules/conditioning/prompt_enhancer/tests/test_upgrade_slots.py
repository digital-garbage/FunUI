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
