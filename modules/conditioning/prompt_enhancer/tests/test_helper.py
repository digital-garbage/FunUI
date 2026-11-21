import threading

from .. import helper, rules


def run(call, scenes=2, answer=None):
    seen = []

    def on_event(ev):
        seen.append(ev)
        if answer is not None:
            threading.Timer(0.01, lambda: rules.decide(ev["id"], answer)).start()
    return helper.Actions(scenes, on_event, wait=2)(call), seen


def test_allowed_change_asks_once_and_says_so():
    out, seen = run({"name": "edit_scene", "args": {"scene": 2, "text": "it sits", "reason": "r"}}, answer=True)
    assert out.startswith("Allowed") and len(seen) == 1
    assert seen[0]["kind"] == "action_request" and "Scene 2" in seen[0]["summary"]


def test_refused_change_tells_the_model_not_to_repeat():
    out, _ = run({"name": "add_scene", "args": {"text": "a cat", "reason": "r"}}, answer=False)
    assert out.startswith("Disallowed")


def test_bad_calls_never_reach_the_person():
    for call in ({"name": "edit_scene", "args": {"scene": 5, "text": "x"}}, {"name": "edit_scene", "args": {"scene": 1}},
                 {"name": "set_clip_length", "args": {"seconds": -1}}, {"name": "add_scene", "args": {"text": "x", "after": 7}}, {"name": "nope", "args": {}}):
        out, seen = run(call)
        assert out.startswith("Error") and not seen


def test_unanswered_change_is_not_made():
    out, _ = run({"name": "set_clip_length", "args": {"seconds": 5}})
    assert out.startswith("No answer")


def test_copy_reference_needs_two_different_scenes():
    out, seen = run({"name": "copy_reference", "args": {"scene": 2, "from_scene": 2}})
    assert out.startswith("Error") and not seen
    out, seen = run({"name": "copy_reference", "args": {"scene": 2, "from_scene": 1, "reason": "r"}}, answer=True)
    assert out.startswith("Allowed") and "same reference" in seen[0]["summary"]


SETTINGS = [{"module": "sampling_attention_bandit", "name": "enabled", "title": "Attention explorer", "label": "Attention explorer", "type": "bool", "now": False, "hint": "Each run tries a small attention change"},
            {"module": "sampling_attention_bandit", "name": "range", "title": "Attention explorer", "label": "Change size", "type": "float", "min": 0.05, "max": 0.5, "now": 0.2, "hint": ""},
            {"module": "post_auto_contrast", "name": "mode", "title": "Auto contrast", "label": "Mode", "type": "enum", "options": ["off", "on"], "now": "off", "hint": "Evens out contrast"}]


def test_find_settings_answers_without_asking_and_set_setting_checks_values():
    acts = helper.Actions(1, lambda ev: None, wait=1, settings=SETTINGS)
    found = acts({"name": "find_settings", "args": {"query": "attention explorer"}})
    assert "sampling_attention_bandit.enabled" in found and "post_auto_contrast" not in found
    assert acts({"name": "find_settings", "args": {"query": "contrast"}}).startswith("post_auto_contrast.mode")
    for bad in ({"module": "x", "setting": "y", "value": 1}, {"module": "sampling_attention_bandit", "setting": "range", "value": 5},
                {"module": "sampling_attention_bandit", "setting": "enabled", "value": "maybe"}, {"module": "post_auto_contrast", "setting": "mode", "value": "wild"}):
        assert acts({"name": "set_setting", "args": bad}).startswith("Error")


def test_allowed_setting_change_carries_the_typed_value():
    seen = []

    def on_event(ev):
        seen.append(ev)
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    out = helper.Actions(1, on_event, wait=2, settings=SETTINGS)({"name": "set_setting", "args": {"module": "sampling_attention_bandit", "setting": "enabled", "value": "true", "reason": "r"}})
    assert out.startswith("Allowed") and seen[0]["args"]["value"] is True
    assert "Attention explorer" in seen[0]["summary"]
