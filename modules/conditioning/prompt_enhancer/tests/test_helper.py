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
    assert out.startswith("Disallowed") and "Ask yourself why" in out


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


def test_project_text_variables_and_shortcuts_are_checked_before_asking():
    acts = helper.Actions(1, lambda ev: None, wait=1, variables=["dog"], shortcuts=[{"name": "sunset", "triggers": ["sunset"], "replacements": ["golden light"]}])
    for bad in ({"name": "set_project_text", "args": {"field": "title", "text": "x"}}, {"name": "set_variable", "args": {"name": "a b", "value": "x"}},
                {"name": "remove_variable", "args": {"name": "cat"}}, {"name": "save_shortcut", "args": {"name": "x", "triggers": [], "replacements": ["y"]}},
                {"name": "remove_shortcut", "args": {"name": "nope"}}):
        assert acts(bad).startswith("Error"), bad
    assert "golden light" in acts({"name": "find_shortcuts", "args": {"query": "sunset"}})
    seen = []

    def on_event(ev):
        seen.append(ev)
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    live = helper.Actions(1, on_event, wait=2, variables=["dog"], shortcuts=[{"name": "sunset", "triggers": ["sunset"], "replacements": ["x"]}])
    assert live({"name": "save_shortcut", "args": {"name": "sunset", "triggers": ["sunset"], "replacements": ["y"], "reason": "r"}}).startswith("Allowed")
    assert seen[0]["args"]["existing"] is True
    assert live({"name": "set_variable", "args": {"name": "$cat", "value": "tabby", "reason": "r"}}).startswith("Allowed")
    assert live({"name": "remove_variable", "args": {"name": "cat", "reason": "r"}}).startswith("Allowed")


INPUTS = [{"slot": "s1", "node": "FunUISampler", "group": "Sampling", "name": "steps", "label": "steps", "type": "INT", "min": 1, "max": 100, "now": 20},
          {"slot": "s1", "node": "FunUISampler", "group": "Sampling", "name": "sampler_name", "label": "sampler_name", "type": "COMBO", "choices": ["euler", "seeds_2"], "now": "euler"},
          {"slot": "L", "node": "UNETLoader", "group": "Models", "name": "unet_name", "label": "unet_name", "type": "COMBO", "choices": ["h3.safetensors"], "now": ""}]


def test_pipeline_inputs_are_looked_up_checked_and_changed_under_one_question():
    acts = helper.Actions(1, lambda ev: None, wait=1, inputs=INPUTS, presets=[{"id": "fast", "title": "H3 fast"}])
    found = acts({"name": "find_node_inputs", "args": {"query": "model unet"}})
    assert "L.unet_name" in found and "NOT SET" in found and "h3.safetensors" in found
    for bad in ([{"slot": "s1", "input": "steps", "value": 500}], [{"slot": "s1", "input": "sampler_name", "value": "nope"}], [{"slot": "zz", "input": "steps", "value": 5}], []):
        assert acts({"name": "set_node_inputs", "args": {"changes": bad}}).startswith("Error"), bad
    assert acts({"name": "apply_preset", "args": {"preset": "other"}}).startswith("Error")
    seen = []

    def on_event(ev):
        seen.append(ev)
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    live = helper.Actions(1, on_event, wait=2, inputs=INPUTS, presets=[{"id": "fast", "title": "H3 fast"}])
    assert live({"name": "set_node_inputs", "args": {"changes": [{"slot": "s1", "input": "steps", "value": "5"}, {"slot": "L", "input": "unet_name", "value": "h3.safetensors"}], "reason": "r"}}).startswith("Allowed")
    assert seen[0]["args"]["changes"][0]["value"] == 5 and "20 -> 5" in seen[0]["summary"]
    assert live({"name": "apply_preset", "args": {"preset": "fast", "reason": "r"}}).startswith("Allowed")
    assert "H3 fast" in seen[1]["summary"]


def test_helper_reads_the_chat_conversation_and_notes():
    from .. import nodes
    from collections import deque
    nodes.CONVOS["p:0"] = {"summary": "talked about dogs", "turns": deque([{"request": "make it warmer", "reply": "ok, warmer light"}]), "seen": set()}
    try:
        text = nodes._chat_memory({"chat": "p:0"}, "no rain")
    finally:
        nodes.CONVOS.pop("p:0", None)
    assert "no rain" in text and "talked about dogs" in text and "make it warmer" in text
    assert nodes._chat_memory({"chat": "none"}, "") == ""


def test_cuts_are_checked_and_counted():
    acts = helper.Actions(2, lambda ev: None, wait=1)
    for bad in ({"name": "cut_scene", "args": {"scene": 3}}, {"name": "cut_scene", "args": {"scene": 1, "at_seconds": -2}},
                {"name": "set_cut_words", "args": {"words": []}}, {"name": "set_cut_words", "args": {"words": ["two words"]}}):
        assert acts(bad).startswith("Error"), bad
    seen = []

    def on_event(ev):
        seen.append(ev)
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    live = helper.Actions(1, on_event, wait=2)
    assert live({"name": "cut_scene", "args": {"scene": 1, "at_seconds": 2, "reason": "r"}}).startswith("Allowed")
    assert live({"name": "cut_scene", "args": {"scene": 2, "reason": "r"}}).startswith("Allowed")      # the cut made a second scene
    assert live({"name": "set_cut_words", "args": {"words": ["qcut"], "reason": "r"}}).startswith("Allowed")
    assert "at 2 seconds" in seen[0]["summary"]


def test_prefix_means_the_anchor_and_the_postfix_switch_takes_on_off():
    seen = []

    def on_event(ev):
        seen.append(ev)
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    acts = helper.Actions(1, on_event, wait=2)
    for name in ("prefix", "Anchor"):
        assert acts({"name": "set_project_text", "args": {"field": name, "text": "cinema", "reason": "r"}}).startswith("Allowed")
        assert seen[-1]["args"]["field"] == "anchor"
    assert acts({"name": "set_project_text", "args": {"field": "postfix_enabled", "text": "off", "reason": "r"}}).startswith("Allowed")
    assert seen[-1]["args"]["text"] is False and "off" in seen[-1]["summary"]
    assert acts({"name": "set_project_text", "args": {"field": "postfix_enabled", "text": "maybe"}}).startswith("Error")
