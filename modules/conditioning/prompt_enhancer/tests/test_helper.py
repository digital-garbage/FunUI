import threading

from .. import helper, rules


def run(call, scenes=2, answer=None):
    seen = []

    def on_event(ev):
        seen.append(ev) if ev.get("kind") != "doing" else None
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
        seen.append(ev) if ev.get("kind") != "doing" else None
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
        seen.append(ev) if ev.get("kind") != "doing" else None
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
        seen.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    live = helper.Actions(1, on_event, wait=2, inputs=INPUTS, presets=[{"id": "fast", "title": "H3 fast"}])
    assert live({"name": "set_node_inputs", "args": {"changes": [{"slot": "s1", "input": "steps", "value": "5"}, {"slot": "L", "input": "unet_name", "value": "h3.safetensors"}], "reason": "r"}}).startswith("Allowed")
    assert seen[0]["args"]["changes"][0]["value"] == 5 and "20 -> 5" in seen[0]["summary"]
    assert live({"name": "apply_preset", "args": {"preset": "fast", "reason": "r"}}).startswith("Allowed")
    assert "H3 fast" in seen[1]["summary"]


def test_helper_reads_notes_but_never_the_chat_conversation():
    from .. import nodes
    from collections import deque
    nodes.CONVOS["p:0"] = {"summary": "talked about dogs", "turns": deque([{"request": "make it warmer", "reply": "ok, warmer light"}]), "seen": set()}
    try:
        text = nodes._chat_memory({"chat": "p:0"}, "no rain")
    finally:
        nodes.CONVOS.pop("p:0", None)
    assert "no rain" in text and "dogs" not in text and "warmer" not in text
    assert nodes._chat_memory({"chat": "none"}, "") == ""


def test_chat_log_is_read_only_through_the_tool():
    from .. import nodes
    from collections import deque
    nodes.CONVOS["p:1"] = {"summary": "talked about dogs", "turns": deque([{"request": "make it warmer", "reply": "ok"}]), "seen": set()}
    try:
        acts = helper.Actions(1, lambda ev: None, chat_log=lambda n: nodes._chat_log({"chat": "p:1"}, n))
        out = acts({"name": "read_chat_log", "arguments": {"turns": 3}})
        assert "dogs" in out and "make it warmer" in out and acts.changes == 0
        assert "read_chat_log" in [t["function"]["name"] for t in helper.tools()]
        assert "Chat tab" in helper.Actions(1, lambda ev: None)({"name": "read_chat_log", "arguments": {}})
    finally:
        nodes.CONVOS.pop("p:1", None)


def test_clear_forgets_the_conversation_and_its_job():
    from .. import nodes, jobs
    from collections import deque
    nodes.CONVOS["helper:x"] = {"summary": "", "turns": deque([{"request": "a", "reply": "b"}]), "seen": set()}
    jobs.JOBS["helper:x"] = object()
    assert nodes.clear_chat("helper:x") is True
    assert "helper:x" not in nodes.CONVOS and "helper:x" not in jobs.JOBS
    assert nodes.clear_chat("helper:x") is False


def test_app_reminders_are_marked_as_the_apps():
    for text in (helper.NUDGE, helper.PLAN_NOTE, helper.FORCE_AGAIN, helper.force_note("change it", False), helper.Actions(1, lambda e: None).plan_note() if False else helper.app_note("x")):
        assert text.startswith("APP NOTE")
    assert "APP NOTE" in helper.HELPER_RULE


def test_cuts_are_checked_and_counted():
    acts = helper.Actions(2, lambda ev: None, wait=1)
    for bad in ({"name": "cut_scene", "args": {"scene": 3}}, {"name": "cut_scene", "args": {"scene": 1, "at_seconds": -2}},
                {"name": "set_cut_words", "args": {"words": []}}, {"name": "set_cut_words", "args": {"words": ["two words"]}}):
        assert acts(bad).startswith("Error"), bad
    seen = []

    def on_event(ev):
        seen.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    live = helper.Actions(1, on_event, wait=2)
    assert live({"name": "cut_scene", "args": {"scene": 1, "at_seconds": 2, "reason": "r"}}).startswith("Allowed")
    assert live({"name": "cut_scene", "args": {"scene": 2, "reason": "r"}}).startswith("Allowed")      # the cut made a second scene
    assert live({"name": "set_cut_words", "args": {"words": ["qcut"], "reason": "r"}}).startswith("Allowed")
    assert "at 2 seconds" in seen[0]["summary"]


def test_prefix_means_the_anchor_and_the_postfix_switch_takes_on_off():
    seen = []

    def on_event(ev):
        seen.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    acts = helper.Actions(1, on_event, wait=2)
    for name in ("prefix", "Anchor"):
        assert acts({"name": "set_project_text", "args": {"field": name, "text": "cinema", "reason": "r"}}).startswith("Allowed")
        assert seen[-1]["args"]["field"] == "anchor"
    assert acts({"name": "set_project_text", "args": {"field": "postfix_enabled", "text": "off", "reason": "r"}}).startswith("Allowed")
    assert seen[-1]["args"]["text"] is False and "off" in seen[-1]["summary"]
    assert acts({"name": "set_project_text", "args": {"field": "postfix_enabled", "text": "maybe"}}).startswith("Error")


def test_set_story_takes_the_whole_text():
    acts = helper.Actions(2, lambda ev: None, wait=1)
    assert acts({"name": "set_story", "args": {"text": "  "}}).startswith("Error")
    assert acts({"name": "set_story", "args": {"text": 5}}).startswith("Error")
    seen = []

    def on_event(ev):
        seen.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    assert helper.Actions(2, on_event, wait=2)({"name": "set_story", "args": {"text": "a\nqcut\nb", "reason": "r"}}).startswith("Allowed")
    assert "Rewrite the global prompt" in seen[0]["summary"]


def test_rules_say_scenes_are_separate_prompts():
    assert "separate prompt" in helper.HELPER_RULE and "one at a time" in helper.HELPER_RULE


def test_run_action_is_checked_against_the_offered_actions():
    acts = [{"id": "generate-all", "label": "Generate"}, {"id": "generate-selected", "label": "Generate selected"}]
    a = helper.Actions(3, lambda ev: None, wait=1, actions=acts)
    assert a({"name": "run_action", "args": {"action": "format-disk"}}).startswith("Error")
    assert a({"name": "run_action", "args": {"action": "generate-selected", "scenes": [4]}}).startswith("Error")
    seen = []

    def on_event(ev):
        seen.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    assert helper.Actions(3, on_event, wait=2, actions=acts)({"name": "run_action", "args": {"action": "generate-selected", "scenes": [2, 3], "reason": "r"}}).startswith("Allowed")
    assert seen[0]["summary"] == "Run: Generate selected on scenes 2, 3"


def test_rules_make_the_global_prompt_a_quick_summary_per_shot():
    assert "never write detailed descriptions of each shot" in helper.HELPER_RULE and "quick summary of the whole video" in helper.HELPER_RULE


def test_announcing_a_change_without_calling_is_noticed():
    assert helper.announced("I will use the `set_story` tool to consolidate the content. I propose the following text:")
    assert helper.announced("I propose a shorter prompt.")
    assert not helper.announced("Done. Scene 2 is longer now.")
    assert "call the tool" in helper.NUDGE.lower() and "Act, do not announce" in helper.HELPER_RULE


def test_calls_are_counted_so_a_missing_call_can_be_seen():
    acts = helper.Actions(1, lambda ev: None, wait=1)
    assert acts.calls == 0
    acts({"name": "find_settings", "args": {"query": "x"}})
    assert acts.calls == 1


def test_a_scene_prefix_is_a_valid_change_on_its_own():
    acts = helper.Actions(2, lambda ev: None, wait=1)
    assert acts({"name": "edit_scene", "args": {"scene": 1}}).startswith("Error")
    seen = []

    def on_event(ev):
        seen.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    live = helper.Actions(2, on_event, wait=2)
    assert live({"name": "edit_scene", "args": {"scene": 1, "prefix": "", "reason": "r"}}).startswith("Allowed")
    assert "(none)" in seen[0]["summary"]
    assert live({"name": "edit_scene", "args": {"scene": 2, "project_prefix": True, "reason": "r"}}).startswith("Allowed")


def test_rules_forbid_shot_labels_in_the_story():
    assert "never labels or numbering such as [Shot 1]" in helper.HELPER_RULE


def test_share_reference_names_a_real_scene():
    acts = helper.Actions(2, lambda ev: None, wait=1)
    assert acts({"name": "share_reference", "args": {"from_scene": 3}}).startswith("Error")
    seen = []

    def on_event(ev):
        seen.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    assert helper.Actions(2, on_event, wait=2)({"name": "share_reference", "args": {"from_scene": 1, "reason": "r"}}).startswith("Allowed")
    assert "every scene" in seen[0]["summary"]


def test_tool_blocks_in_an_answer_are_parsed_run_and_answered():
    text, calls = helper.parse_calls('Sure.\nTOOL: set_clip_length\n{"seconds": 6, "reason": "r"}\nEND_TOOL\nTOOL: find_settings\n{"query": "contrast"}')
    assert text == "Sure." and [c["name"] for c in calls] == ["set_clip_length", "find_settings"] and calls[0]["args"]["seconds"] == 6
    assert helper.parse_calls("TOOL: x\nnot json\nEND_TOOL")[1][0]["error"]
    assert "TOOL: set_clip_length" in helper.text_rule() and "edit_scene(" in helper.text_rule()


def test_text_mode_runs_blocks_until_the_model_answers_in_words():
    acts = helper.Actions(1, lambda ev: None, wait=1, settings=SETTINGS)
    state = {"turns": []}
    asked = []

    def ask(text):
        asked.append((text, [t["reply"] for t in state["turns"]]))
        return ("Auto contrast is off and can be set.", "ok", [], [], "")
    first = ('TOOL: find_settings\n{"query": "contrast"}\nEND_TOOL', "ok", ["note"], [], "")
    reply, status, add, drop, thought = helper.run_text_calls(ask, acts, "turn on auto contrast", state, first)
    assert reply == "Auto contrast is off and can be set." and add == ["note"]
    assert (helper.APP_PREFIX + "TOOL RESULTS:\nfind_settings: post_auto_contrast.mode") in asked[0][0]
    assert asked[0][1] == [first[0]] and state["turns"] == []                       # it saw its own block; nothing of the loop stays in the conversation


def test_tool_arguments_written_loosely_by_a_small_model_are_still_read():
    from .. import helper
    for body, want in [
        ('{"scene": 1, "text": "a face, close-up", "reason": "asked"}', "a face, close-up"),
        ('```json\n{"scene": 1, "text": "a face", "reason": "x",}\n```', "a face"),
        ("{'scene': 1, 'text': 'a face', 'reason': 'x'}", "a face"),
        ('{"scene": 1, "text": "a face\nclose, soft light", "reason": "x"}', "a face\nclose, soft light"),
        ('{"scene": 1, "text": "she says "hello" softly", "reason": "x"}', 'she says "hello" softly'),
        ('{"scene": 1, "text": "a face", "reason": "he said "go""}', "a face"),
    ]:
        got = helper.loose_json(body)
        assert got["text"] == want and got["scene"] == 1, body
    assert helper.parse_calls("TOOL: edit_scene\n" + '{"scene": 2, "text": "x "y" z", "reason": "r"}' + "\nEND_TOOL")[1][0]["args"]["text"] == 'x "y" z'


def test_a_qwen_style_xml_call_and_a_loose_json_call_are_both_read():
    from .. import engine
    got = engine._json_call("<function=edit_scene>\n<parameter=scene>\n1\n</parameter>\n<parameter=text>\na face, close-up\n</parameter>\n</function>")
    assert got == {"name": "edit_scene", "args": {"scene": 1, "text": "a face, close-up"}}
    assert engine._json_call("{'name': 'set_story', 'arguments': {'text': 'a \"b\" c'}}")["args"]["text"] == 'a "b" c'


def test_the_helper_reads_and_changes_the_enhancers_system_prompts_through_the_same_allow(monkeypatch):
    import threading
    from .. import helper, rules
    heard = []

    def on_event(ev):
        heard.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    a = helper.Actions(2, on_event, rules_given={"chat_prompt": "Answer briefly."})
    names = [t["function"]["name"] for t in helper.tools(True)]
    assert "edit_prompt" in names and "read_prompt" in names and "edit_prompt" not in [t["function"]["name"] for t in helper.tools()]
    assert a({"name": "read_prompt", "args": {"field": "chat_prompt"}}) == "Answer briefly."
    out = a({"name": "edit_prompt", "args": {"field": "chat_prompt", "new_text": "Answer briefly for a [[ASPECT_RATIO]] clip.", "reason": "asked"}})
    assert out.startswith("Allowed") and heard[0]["kind"] == "edit_request" and heard[0]["field"] == "chat_prompt"
    assert "[[ASPECT_RATIO]]" in helper.rules_note({"chat_prompt": "x"}) and "main rewrite instructions" in helper.rules_note({"chat_prompt": "x"})
    assert helper.Actions(2, on_event)({"name": "edit_prompt", "args": {}}).startswith("Error")


def test_many_shortcuts_are_changed_renamed_and_deleted_under_one_question():
    lib = [{"name": "a", "triggers": ["aa"], "replacements": ["one"]}, {"name": "b", "triggers": ["bb"], "replacements": ["two"]}, {"name": "c", "triggers": ["cc"], "replacements": ["three"]}]
    acts = helper.Actions(1, lambda ev: None, wait=1, shortcuts=lib)
    for bad in ([], [{"name": "zz", "new_name": "q"}], [{"name": "a"}], [{"name": "a", "new_name": "b"}], [{"name": "a", "triggers": []}],
                [{"name": "a", "category": "x"}, {"name": "a", "category": "y"}]):
        assert acts({"name": "edit_shortcuts", "args": {"changes": bad, "reason": "r"}}).startswith("Error"), bad
    seen = []

    def on_event(ev):
        seen.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    live = helper.Actions(1, on_event, wait=2, shortcuts=lib)
    out = live({"name": "edit_shortcuts", "args": {"reason": "r", "changes": [{"name": "a", "new_name": "alpha"}, {"name": "b", "replacements": ["new two"], "category": "k"}, {"name": "c", "delete": True}]}})
    assert out.startswith("Allowed") and len(seen) == 1                       # one question for the whole batch
    assert "rename to alpha" in seen[0]["summary"] and "delete c" in seen[0]["summary"]
    assert set(live.shortcuts) == {"alpha", "b"} and live.shortcuts["alpha"]["triggers"] == ["aa"] and live.shortcuts["b"]["replacements"] == ["new two"]


def test_shortcut_lookup_says_how_many_there_are_and_pages_on():
    lib = [{"name": f"s{i}", "triggers": [f"t{i}"], "replacements": ["x"]} for i in range(100)]
    acts = helper.Actions(1, lambda ev: None, wait=1, shortcuts=lib)
    first = acts({"name": "find_shortcuts", "args": {"query": ""}})
    assert first.startswith("100 of the library's 100 shortcut(s) match; showing 1 to 40.") and "offset 40" in first and "s39:" in first and "s40:" not in first
    last = acts({"name": "find_shortcuts", "args": {"query": "", "offset": 80}})
    assert "showing 81 to 100" in last and "That is all of them." in last
    assert "library has 100" in acts({"name": "find_shortcuts", "args": {"query": "zzz"}})


def test_shortcut_lookup_finds_an_exact_phrase_in_quotes():
    acts = helper.Actions(1, lambda ev: None, wait=1, shortcuts=[{"name": "a", "triggers": ["x"], "replacements": ["slow dolly in"]}, {"name": "b", "triggers": ["y"], "replacements": ["in slow dolly"]}])
    found = acts({"name": "find_shortcuts", "args": {"query": '"dolly in"'}})
    assert found.startswith("1 of the library's 2") and "a:" in found and "b:" not in found


def test_a_shortcut_name_copied_with_other_case_spacing_or_a_lookalike_is_still_found():
    lib = [{"name": "Rain Walk", "triggers": ["rw"], "replacements": ["x"]}, {"name": "Sun  Set", "triggers": ["ss"], "replacements": ["y"]}, {"name": "Café", "triggers": ["c"], "replacements": ["z"]}]
    seen = []

    def on_event(ev):
        seen.append(ev) if ev.get("kind") != "doing" else None
        threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    acts = helper.Actions(1, on_event, wait=2, shortcuts=lib)
    out = acts({"name": "edit_shortcuts", "args": {"reason": "r", "changes": [{"name": " rain walk ", "new_name": "Walk in rain"}, {"name": "sun set", "delete": True}, {"name": "Café", "category": "k"}]}})
    assert out.startswith("Allowed"), out
    assert [c["name"] for c in seen[0]["args"]["changes"]] == ["Rain Walk", "Sun  Set", "Café"]          # the library's own spelling goes on
    assert set(acts.shortcuts) == {"Walk in rain", "Café"}
    lost = acts({"name": "edit_shortcuts", "args": {"reason": "r", "changes": [{"name": "Walk in rai", "delete": True}, {"name": "nope", "delete": True}]}})
    assert lost.startswith("Error") and "'nope'" in lost and "Nothing was changed" in lost and len(seen) == 1
    assert acts({"name": "remove_shortcut", "args": {"name": "walk IN rain", "reason": "r"}}).startswith("Allowed") and "Walk in rain" not in acts.shortcuts


def test_a_shortcut_name_followed_by_a_description_in_brackets_is_still_found():
    lib = [{"name": "Arms up", "triggers": ["au"], "replacements": ["x"]}, {"name": "Squat (low)", "triggers": ["sq"], "replacements": ["y"]}]
    acts = helper.Actions(1, lambda ev: None, wait=1, shortcuts=lib)
    assert acts.shortcut_named("Arms up (hands over the head)") == "Arms up"
    assert acts.shortcut_named("Squat") == "Squat (low)" and acts.shortcut_named("Squat (deep)") == "Squat (low)"
    assert acts.shortcut_named("Jump (high)") is None


def test_a_checklist_is_ticked_only_after_a_change_and_every_item_can_be_closed_with_a_reason():
    acts = helper.Actions(1, lambda ev: None, wait=1)
    assert acts({"name": "check_item", "args": {"item": 1, "result": "done"}}).startswith("Error: there is no checklist")
    assert acts({"name": "plan", "args": {"items": []}}).startswith("Error")
    assert acts({"name": "plan", "args": {"items": ["rename a", "rename b", "rename c"]}}).startswith("Checklist saved with 3 items")
    assert acts.changes == 0 and acts.open_items() == [1, 2, 3]                 # writing the list is not a change
    assert acts({"name": "check_item", "args": {"item": 1, "result": "done"}}).startswith("Error: no change was made")
    assert acts({"name": "check_item", "args": {"item": 9, "result": "done"}}).startswith("Error: item is a number")
    assert acts({"name": "check_item", "args": {"item": 3, "result": "cannot"}}).startswith("Error: say why")
    assert "marked skipped" in acts({"name": "check_item", "args": {"item": 3, "result": "cannot: the shortcut does not exist"}})
    acts.changes += 1                                                         # a change tool ran
    assert "Still open: 2" in acts({"name": "check_item", "args": {"item": 1, "result": "done"}})
    assert acts.open_items() == [2] and "2. rename b" in acts.plan_note() and "1 of 3" not in acts.plan_note()


def test_the_model_is_sent_back_to_an_unfinished_checklist_until_every_item_is_checked(monkeypatch):
    import threading
    from modules.conditioning.prompt_enhancer import nodes, rules as rl

    class Tiny:
        tok = type("T", (), {"chat_template": "plain chat", "get_vocab": lambda self: {}})()
        def load(self, *a, **k): return None
    monkeypatch.setattr(nodes, "ENGINE", Tiny())
    monkeypatch.setattr(nodes, "EngineClip", lambda *a, **k: type("C", (), {"tools": None, "handler": None})())
    nodes.CONVOS.clear()
    seen = []

    def say(tool, **args):
        import json
        return f"TOOL: {tool}\n{json.dumps(args)}\nEND_TOOL"
    script = iter([say("plan", items=["set $a", "set $b", "set $c"]) + "\n" + say("set_variable", name="a", value="1", reason="r") + "\n" + say("check_item", item=1, result="done"),
                   "That is all.",                                               # it stops with two items open
                   say("set_variable", name="b", value="2", reason="r") + "\n" + say("check_item", item=2, result="done"),
                   "Item 2 is done.",
                   say("check_item", item=3, result="cannot: $c is already set to the same value"),
                   "Finished."])

    def converse(clip, text, **kw):
        seen.append(text)
        return next(script, "Finished."), "ok", [], [], ""
    monkeypatch.setattr(nodes.en, "converse", converse)
    answered = []

    def on_event(ev):
        if ev.get("kind") == "action_request":
            answered.append(ev["tool"])
            threading.Timer(0.01, lambda: rl.decide(ev["id"], True)).start()
    out = nodes.chat_reply("h:plan", "set three variables", engine_model="tiny", max_length=8, on_event=on_event, helper={"sheet": "x", "scenes": 1})
    assert answered == ["set_variable", "set_variable"]
    assert any("checklist is not finished" in t and "2. set $b" in t for t in seen) and any("1. " not in t.split("Still open:")[1] for t in seen if "Still open:" in t)
    assert out["reply"] == "Finished." and "Not finished" not in out["reply"]


def test_even_one_change_is_made_from_a_checklist_the_model_is_asked_for(monkeypatch):
    import json
    import threading
    from modules.conditioning.prompt_enhancer import nodes, rules as rl

    class Tiny:
        tok = type("T", (), {"chat_template": "plain chat", "get_vocab": lambda self: {}})()
        def load(self, *a, **k): return None
    monkeypatch.setattr(nodes, "ENGINE", Tiny())
    monkeypatch.setattr(nodes, "EngineClip", lambda *a, **k: type("C", (), {"tools": None, "handler": None})())
    nodes.CONVOS.clear()
    block = lambda tool, **a: f"TOOL: {tool}\n{json.dumps(a)}\nEND_TOOL"        # noqa: E731
    seen = []
    script = iter([block("set_variable", name="a", value="1", reason="r"), "I will try again.",                    # one change, no list, and it stops
                   block("plan", items=["set $a"]) + "\n" + block("check_item", item=1, result="done"), "Checked."])

    def converse(clip, text, **kw):
        seen.append(text)
        return next(script, "Checked."), "ok", [], [], ""
    monkeypatch.setattr(nodes.en, "converse", converse)
    done = []

    def on_event(ev):
        if ev.get("kind") == "action_request":
            done.append(ev["tool"])
            threading.Timer(0.01, lambda: rl.decide(ev["id"], True)).start()
    out = nodes.chat_reply("h:one", "set the variable a to 1", engine_model="tiny", max_length=8, on_event=on_event, helper={"sheet": "x", "scenes": 1})
    assert done == ["set_variable"] and any("checklist" in t.lower() and "even a single one" in t for t in seen)
    assert "Not finished" not in out["reply"]


def test_a_text_written_again_and_again_is_seen_as_a_loop():
    from .. import gguf
    unit = 'call:edit_shortcuts{changes:[{name:"a"}]} '
    assert gguf.looping("start " + unit * 5)
    assert not gguf.looping("call:edit_shortcuts{changes:[{name:" + "".join(f'"s{i}",' for i in range(80)) + "]}")
    assert not gguf.looping("a short answer")


def test_long_shortcut_batches_are_refused_so_the_model_sends_several_calls():
    acts = helper.Actions(1, lambda ev: None, shortcuts=[{"name": f"s{i}", "triggers": [f"t{i}"], "replacements": ["x"]} for i in range(30)])
    out = acts({"name": "edit_shortcuts", "args": {"changes": [{"name": f"s{i}", "new_name": f"n{i}"} for i in range(21)]}})
    assert "one to 20" in out
