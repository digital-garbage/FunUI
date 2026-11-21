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
