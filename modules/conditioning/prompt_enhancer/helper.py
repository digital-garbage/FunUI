"""The helper: a chat the person opens to ask the model to act on the project, not only to talk.

The model gets a sheet of the project (scenes by number, with prompt and length) and three tools: `edit_scene`, `add_scene` and
`set_clip_length`. Nothing is applied here. Each call is sent to the page as an `action_request` event, shown as a card with Allow and
Disallow, and the answer (the same route as a rule change) goes back to the model. The page does the change in the project.
"""

import threading

from . import rules

MAX_TEXT = 4000
HELPER_RULE = ("You are the helper of a video project. You can act on it with tools. Every change is shown to the user, who allows or refuses it, so propose changes instead of describing them. "
               "Scenes are numbered from 1. This is the project now:\n[[PROJECT_SHEET]]\n"
               "Use edit_scene to change one scene's prompt or length, add_scene to add a scene, set_clip_length to change the default length of every scene. "
               "Lengths are in seconds. Change only what the user asked for. When a tool answers that the user refused, accept it and do not ask again.")


def tools() -> list:
    def fn(name, description, props, required):
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}}}
    scene = {"type": "integer", "description": "The scene number, from 1."}
    reason = {"type": "string", "description": "Why, in one sentence."}
    text = {"type": "string", "description": "The whole prompt text."}
    seconds = {"type": "number", "description": "The length in seconds."}
    return [fn("edit_scene", "Change the prompt and/or the length of one scene.", {"scene": scene, "text": text, "seconds": seconds, "reason": reason}, ["scene", "reason"]),
            fn("add_scene", "Add a new scene after another one (the last when after is left out).", {"text": text, "after": {"type": "integer", "description": "The scene number to add it after."}, "reason": reason}, ["text", "reason"]),
            fn("set_clip_length", "Set the default clip length of the whole project.", {"seconds": seconds, "reason": reason}, ["seconds", "reason"])]


def describe(name, args) -> str:
    """One plain line for the Allow card."""
    if name == "edit_scene":
        parts = ([f"change the prompt to: {str(args.get('text')).strip()}"] if isinstance(args.get("text"), str) and args["text"].strip() else []) + ([f"set the length to {args['seconds']} seconds"] if isinstance(args.get("seconds"), (int, float)) else [])
        return f"Scene {args.get('scene')}: " + " and ".join(parts)
    if name == "add_scene":
        return f"Add a scene after scene {args['after']}: {str(args.get('text')).strip()}" if isinstance(args.get("after"), int) else f"Add a scene at the end: {str(args.get('text')).strip()}"
    return f"Set the default clip length to {args.get('seconds')} seconds"


class Actions:
    """Answers the model's tool calls for one helper request. `scenes`: how many scenes the project has."""

    def __init__(self, scenes, on_event, wait=rules.WAIT):
        self.scenes, self.on_event, self.wait = int(scenes or 0), on_event, wait

    def __call__(self, call) -> str:
        name, args = call.get("name"), call.get("args") or {}
        if name not in ("edit_scene", "add_scene", "set_clip_length"):
            return f"Error: there is no tool called '{name}'."
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 600        # noqa: E731
        if name == "edit_scene":
            if not isinstance(args.get("scene"), int) or not 1 <= args["scene"] <= self.scenes:
                return f"Error: scene must be a number from 1 to {self.scenes}."
            text, secs = args.get("text"), args.get("seconds")
            if (not isinstance(text, str) or not text.strip()) and not num(secs):
                return "Error: give a new text, a length in seconds, or both."
            if isinstance(text, str) and len(text) > MAX_TEXT:
                return f"Error: the text is longer than {MAX_TEXT} characters."
        elif name == "add_scene":
            if not isinstance(args.get("text"), str) or not args["text"].strip() or len(args["text"]) > MAX_TEXT:
                return f"Error: text must be the prompt of the new scene, at most {MAX_TEXT} characters."
            if args.get("after") is not None and (not isinstance(args["after"], int) or not 1 <= args["after"] <= self.scenes):
                return f"Error: after must be a scene number from 1 to {self.scenes}."
        elif not num(args.get("seconds")):
            return "Error: seconds must be a number above 0."
        pid = f"act-{next(rules._IDS)}"
        slot = rules._DECISIONS[pid] = {"event": threading.Event(), "allow": False}
        try:
            self.on_event({"kind": "action_request", "id": pid, "tool": name, "args": args, "summary": describe(name, args), "reason": str(args.get("reason") or "").strip()})
            answered = slot["event"].wait(self.wait)
        finally:
            rules._DECISIONS.pop(pid, None)
        if not answered:
            return "No answer from the user in time, so nothing was changed."
        if slot["allow"]:
            if name == "add_scene":
                self.scenes += 1
            return "Allowed: the change is made in the project."
        return "Disallowed by the user. Nothing was changed. Do not propose the same change again."
