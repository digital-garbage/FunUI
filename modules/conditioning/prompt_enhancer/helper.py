"""The helper: a chat the person opens to ask the model to act on the project, not only to talk.

The model gets a sheet of the project (scenes by number, with prompt and length) and three tools: `edit_scene`, `add_scene` and
`set_clip_length`. Nothing is applied here. Each call is sent to the page as an `action_request` event, shown as a card with Allow and
Disallow, and the answer (the same route as a rule change) goes back to the model. The page does the change in the project.
"""

import re
import threading

from . import rules

MAX_TEXT = 4000
HELPER_RULE = ("You are the helper of a video project. You can act on it with tools. Every change is shown to the user, who allows or refuses it, so propose changes instead of describing them. "
               "Scenes are numbered from 1. This is the project now:\n[[PROJECT_SHEET]]\n"
               "Use edit_scene to change one scene's prompt or length, add_scene to add a scene, set_clip_length to change the default length of every scene, copy_reference to give a scene the same reference pictures and aspect ratio source as another scene (\"same reference\" means the scene before it unless the user says which). Reference pictures are listed per scene in the project. "
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
            fn("copy_reference", "Give a scene the same reference pictures and aspect ratio source picture as another scene.", {"scene": scene, "from_scene": {"type": "integer", "description": "The scene number to copy them from."}, "reason": reason}, ["scene", "from_scene", "reason"]),
            fn("find_settings", "Look up settings of the app by words (a module name, what it does). Answers with each match: module, setting, kind, allowed values and the value now.", {"query": {"type": "string", "description": "Words to look for."}}, ["query"]),
            fn("set_setting", "Change one setting of a module. Use the module and setting names find_settings gave.", {"module": {"type": "string"}, "setting": {"type": "string"}, "value": {"type": ["string", "number", "boolean"], "description": "true or false for a switch, a number, one of the allowed values, or text (for blocks a list such as 2-10,29-46)."}, "reason": reason}, ["module", "setting", "value", "reason"]),
            fn("set_clip_length", "Set the default clip length of the whole project.", {"seconds": seconds, "reason": reason}, ["seconds", "reason"])]


def describe(name, args) -> str:
    """One plain line for the Allow card."""
    if name == "edit_scene":
        parts = ([f"change the prompt to: {str(args.get('text')).strip()}"] if isinstance(args.get("text"), str) and args["text"].strip() else []) + ([f"set the length to {args['seconds']} seconds"] if isinstance(args.get("seconds"), (int, float)) else [])
        return f"Scene {args.get('scene')}: " + " and ".join(parts)
    if name == "add_scene":
        return f"Add a scene after scene {args['after']}: {str(args.get('text')).strip()}" if isinstance(args.get("after"), int) else f"Add a scene at the end: {str(args.get('text')).strip()}"
    if name == "copy_reference":
        return f"Scene {args.get('scene')}: use the same reference pictures as scene {args.get('from_scene')}"
    if name == "set_setting":
        return f"{args.get('title') or args.get('module')}: {args.get('label') or args.get('setting')}: {args.get('now')} -> {args.get('value')}"
    return f"Set the default clip length to {args.get('seconds')} seconds"


class Actions:
    """Answers the model's tool calls for one helper request. `scenes`: how many scenes the project has."""

    def __init__(self, scenes, on_event, wait=rules.WAIT, settings=()):
        self.scenes, self.on_event, self.wait = int(scenes or 0), on_event, wait
        self.settings = {(x["module"], x["name"]): x for x in settings if isinstance(x, dict) and x.get("module") and x.get("name")}

    def find(self, query) -> str:
        words = [w for w in re.findall(r"[a-z0-9]+", str(query or "").lower()) if len(w) > 1]
        scored = []
        for x in self.settings.values():
            hay = " ".join(str(x.get(k) or "") for k in ("module", "title", "name", "label", "hint")).lower().replace("_", " ")
            score = sum(w in hay for w in words)
            if score:
                scored.append((-score, len(scored), x))
        if not scored:
            return "No setting matches. Try other words."
        return "\n".join(line(x) for _, _, x in sorted(scored, key=lambda t: t[:2])[:15])

    def check(self, args):
        """-> (the setting, the value as it will be saved) or an error text."""
        spec = self.settings.get((str(args.get("module")), str(args.get("setting"))))
        if spec is None:
            return f"Error: there is no setting {args.get('module')}.{args.get('setting')}. Call find_settings first.", None
        kind, v = spec.get("type"), args.get("value")
        if kind == "bool":
            v = {"true": True, "false": False}.get(v.strip().lower()) if isinstance(v, str) else v
            return (None, v) if isinstance(v, bool) else ("Error: this setting is a switch, give true or false.", None)
        if kind in ("int", "float"):
            try:
                v = float(v) if not isinstance(v, bool) else None
            except (TypeError, ValueError):
                v = None
            if v is None or v != v:
                return "Error: this setting takes a number.", None
            lo, hi = spec.get("min"), spec.get("max")
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                return f"Error: the value must be between {lo} and {hi}.", None
            return None, int(round(v)) if kind == "int" else v
        if kind == "enum":
            return (None, v) if v in (spec.get("options") or []) else (f"Error: the value must be one of: {', '.join(map(str, spec.get('options') or []))}.", None)
        return (None, v.strip()) if isinstance(v, str) and len(v) <= 2000 else ("Error: this setting takes text of at most 2000 characters.", None)

    def __call__(self, call) -> str:
        name, args = call.get("name"), call.get("args") or {}
        if name not in ("edit_scene", "add_scene", "copy_reference", "set_clip_length", "find_settings", "set_setting"):
            return f"Error: there is no tool called '{name}'."
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 600        # noqa: E731
        if name == "find_settings":
            return self.find(args.get("query"))
        if name == "set_setting":
            err, value = self.check(args)
            if err:
                return err
            spec = self.settings[(str(args["module"]), str(args["setting"]))]
            if value == spec.get("now"):
                return "Nothing to change: it already has that value."
            args = {**args, "value": value, "now": spec.get("now"), "title": spec.get("title"), "label": spec.get("label")}
        elif name == "edit_scene":
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
        elif name == "copy_reference":
            a, b = args.get("scene"), args.get("from_scene")
            if not all(isinstance(v, int) and 1 <= v <= self.scenes for v in (a, b)) or a == b:
                return f"Error: scene and from_scene must be two different scene numbers from 1 to {self.scenes}."
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


def line(x) -> str:
    """One setting as the model reads it."""
    kind = x.get("type")
    allowed = f"one of {', '.join(map(str, x['options']))}" if kind == "enum" else f"{x['min']} to {x['max']}" if x.get("min") is not None and x.get("max") is not None else kind
    return f"{x['module']}.{x['name']} ({x.get('title') or ''}: {x.get('label') or x['name']}; {allowed}; now {x.get('now')}) {str(x.get('hint') or '')[:140]}"
