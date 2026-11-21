"""The helper: a chat the person opens to ask the model to act on the project, not only to talk.

The model gets a sheet of the project (scenes by number, with prompt and length) and three tools: `edit_scene`, `add_scene` and
`set_clip_length`. Nothing is applied here. Each call is sent to the page as an `action_request` event, shown as a card with Allow and
Disallow, and the answer (the same route as a rule change) goes back to the model. The page does the change in the project.
"""

import re
import threading

from . import rules

MAX_TEXT = 4000
TEXTS = {"anchor": "the anchor", "postfix": "the postfix", "negative": "the negative prompt", "postfix_enabled": "the postfix switch"}
ALIASES = {"prefix": "anchor", "main_prompt": "anchor", "negative_prompt": "negative", "postfix_on": "postfix_enabled", "postfix_switch": "postfix_enabled"}
EXTRA = ("run_action", "set_story", "cut_scene", "set_cut_words", "set_node_inputs", "apply_preset", "set_project_text", "set_variable", "remove_variable", "save_shortcut", "remove_shortcut")
HELPER_RULE = ("You are the helper of a video project. You can act on it with tools. Every change is shown to the user, who allows or refuses it, so propose changes instead of describing them. "
               "Scenes are numbered from 1. This is the project now:\n[[PROJECT_SHEET]]\n"
               "Use edit_scene to change one scene's prompt, length or own prefix (a scene may have its own prefix instead of the project anchor; the sheet shows it), add_scene to add a scene, set_clip_length to change the default length of every scene, copy_reference to give a scene the same reference pictures and aspect ratio source as another scene (\"same reference\" means the scene before it unless the user says which). Reference pictures are listed per scene in the project. "
               "The global prompt is every scene prompt in order as one text, with the cut word written between scenes; rewrite it with set_story (the whole new text, scenes separated by the cut word on a line of its own). Every part of the Story between two cut words is a separate prompt: its own scene, made as its own clip at its own time, so it must stand on its own and never needs to be handled together with the others. The global prompt (the Composer's first tab) is a quick summary of the whole video, one short prompt per shot. Keep each shot short, a line or two with the main action and look, and never write detailed descriptions of each shot there; the detail comes later. In the global prompt write nothing but the scene prompts and the cut word between them: never labels or numbering such as [Shot 1] or Scene 2, and no headings. Change only the scenes the user means, one at a time with edit_scene, and use set_story only when the user asks for the whole Story to be rewritten. Do not merge scenes, move details between them or make them agree with each other unless asked. You can also run things for the user: the actions of the app (Generate, render, undo and so on) are listed in the project; start one with run_action, naming scenes first when the action works on picked scenes (Generate selected). run_action only starts the action and does not wait for it to finish; never start Generate unless the user asked for it. Cuts: the global prompt is cut into scenes at the cut words (the first one is written between scenes, change them with set_cut_words); a scene can also be cut in two on the timeline with cut_scene, and the halves stay one generated clip (the project says which scenes belong to one clip). "
               "The project has an anchor (also called the prefix: one text put in front of every scene prompt), a postfix (put after it, with an on/off switch), a negative prompt and $variables; change them with set_project_text (field anchor, postfix, negative or postfix_enabled with the text true or false), set_variable and remove_variable. Shortcuts (trigger words that expand into phrases) are looked up with find_shortcuts and changed with save_shortcut and remove_shortcut; a shortcut is shared by every project, so change one only when asked. "
               "Every other setting of the app (the modules listed in the project, such as the attention explorer, auto contrast or the sampler) is reached with find_settings, to look a setting up by words, and set_setting, to change it. Always call find_settings first and use the module and setting names it gives; never guess them. "
               "The pipeline (the nodes behind Generate: model files, sampler, steps, resolution) is reached with find_node_inputs, to look inputs up with their allowed values, and set_node_inputs, to change several at once under one question. To pick models, find the loader inputs, choose the files that fit (an unset one says so) and set them together. apply_preset replaces the whole pipeline with one of the starting points listed in the project; use it only when asked. "
               "Act, do not announce: when you decide on a change, call the tool in that same turn. Never write that you will use a tool, never describe the call and never put the proposed text in your answer instead of calling. Lengths are in seconds. Change only what the user asked for. When a tool answers that the user refused, think about why and correct yourself, as the tool says.")


def tools() -> list:
    def fn(name, description, props, required):
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}}}
    scene = {"type": "integer", "description": "The scene number, from 1."}
    reason = {"type": "string", "description": "Why, in one sentence."}
    text = {"type": "string", "description": "The whole prompt text."}
    seconds = {"type": "number", "description": "The length in seconds."}
    return [fn("edit_scene", "Change the prompt, the length and/or the prefix of one scene. A scene can have its own prefix instead of the project anchor.", {"scene": scene, "text": text, "seconds": seconds, "prefix": {"type": "string", "description": "The scene's own prefix, put in front of its prompt instead of the project anchor. An empty text means no prefix at all for this scene."}, "project_prefix": {"type": "boolean", "description": "true: the scene goes back to using the project anchor."}, "reason": reason}, ["scene", "reason"]),
            fn("add_scene", "Add a new scene after another one (the last when after is left out).", {"text": text, "after": {"type": "integer", "description": "The scene number to add it after."}, "reason": reason}, ["text", "reason"]),
            fn("copy_reference", "Give a scene the same reference pictures and aspect ratio source picture as another scene.", {"scene": scene, "from_scene": {"type": "integer", "description": "The scene number to copy them from."}, "reason": reason}, ["scene", "from_scene", "reason"]),
            fn("set_project_text", "Set the project's anchor (= prefix), postfix, negative prompt, or switch the postfix on or off (field postfix_enabled, text true or false). The whole new text; empty clears it.", {"field": {"type": "string", "enum": list(TEXTS)}, "text": {"type": "string"}, "reason": reason}, ["field", "text", "reason"]),
            fn("set_variable", "Add a $variable of the project or change its value.", {"name": {"type": "string", "description": "The name without the $."}, "value": {"type": "string"}, "reason": reason}, ["name", "value", "reason"]),
            fn("remove_variable", "Remove a $variable from the project.", {"name": {"type": "string"}, "reason": reason}, ["name", "reason"]),
            fn("find_shortcuts", "Look up shortcuts by words in their name, triggers or replacements.", {"query": {"type": "string"}}, ["query"]),
            fn("save_shortcut", "Add a shortcut or replace the one with the same name.", {"name": {"type": "string"}, "triggers": {"type": "array", "items": {"type": "string"}, "description": "Words or phrases that fire it."}, "replacements": {"type": "array", "items": {"type": "string"}, "description": "What it expands into; with several, one is drawn each time."}, "category": {"type": "string"}, "reason": reason}, ["name", "triggers", "replacements", "reason"]),
            fn("remove_shortcut", "Delete a shortcut of the library.", {"name": {"type": "string"}, "reason": reason}, ["name", "reason"]),
            fn("run_action", "Start one of the app's actions (listed in the project), such as generating or rendering. Optionally pick scenes first.", {"action": {"type": "string", "description": "The id of the action."}, "scenes": {"type": "array", "items": {"type": "integer"}, "description": "Scene numbers to pick first, for actions that work on the picked scenes."}, "reason": reason}, ["action", "reason"]),
            fn("set_story", "Rewrite the whole global prompt (all scene prompts in one text, scenes separated by the cut word on its own line). Scenes are matched to the earlier ones by order.", {"text": {"type": "string"}, "reason": reason}, ["text", "reason"]),
            fn("cut_scene", "Cut a scene in two on the timeline. Both halves stay one generated clip.", {"scene": scene, "at_seconds": {"type": "number", "description": "Where to cut, in seconds from the start of the scene. The middle when left out."}, "reason": reason}, ["scene", "reason"]),
            fn("set_cut_words", "Replace the cut words of the global prompt (a whole word typed in it starts a new scene).", {"words": {"type": "array", "items": {"type": "string"}}, "reason": reason}, ["words", "reason"]),
            fn("find_node_inputs", "Look up inputs of the pipeline's nodes by words (a node, a group, an input such as steps or sampler). Answers with each match: slot, input, allowed values and the value now.", {"query": {"type": "string"}}, ["query"]),
            fn("set_node_inputs", "Change inputs of pipeline nodes, all under one question to the user.", {"changes": {"type": "array", "items": {"type": "object", "properties": {"slot": {"type": "string"}, "input": {"type": "string"}, "value": {"type": ["string", "number", "boolean"]}}, "required": ["slot", "input", "value"]}}, "reason": reason}, ["changes", "reason"]),
            fn("apply_preset", "Replace the whole pipeline with one of the listed starting points.", {"preset": {"type": "string", "description": "The id of the starting point."}, "reason": reason}, ["preset", "reason"]),
            fn("find_settings", "Look up settings of the app by words (a module name, what it does). Answers with each match: module, setting, kind, allowed values and the value now.", {"query": {"type": "string", "description": "Words to look for."}}, ["query"]),
            fn("set_setting", "Change one setting of a module. Use the module and setting names find_settings gave.", {"module": {"type": "string"}, "setting": {"type": "string"}, "value": {"type": ["string", "number", "boolean"], "description": "true or false for a switch, a number, one of the allowed values, or text (for blocks a list such as 2-10,29-46)."}, "reason": reason}, ["module", "setting", "value", "reason"]),
            fn("set_clip_length", "Set the default clip length of the whole project.", {"seconds": seconds, "reason": reason}, ["seconds", "reason"])]


def describe(name, args) -> str:
    """One plain line for the Allow card."""
    if name == "edit_scene":
        parts = ([f"change the prompt to: {str(args.get('text')).strip()}"] if isinstance(args.get("text"), str) and args["text"].strip() else []) + ([f"set the length to {args['seconds']} seconds"] if isinstance(args.get("seconds"), (int, float)) else []) + ([f"set its own prefix to: {args['prefix'].strip() or '(none)'}"] if isinstance(args.get("prefix"), str) else []) + (["use the project prefix again"] if args.get("project_prefix") is True else [])
        return f"Scene {args.get('scene')}: " + " and ".join(parts)
    if name == "add_scene":
        return f"Add a scene after scene {args['after']}: {str(args.get('text')).strip()}" if isinstance(args.get("after"), int) else f"Add a scene at the end: {str(args.get('text')).strip()}"
    if name == "copy_reference":
        return f"Scene {args.get('scene')}: use the same reference pictures as scene {args.get('from_scene')}"
    if name == "set_setting":
        return f"{args.get('title') or args.get('module')}: {args.get('label') or args.get('setting')}: {args.get('now')} -> {args.get('value')}"
    if name == "run_action":
        picked = args.get("scenes") or []
        return f"Run: {args.get('label') or args.get('action')}" + (f" on scene{'s' if len(picked) > 1 else ''} {', '.join(map(str, picked))}" if picked else "")
    if name == "set_story":
        return "Rewrite the global prompt (all scenes): " + str(args.get("text")).strip()
    if name == "cut_scene":
        return f"Cut scene {args.get('scene')} in two" + (f" at {args['at_seconds']} seconds" if isinstance(args.get("at_seconds"), (int, float)) else " in the middle")
    if name == "set_cut_words":
        return "Set the cut words of the global prompt to: " + ", ".join(args.get("words") or [])
    if name == "set_node_inputs":
        return "Change the pipeline: " + "; ".join(f"{c['slot']}.{c['input']}: {c.get('now')} -> {c['value']}" for c in args.get("changes") or [])
    if name == "apply_preset":
        return f"Replace the whole pipeline with the starting point '{args.get('title') or args.get('preset')}'. Every file and value set in the pipeline is cleared"
    if name == "set_project_text":
        if args.get("field") == "postfix_enabled":
            return f"Switch the postfix {'on' if args.get('text') else 'off'}"
        return f"Set {TEXTS.get(args.get('field'))} to: {args.get('text') or '(empty)'}"
    if name == "set_variable":
        return f"Set ${args.get('name')} to: {args.get('value')}"
    if name == "remove_variable":
        return f"Remove ${args.get('name')}"
    if name == "save_shortcut":
        return f"Save shortcut {args.get('name')}: {', '.join(args.get('triggers') or [])} -> {' | '.join(args.get('replacements') or [])}"
    if name == "remove_shortcut":
        return f"Delete shortcut {args.get('name')}"
    return f"Set the default clip length to {args.get('seconds')} seconds"


class Actions:
    """Answers the model's tool calls for one helper request. `scenes`: how many scenes the project has."""

    def __init__(self, scenes, on_event, wait=rules.WAIT, settings=(), variables=(), shortcuts=(), inputs=(), presets=(), actions=()):
        self.scenes, self.on_event, self.wait = int(scenes or 0), on_event, wait
        self.calls = 0
        self.inputs = {(x["slot"], x["name"]): x for x in inputs if isinstance(x, dict) and x.get("slot") and x.get("name")}
        self.actions = {str(x["id"]): x for x in actions if isinstance(x, dict) and x.get("id")}
        self.presets = {str(x["id"]): x for x in presets if isinstance(x, dict) and x.get("id")}
        self.variables = {str(v) for v in variables}
        self.shortcuts = {str(x["name"]): x for x in shortcuts if isinstance(x, dict) and x.get("name")}
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

    def find_input(self, query) -> str:
        words = [w for w in re.findall(r"[a-z0-9]+", str(query or "").lower()) if len(w) > 1]
        hits = []
        for x in self.inputs.values():
            hay = " ".join(str(x.get(k) or "") for k in ("slot", "node", "group", "name", "label")).lower().replace("_", " ")
            if words and any(w in hay for w in words):
                hits.append((-sum(w in hay for w in words), len(hits), x))
        return "\n".join(node_line(x) for _, _, x in sorted(hits, key=lambda t: t[:2])[:20]) or "No pipeline input matches. Try other words (a node name, steps, sampler, model)."

    def check_input(self, change):
        """-> an error text, or None; `change` gains "now" and the typed value."""
        spec = self.inputs.get((str(change.get("slot")), str(change.get("input"))))
        if spec is None:
            return f"there is no input {change.get('slot')}.{change.get('input')} (call find_node_inputs first)"
        kind, v = spec.get("type"), change.get("value")
        if kind == "COMBO":
            if v not in (spec.get("choices") or []):
                return f"{change['slot']}.{change['input']} must be one of the listed files or values"
        elif kind in ("INT", "FLOAT"):
            try:
                v = float(v) if not isinstance(v, bool) else None
            except (TypeError, ValueError):
                v = None
            if v is None or v != v or (spec.get("min") is not None and v < spec["min"]) or (spec.get("max") is not None and v > spec["max"]):
                return f"{change['slot']}.{change['input']} takes a number from {spec.get('min')} to {spec.get('max')}"
            v = int(round(v)) if kind == "INT" else v
        elif kind == "BOOLEAN":
            v = {"true": True, "false": False}.get(v.strip().lower()) if isinstance(v, str) else v
            if not isinstance(v, bool):
                return f"{change['slot']}.{change['input']} is true or false"
        elif not isinstance(v, str) or len(v) > MAX_TEXT:
            return f"{change['slot']}.{change['input']} takes text"
        change["value"], change["now"] = v, spec.get("now")
        return None

    def find_shortcut(self, query) -> str:
        words = [w for w in re.findall(r"[a-z0-9]+", str(query or "").lower()) if len(w) > 1]
        hits = [x for x in self.shortcuts.values() if words and all(w in (x["name"] + " " + " ".join(x.get("triggers") or []) + " " + " ".join(x.get("replacements") or [])).lower() for w in words)]
        return "\n".join(f"{x['name']}: {', '.join(x.get('triggers') or [])} -> {' | '.join(x.get('replacements') or [])}{'' if x.get('enabled', True) else ' (off)'}" for x in hits[:15]) or "No shortcut matches."

    def extra(self, name, args):
        """-> an error text, or None when the call is valid (args may gain fields the page needs)."""
        text = lambda v, n=MAX_TEXT: isinstance(v, str) and len(v) <= n        # noqa: E731
        if name == "run_action":
            spec = self.actions.get(str(args.get("action")))
            if spec is None:
                return f"Error: there is no action '{args.get('action')}'. They are: {', '.join(self.actions)}."
            picked = args.get("scenes")
            if picked is not None and not (isinstance(picked, list) and all(isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= self.scenes for n in picked)):
                return f"Error: scenes are numbers from 1 to {self.scenes}."
            args["label"] = spec.get("label")
            return None
        if name == "set_story":
            return None if text(args.get("text"), 20000) and args["text"].strip() else "Error: text must be the whole new Story, at most 20000 characters."
        if name == "cut_scene":
            if not isinstance(args.get("scene"), int) or not 1 <= args["scene"] <= self.scenes:
                return f"Error: scene must be a number from 1 to {self.scenes}."
            at = args.get("at_seconds")
            return None if at is None or (isinstance(at, (int, float)) and not isinstance(at, bool) and 0 < at <= 600) else "Error: at_seconds must be a positive number."
        if name == "set_cut_words":
            words = args.get("words")
            ok = isinstance(words, list) and 0 < len(words) <= 10 and all(isinstance(w, str) and re.fullmatch(r"[A-Za-z0-9_]{1,30}", w.strip()) for w in words)
            args["words"] = [w.strip() for w in words] if ok else words
            return None if ok else "Error: words is a list of one to ten single words (letters, digits, _)."
        if name == "set_node_inputs":
            changes = args.get("changes")
            if not isinstance(changes, list) or not 0 < len(changes) <= 40 or not all(isinstance(c, dict) for c in changes):
                return "Error: changes is a list of {slot, input, value}."
            args["changes"] = changes = [dict(c) for c in changes]
            bad = [e for e in (self.check_input(c) for c in changes) if e]
            return "Error: " + "; ".join(bad[:3]) if bad else None
        if name == "apply_preset":
            spec = self.presets.get(str(args.get("preset")))
            if spec is None:
                return f"Error: there is no starting point '{args.get('preset')}'. They are: {', '.join(self.presets)}."
            args["title"] = spec.get("title")
            return None
        if name == "set_project_text":
            args["field"] = ALIASES.get(str(args.get("field")).strip().lower(), str(args.get("field")).strip().lower())
            if args["field"] == "postfix_enabled":
                on = {"true": True, "on": True, "false": False, "off": False}.get(str(args.get("text")).strip().lower())
                if on is None:
                    return "Error: for postfix_enabled the text is true or false."
                args["text"] = on
                return None
            return None if args.get("field") in TEXTS and text(args.get("text")) else f"Error: field is one of {', '.join(TEXTS)} and text is at most {MAX_TEXT} characters."
        if name in ("set_variable", "remove_variable"):
            if not isinstance(args.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_]{1,40}", args["name"].lstrip("$")):
                return "Error: a variable name is letters, digits and _ without the $."
            args["name"] = args["name"].lstrip("$")
            if name == "remove_variable":
                return None if args["name"] in self.variables else f"Error: there is no variable ${args['name']}."
            return None if text(args.get("value")) else "Error: value must be text."
        if name == "save_shortcut":
            ok = lambda v: isinstance(v, list) and v and all(isinstance(x, str) and x.strip() and len(x) <= 500 for x in v)        # noqa: E731
            if not text(args.get("name"), 80) or not args["name"].strip() or not ok(args.get("triggers")) or not ok(args.get("replacements")):
                return "Error: a shortcut needs a name, at least one trigger and at least one replacement."
            args["existing"] = args["name"] in self.shortcuts
            return None
        return None if args.get("name") in self.shortcuts else f"Error: there is no shortcut called '{args.get('name')}'."        # remove_shortcut

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
        self.calls += 1
        if name not in ("edit_scene", "add_scene", "copy_reference", "set_clip_length", "find_settings", "set_setting", "find_shortcuts", "find_node_inputs") + EXTRA:
            return f"Error: there is no tool called '{name}'."
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 600        # noqa: E731
        if name == "find_settings":
            return self.find(args.get("query"))
        if name == "find_shortcuts":
            return self.find_shortcut(args.get("query"))
        if name == "find_node_inputs":
            return self.find_input(args.get("query"))
        if name in EXTRA:
            args = dict(args)
            err = self.extra(name, args)
            if err:
                return err
        elif name == "set_setting":
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
            pre = args.get("prefix")
            if (not isinstance(text, str) or not text.strip()) and not num(secs) and not isinstance(pre, str) and args.get("project_prefix") is not True:
                return "Error: give a new text, a length in seconds, a prefix, or project_prefix."
            if isinstance(pre, str) and len(pre) > MAX_TEXT:
                return f"Error: the prefix is longer than {MAX_TEXT} characters."
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
            elif name == "cut_scene":
                self.scenes += 1
            elif name == "set_variable":
                self.variables.add(args["name"])
            elif name == "remove_variable":
                self.variables.discard(args["name"])
            elif name == "save_shortcut":
                self.shortcuts[args["name"]] = args
            elif name == "remove_shortcut":
                self.shortcuts.pop(args["name"], None)
            return "Allowed: the change is made in the project."
        return ("Disallowed by the user. Nothing was changed. Ask yourself why: compare this change with what the user asked and with the project as it is, and find what you got wrong "
                "(the wrong scene, a value too far, more than was asked). Then propose one corrected change that fixes it, or, if you cannot tell, ask the user one short question. Never propose the same change again.")


def line(x) -> str:
    """One setting as the model reads it."""
    kind = x.get("type")
    allowed = f"one of {', '.join(map(str, x['options']))}" if kind == "enum" else f"{x['min']} to {x['max']}" if x.get("min") is not None and x.get("max") is not None else kind
    return f"{x['module']}.{x['name']} ({x.get('title') or ''}: {x.get('label') or x['name']}; {allowed}; now {x.get('now')}) {str(x.get('hint') or '')[:140]}"


def node_line(x) -> str:
    """One pipeline input as the model reads it."""
    kind = x.get("type")
    allowed = f"one of {', '.join(map(str, x.get('choices') or []))}" if kind == "COMBO" else f"{x.get('min')} to {x.get('max')}" if x.get("min") is not None and x.get("max") is not None else str(kind).lower()
    now = x.get("now")
    return f"{x['slot']}.{x['name']} ({x.get('node')}, {x.get('group') or 'no group'}; {allowed}; now {'NOT SET' if now in (None, '') else now})"


NUDGE = "You described a change but did not call a tool, so nothing happened. Call the tool now, with the full arguments, and write nothing else."
_TOOL_WORDS = re.compile(r"\b(" + "|".join(["edit_scene", "add_scene", "copy_reference", "set_clip_length", "set_story", "cut_scene", "set_cut_words", "set_project_text", "set_variable", "remove_variable",
                                              "save_shortcut", "remove_shortcut", "set_setting", "set_node_inputs", "apply_preset", "run_action"]) + r")\b|\bI (?:will|would|propose|suggest)\b", re.I)


def announced(reply) -> bool:
    """Whether an answer talks about a change (a tool name, or "I will / I propose") instead of making it."""
    return bool(_TOOL_WORDS.search(str(reply or "")))
