"""The helper: a chat the person opens to ask the model to act on the project, not only to talk.

The model gets a sheet of the project (scenes by number, with prompt and length) and three tools: `edit_scene`, `add_scene` and
`set_clip_length`. Nothing is applied here. Each call is sent to the page as an `action_request` event, shown as a card with Allow and
Disallow, and the answer (the same route as a rule change) goes back to the model. The page does the change in the project.
"""

import json
import difflib
import re
import unicodedata
import threading

from . import rules

MAX_TEXT = 4000
FIND_PAGE = 40
TEXTS = {"anchor": "the anchor", "postfix": "the postfix", "negative": "the negative prompt", "postfix_enabled": "the postfix switch"}
ALIASES = {"prefix": "anchor", "main_prompt": "anchor", "negative_prompt": "negative", "postfix_on": "postfix_enabled", "postfix_switch": "postfix_enabled"}
EXTRA = ("share_reference", "run_action", "set_story", "cut_scene", "set_cut_words", "set_node_inputs", "apply_preset", "set_project_text", "set_variable", "remove_variable", "save_shortcut", "remove_shortcut", "edit_shortcuts")
APP_PREFIX = "APP NOTE (from the app, not from the user; the user has written nothing new):\n"


def app_note(text) -> str:
    """Mark a reminder as the app's, so the model never takes it for something the user said."""
    return APP_PREFIX + str(text)


HELPER_RULE = ("You are the helper of a video project. You can act on it with tools. Every change is shown to the user, who allows or refuses it, so propose changes instead of describing them. "
               "Scenes are numbered from 1. This is the project now:\n[[PROJECT_SHEET]]\nEarlier messages, answers and tool results in this chat are history: they already happened and are finished. Answer only the newest message, never redo an earlier change, and trust the project above over what the history says about it. "
               "Use edit_scene to change one scene's prompt, length or own prefix (a scene may have its own prefix instead of the project anchor; the sheet shows it), add_scene to add a scene, set_clip_length to change the default length of every scene, copy_reference to give one scene the same reference pictures and aspect ratio source as another scene, share_reference to give every scene the references of one scene (references pinned to the whole project already apply to every scene and are shown apart) (\"same reference\" means the scene before it unless the user says which). Reference pictures are listed per scene in the project. "
               "The global prompt is every scene prompt in order as one text, with the cut word written between scenes; rewrite it with set_story (the whole new text, scenes separated by the cut word on a line of its own). Every part of the Story between two cut words is a separate prompt: its own scene, made as its own clip at its own time, so it must stand on its own and never needs to be handled together with the others. The global prompt (the Composer's first tab) is a quick summary of the whole video, one short prompt per shot. Keep each shot short, a line or two with the main action and look, and never write detailed descriptions of each shot there; the detail comes later. In the global prompt write nothing but the scene prompts and the cut word between them: never labels or numbering such as [Shot 1] or Scene 2, and no headings. Change only the scenes the user means, one at a time with edit_scene, and use set_story only when the user asks for the whole Story to be rewritten. Do not merge scenes, move details between them or make them agree with each other unless asked. You can also run things for the user: the actions of the app (Generate, render, undo and so on) are listed in the project; start one with run_action, naming scenes first when the action works on picked scenes (Generate selected). run_action only starts the action and does not wait for it to finish; never start Generate unless the user asked for it. Cuts: the global prompt is cut into scenes at the cut words (the first one is written between scenes, change them with set_cut_words); a scene can also be cut in two on the timeline with cut_scene, and the halves stay one generated clip (the project says which scenes belong to one clip). "
               "The project has an anchor (also called the prefix: one text put in front of every scene prompt), a postfix (put after it, with an on/off switch), a negative prompt and $variables; change them with set_project_text (field anchor, postfix, negative or postfix_enabled with the text true or false), set_variable and remove_variable. Shortcuts (trigger words that expand into phrases) are looked up with find_shortcuts and changed with save_shortcut and remove_shortcut, and many at once (edits, renames, deletes) with edit_shortcuts; a shortcut is shared by every project, so change one only when asked. "
               "Every other setting of the app (the modules listed in the project, such as the attention explorer, auto contrast or the sampler) is reached with find_settings, to look a setting up by words, and set_setting, to change it. Always call find_settings first and use the module and setting names it gives; never guess them. "
               "The pipeline (the nodes behind Generate: model files, sampler, steps, resolution) is reached with find_node_inputs, to look inputs up with their allowed values, and set_node_inputs, to change several at once under one question. To pick models, find the loader inputs, choose the files that fit (an unset one says so) and set them together. apply_preset replaces the whole pipeline with one of the starting points listed in the project; use it only when asked. "
               "Act, do not announce: when you decide on a change, call the tool in that same turn. Never write that you will use a tool, never describe the call and never put the proposed text in your answer instead of calling. Lengths are in seconds. Change only what the user asked for. Every request for a change is done from a checklist, even a single change: first call plan with one item per change, or per batch of up to 60 changes (one item for one change), then do the items in order and call check_item for each one when its change is made. Messages that start with APP NOTE come from the app, never from the user: they are reminders of these rules, not requests. You never see the Chat tab conversation unless you call read_chat_log, and you call it only when the user explicitly asks you to look at or use that chat log. The app will not let you stop until every item is checked, so check an item only when its change was really made, or give the reason it cannot be done. When a tool answers that the user refused, think about why and correct yourself, as the tool says.")


GROUPS = {"scenes": ("edit_scene", "add_scene", "copy_reference", "set_clip_length", "set_story", "cut_scene", "set_cut_words", "share_reference"),
          "text": ("set_project_text",),
          "variables": ("set_variable", "remove_variable"),
          "shortcuts": ("find_shortcuts", "save_shortcut", "remove_shortcut", "edit_shortcuts"),
          "settings": ("find_settings", "set_setting", "find_node_inputs", "set_node_inputs", "apply_preset", "run_action"),
          "rules": ("read_prompt", "edit_prompt")}
_WORDS = {"scenes": r"scene|shot|clip|second|length|story|cut|picture|image|reference|aspect|frame|prompt",
          "text": r"anchor|prefix|postfix|negative|prompt",
          "variables": r"variable|\$\w",
          "shortcuts": r"shortcut",
          "settings": r"setting|node|preset|run\b|generate|sampler|steps?\b|seed|resolution|width|height|fps|lora",
          "rules": r"system prompt|rules?\b|instruction|reference_intro|macro|\[\[|enhancer|prompts? (?:for|of) the (?:model|enhancer)"}
_CHANGE = re.compile(r"\b(change|update|edit|make|set|add|remove|delete|rewrite|replace|fix|apply|put|turn|switch|inherit|use|let|tell|teach|rename|shorten|lengthen|extend|copy|run|save|create|merge|split|correct|adjust|modify|improve|tweak|reword|rephrase|include|exclude|ensure|stop|avoid|drop|enable|disable|increase|decrease|reduce|raise|lower)\b", re.I)


READS = ("read_chat_log", "read_prompt", "find_settings", "find_shortcuts", "find_node_inputs", "plan", "check_item")
MAX_PLAN_ROUNDS = 40         # how many times the app sends the model back to an unfinished checklist
_CALLBACK = re.compile(r"didn'?t|did not|not (?:done|changed|call)|nothing (?:changed|happened)|call the tool|do it|go ahead|correct (?:it|them)|fix (?:it|them)|so\b.*\?", re.I)


def intent(message) -> set:
    """Which kinds of tool a request is about, from its words: scenes, text, variables, shortcuts, settings, rules. Empty when it names none of them."""
    text = str(message or "").lower()
    return {g for g, words in _WORDS.items() if re.search(words, text)}


def wants_change(message, last_reply="") -> bool:
    """A request that asks for something to be done (not a question), so an answer without a change has failed it. A follow-up such as "will you correct them?"
    or "you didn't call the tool" counts when the helper's last answer only announced a change."""
    text = str(message or "").strip()
    if not text:
        return False
    if announced(last_reply) and _CALLBACK.search(text):
        return True
    return not text.endswith("?") and bool(_CHANGE.search(text))


def tools(with_rules=False, only=None) -> list:
    """The helper's tools; with the enhancer's own system prompts (rules) handed over, also read_prompt and edit_prompt. `only` (kinds from intent) keeps just
    the tools of those kinds, so the model sees the ones that fit the request; empty or None keeps all."""
    every = _tools() + (rules.tools() if with_rules else [])
    if not only:
        return every
    keep = {n for g in only for n in GROUPS.get(g, ())}
    return [t for t in every if t["function"]["name"] in keep or not any(t["function"]["name"] in names for names in GROUPS.values())]


FORCE_AGAIN = app_note("Still nothing was changed. Call the tool that makes the change now, with all its arguments, and write nothing else.")


PLAN_NOTE = app_note("Every change is done from a checklist, even a single one. Call plan now with one item per change (one item if there is only one; a batch of up to 60 changes may be one item). "
             "Then do the items, calling check_item for each one when its change is made. If a change was already made, plan it and check it. Write nothing else.")


def force_note(message, with_rules=False, context="") -> str:
    """The reminder for a request that asked for a change and got none: which tools fit, and that one must be called now."""
    names = ", ".join(t["function"]["name"] for t in tools(with_rules, intent(f"{message} {context}")))
    return app_note("The user asked for a change and nothing was changed, because you did not call a tool. Call the fitting tool now, with the full arguments"
            + (f" (the tools that fit: {names})" if names else "") + ", and write nothing else."
            + (" To change a rule: edit_prompt takes field (the rule's name, instructions for the main rewrite rule), new_text (the whole rule, with only the asked change made) and reason. "
               "If you have not read the rule yet, the answer to edit_prompt shows it, so call edit_prompt directly." if "edit_prompt" in names else ""))


def rules_note(given) -> str:
    """Tells the helper which system prompts of the enhancer it may read and change, and which macros they can use."""
    from . import enhance
    macros = ", ".join(f"[[{k}]]" for k in enhance.MACROS)
    return ("\n\nThe enhancer's own system prompts (rules) can be read with read_prompt and changed with edit_prompt, which gives the whole new text: read the rule first, change only what was asked and keep the rest word for word. "
            "The user sees the change and may refuse it. When the user says \"the system prompt\" without naming one, they mean the main rewrite instructions (instructions); reference_intro only says how shortcut and lorebook entries are introduced, so change it only when the user is talking about that. In the app the rules are named: instructions = Rewrite instructions, marks_prompt = Markup rule, fixed_prompt = Fixed text around it, reference_intro = Reference intro, chat_prompt = Chat answers, notes_prompt = Notes rule, compact_prompt = Conversation notes. The rules are:\n" + rules.sheet(given) + "\nA rule may use these macros, which the app fills in for each request: " + macros + ".")


def _tools() -> list:
    def fn(name, description, props, required):
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}}}
    scene = {"type": "integer", "description": "The scene number, from 1."}
    reason = {"type": "string", "description": "Why, in one sentence."}
    text = {"type": "string", "description": "The whole prompt text."}
    seconds = {"type": "number", "description": "The length in seconds."}
    return [fn("edit_scene", "Change the prompt, the length and/or the prefix of one scene. A scene can have its own prefix instead of the project anchor.", {"scene": scene, "text": text, "seconds": seconds, "prefix": {"type": "string", "description": "The scene's own prefix, put in front of its prompt instead of the project anchor. An empty text means no prefix at all for this scene."}, "project_prefix": {"type": "boolean", "description": "true: the scene goes back to using the project anchor."}, "reason": reason}, ["scene", "reason"]),
            fn("add_scene", "Add a new scene after another one (the last when after is left out). It gets the same reference pictures and aspect ratio source as the first scene.", {"text": text, "after": {"type": "integer", "description": "The scene number to add it after."}, "reason": reason}, ["text", "reason"]),
            fn("copy_reference", "Give a scene the same reference pictures and aspect ratio source picture as another scene.", {"scene": scene, "from_scene": {"type": "integer", "description": "The scene number to copy them from."}, "reason": reason}, ["scene", "from_scene", "reason"]),
            fn("set_project_text", "Set the project's anchor (= prefix), postfix, negative prompt, or switch the postfix on or off (field postfix_enabled, text true or false). The whole new text; empty clears it.", {"field": {"type": "string", "enum": list(TEXTS)}, "text": {"type": "string"}, "reason": reason}, ["field", "text", "reason"]),
            fn("set_variable", "Add a $variable of the project or change its value.", {"name": {"type": "string", "description": "The name without the $."}, "value": {"type": "string"}, "reason": reason}, ["name", "value", "reason"]),
            fn("remove_variable", "Remove a $variable from the project.", {"name": {"type": "string"}, "reason": reason}, ["name", "reason"]),
            fn("plan", "Write the checklist for a request that needs more than three separate changes: one item per change, or per batch of up to 60 changes. Call this first, before the changes.", {"items": {"type": "array", "items": {"type": "string"}, "description": "The steps, in the order they will be done."}}, ["items"]),
            fn("check_item", "Tick one item of the checklist once its change was made, or say why it cannot be done.", {"item": {"type": "integer", "description": "The item's number, from 1."}, "result": {"type": "string", "description": "done, or 'cannot: ' and the reason."}}, ["item", "result"]),
            fn("read_chat_log", "Read the conversation of the user's Chat tab (the enhancer's own chat, not this one), newest part last. Call it ONLY when the user explicitly asks you to look at, use or adapt to that chat log; never on your own.", {"turns": {"type": "integer", "description": "How many of the latest turns to read (default 6, at most 20)."}}, []),
            fn("find_shortcuts", "Look up shortcuts by words in their name, triggers or replacements; an empty query lists all, and words in quotes are searched as an exact phrase. Shows 40 at a time and says how many there are in all: when asked for all of them, page on with offset until you have seen every one.", {"query": {"type": "string"}, "offset": {"type": "integer", "description": "How many matches to skip, to get the next page."}}, []),
            fn("save_shortcut", "Add a shortcut or replace the one with the same name.", {"name": {"type": "string"}, "triggers": {"type": "array", "items": {"type": "string"}, "description": "Words or phrases that fire it."}, "replacements": {"type": "array", "items": {"type": "string"}, "description": "What it expands into; with several, one is drawn each time."}, "category": {"type": "string"}, "reason": reason}, ["name", "triggers", "replacements", "reason"]),
            fn("edit_shortcuts", "Change, rename or delete many shortcuts at once; the user allows the whole batch with one answer. Each change names an existing shortcut. Leave out what stays as it is.",
               {"changes": {"type": "array", "description": "Up to 60 changes; for more, call again with the next ones.", "items": {"type": "object", "properties": {"name": {"type": "string", "description": "The shortcut's current name."}, "new_name": {"type": "string"}, "triggers": {"type": "array", "items": {"type": "string"}, "description": "The whole new list."}, "replacements": {"type": "array", "items": {"type": "string"}, "description": "The whole new list."}, "category": {"type": "string"}, "delete": {"type": "boolean"}}, "required": ["name"]}}, "reason": reason}, ["changes", "reason"]),
            fn("remove_shortcut", "Delete a shortcut of the library.", {"name": {"type": "string"}, "reason": reason}, ["name", "reason"]),
            fn("share_reference", "Give every scene the same reference pictures as one scene (what the media bin calls adding a reference to all scenes).", {"from_scene": {"type": "integer", "description": "The scene number to take the references from."}, "reason": reason}, ["from_scene", "reason"]),
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
    if name == "share_reference":
        return f"Give every scene the reference pictures of scene {args.get('from_scene')}"
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
    if name == "edit_shortcuts":
        out = []
        for c in args.get("changes") or []:
            if c.get("delete"):
                out.append(f"delete {c['name']}")
                continue
            what = [f"rename to {c['new_name']}"] if c.get("new_name") and c["new_name"] != c["name"] else []
            what += [f"{k} {'; '.join(c[k]) if isinstance(c[k], list) else c[k]}" for k in ("triggers", "replacements", "category") if k in c]
            out.append(f"{c['name']}: " + (", ".join(what) or "no change"))
        return f"Change {len(out)} shortcut(s):\n" + "\n".join(out)
    return f"Set the default clip length to {args.get('seconds')} seconds"


class Actions:
    """Answers the model's tool calls for one helper request. `scenes`: how many scenes the project has."""

    def __init__(self, scenes, on_event, wait=rules.WAIT, settings=(), variables=(), shortcuts=(), inputs=(), presets=(), actions=(), rules_given=None, chat_log=None):
        self.chat_log = chat_log
        self.scenes, self.on_event, self.wait = int(scenes or 0), on_event, wait
        self.editor = rules.Editor(rules_given, on_event, wait) if rules_given else None
        self.calls = 0
        self.changes = 0                                   # calls that change something: looking a rule or setting up is not one
        self.plan, self.plan_changes = [], 0               # the checklist the model wrote for a long request, and how many changes were made when it did
        self.inputs = {(x["slot"], x["name"]): x for x in inputs if isinstance(x, dict) and x.get("slot") and x.get("name")}
        self.actions = {str(x["id"]): x for x in actions if isinstance(x, dict) and x.get("id")}
        self.presets = {str(x["id"]): x for x in presets if isinstance(x, dict) and x.get("id")}
        self.variables = {str(v) for v in variables}
        self.shortcuts = {str(x["name"]): x for x in shortcuts if isinstance(x, dict) and x.get("name")}
        self.settings = {(x["module"], x["name"]): x for x in settings if isinstance(x, dict) and x.get("module") and x.get("name")}

    def open_items(self) -> list:
        return [i for i, x in enumerate(self.plan, 1) if x["state"] == "open"]

    def planning(self, name, args) -> str:
        """`plan` writes the checklist, `check_item` ticks an item. They change nothing in the project; the app holds the model to the list."""
        if name == "plan":
            items = args.get("items")
            if isinstance(items, str):
                items = [x.strip(" -*\t0123456789.)") for x in items.splitlines() if x.strip()]
            if not isinstance(items, list) or not 1 <= len(items) <= 60 or not all(isinstance(x, str) and x.strip() for x in items):
                return "Error: items is a list of one to 60 texts, one per change or batch of changes."
            self.plan = [{"text": x.strip()[:300], "state": "open", "note": ""} for x in items]
            self.plan_changes = 0                          # any change made in this request counts, whether it came before the list or after
            self.on_event({"kind": "status", "text": f"Checklist: 0 of {len(self.plan)} done"})
            self.show_plan()
            return f"Checklist saved with {len(self.plan)} items. Do them in order: make the change for an item, then call check_item for it. You may answer the user only when every item is checked."
        if not self.plan:
            return "Error: there is no checklist yet. Call plan first."
        n, result = args.get("item"), str(args.get("result") or "").strip()
        if isinstance(n, str) and n.strip().isdigit():
            n = int(n)
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= len(self.plan):
            return f"Error: item is a number from 1 to {len(self.plan)}."
        if not result:
            return "Error: result is 'done', or 'cannot: ' and the reason."
        if re.match(r"(?i)\s*(cannot|can't|skipped|refused|not possible)", result):
            if len(result) < 12:
                return "Error: say why it cannot be done, after 'cannot: '."
            state = "skipped"
        elif self.changes <= self.plan_changes:
            return "Error: no change was made since the checklist was written, so nothing can be done yet. Make the change with its tool first, then check the item."
        else:
            state = "done"
        self.plan[n - 1].update(state=state, note=result[:200])
        left = self.open_items()
        self.on_event({"kind": "status", "text": f"Checklist: {len(self.plan) - len(left)} of {len(self.plan)} done"})
        self.show_plan()
        return f"Item {n} marked {state}. " + (f"Still open: {', '.join(str(i) for i in left[:20])}." if left else "Every item is checked: now tell the user in one or two sentences what was done and what could not be.")

    def read_chat_log(self, turns) -> str:
        n = min(20, max(1, int(turns))) if isinstance(turns, (int, float)) and not isinstance(turns, bool) else 6
        got = self.chat_log(n) if self.chat_log else ""
        return got or "The Chat tab conversation is empty."

    def show_plan(self):
        """The checklist as it stands, for the page to keep pinned."""
        self.on_event({"kind": "plan", "items": [{"text": x["text"], "state": x["state"], "note": x["note"]} for x in self.plan]})

    def plan_note(self) -> str:
        left = self.open_items()
        lines = "\n".join(f"{i}. {self.plan[i - 1]['text']}" for i in left[:12])
        return app_note(f"Your checklist is not finished: {len(self.plan) - len(left)} of {len(self.plan)} items are checked. Still open:\n{lines}\n"
                "Do the next open item now: call the tool for it with all its arguments, then call check_item for it. If an item cannot be done, call check_item with result 'cannot: ' and the reason. "
                "Do not write to the user until every item is checked.")

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

    @staticmethod
    def _norm(name) -> str:
        return " ".join(unicodedata.normalize("NFKC", str(name or "")).casefold().replace("\u200b", "").split())

    def shortcut_named(self, name):
        """-> the library's own spelling of a shortcut name the model wrote, or None. The model copies names from a lookup and may change the case, the spacing or a
        look-alike character; a name that is only different in those ways, or a name the lookup cut off that has one match, is still that shortcut."""
        name = str(name or "")
        if name in self.shortcuts:
            return name
        bare = lambda n: self._norm(re.sub(r"\s*[\(\[（][^()\[\]（）]*[\)\]）]\s*$", "", str(n or "")))        # noqa: E731 -- a name followed by a description in brackets
        for flat in (self._norm, bare):                       # first the name as written, then with a bracketed note taken off either side
            key = flat(name)
            same = [n for n in self.shortcuts if flat(n) == key]
            if len(same) == 1 and key:
                return same[0]
        key = bare(name)
        near = [n for n in self.shortcuts if key and bare(n).startswith(key)]
        return near[0] if len(near) == 1 else None

    def unknown_shortcut(self, name) -> str:
        close = difflib.get_close_matches(str(name or ""), list(self.shortcuts), n=5, cutoff=0.5)
        return f"Error: there is no shortcut called '{name}'." + (f" Did you mean: {'; '.join(close)}?" if close else " Call find_shortcuts to see the names.")

    def find_shortcut(self, query, offset=0) -> str:
        """A page of the matches, with how many there are and where to go on, so a request for all of them is not cut short unnoticed. An empty query lists them all."""
        q = str(query or "").lower()
        phrases = [p.strip() for p in re.findall(r'"([^"]+)"', q) if p.strip()]       # "in quotes": that exact run of text, anywhere in a shortcut
        words = [w for w in re.findall(r"[a-z0-9]+", re.sub(r'"[^"]*"', " ", q)) if len(w) > 1]
        total = len(self.shortcuts)
        hits = [x for x in self.shortcuts.values() if all(w in (x["name"] + " " + " ".join(x.get("triggers") or []) + " " + " ".join(x.get("replacements") or [])).lower() for w in [*words, *phrases])]
        if not hits:
            return f"No shortcut matches. The library has {total} shortcut(s)."
        start = offset if isinstance(offset, int) and not isinstance(offset, bool) and offset > 0 else 0
        page = hits[start:start + FIND_PAGE]
        if not page:
            return f"{len(hits)} shortcut(s) match, so offset {start} is past the end."
        rows = "\n".join(f"{x['name']}: {', '.join(x.get('triggers') or [])} -> {' | '.join(x.get('replacements') or [])}{'' if x.get('enabled', True) else ' (off)'}" for x in page)
        end = start + len(page)
        head = f"{len(hits)} of the library's {total} shortcut(s) match; showing {start + 1} to {end}."
        return head + (f" That is not all of them: call find_shortcuts again with offset {end} for the next {min(FIND_PAGE, len(hits) - end)}, until you have seen all {len(hits)}." if end < len(hits) else " That is all of them.") + "\n" + rows

    def extra(self, name, args):
        """-> an error text, or None when the call is valid (args may gain fields the page needs)."""
        text = lambda v, n=MAX_TEXT: isinstance(v, str) and len(v) <= n        # noqa: E731
        if name == "share_reference":
            return None if isinstance(args.get("from_scene"), int) and 1 <= args["from_scene"] <= self.scenes else f"Error: from_scene must be a number from 1 to {self.scenes}."
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
            own = self.shortcut_named(args["name"])
            args["existing"] = own is not None
            if own:
                args["name"] = own
            return None
        if name == "edit_shortcuts":
            changes = args.get("changes")
            if not isinstance(changes, list) or not 0 < len(changes) <= 60 or not all(isinstance(c, dict) for c in changes):
                return "Error: changes is a list of one to 60 items, each with the shortcut's name."
            ok = lambda v: isinstance(v, list) and v and all(isinstance(x, str) and x.strip() and len(x) <= 500 for x in v)        # noqa: E731
            known, taken, out = dict(self.shortcuts), set(), []
            lost = [str(c.get("name") or "") for c in changes if self.shortcut_named(c.get("name")) is None]
            if lost:
                more = f" (and {len(lost) - 5} more)" if len(lost) > 5 else ""
                return "Error: " + " ".join(self.unknown_shortcut(n)[len("Error: "):] for n in lost[:5]) + more + " Nothing was changed."
            for c in changes:
                old = self.shortcut_named(c.get("name"))
                if old in taken:
                    return f"Error: '{old}' appears twice in changes."
                taken.add(old)
                item = {"name": old}
                if c.get("delete") is True:
                    out.append({**item, "delete": True})
                    continue
                for key in ("triggers", "replacements"):
                    if key in c and not ok(c[key]):
                        return f"Error: {key} of '{old}' must be a non-empty list of texts."
                    if key in c:
                        item[key] = [x.strip() for x in c[key]]
                if "new_name" in c:
                    new = c["new_name"]
                    if not text(new, 80) or not new.strip():
                        return f"Error: new_name of '{old}' must be text of at most 80 characters."
                    clash = new.strip() != old and (new.strip() in known or new.strip() in {str(x.get('new_name') or '').strip() for x in out})
                    if clash:
                        return f"Error: a shortcut named '{new.strip()}' already exists."
                    item["new_name"] = new.strip()
                if "category" in c:
                    item["category"] = str(c["category"] or "").strip()[:80]
                if len(item) == 1:
                    return f"Error: the change for '{old}' changes nothing."
                out.append(item)
            args["changes"] = out
            return None
        own = self.shortcut_named(args.get("name"))        # remove_shortcut
        if own is None:
            return self.unknown_shortcut(args.get("name"))
        args["name"] = own
        return None

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
        """Answer one tool call; the page hears which tool is at work (kind "doing") until it is done."""
        self.on_event({"kind": "doing", "tool": str(call.get("name") or "")})
        try:
            return self._call(call)
        finally:
            self.on_event({"kind": "doing", "tool": ""})

    def _call(self, call) -> str:
        name, args = call.get("name"), call.get("args") or {}
        self.calls += 1
        if call.get("unreadable"):
            return ("Error: the call to " + str(name) + " could not be read, so nothing happened. Write it again with each text argument between the quote marks the format uses, "
                    "lists in [ ] and objects in { }, and with every argument the tool needs. What you wrote starts: " + str(call["unreadable"])[:160])
        if name not in READS:
            self.changes += 1
        if name in ("plan", "check_item"):
            return self.planning(name, args)
        if name == "read_chat_log":
            return self.read_chat_log(args.get("turns"))
        if name in ("read_prompt", "edit_prompt"):
            return self.editor(call) if self.editor else "Error: the enhancer's system prompts are not available."
        if name not in ("edit_scene", "add_scene", "copy_reference", "set_clip_length", "find_settings", "set_setting", "find_shortcuts", "find_node_inputs") + EXTRA:
            return f"Error: there is no tool called '{name}'."
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 600        # noqa: E731
        if name == "find_settings":
            return self.find(args.get("query"))
        if name == "find_shortcuts":
            return self.find_shortcut(args.get("query"), args.get("offset"))
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
            elif name == "edit_shortcuts":
                for c in args["changes"]:
                    cur = self.shortcuts.pop(c["name"], None)
                    if c.get("delete") or cur is None:
                        continue
                    new = {**cur, **{k: v for k, v in c.items() if k not in ("name", "new_name")}, "name": c.get("new_name") or c["name"]}
                    self.shortcuts[new["name"]] = new
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


NUDGE = app_note("You described a change but did not call a tool, so nothing happened. Call the tool now, with the full arguments, and write nothing else.")
_TOOL_WORDS = re.compile(r"\b(" + "|".join(["edit_scene", "add_scene", "copy_reference", "set_clip_length", "set_story", "cut_scene", "set_cut_words", "set_project_text", "set_variable", "remove_variable",
                                              "save_shortcut", "remove_shortcut", "edit_shortcuts", "set_setting", "set_node_inputs", "apply_preset", "run_action"]) + r")\b|\bI (?:will|would|propose|suggest)\b", re.I)


def announced(reply) -> bool:
    """Whether an answer talks about a change (a tool name, or "I will / I propose") instead of making it."""
    return bool(_TOOL_WORDS.search(str(reply or "")))


_TOOL_BLOCK = re.compile(r"(?<![A-Z_])TOOL[ \t]*:[ \t]*([A-Za-z_]+)[ \t]*\n(.*?)(?:\n?[ \t]*END_TOOL|\Z)", re.S)


def text_rule(with_rules=False) -> str:
    """How a model without tool calls makes its changes: it writes TOOL blocks, the app runs them and answers with the results."""
    lines = []
    for t in tools(with_rules):
        f = t["function"]
        props = ", ".join(f["parameters"]["properties"])
        lines.append(f"- {f['name']}({props}): {f['description']}")
    return ("\n\nYou have no built-in tool calls, so make every change or lookup by writing a block in your answer, exactly like this, one block per call:\n"
            "TOOL: set_clip_length\n{\"seconds\": 6, \"reason\": \"why\"}\nEND_TOOL\n"
            "The line after TOOL: holds the tool name, the next lines are its arguments as JSON. Write no text of your own before the blocks. The app shows each change to the user, runs it if allowed and "
            "answers you with TOOL RESULTS; then write the next block, or tell the user in one or two sentences what happened. Tools:\n" + "\n".join(lines))


def loose_json(body):
    """JSON as small models write it: code fences, smart or single quotes, newlines inside strings, trailing commas and unescaped quotes inside a text are all read. Raises ValueError when nothing can be made of it."""
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", str(body).strip())
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    for attempt in (t, re.sub(r",\s*([}\]])", r"\1", t)):
        try:
            return json.loads(attempt, strict=False)
        except ValueError:
            pass
    try:
        import ast
        got = ast.literal_eval(t)
        if isinstance(got, dict):
            return got
    except (ValueError, SyntaxError):
        pass
    keys = list(re.finditer(r"""["']([A-Za-z_]+)["']\s*:\s*""", t))
    if not keys:
        raise ValueError("no arguments")
    out = {}
    for i, m in enumerate(keys):
        end = keys[i + 1].start() if i + 1 < len(keys) else len(t)
        raw = t[m.end():end].strip().rstrip("}").rstrip().rstrip(",").strip()
        if raw[:1] in "\"'" and raw[-1:] == raw[:1] and len(raw) >= 2:
            out[m.group(1)] = raw[1:-1].replace('\\"', '"').replace("\\n", "\n")
        else:
            try:
                out[m.group(1)] = json.loads(raw.lower() if raw.lower() in ("true", "false", "null") else raw)
            except ValueError:
                out[m.group(1)] = raw.strip("\"'")
    return out


def parse_calls(text):
    """-> (the answer without TOOL blocks, [{"name", "args"}]). A block whose arguments are not JSON carries "error"."""
    found = []

    def take(m):
        body = m.group(2).strip()
        try:
            args = loose_json(body) if body else {}
            found.append({"name": m.group(1), "args": args if isinstance(args, dict) else {}, **({} if isinstance(args, dict) else {"error": "the arguments must be a JSON object"})})
        except ValueError:
            found.append({"name": m.group(1), "args": {}, "error": "the arguments are not valid JSON"})
        return ""
    clean = _TOOL_BLOCK.sub(take, str(text or ""))
    return re.sub(r"\n{3,}", "\n\n", clean).strip(), found


def run_text_calls(ask, actions, message, state, first, rounds=4):
    """The text-mode loop for a model without tool calls: run its TOOL blocks, tell it the results, until it answers in words. -> the last result tuple."""
    reply, status, add, drop, thought = first
    pushed, ran = 0, []
    try:
        for _ in range(rounds):
            text, calls = parse_calls(reply)
            if not calls:
                break
            results = []
            for call in calls:
                out = f"Error: {call['error']}." if call.get("error") else actions(call)
                results.append(f"{call['name']}: {out}")
                ran.append(f"{call['name']}: {str(out).splitlines()[0][:120]}")
            state["turns"].append({"request": str(message).strip(), "reply": reply, "kind": "chat"})
            pushed += 1
            message = app_note("TOOL RESULTS:\n" + "\n".join(results) + "\nNow write the next TOOL block, or tell the user in one or two sentences what happened.")
            reply, status, more_add, more_drop, thought = ask(message)
            add, drop = add + more_add, drop + more_drop
    finally:
        for _ in range(pushed):
            state["turns"].pop()
    text, _calls = parse_calls(reply)
    return text or "Done.", status, add, drop, thought
