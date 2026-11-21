"""The enhancer's own rules as something the model can read and, with the person's approval, change.

Every system prompt of the Enhance tab is a "rule" with a name and a purpose. The model gets two tools: `read_prompt` (see one in
full) and `edit_prompt` (propose a new text). A proposal is never applied here: it is sent to the page as an `edit_request` event,
shown as a before/after diff, and waits for Allow or Disallow. The page saves an allowed text into the project; the model is told
the answer and carries on.
"""

import itertools
import re
import threading

FIELDS = {
    "instructions": "What you do with each prompt you are given to rewrite.",
    "marks_prompt": "How you keep (word:1.5) and [a|b] markup in a rewrite.",
    "fixed_prompt": "How you treat the fixed text before and after the part you rewrite.",
    "reference_intro": "How you read the shortcut and lorebook entries sent along with a prompt.",
    "conversation_prompt": "The conversation rules when a plain encoder writes the rewrites.",
    "turns_prompt": "The conversation rules when the rewrites are written as real chat turns.",
    "revision_prompt": "How you apply the user's comments when there is no conversation.",
    "rating_prefix": "The words in front of the user's rating.",
    "chat_prompt": "How you answer the user in the Chat tab.",
    "notes_prompt": "How you add and drop the user's standing Notes.",
    "compact_prompt": "How a long conversation is shortened into notes.",
    "edit_prompt": "These rules about reading and changing your own rules.",
}
MAX_TEXT = 8000
WAIT = 600                      # seconds a proposal waits for an answer before it counts as not allowed
_DECISIONS = {}                 # proposal id -> {"event": Event, "allow": bool}
_IDS = itertools.count(1)


def sheet(rules=None) -> str:
    """The rules as a list for the model: name and purpose (the full text comes from read_prompt)."""
    return "\n".join(f"- {name}: {purpose}" for name, purpose in FIELDS.items() if rules is None or name in rules)


def tools() -> list:
    """The two tools, in the JSON-schema form every transformers chat template accepts."""
    field = {"type": "string", "enum": list(FIELDS), "description": "Which rule."}
    return [{"type": "function", "function": {"name": "read_prompt", "description": "Read one of your own rules in full.",
                                              "parameters": {"type": "object", "properties": {"field": field}, "required": ["field"]}}},
            {"type": "function", "function": {"name": "edit_prompt", "description": "Propose a new text for one of your own rules. The user sees the change and may refuse it.",
                                              "parameters": {"type": "object", "properties": {"field": field, "new_text": {"type": "string", "description": "The whole new text of the rule."},
                                                                                              "reason": {"type": "string", "description": "Why, in one sentence."}}, "required": ["field", "new_text", "reason"]}}}]


_EDIT_BLOCK = re.compile(r"EDIT_RULE[ \t]*:?[ \t]*([A-Za-z_]+)[ \t]*[,:]?[ \t]*\n?(.*?)\n?[ ,\t]*END_EDIT_RULE", re.S)


def parse_edits(text):
    """The rule changes a model wrote in its answer instead of calling the tool: `EDIT_RULE: name`, the whole new text, `END_EDIT_RULE`.
    -> (the answer without those blocks, [{"field", "new_text"}])."""
    found = []

    def take(m):
        found.append({"field": m.group(1).strip(), "new_text": m.group(2).strip()})
        return ""
    clean = _EDIT_BLOCK.sub(take, str(text or ""))
    return re.sub(r"\n{3,}", "\n\n", clean).strip(), found


def decide(proposal, allow) -> bool:
    """The page's answer to a proposal. -> whether a proposal was waiting for it."""
    slot = _DECISIONS.get(str(proposal))
    if slot is None:
        return False
    slot["allow"] = bool(allow)
    slot["event"].set()
    return True


class Editor:
    """Answers the model's tool calls for one chat request. `rules`: {name: the text now in force}; `on_event` carries proposals to the page."""

    def __init__(self, rules, on_event, wait=WAIT):
        self.rules, self.on_event, self.wait = dict(rules or {}), on_event, wait

    def __call__(self, call) -> str:
        name, args = call.get("name"), call.get("args") or {}
        field = str(args.get("field") or "")
        if field not in FIELDS or field not in self.rules:
            return f"Error: there is no rule called '{field}'. The rules are: {', '.join(n for n in FIELDS if n in self.rules)}."
        if name == "read_prompt":
            return self.rules[field]
        if name != "edit_prompt":
            return f"Error: there is no tool called '{name}'."
        new = args.get("new_text")
        if not isinstance(new, str) or not new.strip():
            return "Error: new_text must be the whole new text of the rule."
        if len(new) > MAX_TEXT:
            return f"Error: the text is longer than {MAX_TEXT} characters."
        if new.strip() == self.rules[field].strip():
            return "Nothing to change: that is already the text of the rule."
        pid = f"edit-{next(_IDS)}"
        slot = _DECISIONS[pid] = {"event": threading.Event(), "allow": False}
        try:
            self.on_event({"kind": "edit_request", "id": pid, "field": field, "old": self.rules[field], "new": new.strip(), "reason": str(args.get("reason") or "").strip()})
            answered = slot["event"].wait(self.wait)
        finally:
            _DECISIONS.pop(pid, None)
        if not answered:
            return "No answer from the user in time, so nothing was changed."
        if slot["allow"]:
            self.rules[field] = new.strip()
            return "Allowed: the user saved the change. It applies from the next request."
        return "Disallowed by the user. Nothing was changed. Do not propose the same change again."
