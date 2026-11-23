"""The Enhance Prompt node: an LLM rewrites the finished prompt before it is encoded."""

import collections
import json
import re
import threading
import time

from comfy_api.latest import io

from ..._core import log
from . import enhance as en, history
from .engine import ENGINE, EngineClip

#: The runs' rewrites and chat messages, newest last: what the Composer's tabs show. Kept in history.py's file, so a restart keeps them.
RUNS = collections.deque(maxlen=history.MAX_RUNS)
#: "Send everything into chat": the earlier turns {request, reply} of each conversation, kept in the same file.
CONVOS = collections.OrderedDict()
MAX_CONVOS = 20
history.load(RUNS, CONVOS, en.CONVO_KEEP)


_MARKUP = re.compile(r"\([^()]*:\s*[\d.]+\)|\[[^\]]*\|[^\]]*\]|⟦[^⟧]*⟧")
_MASK = re.compile(r"\{M(\d+)\}")
SCOPES = ("whole", "own")       # whole: the model rewrites everything it is given; own: only the text between the prefix and the postfix


def mask_markup(text):
    """-> (text, marks): each (word:1.5) / [a|b] span becomes {M1}, {M2}... so the model cannot rewrite it."""
    marks = []

    def take(m):
        marks.append(m.group(0))
        return "{M%d}" % len(marks)
    return _MARKUP.sub(take, str(text or "")), marks


def unmask_markup(text, marks):
    """-> text with every mark put back, or None when the model lost or doubled one."""
    seen = [int(n) for n in _MASK.findall(text)]
    if sorted(seen) != list(range(1, len(marks) + 1)):
        return None
    return _MASK.sub(lambda m: marks[int(m.group(1)) - 1], text)


_LOADED = {}      # one entry: the last separate encoder, so a second Generate does not load a 9 GB file again


def _own_encoder(file, family):
    """The encoder the person picked for the enhancer, loaded the way the pipeline's own loader loads one."""
    if _LOADED.get("key") != (file, family):
        from ...loaders.clip import nodes as clip_nodes        # here, not at the top: the enhancer must load even when the CLIP loader module did not
        _LOADED.clear()
        _LOADED.update(key=(file, family), clip=clip_nodes.FunUICLIPLoader.execute(clip_name1=file, type=family).args[0])
    return _LOADED["clip"]


def _stop_if_interrupted():
    """Hand ComfyUI's Stop on: it raises its own interrupt, which ends the queue item cleanly instead of going on to the sampler."""
    try:
        import comfy.model_management as mm
        stop = mm.throw_exception_if_processing_interrupted
    except Exception:                                               # noqa: BLE001 -- not inside a working ComfyUI
        return
    stop()


def fit(clip, state, fixed_chars, max_length, compact, context_tokens, prompts=None, macros=None):
    """-> (about how many tokens the next request fills, the context window, a status note, whether it was summarised now).
    The conversation is summarised first when the request would fill COMPACT_AT of the window (4 characters a token, the answer's room included)."""
    # A small local model slows down and loses focus on a very long context, so what it reports is only a ceiling: summarising
    # works to DEFAULT_WINDOW unless the person sets a size.
    window = int(context_tokens or 0) or min(en.context_window(clip) or en.DEFAULT_WINDOW, en.DEFAULT_WINDOW)
    keep = en.CONVO_KEEP if compact else en.CONVO_TURNS
    size = lambda: (fixed_chars + len(en.convo_block(state["turns"] if state else (), state["summary"] if state else "", keep))) // en.CHARS_PER_TOKEN + (int(max_length) or 1024)    # noqa: E731
    note, did = "", False
    if state and compact and state["turns"] and size() > en.COMPACT_AT * window:
        summary = en.compact(clip, state["summary"], list(state["turns"]), prompts=prompts, macros=macros)
        if summary:
            state["summary"] = summary
            state["turns"].clear()
            note, did = "; the earlier conversation was summarised to fit the context window", True
        else:
            note = "; the conversation is near the context window and could not be summarised"
    return size(), window, note, did


def _pictures(*images):
    """The connected pictures: none, one, or a list when several are connected."""
    got = [i for i in images if i is not None]
    return got[0] if len(got) == 1 else (got or None)


def _chat_reply(conversation, message, *, stop, notes="", prompts=None, comments=(), engine_model="", on_event=None, rules=None, self_edit=False, speed=None, use_comfy=False, encoder_file="", encoder_family="stable_diffusion", compact=True, context_tokens=0, max_length=400, thinking=False, video_length="", aspect_ratio="", helper=None, **sampling):
    """One live answer in a conversation, from the encoder the person picked. -> {"ok", "reply", "status"}. The exchange joins the conversation."""
    key = str(conversation or "").strip()
    editing, editor, text_mode, actions = False, None, False, None
    if not key:
        return {"ok": False, "reply": "", "status": "no conversation"}
    engine_model = "" if use_comfy else (str(engine_model or "").strip() or (f"comfy:{encoder_file}" if encoder_file else ""))
    speed = speed or {}
    if engine_model:
        try:
            ENGINE.load(engine_model, speed.get("precision", "auto"))
        except Exception as e:                                      # noqa: BLE001
            if not encoder_file:
                return {"ok": False, "reply": "", "status": f"could not load the engine model {engine_model}: {e}"}
            engine_model = ""                                       # an older project's encoder: ComfyUI's own loader reads it
        else:
            clip = EngineClip(ENGINE, on_event, live="chat", cache=speed.get("prefix_cache", True), lookup=speed.get("lookup_tokens", 0))
            if helper and on_event:                                  # the helper acts on the project: each call is asked of the person
                from . import engine as eng, helper as hp
                template = getattr(ENGINE.tok, "chat_template", "") or ""
                actions = hp.Actions(helper.get("scenes"), on_event, settings=helper.get("settings") or (), variables=helper.get("variables") or (), shortcuts=helper.get("shortcuts") or (), inputs=helper.get("inputs") or (), presets=helper.get("presets") or (), actions=helper.get("actions") or (), rules_given=helper.get("rules") or None)
                if "tools" in template and eng.dialect_for(ENGINE.tok.get_vocab(), template).call_open:
                    clip.tools, clip.handler = hp.tools(bool(helper.get("rules")), hp.intent(message)), actions
                else:                                                # no tool calls: it writes TOOL blocks in its answer and the app runs them
                    text_mode = True
            elif self_edit and rules and on_event:                     # the model may read and propose changes to its own rules, each one asked of the person
                from . import engine as eng, rules as rl
                template = getattr(ENGINE.tok, "chat_template", "") or ""
                editor = rl.Editor(rules, on_event)
                editing = True
                if "tools" in template and eng.dialect_for(ENGINE.tok.get_vocab(), template).call_open:        # the template must render tools and the model must have a call format
                    clip.tools, clip.handler = rl.tools(), editor
                else:
                    on_event({"kind": "status", "text": "this model has no tool calls: it proposes a rule change in its answer instead"})
    if not engine_model:
        if not encoder_file:
            return {"ok": False, "reply": "", "status": "pick a model in Composer > Enhance: the pipeline's own encoder cannot chat"}
        try:
            clip = _own_encoder(encoder_file, encoder_family)
        except Exception as e:                                      # noqa: BLE001
            return {"ok": False, "reply": "", "status": f"could not load the encoder {encoder_file}: {e}"}
    state = CONVOS.setdefault(key, {"summary": "", "turns": collections.deque(maxlen=en.CONVO_KEEP), "seen": set()})
    CONVOS.move_to_end(key)
    sheet = __import__(__package__ + ".rules", fromlist=["sheet"]).sheet(rules) if editing else ""
    macros = en.macro_values(state["turns"], state["summary"], comments, notes, "", "", message, rules=sheet, length=video_length, aspect=aspect_ratio)
    used, window, note, did = fit(clip, state, len(en.pick(prompts, "chat_prompt")) + len(en.pick(prompts, "notes_prompt")) + len(str(message)), max_length, compact, context_tokens, prompts, macros)
    if did:
        macros = en.macro_values(state["turns"], state["summary"], comments, notes, "", "", message, rules=sheet, length=video_length, aspect=aspect_ratio)      # the shortened turns are in the notes now
    keep = en.CONVO_KEEP if compact else en.CONVO_TURNS
    hp_rule = __import__(__package__ + ".helper", fromlist=["x"]).HELPER_RULE.replace("[[PROJECT_SHEET]]", str((helper or {}).get("sheet") or "(empty)") + _chat_memory(helper, notes)) + (__import__(__package__ + ".helper", fromlist=["x"]).text_rule(bool(helper.get("rules"))) if text_mode else "") + (__import__(__package__ + ".helper", fromlist=["x"]).rules_note(helper.get("rules")) if helper and helper.get("rules") else "") if helper else ""
    def ask(text):
        stop.check()
        out = _ask(text)
        stop.check()                                                  # the writing was cut short by the Stop: what it wrote so far is not an answer
        return out
    _ask = lambda text: en.converse(clip, text, convo={"summary": state["summary"], "turns": list(state["turns"]), "keep": keep}, notes=notes, prompts=prompts, macros=macros, system_extra=hp_rule if helper else (en.pick(prompts, "edit_prompt") if clip.tools else en.EDIT_TEXT_RULE) if editing else "",
                                           max_length=max_length, thinking=thinking, **sampling)
    reply, status, add, drop, thought = ask(message)
    if helper and text_mode and actions is not None:
        reply, status, add, drop, thought = __import__(__package__ + ".helper", fromlist=["x"]).run_text_calls(ask, actions, message, state, (reply, status, add, drop, thought))
    if helper and clip.tools and getattr(clip.handler, "calls", 1) == 0:
        hp = __import__(__package__ + ".helper", fromlist=["x"])
        asked_only = str(message).strip().endswith("?") and not hp.announced(reply)         # a plain question may simply be answered
        if not asked_only:
            nudged = hp.announced(reply)
            if nudged:
                # the model wrote about a change instead of calling the tool: it is told once to call it (its own answer stays in the conversation for that turn)
                state["turns"].append({"request": str(message).strip(), "reply": reply, "kind": "chat"})
                try:
                    second = ask(hp.NUDGE)
                finally:
                    state["turns"].pop()
                if second[0] or clip.handler.calls:
                    reply, status, add, drop, thought = second[0] or "Done.", second[1], add + second[2], drop + second[3], second[4]
            if not clip.handler.calls and (not nudged or hp.announced(reply)):
                # the model has tool calls on paper but does not use them: it is asked again in text mode, writing TOOL blocks that the app runs
                clip.tools, text_mode, actions = None, True, clip.handler
                hp_rule += hp.text_rule(bool(helper.get("rules")))
                reply, status, add, drop, thought = hp.run_text_calls(ask, actions, message, state, ask(message))
    if helper and actions is not None and not getattr(actions, "plan", None):
        hp = __import__(__package__ + ".helper", fromlist=["x"])
        before = list(state["turns"])[-1] if state["turns"] else {}
        cut_short = str(before.get("reply", "")).startswith("(the user stopped")
        if (hp.wants_change(message, before.get("reply", "")) or (cut_short and hp.wants_change(before.get("request", ""))) or (hp.announced(reply) and not str(message).strip().endswith("?"))):
            # a request for a change is done from a checklist at all times, even one change: the model is asked for the list if it did not write one
            if not text_mode:
                clip.tools = hp.tools(bool(helper.get("rules")))
            state["turns"].append({"request": str(message).strip(), "reply": reply or "(no answer)", "kind": "chat"})
            try:
                for _ in range(2):
                    got = ask(hp.PLAN_NOTE)
                    if text_mode:
                        got = hp.run_text_calls(ask, actions, hp.PLAN_NOTE, state, got)
                    if actions.plan:
                        reply, status, add, drop, thought = got[0] or reply, got[1], add + got[2], drop + got[3], got[4]
                        break
            finally:
                state["turns"].pop()
    if helper and actions is not None and not getattr(actions, "changes", 0):
        hp = __import__(__package__ + ".helper", fromlist=["x"])
        before = list(state["turns"])[-1] if state["turns"] else {}
        cut_short = str(before.get("reply", "")).startswith("(the user stopped")     # a new message that interrupted a change request is the next part of it
        if hp.wants_change(message, before.get("reply", "")) or (cut_short and hp.wants_change(before.get("request", ""))) or (hp.announced(reply) and not str(message).strip().endswith("?")):
            # the request asked for something to be done and every try ended without a change (an answer in words, or only a look at a rule): one more round that names the fitting tools
            note = hp.force_note(message, bool(helper.get("rules")), f"{before.get('request', '')} {before.get('reply', '')}")
            if not text_mode:
                clip.tools = hp.tools(bool(helper.get("rules")))
            state["turns"].append({"request": str(message).strip(), "reply": reply or "(no answer)", "kind": "chat"})
            try:
                for attempt in range(3):                              # a rule read first, or a call that was refused as incomplete, gets the next round
                    forced = ask(note if not attempt else hp.FORCE_AGAIN)
                    if text_mode:
                        forced = hp.run_text_calls(ask, actions, note, state, forced)
                    if actions.changes:
                        break
            finally:
                state["turns"].pop()
            if actions.changes:
                reply, status, add, drop, thought = forced[0] or "Done.", forced[1], add + forced[2], drop + forced[3], forced[4]
    if helper and actions is not None and getattr(actions, "plan", None) and actions.open_items():
        # a checklist the model wrote is not finished: it is sent back to the open items until every one is checked (or it keeps making no headway)
        hp = __import__(__package__ + ".helper", fromlist=["x"])
        idle = 0
        for _ in range(hp.MAX_PLAN_ROUNDS):
            todo = actions.open_items()
            if not todo:
                break
            mark = (actions.calls, len(todo))
            note = actions.plan_note()
            if not text_mode:
                clip.tools = hp.tools(bool(helper.get("rules")))
            state["turns"].append({"request": str(message).strip(), "reply": reply or "(no answer)", "kind": "chat"})
            try:
                got = ask(note)
                if text_mode:
                    got = hp.run_text_calls(ask, actions, note, state, got)
            finally:
                state["turns"].pop()
            reply, status, add, drop, thought = got[0] or reply, got[1], add + got[2], drop + got[3], got[4]
            idle = idle + 1 if (actions.calls, len(actions.open_items())) == mark else 0
            if idle >= 3:
                break
        left = actions.open_items()
        if left:
            reply = (reply or "").rstrip() + f"\n\n(Not finished: items {', '.join(str(i) for i in left[:20])} of {len(actions.plan)} on the checklist are still open.)"
        elif not reply or reply == "Done.":
            reply = f"Done: all {len(actions.plan)} checklist items are checked."
    if helper and reply and actions is not None and not getattr(actions, "changes", 0) and not str(message).strip().endswith("?") and (not getattr(actions, "calls", 0) or __import__(__package__ + ".helper", fromlist=["x"]).wants_change(message)):
        reply += "\n\n(Nothing was changed: the model did not call a tool.)"          # a request that ended with no call must not read as done
    if helper and not reply and actions is not None and getattr(actions, "changes", 0):          # it acted and then said nothing: that is not a failure
        reply, status = "Done. The changes it asked for were handled above.", "ok"
    if helper and not reply and actions is not None and getattr(actions, "calls", 0) and not getattr(actions, "changes", 0):          # it only looked: say so instead of "nothing"
        reply, status = "I looked, but changed nothing.", "ok"
    if editing and editor is not None and reply:                      # a change written into the answer (a model that does not call the tool): the same approval as a call
        from . import rules as rl
        reply, written = rl.parse_edits(reply)
        outcome = []
        for call in written:
            verdict = editor({"name": "edit_prompt", "args": {**call, "reason": "proposed in the answer"}})
            outcome.append(f"{call['field']}: {'saved' if verdict.startswith('Allowed') else 'not changed'}")
        if outcome:
            reply = (reply + "\n\n" if reply else "") + "Rule change " + "; ".join(outcome) + "."
    ok = bool(reply)
    if ok:
        state["turns"].append({"request": str(message).strip(), "reply": reply, "kind": "chat"})
    _log({"prompt_id": f"chat-{time.time_ns()}", "kind": "chat", "conversation": key, "ok": ok, "status": status + note, "before": str(message), "after": reply,
                 "asked": str(message).strip(), "answered": reply, "remember": add, "forget": drop, "compacted": did, "summary": state["summary"] if did else "", "thinking": thought, "sent": "", "context": {"used": used, "window": window, "compact": bool(compact)}})
    while len(CONVOS) > MAX_CONVOS:
        CONVOS.popitem(last=False)
    return {"ok": ok, "reply": reply, "status": status + note, "remember": add, "forget": drop, "compacted": did}


class Interrupted(Exception):
    """The person stopped the helper, or sent a new message while it was still working."""


class Stop(threading.Event):
    """One helper request's stop flag, and the proposals it is waiting on (they are declined when it stops)."""

    def __init__(self):
        super().__init__()
        self.pending, self.done = set(), False

    def check(self):
        if self.is_set():
            raise Interrupted()

    def interrupt(self):
        self.set()
        from . import rules
        for pid in list(self.pending):
            rules.decide(pid, False)


STOPS, LOCKS = {}, {}


def stop_chat(conversation) -> bool:
    """Stop what the helper is doing for this conversation. -> whether a request was running."""
    stop = STOPS.get(str(conversation or "").strip())
    if stop is None or stop.done:
        return False
    stop.interrupt()
    return True


def chat_reply(conversation, message, *, on_event=None, **kw):
    """One chat or helper request. A new request of a conversation that is still working stops that one first and then runs, so the model
    reads what it was asked before and the new message together; the stopped request stays in the conversation as unanswered."""
    from . import engine as eng
    key = str(conversation or "").strip()
    if not key:
        return _chat_reply(conversation, message, stop=Stop(), on_event=on_event, **kw)
    stop_chat(key)
    with LOCKS.setdefault(key, threading.Lock()):
        stop = STOPS[key] = Stop()

        def tap(ev):
            if ev.get("kind") in ("action_request", "edit_request") and ev.get("id"):
                stop.pending.add(ev["id"])
            if on_event:
                on_event(ev)
        eng.ASKING.stop = stop
        try:
            return _chat_reply(conversation, message, stop=stop, on_event=tap if on_event else None, **kw)
        except Interrupted:
            state = CONVOS.setdefault(key, {"summary": "", "turns": collections.deque(maxlen=en.CONVO_KEEP), "seen": set()})
            state["turns"].append({"request": str(message).strip(), "reply": "(the user stopped this request before it was answered)", "kind": "chat"})
            return {"ok": False, "reply": "", "status": "stopped", "stopped": True, "remember": [], "forget": [], "compacted": False}
        finally:
            eng.ASKING.stop = None
            stop.done = True


def _chat_memory(helper, notes) -> str:
    """What the helper may read of the Chat tab: its standing notes, the summary of the earlier conversation and the latest turns (it never writes there)."""
    chat = CONVOS.get(str((helper or {}).get("chat") or ""))
    parts = [("\n\nNOTES (the user's standing instructions):\n" + str(notes).strip()) if str(notes or "").strip() else ""]
    if chat:
        turns = "\n".join(f"User: {t['request']}\nModel: {t['reply']}" for t in list(chat["turns"])[-6:])
        parts.append("\n\nTHE CHAT TAB CONVERSATION (read only, for what the user said before; it is not a field of the project):\n" + ((("Condensed earlier part of that conversation: " + chat["summary"] + "\n") if chat["summary"] else "") + turns)[-6000:])
    return "".join(parts)


_LAST_CHAT_ID = [max([int(r.get("chat_id") or 0) for r in RUNS] or [0])]       # after a restart the numbers go on above the saved ones


def chat_id() -> int:
    """A number for each message of the Chat tab, rising with every call (milliseconds, pushed up by one when two come in the same one)."""
    _LAST_CHAT_ID[0] = max(int(time.time() * 1000), _LAST_CHAT_ID[0] + 1)
    return _LAST_CHAT_ID[0]


def _log(entry):
    """Keep a run for the Chat and Enhance tabs, with its chat id."""
    RUNS.append({**entry, "chat_id": chat_id()})
    history.save(RUNS, CONVOS)


def _prompt_id():
    try:
        from server import PromptServer
        return str(PromptServer.instance.last_prompt_id or "")
    except Exception:                                               # noqa: BLE001
        return ""


class FunUIEnhancePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunUIEnhancePrompt",
            display_name="FunUI Enhance Prompt",
            category="FunUI/Conditioning",
            description="Rewrite the finished prompt with a language model before it is encoded. "
                        "Off, or on any failure, the prompt passes through unchanged.",
            inputs=[
                io.Clip.Input("clip", tooltip="A text encoder that can generate text (Gemma, Qwen...)."),
                io.String.Input("text", multiline=True, default=""),
                io.Boolean.Input("enabled", default=False,
                                 tooltip="Off: the prompt passes through and the model is not run."),
                io.String.Input("engine_model", default="", optional=True,
                                tooltip="A Hugging Face model (owner/name) or a folder, run by FunUI's own transformers engine instead of the encoder. Empty uses the encoder."),
                io.String.Input("instructions", multiline=True, default="", optional=True,
                                tooltip="What the model is told to do. Empty uses the built-in instructions."),
                io.String.Input("chat_prompt", multiline=True, default="", optional=True,
                                tooltip="The chat answers' instructions (Chat tab, Send). Empty uses the built-in text."),
                io.String.Input("notes_prompt", multiline=True, default="", optional=True,
                                tooltip="How the model is told to add and drop Notes. Used with the chat instructions. Empty uses the built-in text."),
                io.String.Input("compact_prompt", multiline=True, default="", optional=True,
                                tooltip="What the model that shortens a long conversation is told. Empty uses the built-in text."),
                io.String.Input("revision_prompt", multiline=True, default="", optional=True,
                                tooltip="How the model is told to apply your comments, in a one-time request. Empty uses the built-in text."),
                io.String.Input("conversation_prompt", multiline=True, default="", optional=True,
                                tooltip="The conversation rules sent to an encoder that takes no chat turns. Empty uses the built-in text."),
                io.String.Input("turns_prompt", multiline=True, default="", optional=True,
                                tooltip="The conversation rules sent to Gemma 4, which takes real chat turns. Empty uses the built-in text."),
                io.String.Input("fixed_prompt", multiline=True, default="", optional=True,
                                tooltip="Tells the model about the fixed text around your own text (Only my own text mode). Empty uses the built-in text."),
                io.Boolean.Input("use_comfy", default=False, optional=True,
                                 tooltip="Writes with ComfyUI's own text generation (the encoder from the pipeline or the picked file) instead of FunUI's engine. The engine is not used."),
                io.Combo.Input("precision", options=["auto", "fp8", "bf16"], default="auto", optional=True,
                               tooltip="How a single fp8 model file is run. bf16 is faster but needs twice the memory; fp8 is the smallest; auto picks bf16 when the graphics card has room."),
                io.Boolean.Input("prefix_cache", default=True, optional=True,
                                 tooltip="Reuses the start of the request that did not change since the last one, so only the new part is read again."),
                io.Int.Input("lookup_tokens", default=0, min=0, max=10, optional=True,
                             tooltip="Words guessed from the prompt and checked in one step. Helps only when a rewrite mostly repeats the prompt; slower otherwise. 0 = off."),
                io.Int.Input("refine_passes", default=0, min=0, max=3, optional=True,
                             tooltip="How many times the model checks its own rewrite against the rules and improves it. Each pass is another model call, so a rewrite takes longer."),
                io.String.Input("refine_prompt", multiline=True, default="", optional=True,
                                tooltip="What the model is told for each self-check pass. Empty uses the built-in text."),
                io.Boolean.Input("self_edit", default=False, optional=True,
                                 tooltip="Lets the chat model read and propose changes to its own system prompts. Every change is shown to you first and applies only if you allow it."),
                io.String.Input("edit_prompt", multiline=True, default="", optional=True,
                                tooltip="Tells the chat model it can read and propose changes to its own rules. Empty uses the built-in text."),
                io.String.Input("marks_prompt", multiline=True, default="", optional=True,
                                tooltip="The rule that keeps (word:1.5) and [a|b] markup. Empty uses the built-in text."),
                io.Int.Input("max_length", default=0, min=0, max=4096, step=8, optional=True,
                             tooltip="The longest rewrite, in tokens. 0 = no limit."),
                io.Boolean.Input("greedy", default=False, optional=True,
                                 tooltip="Always the likeliest word. Hides the sampling dials below."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05, optional=True),
                io.Float.Input("top_p", default=0.92, min=0.0, max=1.0, step=0.01, optional=True),
                io.Int.Input("top_k", default=50, min=1, max=1000, optional=True,
                             tooltip="top_p and min_p only apply while top_k is above 0."),
                io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01, optional=True),
                io.Float.Input("repetition_penalty", default=1.3, min=1.0, max=2.0, step=0.05, optional=True),
                io.Float.Input("presence_penalty", default=0.0, min=0.0, max=2.0, step=0.05, optional=True),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, optional=True,
                             tooltip="0 = a fresh rewrite every run. Any other number repeats the same rewrite."),
                io.Boolean.Input("thinking", default=False, optional=True,
                                 tooltip="Let a thinking model reason first. Its reasoning eats the token budget."),
                io.Boolean.Input("thinking_comments", default=True, optional=True,
                                 tooltip="Enable thinking on a rewrite that has comments to follow. Off: such a rewrite does not think."),
                io.Boolean.Input("thinking_chat", default=True, optional=True,
                                 tooltip="Enable thinking on messages of the Chat tab."),
                io.Boolean.Input("thinking_helper", default=True, optional=True,
                                 tooltip="Enable thinking in the Helper."),
                io.Int.Input("helper_max_length", default=0, min=0, max=16384, step=8, optional=True,
                             tooltip="The most tokens the Helper may write in one answer. 0 means no limit."),
                # Text, not a list: a file that has gone must not stop a run where Enhance is off. The Enhance panel offers the files.
                io.String.Input("encoder_file", default="", optional=True,
                                tooltip="A text encoder file just for rewriting (Gemma, Qwen...). Empty uses the model's own encoder."),
                io.String.Input("encoder_family", default="stable_diffusion", optional=True,
                                tooltip="Which family that file is for. Only used with an encoder file."),
                io.Image.Input("image", optional=True, tooltip="Shown to the model alongside the prompt."),
                io.Image.Input("image_2", optional=True, tooltip="A second picture shown to the model (a model that reads several pictures)."),
                io.Image.Input("image_3", optional=True, tooltip="A third picture shown to the model."),
                io.Image.Input("image_4", optional=True, tooltip="A fourth picture shown to the model."),
                io.Boolean.Input("use_image", default=True, optional=True,
                                 tooltip="Send the picture (when one is connected) to the model."),
                io.String.Input("shortcuts", multiline=True, default="", optional=True,
                                tooltip="Shortcut names to hand the model as reference, one per line."),
                io.String.Input("lorebooks", multiline=True, default="", optional=True,
                                tooltip="Paths to lorebook or shortcuts JSON files, one per line."),
                io.String.Input("reference_intro", multiline=True, default="", optional=True,
                                tooltip="The line above the reference entries. Empty uses the built-in one."),
                io.Boolean.Input("name_entries", default=True, optional=True,
                                 tooltip="Tell the model the name of each shortcut or lore entry. Off: it gets only their text, so it has no name to write into the prompt."),
                io.String.Input("chat", multiline=True, default="", optional=True,
                                tooltip="Your comments on earlier rewrites (Composer > Chat)."),
                io.Combo.Input("style", options=list(en.STYLES), default="default", optional=True,
                               tooltip="default: the built-in instructions. dynamic: adds a concrete action, a change over time and secondary motion to every shot. Typed Instructions override both."),
                io.Combo.Input("scope", options=list(SCOPES), default="whole", optional=True,
                               tooltip="whole: the model rewrites the whole prompt. own: only the text between the global prefix and postfix is rewritten; they are kept as written."),
                io.Boolean.Input("last_frame", default=False, optional=True,
                                 tooltip="Shows the model the previous scene's last frame instead of the first reference. Only works while Continuity carries that frame."),
                io.Boolean.Input("to_chat", default=False, optional=True,
                                 tooltip="Sends every request as part of one running conversation (earlier requests and answers come along) instead of a one-time request."),
                io.String.Input("conversation", default="", optional=True,
                                tooltip="Which conversation the requests belong to (set by the app). Only used with Send everything into chat."),
                io.String.Input("scene", default="", optional=True,
                                tooltip="Which scene this request is for (set by the app). A new request for the same scene replaces that scene's earlier turn in the conversation."),
                io.Boolean.Input("compact", default=True, optional=True,
                                 tooltip="When a request nearly fills the context window, the conversation so far is summarised and the summary is sent instead of every turn."),
                io.Int.Input("context_tokens", default=0, min=0, max=1048576, optional=True,
                             tooltip="Summarising starts at 80% of this. 0 uses 32768, or the model's own size if smaller."),
                io.Boolean.Input("pass_rating", default=False, optional=True,
                                 tooltip="Adds one line with the rating of the last rated scene before this one to the end of the request."),
                io.Boolean.Input("comment_button", default=False, optional=True,
                                 tooltip="Shows a Comment button next to a scene's rating: a comment on the last rewrite, the same as one in the Chat tab (used by the app)."),
                io.String.Input("notes", multiline=True, default="", optional=True,
                                tooltip="Standing notes that apply to every rewrite, one per line (kept by the app: edited in the Chat tab)."),
                io.String.Input("rating_prefix", default="", optional=True,
                                tooltip="The words in front of the rating (set in the app). Empty uses the built-in line."),
                io.String.Input("rating_line", default="", optional=True,
                                tooltip="The rating line (set by the app). Only used with Pass the rating."),
                io.String.Input("head", default="", optional=True,
                                tooltip="The global prefix as it stands at the start of the prompt (set by the app). Only used with scope own."),
                io.String.Input("tail", default="", optional=True,
                                tooltip="The global postfix as it stands at the end of the prompt (set by the app). Only used with scope own."),
                io.String.Input("video_length", default="", optional=True,
                                tooltip="How long the clip being made is, as text such as '5.0 seconds (121 frames at 24 fps)' (set by the app). The VIDEO_LENGTH macro."),
                io.String.Input("aspect_ratio", default="", optional=True,
                                tooltip="The clip's target aspect ratio, as text such as '16:9 landscape' (set by the app). The ASPECT_RATIO macro."),
            ],
            outputs=[io.String.Output(display_name="enhanced"),
                     io.String.Output(display_name="status")],
        )

    @classmethod
    def fingerprint_inputs(cls, seed=0, enabled=False, **_):
        # Seed 0 means "a new rewrite every run", and ComfyUI would otherwise hand
        # back the cached one for identical inputs.
        return time.time() if enabled and not seed else ""

    @classmethod
    def execute(cls, clip, text, enabled=False, engine_model="", instructions="", max_length=0, greedy=False,
                temperature=0.7, top_p=0.92, top_k=50, min_p=0.05, repetition_penalty=1.3,
                presence_penalty=0.0, seed=0, thinking=False, thinking_comments=True, thinking_chat=True, thinking_helper=True, helper_max_length=0, encoder_file="", encoder_family="stable_diffusion", image=None, image_2=None, image_3=None, image_4=None, use_image=True, shortcuts="",
                lorebooks="", reference_intro="", name_entries=True, chat="", style="default", scope="whole", last_frame=False, pass_rating=False, comment_button=False, notes="", video_length="", aspect_ratio="", rating_prefix="", rating_line="", to_chat=False, conversation="", scene="", compact=True, context_tokens=0, head="", tail="",
                chat_prompt="", notes_prompt="", compact_prompt="", revision_prompt="", conversation_prompt="", turns_prompt="", marks_prompt="", fixed_prompt="", edit_prompt="", self_edit=False, refine_passes=0, refine_prompt="", precision="auto", prefix_cache=True, lookup_tokens=0, use_comfy=False) -> io.NodeOutput:
        if not enabled:
            # still the final prompt, and the Composer's Enhance tab shows what was encoded either way
            _log({"prompt_id": _prompt_id(), "status": "off: the prompt ran as typed", "before": text, "after": text,
                         "thinking": "", "sent": "", "ok": True})
            return io.NodeOutput(text, "off")
        prompts = {"chat_prompt": chat_prompt, "notes_prompt": notes_prompt, "compact_prompt": compact_prompt, "revision_prompt": revision_prompt,
                   "conversation_prompt": conversation_prompt, "turns_prompt": turns_prompt, "marks_prompt": marks_prompt, "fixed_prompt": fixed_prompt, "edit_prompt": edit_prompt, "refine_prompt": refine_prompt}
        text = str(text or "")
        pre = post = ""
        body = text
        if scope == "own":              # only what sits between the global prefix and postfix goes to the model
            pre, post = str(head or "").strip(), str(tail or "").strip()
            if (pre and not text.startswith(pre)) or (post and not text.endswith(post)) or len(pre) + len(post) > len(text):
                log.warning("FunUI Prompt enhancer", "the global prefix/postfix could not be found in the prompt; the original prompt is used")
                _log({"prompt_id": _prompt_id(), "status": "the prefix/postfix could not be found in the prompt; prompt unchanged",
                             "before": text, "after": text, "thinking": "", "sent": "", "ok": False})
                return io.NodeOutput(text, "prefix/postfix not found")
            body = text[len(pre):len(text) - len(post)].strip()
        engine_model = "" if use_comfy else (str(engine_model or "").strip() or (f"comfy:{encoder_file}" if encoder_file else ""))          # a file picked from ComfyUI's list runs on the engine
        if engine_model:
            try:
                ENGINE.load(engine_model, precision)
                clip = EngineClip(ENGINE, live="rewrite", cache=prefix_cache, lookup=lookup_tokens)
            except Exception as e:                                  # noqa: BLE001 -- the prompt must still go through
                if not encoder_file:
                    log.warning("FunUI Prompt enhancer", f"could not load the engine model {engine_model}: {e}; the original prompt is used")
                    _log({"prompt_id": _prompt_id(), "status": f"could not load the engine model {engine_model}: {e}", "before": text,
                          "after": text, "thinking": "", "sent": "", "ok": False})
                    return io.NodeOutput(text, "engine failed")
                engine_model = ""                                   # an older project's encoder: ComfyUI's own loader reads it
        if not engine_model and encoder_file:
            try:
                clip = _own_encoder(encoder_file, encoder_family)
            except Exception as e:                                  # noqa: BLE001 -- the prompt must still go through
                log.warning("FunUI Prompt enhancer", f"could not load {encoder_file}: {e}; the original prompt is used")
                _log({"prompt_id": _prompt_id(), "status": f"could not load the encoder {encoder_file}: {e}", "before": text,
                             "after": text, "thinking": "", "sent": "", "ok": False})
                return io.NodeOutput(text, "encoder failed")
        lines = lambda s: [x for x in str(s or "").splitlines() if x.strip()]    # noqa: E731
        groups = en.sources(lines(shortcuts), lines(lorebooks))
        try:
            rounds = json.loads(chat) if str(chat or "").strip() else []
        except ValueError:
            rounds = []
            log.warning("FunUI Prompt enhancer", "the Chat comments are not readable; ignored")
        if not isinstance(rounds, list):
            rounds = []
        key = str(conversation or "").strip() if to_chat else ""
        reference_text = en.reference(body, groups, reference_intro, names=name_entries is not False)
        masked, marks = mask_markup(body)
        system = str(instructions or "").strip() or en.STYLES.get(style, en.SYSTEM_PROMPT)
        if marks:
            system += "\n\n" + en.pick(prompts, "marks_prompt")
        if pre or post:               # the model never sees them in the text it rewrites, so it is told they are there
            system += "\n\n" + en.pick(prompts, "fixed_prompt")
        asked = "\n".join(x for x in (body, rating_line if pass_rating else "") if x)      # what the person wrote, with the rating line that went with it
        history, compacted, context, did = "", "", None, False
        state = CONVOS.get(key) if key else None
        scene = str(scene or "").strip()
        before_turns = None
        if state and scene:            # an edited or repeated prompt replaces this scene's earlier turn: the old version must not pull the new one back
            before_turns = list(state["turns"])
            state["turns"] = collections.deque((t for t in before_turns if t.get("scene") != scene), maxlen=en.CONVO_KEEP)
        # In a conversation every comment is part of it, in order: the ones it has not seen yet go in with this request (and so into
        # its turn), and the next request does not repeat them. Outside one, the comments are the Chat tab's labelled block.
        fresh = []
        if key:
            seen = state["seen"] if state else set()
            fresh = [(f"{i}:{r.get('id') or r['comment']}", str(r["comment"]).strip()) for i, r in enumerate(rounds)
                     if isinstance(r, dict) and str(r.get("comment") or "").strip() and f"{i}:{r.get('id') or r['comment']}" not in seen]
        macros = en.macro_values(state["turns"] if state else (), state["summary"] if state else "", [r.get("comment") for r in rounds if isinstance(r, dict)], notes, body, rating_line if pass_rating else "", prefix=pre, postfix=post, length=video_length, aspect=aspect_ratio)
        feedback = ("Feedback on your earlier rewrite, apply it:\n" + "\n".join("- " + c for _, c in fresh)) if fresh else ""
        note_text = "\n".join(x for x in (feedback, rating_line if pass_rating else "") if x)
        if key:
            keep = en.CONVO_KEEP if compact else en.CONVO_TURNS
            fixed = len(system) + len(masked) + len(reference_text) + len(note_text)
            used, window, compacted, did = fit(clip, state, fixed, max_length, compact, context_tokens, prompts, macros)
            if did:
                macros = en.macro_values(state["turns"], state["summary"], [r.get("comment") for r in rounds if isinstance(r, dict)], notes, body, rating_line if pass_rating else "", prefix=pre, postfix=post, length=video_length, aspect=aspect_ratio)
            if state:
                history = en.convo_block(state["turns"], state["summary"], keep)
            context = {"used": used, "window": window, "compact": bool(compact)}
        out, status, info = en.enhance(
            clip, masked, system=system,
            reference_text=reference_text, chat="" if key else en.chat_block(rounds), style=style,
            note=note_text, notes=notes, prompts=prompts, macros=macros, refine=refine_passes,
            history=history, convo={"summary": state["summary"], "turns": list(state["turns"]), "keep": keep} if state else None,
            seed=seed, image=_pictures(image, image_2, image_3, image_4) if use_image else None, thinking=(bool(thinking_comments) if fresh or any(isinstance(r, dict) and str(r.get("comment") or "").strip() for r in rounds) else bool(thinking)), max_length=max_length,
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
            do_sample=not greedy)
        _stop_if_interrupted()
        if info["ok"]:
            back = unmask_markup(out, marks)
            if back is None:
                info.update(ok=False, after=text)
                status = "the model changed the weights / [a|b] marks in the prompt; prompt unchanged"
            else:
                final = " ".join(x for x in (pre, back, post) if x)
                info.update(before=text, after=final, asked=asked, answered=back)      # asked/answered: what the model was given and said, without the global prefix and postfix
                out = final
        if not info["ok"] and before_turns is not None:
            state["turns"] = collections.deque(before_turns, maxlen=en.CONVO_KEEP)          # nothing was replaced: the earlier turn stays
        if not info["ok"]:
            # The prompt went through UNCHANGED and the person must know why.
            out = text
            info["before"] = info["after"] = text
            log.warning("FunUI Prompt enhancer", f"{status}; the original prompt is used")
        if key and info["ok"]:
            kept = CONVOS.setdefault(key, {"summary": "", "turns": collections.deque(maxlen=en.CONVO_KEEP), "seen": set()})
            turns = kept["turns"]
            kept["seen"].update(k for k, _ in fresh)
            CONVOS.move_to_end(key)
            turns.append({"request": "\n".join(x for x in (body, note_text) if x), "reply": info.get("answered") or out,
                          "prompt": body, "feedback": [c for _, c in fresh], "rating": rating_line if pass_rating else "", **({"scene": scene} if scene else {})})
            while len(CONVOS) > MAX_CONVOS:
                CONVOS.popitem(last=False)
        _log({"prompt_id": _prompt_id(), "status": status + compacted, "conversation": key, "context": context, "compacted": did, "summary": state["summary"] if (did and state) else "", **info})
        return io.NodeOutput(out, status)
