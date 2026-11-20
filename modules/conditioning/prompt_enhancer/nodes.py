"""The Enhance Prompt node: an LLM rewrites the finished prompt before it is encoded."""

import collections
import json
import re
import time

from comfy_api.latest import io

from ..._core import log
from . import enhance as en

#: The last runs' rewrites, newest last: what the Composer's readout shows. In
#: memory on purpose -- it is the last run's answer, not state to keep.
RUNS = collections.deque(maxlen=20)
#: "Send everything into chat": the earlier turns {request, reply} of each conversation, in memory only (a restart starts them over).
CONVOS = collections.OrderedDict()
MAX_CONVOS = 20


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


def fit(clip, state, fixed_chars, max_length, compact, context_tokens):
    """-> (about how many tokens the next request fills, the context window, a status note, whether it was summarised now).
    The conversation is summarised first when the request would fill COMPACT_AT of the window (4 characters a token, the answer's room included)."""
    # A small local model slows down and loses focus on a very long context, so what it reports is only a ceiling: summarising
    # works to DEFAULT_WINDOW unless the person sets a size.
    window = int(context_tokens or 0) or min(en.context_window(clip) or en.DEFAULT_WINDOW, en.DEFAULT_WINDOW)
    keep = en.CONVO_KEEP if compact else en.CONVO_TURNS
    size = lambda: (fixed_chars + len(en.convo_block(state["turns"] if state else (), state["summary"] if state else "", keep))) // en.CHARS_PER_TOKEN + int(max_length)    # noqa: E731
    note, did = "", False
    if state and compact and state["turns"] and size() > en.COMPACT_AT * window:
        summary = en.compact(clip, state["summary"], list(state["turns"]))
        if summary:
            state["summary"] = summary
            state["turns"].clear()
            note, did = "; the earlier conversation was summarised to fit the context window", True
        else:
            note = "; the conversation is near the context window and could not be summarised"
    return size(), window, note, did


def chat_reply(conversation, message, *, notes="", encoder_file, encoder_family="stable_diffusion", compact=True, context_tokens=0, max_length=400, thinking=False, **sampling):
    """One live answer in a conversation, from the encoder the person picked. -> {"ok", "reply", "status"}. The exchange joins the conversation."""
    key = str(conversation or "").strip()
    if not key:
        return {"ok": False, "reply": "", "status": "no conversation"}
    if not encoder_file:
        return {"ok": False, "reply": "", "status": "pick an Encoder in Composer > Enhance: the model's own encoder cannot chat"}
    try:
        clip = _own_encoder(encoder_file, encoder_family)
    except Exception as e:                                          # noqa: BLE001
        return {"ok": False, "reply": "", "status": f"could not load the encoder {encoder_file}: {e}"}
    state = CONVOS.setdefault(key, {"summary": "", "turns": collections.deque(maxlen=en.CONVO_KEEP), "seen": set()})
    CONVOS.move_to_end(key)
    used, window, note, did = fit(clip, state, len(en.CHAT_REPLY_SYSTEM) + len(str(message)), max_length, compact, context_tokens)
    keep = en.CONVO_KEEP if compact else en.CONVO_TURNS
    reply, status, add, drop = en.converse(clip, message, convo={"summary": state["summary"], "turns": list(state["turns"]), "keep": keep}, notes=notes,
                                           max_length=max_length, thinking=thinking, **sampling)
    ok = bool(reply)
    if ok:
        state["turns"].append({"request": str(message).strip(), "reply": reply, "kind": "chat"})
    RUNS.append({"prompt_id": f"chat-{time.time_ns()}", "kind": "chat", "conversation": key, "ok": ok, "status": status + note, "before": str(message), "after": reply,
                 "asked": str(message).strip(), "answered": reply, "remember": add, "forget": drop, "compacted": did, "summary": state["summary"] if did else "", "thinking": "", "sent": "", "context": {"used": used, "window": window, "compact": bool(compact)}})
    while len(CONVOS) > MAX_CONVOS:
        CONVOS.popitem(last=False)
    return {"ok": ok, "reply": reply, "status": status + note, "remember": add, "forget": drop, "compacted": did}


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
                io.String.Input("instructions", multiline=True, default="", optional=True,
                                tooltip="What the model is told to do. Empty uses the built-in instructions."),
                io.Int.Input("max_length", default=400, min=32, max=4096, step=8, optional=True,
                             tooltip="The longest rewrite, in tokens."),
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
                # Text, not a list: a file that has gone must not stop a run where Enhance is off. The Enhance panel offers the files.
                io.String.Input("encoder_file", default="", optional=True,
                                tooltip="A text encoder file just for rewriting (Gemma, Qwen...). Empty uses the model's own encoder."),
                io.String.Input("encoder_family", default="stable_diffusion", optional=True,
                                tooltip="Which family that file is for. Only used with an encoder file."),
                io.Image.Input("image", optional=True, tooltip="Shown to the model alongside the prompt."),
                io.Boolean.Input("use_image", default=True, optional=True,
                                 tooltip="Send the picture (when one is connected) to the model."),
                io.String.Input("shortcuts", multiline=True, default="", optional=True,
                                tooltip="Shortcut names to hand the model as reference, one per line."),
                io.String.Input("lorebooks", multiline=True, default="", optional=True,
                                tooltip="Paths to lorebook or shortcuts JSON files, one per line."),
                io.String.Input("reference_intro", multiline=True, default="", optional=True,
                                tooltip="The line above the reference entries. Empty uses the built-in one."),
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
    def execute(cls, clip, text, enabled=False, instructions="", max_length=400, greedy=False,
                temperature=0.7, top_p=0.92, top_k=50, min_p=0.05, repetition_penalty=1.3,
                presence_penalty=0.0, seed=0, thinking=False, encoder_file="", encoder_family="stable_diffusion", image=None, use_image=True, shortcuts="",
                lorebooks="", reference_intro="", chat="", style="default", scope="whole", last_frame=False, pass_rating=False, comment_button=False, notes="", rating_prefix="", rating_line="", to_chat=False, conversation="", compact=True, context_tokens=0, head="", tail="") -> io.NodeOutput:
        if not enabled:
            # still the final prompt, and the Composer's Enhance tab shows what was encoded either way
            RUNS.append({"prompt_id": _prompt_id(), "status": "off: the prompt ran as typed", "before": text, "after": text,
                         "thinking": "", "sent": "", "ok": True})
            return io.NodeOutput(text, "off")
        text = str(text or "")
        pre = post = ""
        body = text
        if scope == "own":              # only what sits between the global prefix and postfix goes to the model
            pre, post = str(head or "").strip(), str(tail or "").strip()
            if (pre and not text.startswith(pre)) or (post and not text.endswith(post)) or len(pre) + len(post) > len(text):
                log.warning("FunUI Prompt enhancer", "the global prefix/postfix could not be found in the prompt; the original prompt is used")
                RUNS.append({"prompt_id": _prompt_id(), "status": "the prefix/postfix could not be found in the prompt; prompt unchanged",
                             "before": text, "after": text, "thinking": "", "sent": "", "ok": False})
                return io.NodeOutput(text, "prefix/postfix not found")
            body = text[len(pre):len(text) - len(post)].strip()
        if encoder_file:
            try:
                clip = _own_encoder(encoder_file, encoder_family)
            except Exception as e:                                  # noqa: BLE001 -- the prompt must still go through
                log.warning("FunUI Prompt enhancer", f"could not load {encoder_file}: {e}; the original prompt is used")
                RUNS.append({"prompt_id": _prompt_id(), "status": f"could not load the encoder {encoder_file}: {e}", "before": text,
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
        reference_text = en.reference(body, groups, reference_intro)
        masked, marks = mask_markup(body)
        system = str(instructions or "").strip() or en.STYLES.get(style, en.SYSTEM_PROMPT)
        if marks:
            system += "\n\n" + en.MARKS_RULE
        asked = "\n".join(x for x in (body, rating_line if pass_rating else "") if x)      # what the person wrote, with the rating line that went with it
        history, compacted, context, did = "", "", None, False
        state = CONVOS.get(key) if key else None
        # In a conversation every comment is part of it, in order: the ones it has not seen yet go in with this request (and so into
        # its turn), and the next request does not repeat them. Outside one, the comments are the Chat tab's labelled block.
        fresh = []
        if key:
            seen = state["seen"] if state else set()
            fresh = [(f"{i}:{r.get('id') or r['comment']}", str(r["comment"]).strip()) for i, r in enumerate(rounds)
                     if isinstance(r, dict) and str(r.get("comment") or "").strip() and f"{i}:{r.get('id') or r['comment']}" not in seen]
        feedback = ("Feedback on your earlier rewrite, apply it:\n" + "\n".join("- " + c for _, c in fresh)) if fresh else ""
        note_text = "\n".join(x for x in (feedback, rating_line if pass_rating else "") if x)
        if key:
            keep = en.CONVO_KEEP if compact else en.CONVO_TURNS
            fixed = len(system) + len(masked) + len(reference_text) + len(note_text)
            used, window, compacted, did = fit(clip, state, fixed, max_length, compact, context_tokens)
            if state:
                history = en.convo_block(state["turns"], state["summary"], keep)
            context = {"used": used, "window": window, "compact": bool(compact)}
        out, status, info = en.enhance(
            clip, masked, system=system,
            reference_text=reference_text, chat="" if key else en.chat_block(rounds), style=style,
            note=note_text, notes=notes,
            history=history, convo={"summary": state["summary"], "turns": list(state["turns"]), "keep": keep} if state else None,
            seed=seed, image=image if use_image else None, thinking=thinking, max_length=max_length,
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
            do_sample=not greedy)
        if info["ok"]:
            back = unmask_markup(out, marks)
            if back is None:
                info.update(ok=False, after=text)
                status = "the model changed the weights / [a|b] marks in the prompt; prompt unchanged"
            else:
                final = " ".join(x for x in (pre, back, post) if x)
                info.update(before=text, after=final, asked=asked, answered=back)      # asked/answered: what the model was given and said, without the global prefix and postfix
                out = final
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
            turns.append({"request": "\n".join(x for x in (body, note_text) if x), "reply": info.get("answered") or out})
            while len(CONVOS) > MAX_CONVOS:
                CONVOS.popitem(last=False)
        RUNS.append({"prompt_id": _prompt_id(), "status": status + compacted, "conversation": key, "context": context, "compacted": did, "summary": state["summary"] if (did and state) else "", **info})
        return io.NodeOutput(out, status)
