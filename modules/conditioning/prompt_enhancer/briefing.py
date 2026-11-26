"""Briefing: before a rewrite, the model reads everything it was given and may ask the person one question; the answer goes into the request.

The run waits here while the page shows the question (a small window with an input, Cancel and Send). Cancel, a timeout or a page that is gone
all mean "no answer": the rewrite goes on without it."""
import itertools
import threading
import time

WAIT = 900                                                  # seconds the run waits for the person

_IDS = itertools.count(1)
_SLOTS = {}

QUESTION_SYSTEM = (
    "You get everything a writer was given to rewrite one prompt for a video model: the instructions, the reference material, the notes and the prompt itself. "
    "Do not rewrite anything. Decide whether ONE short question to the person would clearly improve the rewrite, for example a detail that is missing "
    "or two readings that give very different results. If so, write only that question, in the person's language, in one or two sentences. "
    "If the material is enough, write only the word NONE.")


KEEP = 1800                                                 # seconds an answer is remembered outside a conversation: the next runs do not ask it again
_ASKED = []                                                 # [(when, question, answer)], newest last
_BOOK = {}                                                  # conversation -> [(question, answer)]: kept as long as the conversation is


def asked(key="") -> list:
    """What was asked lately (in this conversation, for the whole of it), and what was answered ("" for no answer): [(question, answer)]."""
    if key:
        return list(_BOOK.get(key, ()))
    now = time.time()
    _ASKED[:] = [x for x in _ASKED if now - x[0] < KEEP][-20:]
    return [(q, a) for _, q, a in _ASKED]


def same(a, b) -> bool:
    """Whether two questions are the same one, word for word without caring for case or punctuation."""
    norm = lambda t: " ".join("".join(ch if ch.isalnum() else " " for ch in str(t).lower()).split())      # noqa: E731
    return norm(a) == norm(b)


def remember(question, reply, key=""):
    if key:
        _BOOK.setdefault(key, []).append((question, reply))
        del _BOOK[key][:-30]
        while len(_BOOK) > 20:                              # the longest unused conversations go first
            _BOOK.pop(next(iter(_BOOK)))
    else:
        _ASKED.append((time.time(), question, reply))


def forget(key):
    """A conversation that was reset takes its questions with it."""
    return _BOOK.pop(key, None) is not None


def clean(text) -> str:
    """The question a model wrote -> the question, or "" when it asked nothing."""
    out = " ".join(str(text or "").replace("<think>", " ").split()).strip().strip('"')
    if not out or out.upper().rstrip(".!") in ("NONE", "NO QUESTION", "NO QUESTIONS"):
        return ""
    return out[:600]


def answer(slot_id, text) -> bool:
    """The page's answer: a text, or None / empty for Cancel. -> whether a question was waiting for it."""
    slot = _SLOTS.get(str(slot_id))
    if slot is None:
        return False
    slot["text"] = str(text).strip() if text else ""
    slot["event"].set()
    return True


def ask(question, prompt_id, send, check=lambda: None, wait=WAIT) -> str:
    """Show `question` on the page that queued this run and wait for the answer. -> the answer, "" when there is none.
    `send(event, data)` reaches the page; `check()` raises when the run was stopped."""
    slot_id = str(next(_IDS))
    slot = _SLOTS[slot_id] = {"event": threading.Event(), "text": ""}
    try:
        send("funui.briefing", {"prompt_id": prompt_id, "id": slot_id, "question": question})
        waited = 0.0
        while not slot["event"].wait(0.5):
            check()
            waited += 0.5
            if waited >= wait:
                break
        return slot["text"]
    finally:
        _SLOTS.pop(slot_id, None)
        try:
            send("funui.briefing_done", {"prompt_id": prompt_id, "id": slot_id})
        except Exception:                                   # noqa: BLE001 -- a closed page must not fail the run
            pass
