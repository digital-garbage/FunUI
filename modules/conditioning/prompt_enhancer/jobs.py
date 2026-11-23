"""A Helper request that outlives the page that asked it. The request runs on the server; what it writes is kept as it comes, so a page that was
refreshed (or opened later) can attach, see everything so far and follow the rest. A question still waiting for the person (Allow / Disallow) is
shown again on attaching; one already answered is not."""
import threading
import time

KEEP_CHARS = 200_000          # of text kept per request: the oldest text goes first
MAX_JOBS = 20

JOBS: dict = {}


class Job:
    def __init__(self, conversation, message):
        self.conversation, self.message, self.started = str(conversation), str(message), time.time()
        self.events, self.done, self.seen = [], False, False
        self.lock = threading.Lock()
        self.subs = []                      # (loop, asyncio.Queue) of the pages that are following it

    def push(self, ev):
        """One thing the request wrote or asked. Pieces of the same kind of text are joined, so the record stays small."""
        with self.lock:
            last = self.events[-1] if self.events else None
            if ev.get("kind") in ("thinking", "answer") and last and last.get("kind") == ev["kind"]:
                last["text"] = str(last.get("text", "")) + str(ev.get("text", ""))
            else:
                self.events.append(dict(ev))
            over = sum(len(str(e.get("text", ""))) for e in self.events) - KEEP_CHARS
            for e in [e for e in self.events if e.get("kind") in ("thinking", "answer")]:        # the oldest writing goes first
                if over <= 0:
                    break
                over -= len(e.get("text", ""))
                self.events.remove(e)
            subs = list(self.subs)
        for loop, queue in subs:
            loop.call_soon_threadsafe(queue.put_nowait, ev)

    def finish(self, result):
        self.push({"kind": "result", "result": result})
        with self.lock:
            self.done, subs = True, list(self.subs)
        for loop, queue in subs:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    def attach(self, loop, queue):
        """-> what was written so far, with the questions still open (the page then hears the rest through `queue`)."""
        from . import rules
        with self.lock:
            backlog = [dict(e) for e in self.events if e.get("kind") not in ("action_request", "edit_request") or e.get("id") in rules._DECISIONS]
            if not self.done:
                self.subs.append((loop, queue))
        return backlog

    def detach(self, loop, queue):
        with self.lock:
            self.subs = [s for s in self.subs if s[1] is not queue]


def start(conversation, message) -> Job:
    job = Job(conversation, message)
    for key in [k for k, j in JOBS.items() if j.done and j.seen][:max(0, len(JOBS) - MAX_JOBS + 1)]:
        JOBS.pop(key, None)
    JOBS[str(conversation)] = job
    return job


def waiting(conversation):
    """The request of a conversation that a page has not yet been given the end of, or None."""
    job = JOBS.get(str(conversation or "").strip())
    return job if job is not None and not (job.done and job.seen) else None
