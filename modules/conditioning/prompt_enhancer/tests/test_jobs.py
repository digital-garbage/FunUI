import asyncio

from .. import jobs, rules


def test_what_a_request_wrote_is_kept_joined_and_replayed_to_a_page_that_attaches_late():
    job = jobs.Job("h:1", "do it")
    for ev in ({"kind": "thinking", "text": "a "}, {"kind": "thinking", "text": "b"}, {"kind": "answer", "text": "x"}, {"kind": "status", "text": "step"}):
        job.push(ev)
    loop = asyncio.new_event_loop()
    try:
        queue = asyncio.Queue()
        backlog = job.attach(loop, queue)
        assert backlog == [{"kind": "thinking", "text": "a b"}, {"kind": "answer", "text": "x"}, {"kind": "status", "text": "step"}]
        job.push({"kind": "answer", "text": "y"})                                  # what comes after is heard through the queue
        loop.run_until_complete(asyncio.sleep(0))
        assert queue.get_nowait() == {"kind": "answer", "text": "y"}
        job.finish({"ok": True, "reply": "done"})
        loop.run_until_complete(asyncio.sleep(0))
        assert queue.get_nowait()["kind"] == "result" and queue.get_nowait() is None and job.done
    finally:
        loop.close()


def test_a_question_still_open_is_shown_again_and_one_that_was_answered_is_not():
    job = jobs.Job("h:2", "do it")
    rules._DECISIONS["act-open"] = {"event": None, "allow": False}
    try:
        job.push({"kind": "action_request", "id": "act-open", "tool": "set_variable"})
        job.push({"kind": "action_request", "id": "act-gone", "tool": "set_variable"})
        loop = asyncio.new_event_loop()
        try:
            ids = [e["id"] for e in job.attach(loop, asyncio.Queue())]
        finally:
            loop.close()
        assert ids == ["act-open"]
    finally:
        rules._DECISIONS.pop("act-open", None)


def test_the_record_stays_small_and_a_finished_request_is_forgotten_once_its_end_was_seen():
    job = jobs.Job("h:3", "x")
    for _ in range(300):
        job.push({"kind": "thinking", "text": "t" * 1000})
        job.push({"kind": "status", "text": "s"})                                  # a different kind in between: the pieces are not joined
    assert sum(len(e.get("text", "")) for e in job.events) <= jobs.KEEP_CHARS
    jobs.JOBS.clear()
    started = jobs.start("h:3", "x")
    assert jobs.waiting("h:3") is started
    started.finish({"ok": True})
    assert jobs.waiting("h:3") is started            # nobody has been given the end yet
    started.seen = True
    assert jobs.waiting("h:3") is None
    jobs.JOBS.clear()


def test_a_helper_request_goes_on_when_the_page_goes_away_and_the_next_page_follows_it(monkeypatch):
    import asyncio as aio
    import json
    import threading

    import pytest
    web = pytest.importorskip("aiohttp.web")
    from aiohttp.test_utils import TestClient, TestServer
    import importlib
    mod = importlib.import_module(jobs.__package__)
    from .. import nodes
    jobs.JOBS.clear()
    release, wrote = threading.Event(), threading.Event()

    def fake(conversation, message, *, on_event=None, **kw):
        on_event({"kind": "thinking", "text": "first thought "})
        wrote.set()
        release.wait(10)
        on_event({"kind": "answer", "text": "late words"})
        return {"ok": True, "reply": "all done", "status": "ok"}
    monkeypatch.setattr(nodes, "chat_reply", fake)

    class Table:
        def __init__(self, app):
            self.app = app

        def get(self, path):
            return lambda fn: self.app.router.add_get(path, fn) and fn

        def post(self, path):
            return lambda fn: self.app.router.add_post(path, fn) and fn

    async def read(resp, stop_after=None):
        out = []
        async for raw in resp.content:
            line = raw.decode().strip()
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
                if stop_after and stop_after(out):
                    break
        return out

    async def scenario():
        app = web.Application()
        mod.routes(Table(app), "/m", web)
        async with TestClient(TestServer(app)) as client:
            first = await client.post("/m/chat_stream", json={"conversation": "h:live", "message": "go", "engine_model": "a/b::q.gguf", "helper": {"sheet": "x", "scenes": 1}})
            await read(first, lambda out: any(e.get("kind") == "thinking" for e in out))
            first.close()                                                    # the page is refreshed
            await aio.sleep(0.1)
            assert wrote.is_set() and not release.is_set()
            attached = await client.get("/m/chat_attach", params={"conversation": "h:live"})
            release.set()                                                    # the model goes on while the new page follows it
            got = await read(attached)
            kinds = [e["kind"] for e in got]
            assert kinds[0] == "begin" and got[0]["message"] == "go"
            assert "".join(e.get("text", "") for e in got if e["kind"] == "thinking") == "first thought "
            assert "late words" in "".join(e.get("text", "") for e in got if e["kind"] == "answer") and got[-1]["result"]["reply"] == "all done"
            again = await client.get("/m/chat_attach", params={"conversation": "h:live"})
            assert (await again.json()) == {"running": False}               # its end was seen: nothing more to follow
    aio.run(scenario())
    jobs.JOBS.clear()
