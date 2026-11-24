"""Prompt enhancer: a language model rewrites the finished prompt before it is
encoded. A node-only module, with a small readout route for the Composer.

Sits after shortcut expansion and `$variables` (the app resolves both before the
text reaches the pipeline), so the model is handed exactly the text that would
otherwise have been encoded. Off by default. Every failure hands the ORIGINAL
prompt on and says so in the log: a rewrite that did not happen must never empty
the prompt.

One generation per scene -- v5 runs a scene per generation, so a multi-scene
project is never rewritten as one blob and re-split.

Whether a given text encoder writes sensible prose is an open question per model
(H3's conditioning checkpoint has no trained decoder head; Gemma3/Qwen3 do). v4's
result: confirmed working on Gemma3-12B. v5 on a rental: works with Gemma 4 E4B (a real rewrite in 2.5 s); H3's own encoder writes no prose, so there the run keeps your prompt and says why. On H3, pick a chat encoder (Gemma 4) in Composer > Enhance > Encoder: verified end to end.
"""

import time
from .nodes import RUNS, FunUIEnhancePrompt

ID = "conditioning_prompt_enhancer"
TITLE = "Prompt enhancer"
STAGE = "conditioning"
CATEGORY = "conditioning"
STATUS = "experimental"

NODES = [FunUIEnhancePrompt]


def routes(table, base, web):
    @table.get(base + "/defaults")
    async def _defaults(_req):
        """What an empty field means, so the Composer can show it."""
        from . import enhance
        return web.json_response({"instructions": enhance.SYSTEM_PROMPT,
                                  "reference_intro": enhance.REFERENCE_INTRO,
                                  "styles": enhance.STYLES, "prompts": enhance.defaults(), "macros": enhance.MACROS})

    @table.get(base + "/runs")
    async def _runs(req):
        """The last runs' rewrites, newest last. `?prompt_id=` narrows to one run."""
        want = req.rel_url.query.get("prompt_id")
        runs = [r for r in RUNS if not want or r["prompt_id"] == want]
        return web.json_response({"runs": runs, "now": int(time.time() * 1000)})        # `now`: the app sets its own comments by this clock

    def speed_of(body):
        """The speed settings a request carries: precision, prefix cache, prompt-lookup count."""
        try:
            look = max(0, min(10, int(body.get("lookup_tokens", 0))))
        except (TypeError, ValueError):
            look = 0
        return dict(precision=str(body.get("precision") or "auto"), prefix_cache=body.get("prefix_cache") is not False, lookup_tokens=look)

    def chat_args(body):
        """The request body of a chat message -> chat_reply's arguments."""
        from . import enhance
        keys = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
        sampling = {k: body[k] for k in keys if isinstance(body.get(k), (int, float)) and not isinstance(body.get(k), bool)}
        sampling["do_sample"] = body.get("greedy") is not True
        if isinstance(body.get("seed"), int) and not isinstance(body.get("seed"), bool):
            sampling["seed"] = body["seed"]
        rules = {k: str(v) for k, v in (body.get("rules") or {}).items() if isinstance(v, str)} if isinstance(body.get("rules"), dict) else {}
        budget = body.get("thinking_budget")
        return dict(thinking_budget=budget if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0 else 0, rules=rules, self_edit=body.get("self_edit") is True, notes=str(body.get("notes") or ""), video_length=str(body.get("video_length") or "")[:200], aspect_ratio=str(body.get("aspect_ratio") or "")[:200], encoder_file=str(body.get("encoder_file") or ""), engine_model=str(body.get("engine_model") or "").strip(), speed=speed_of(body), use_comfy=body.get("use_comfy") is True,
                    encoder_family=str(body.get("encoder_family") or "stable_diffusion"), prompts={k: body.get(k) for k in enhance.PROMPT_FIELDS},
                    comments=[str(x) for x in (body.get("comments") or []) if isinstance(x, str)], compact=body.get("compact") is not False,
                    helper=({"sheet": str((body.get("helper") or {}).get("sheet") or "")[:12000], "scenes": int((body.get("helper") or {}).get("scenes") or 0), "actions": [x for x in ((body.get("helper") or {}).get("actions") or [])[:100] if isinstance(x, dict)], "chat": str((body.get("helper") or {}).get("chat") or "")[:200], "rules": {str(k): str(v)[:8000] for k, v in list((((body.get("helper") or {}).get("rules")) or {}).items())[:40] if isinstance(v, str)} if isinstance((body.get("helper") or {}).get("rules"), dict) else {}, "inputs": [x for x in ((body.get("helper") or {}).get("inputs") or [])[:4000] if isinstance(x, dict)], "presets": [x for x in ((body.get("helper") or {}).get("presets") or [])[:100] if isinstance(x, dict)], "variables": [str(x)[:40] for x in ((body.get("helper") or {}).get("variables") or [])[:300]], "shortcuts": [x for x in ((body.get("helper") or {}).get("shortcuts") or [])[:3000] if isinstance(x, dict)], "settings": [x for x in ((body.get("helper") or {}).get("settings") or [])[:3000] if isinstance(x, dict)]} if isinstance(body.get("helper"), dict) else None), context_tokens=int(body.get("context_tokens") or 0), max_length=int(body.get("max_length") or 0), thinking=(body.get("thinking_helper") if isinstance(body.get("helper"), dict) else body.get("thinking_chat")) is not False, **sampling)

    async def chat_body(req):
        try:
            body = await req.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not str(body.get("message") or "").strip() or not str(body.get("conversation") or "").strip():
            return None
        return body

    @table.post(base + "/chat")
    async def _chat(req):
        """A live answer in a conversation (the Chat tab): the encoder picked in Enhance answers under its own chat prompt."""
        import asyncio
        from . import nodes
        body = await chat_body(req)
        if body is None:
            return web.json_response({"ok": False, "reply": "", "status": "a conversation and a message are needed"}, status=400)
        args = chat_args(body)
        if not args["engine_model"]:                     # ComfyUI's encoder shares its memory with a generation; the engine has its own
            try:
                from server import PromptServer
                if PromptServer.instance.prompt_queue.get_tasks_remaining() > 0:
                    return web.json_response({"ok": False, "reply": "", "status": "a generation is running: the encoder shares its memory, so the reply waits for it"}, status=409)
            except Exception:                                       # noqa: BLE001 -- not inside ComfyUI: nothing to wait for
                pass
        result = await asyncio.get_running_loop().run_in_executor(None, lambda: nodes.chat_reply(body["conversation"], body["message"], **args))
        return web.json_response(result, status=200 if result["ok"] else 422)

    @table.post(base + "/chat_stream")
    async def _chat_stream(req):
        """The same answer, sent as it forms: server-sent events {"kind": status | thinking | answer | call | result}. Needs an engine model."""
        import asyncio
        import json
        import threading
        from . import engine as eng, nodes
        body = await chat_body(req)
        if body is None or body.get("use_comfy") is True or not str(body.get("engine_model") or "").strip():
            return web.json_response({"ok": False, "reply": "", "status": "a conversation, a message and an engine model are needed"}, status=400)
        args = chat_args(body)
        from . import jobs
        loop, queue = asyncio.get_running_loop(), asyncio.Queue()
        job = jobs.start(body["conversation"], body["message"]) if isinstance(body.get("helper"), dict) else None     # a helper request outlives the page that asked it
        if job is not None:
            job.attach(loop, queue)
            push = job.push
        else:
            push = lambda ev: loop.call_soon_threadsafe(queue.put_nowait, ev)       # noqa: E731

        def work():
            try:
                done = threading.Event()
                if eng.ENGINE.model is None or eng.ENGINE.source != args["engine_model"] or eng.ENGINE.precision != args["speed"]["precision"]:
                    push({"kind": "status", "text": f"loading {args['engine_model']} (the first time downloads it)"})

                    def tick():             # the page sees the download and the loading move, not one line that looks stuck
                        while not done.wait(2):
                            st = eng.ENGINE.status()
                            if st["phase"] == "downloading":
                                push({"kind": "status", "text": f"downloading: {st['downloaded'] / 2 ** 30:.1f}" + (f" of {st['total'] / 2 ** 30:.1f}" if st["total"] else "") + " GB"})
                            elif st["phase"] == "loading":
                                push({"kind": "status", "text": st["message"]})
                    threading.Thread(target=tick, daemon=True, name="funui-engine-status").start()
                try:
                    result = nodes.chat_reply(body["conversation"], body["message"], on_event=push, **args)
                finally:
                    done.set()
            except Exception as exc:                                # noqa: BLE001 -- the stream must end with a result
                result = {"ok": False, "reply": "", "status": str(exc)}
            if job is not None:
                job.finish(result)
            else:
                push({"kind": "result", "result": result})
                push(None)
        loop.run_in_executor(None, work)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        await resp.prepare(req)
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                await resp.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                if job is not None and ev.get("kind") == "result":
                    job.seen = True
        except (ConnectionResetError, asyncio.CancelledError) as gone:
            if job is not None:
                job.detach(loop, queue)                              # the helper goes on without the page; the next page attaches to it
                if isinstance(gone, ConnectionResetError):
                    return resp                                      # not an error: the page left
                raise
            eng.ENGINE.cancel.set()                                  # the page went away: stop writing
            raise
        await resp.write_eof()
        return resp

    @table.get(base + "/chat_attach")
    async def _chat_attach(req):
        """Follow a helper request that is still running (or finished while no page was looking): `?conversation=`. Server-sent events, the same as chat_stream,
        starting with {"kind": "begin", "message", "seconds"} and everything written so far. {"running": false} when there is nothing to follow."""
        import asyncio
        import json
        from . import jobs
        job = jobs.waiting(req.rel_url.query.get("conversation"))
        if job is None:
            return web.json_response({"running": False})
        loop, queue = asyncio.get_running_loop(), asyncio.Queue()
        backlog = job.attach(loop, queue)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        await resp.prepare(req)
        send = lambda ev: resp.write(b"data: " + json.dumps(ev).encode() + b"\n\n")      # noqa: E731
        try:
            await send({"kind": "begin", "message": job.message, "seconds": int(time.time() - job.started)})
            for ev in backlog:
                await send(ev)
                if ev.get("kind") == "result":
                    job.seen = True
            while not job.done or not queue.empty():
                try:
                    ev = await asyncio.wait_for(queue.get(), 30)
                except asyncio.TimeoutError:
                    await resp.write(b": waiting\n\n")
                    continue
                if ev is None:
                    break
                await send(ev)
                if ev.get("kind") == "result":
                    job.seen = True
        except (ConnectionResetError, asyncio.CancelledError) as gone:
            job.detach(loop, queue)
            if isinstance(gone, ConnectionResetError):
                return resp
            raise
        job.detach(loop, queue)
        await resp.write_eof()
        return resp

    @table.post(base + "/chat_stop")
    async def _chat_stop(req):
        """Stop what the helper is doing for a conversation: {"conversation"}. The stream of that request ends with a "stopped" result."""
        from . import nodes
        try:
            body = await req.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not str(body.get("conversation") or "").strip():
            return web.json_response({"ok": False}, status=400)
        return web.json_response({"ok": nodes.stop_chat(body["conversation"])})

    @table.post(base + "/chat_clear")
    async def _chat_clear(req):
        """Forget a conversation on the server (its turns, its job, its running request): {"conversation"}."""
        from . import nodes
        try:
            body = await req.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not str(body.get("conversation") or "").strip():
            return web.json_response({"ok": False}, status=400)
        return web.json_response({"ok": True, "had": nodes.clear_chat(body["conversation"])})

    @table.post(base + "/edit_decision")
    async def _edit_decision(req):
        """The person's answer to a rule change the model proposed: {"id", "allow"}."""
        from . import rules
        try:
            body = await req.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not body.get("id"):
            return web.json_response({"ok": False}, status=400)
        return web.json_response({"ok": rules.decide(body["id"], body.get("allow") is True)})

    @table.get(base + "/engine")
    async def _engine_status(_req):
        """Where the engine model is: idle, downloading (with bytes), loading, ready or error."""
        from . import engine as eng
        return web.json_response(eng.ENGINE.status())

    @table.post(base + "/engine/load")
    async def _engine_load(req):
        """Download (once) and load the engine model in the background: {"source": "owner/name" or a folder}."""
        from . import engine as eng
        try:
            body = await req.json()
        except ValueError:
            body = None
        source = str((body or {}).get("source") or "").strip() if isinstance(body, dict) else ""
        if not source:
            return web.json_response({"ok": False, "status": "an engine model is needed"}, status=400)
        started = eng.ENGINE.load_async(source, str((body or {}).get("precision") or "auto"))
        return web.json_response({"ok": True, "started": started, **eng.ENGINE.status()})

    @table.post(base + "/engine/download")
    async def _engine_download(req):
        """Download a model's files only, without loading it into memory: {"source": "owner/name::file.gguf|..."}."""
        from . import engine as eng
        try:
            body = await req.json()
        except ValueError:
            body = None
        source = str((body or {}).get("source") or "").strip() if isinstance(body, dict) else ""
        if not source:
            return web.json_response({"ok": False, "status": "an engine model is needed"}, status=400)
        started = eng.ENGINE.download_async(source)
        return web.json_response({"ok": True, "started": started, **eng.ENGINE.status()})

    @table.get(base + "/engine/files")
    async def _engine_files(req):
        """The weight files of a Hugging Face repo with their sizes, so one of them can be picked: ?repo=owner/name."""
        import asyncio
        import re
        repo = req.rel_url.query.get("repo", "").strip()
        if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo):
            return web.json_response({"files": [], "why": "write the repo as owner/name"})

        def look():
            from huggingface_hub import HfApi
            info = HfApi().model_info(repo, files_metadata=True)
            return [{"name": s.rfilename, "size": s.size or 0} for s in info.siblings if s.rfilename.endswith((".safetensors", ".gguf"))]
        try:
            return web.json_response({"files": await asyncio.get_running_loop().run_in_executor(None, look), "why": ""})
        except Exception as exc:                                    # noqa: BLE001 -- said in the panel
            return web.json_response({"files": [], "why": f"could not read {repo}: {exc}"})

    @table.get(base + "/engine/live")
    async def _engine_live(_req):
        """What the engine is writing now (or wrote last): {id, kind, label, thinking, answer, status, done}. The tabs poll it to show a rewrite as it forms."""
        from . import engine as eng
        return web.json_response(eng.LIVE.read())

    @table.get(base + "/engine/models")
    async def _engine_models(_req):
        """The models the engine downloaded, with their sizes."""
        import asyncio
        from . import engine as eng
        return web.json_response({"models": await asyncio.get_running_loop().run_in_executor(None, eng.ENGINE.downloads)})

    @table.post(base + "/engine/delete")
    async def _engine_delete(req):
        """Remove one downloaded model: {"name": folder name from /engine/models}."""
        import asyncio
        from . import engine as eng
        try:
            body = await req.json()
        except ValueError:
            body = None
        try:
            await asyncio.get_running_loop().run_in_executor(None, eng.ENGINE.delete, str((body or {}).get("name") or "") if isinstance(body, dict) else "")
        except (ValueError, OSError) as exc:
            return web.json_response({"ok": False, "status": str(exc)}, status=400)
        return web.json_response({"ok": True})

    @table.post(base + "/engine/unload")
    async def _engine_unload(_req):
        from . import engine as eng
        eng.ENGINE.unload()
        return web.json_response(eng.ENGINE.status())

    @table.get(base + "/encoders")
    async def _encoders(_req):
        """Encoder files and families the Enhance panel can offer (read fresh: files come and go)."""
        try:
            from ...loaders.clip import nodes as clip_nodes
            return web.json_response({"files": clip_nodes.encoder_files(), "families": clip_nodes.clip_types()})
        except Exception as exc:                                    # noqa: BLE001 -- the CLIP loader module is another module; its absence empties the picker, not the panel
            return web.json_response({"files": [], "families": [], "why": f"the CLIP loader module is not available: {exc}"})


PROVIDES = {"routes": routes}
