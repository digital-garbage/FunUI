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
                                  "styles": enhance.STYLES})

    @table.get(base + "/runs")
    async def _runs(req):
        """The last runs' rewrites, newest last. `?prompt_id=` narrows to one run."""
        want = req.rel_url.query.get("prompt_id")
        runs = [r for r in RUNS if not want or r["prompt_id"] == want]
        return web.json_response({"runs": runs, "now": int(time.time() * 1000)})        # `now`: the app sets its own comments by this clock

    @table.post(base + "/chat")
    async def _chat(req):
        """A live answer in a conversation (the Chat tab): the encoder picked in Enhance answers under its own chat prompt."""
        import asyncio
        from . import nodes
        try:
            body = await req.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not str(body.get("message") or "").strip() or not str(body.get("conversation") or "").strip():
            return web.json_response({"ok": False, "reply": "", "status": "a conversation and a message are needed"}, status=400)
        try:
            from server import PromptServer
            if PromptServer.instance.prompt_queue.get_tasks_remaining() > 0:
                return web.json_response({"ok": False, "reply": "", "status": "a generation is running: the encoder shares its memory, so the reply waits for it"}, status=409)
        except Exception:                                           # noqa: BLE001 -- not inside ComfyUI: nothing to wait for
            pass
        keys = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
        sampling = {k: body[k] for k in keys if isinstance(body.get(k), (int, float)) and not isinstance(body.get(k), bool)}
        sampling["do_sample"] = body.get("greedy") is not True
        if isinstance(body.get("seed"), int) and not isinstance(body.get("seed"), bool):
            sampling["seed"] = body["seed"]

        def work():
            return nodes.chat_reply(body["conversation"], body["message"], notes=str(body.get("notes") or ""), encoder_file=str(body.get("encoder_file") or ""),
                                    encoder_family=str(body.get("encoder_family") or "stable_diffusion"), compact=body.get("compact") is not False,
                                    context_tokens=int(body.get("context_tokens") or 0), max_length=int(body.get("max_length") or 400),
                                    thinking=body.get("thinking") is True, **sampling)
        result = await asyncio.get_running_loop().run_in_executor(None, work)
        return web.json_response(result, status=200 if result["ok"] else 422)

    @table.get(base + "/encoders")
    async def _encoders(_req):
        """Encoder files and families the Enhance panel can offer (read fresh: files come and go)."""
        try:
            from ...loaders.clip import nodes as clip_nodes
            return web.json_response({"files": clip_nodes.encoder_files(), "families": clip_nodes.clip_types()})
        except Exception as exc:                                    # noqa: BLE001 -- the CLIP loader module is another module; its absence empties the picker, not the panel
            return web.json_response({"files": [], "families": [], "why": f"the CLIP loader module is not available: {exc}"})


PROVIDES = {"routes": routes}
