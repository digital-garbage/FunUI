"""Block map: where in the model, and how hard, a per-block feature should act.

A profile run (switched on from Settings > Refinement & Taste, off by default) nudges one group of
blocks at a time and measures how far the picture guess moves and how much of that move is fine-detail
noise (see measure.py). It costs extra model calls on three steps of one run, then writes a map for this
model. Features that act on chosen blocks (Attention explorer, REINS, ...) read the map through the
model's transformer_options and scale their strength block by block, around 1 on average, so a strength
setting keeps its meaning. With no map, or a flat one, every multiplier is 1 and nothing changes.
"""

from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, streams
from . import measure

ID = "block_map"
TITLE = "Block map"
MOUNT = "generation.sampling"
STAGE = "post"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["dit_block_hooks"]
# Innermost, like the other measurement modules: only real forwards are nudged.
AFTER = ["video_detail", "late_branch", "score_slider", "sharpen"]

PUBLISHED = "funui_block_map"

SETTINGS = {
    "follow": {
        "type": "bool", "default": True,
        "label": "Follow the block map",
        "hint": "Per-block features spread their strength by how much each block can take, from a profile of this model. "
                "Nothing changes until a profile has been made.",
    },
}


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunUI Block map", message)


def install(patcher, values, key):
    n = dit_hooks.block_count(patcher)
    if n == 0:
        return None
    view = None
    data = measure.load(n) if values.get("follow", True) else None
    if data:
        try:
            view = measure.MapView(data)
        except (KeyError, TypeError, ValueError):
            _say("the saved map for this model is unreadable and is not used")
    dit_hooks.publish(patcher, PUBLISHED, view)
    note = []
    if view is not None:
        for kind in measure.KINDS:
            if view.has(kind):
                note.append(f"{kind}: " + ("flat, no effect" if view.flat(kind) else "in use"))

    if note:
        log.once(f"{ID}:read:{n}:{'|'.join(note)}", log.INFO, "FunUI Block map",
                 f"following the saved map of {n} blocks ({', '.join(note)})")
    if not measure.enabled():
        measure.problem = None
        return ("block map " + ", ".join(note)) if note else None

    measure.problem = None
    live = {"state": measure.Probe(n)}

    def hook_for(block):
        def hook(args, extra):
            return measure.block_hook(live["state"], block)(args, extra)
        return hook

    for b in range(n):
        dit_hooks.add_block_hook(patcher, key, b, hook_for(b))

    def attention(func, *a, **kw):
        return measure.attention_override(live["state"])(func, *a, **kw)

    dit_hooks.add_attention_override(patcher, key, attention)
    dit_hooks.without_compiler(patcher, key)

    def outer(executor, *args, **kwargs):
        live["state"] = measure.Probe(n)
        try:
            return executor(*args, **kwargs)
        finally:
            res = measure.result(live["state"])
            if res is not None:
                measure.save(res)
                measure.set_enabled(False)              # one profile per switch: later runs follow the map, they do not redo it
                log.once(f"{ID}:saved:{n}", log.INFO, "FunUI Block map",
                         f"Active | profile of {n} blocks saved and the switch turned off; later runs follow it")
            else:
                measure.problem = "the profile run measured nothing: the model's picture could not be read"
                log.once(f"{ID}:none", log.INFO, "FunUI Block map", "Inactive | " + measure.problem)

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, outer)

    def apply_model(executor, x, t, *args, **kwargs):
        out = executor(x, t, *args, **kwargs)
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options") or {}
        st = live["state"]
        where = dit_hooks.current_step(to)
        if where is None or dit_hooks.probing(to) or st.kind is not None:
            return out
        point = measure.point_of(*where)
        if point is None or point in st.done:
            return out
        st.done.add(point)
        base = streams.video_of(out, named)
        if base is None:
            return out
        video = base[0]
        st.size = [int(video.shape[-2]), int(video.shape[-1])]
        if point == min(st.done):
            log.once(f"{ID}:cost", log.INFO, "FunUI Block map",
                     f"profiling: {2 * len(measure.groups(n))} extra model calls on each of {len(measure.POINTS)} steps")
        probe_args, probe_kwargs = streams.with_options(args, kwargs, {**to, dit_hooks.PROBE: True})
        for kind in measure.KINDS:
            for g, span in enumerate(measure.groups(n)):
                st.kind, st.group, st.block = kind, span, 0
                try:
                    probed = streams.video_of(executor(x, t, *probe_args, **probe_kwargs), named)
                    if probed is not None:
                        st.cells[(kind, g, point)] = measure.response(video, probed[0])
                except Exception as exc:                  # noqa: BLE001 -- a probe never costs the step
                    _say(f"a probe pass failed ({type(exc).__name__}: {exc}); that cell is not measured")
                finally:
                    st.kind = None
        return out

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return "profiling blocks on this run (switched on in Settings > Refinement & Taste)"


def routes(table, base, web):
    def view(n=None):
        data = measure.load(n) if n else None
        return {"enabled": measure.enabled(), "problem": measure.problem,
                "maps": sorted(p.name for p in measure.FOLDER.glob("*_blocks.json")) if measure.FOLDER.exists() else []}

    @table.get(base + "/status")
    async def _status(_req):
        return web.json_response(view())

    @table.post(base + "/enabled")
    async def _enabled(req):
        try:
            body = await req.json()
        except Exception:                                 # noqa: BLE001
            body = {}
        measure.set_enabled((body or {}).get("enabled") is True)
        return web.json_response(view())

    @table.post(base + "/clear")
    async def _clear(_req):
        measure.clear()
        return web.json_response(view())

    @table.get(base + "/export/{blocks}")
    async def _export(req):
        try:
            data = measure.load(int(req.match_info["blocks"]))
        except ValueError:
            data = None
        if data is None:
            return web.json_response({"why": "no map for that block count"}, status=404)
        return web.json_response(data, headers={"Content-Disposition": 'attachment; filename="block_map.json"'})


def cache_key():
    return "on" if measure.enabled() else "off"


PROVIDES = {"modifier": install, "routes": routes, "cache_key": cache_key}
