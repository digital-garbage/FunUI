"""Identity bank: a liked clip's look is kept and leaned on in the next ones (LTX).

BachVid ("raw K/V identity bank", arXiv 2510.21696): the attention keys and values a liked run used at a few
blocks are the look the model had settled on. Later runs lean toward them at the same positions, with no
reference picture and no training. KV-Lock ("When to Lock Attention", arXiv 2603.09657): the lean is
scheduled by how unsettled the picture is, harder when it drifts and softer when it has settled.

Every run's last step is kept on the taste key until you rate it. A like adds it to the bank (the average
of the liked runs); a dislike is ignored. The bank is only used on a run of the same size (same tokens), and
the run says when it is not. Overlap is a lerp of the running keys and values toward the bank's, so no
sequence length or RoPE changes.

Not the paper's concat-attend and not its DDIM inversion (a few-step distilled model inverts poorly): this
is the v4 form, which was validated on LTX 2.3. KV-Lock here reads how much the model's prediction moves
between steps, relative to its own average in the run, instead of v4's absolute threshold.

Memory: one picture-token row of keys and values per kept block per remembered run, in half precision
(about 100 MB per block for a 768x512, 121-frame clip), so the defaults keep two blocks and three runs.
"""

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, registry, streams

ID = "identity_bank"
TITLE = "Identity bank"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["block_scope", "ltx_av"]
USES = ["taste_store"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Identity bank",
        "hint": "Keeps the look of the clips you like and leans on it in the next ones: the same character and place without a reference picture. Needs a Taste key and one liked clip of the same size.",
    },
    "strength": {
        "type": "float", "default": 0.1, "min": 0.0, "max": 0.6, "step": 0.02,
        "label": "How hard", "ui": "slider",
        "hint": "0 = only keep. Higher locks harder and costs variety; past 0.3 the picture starts to repeat the liked one.",
        "when": {"enabled": True},
    },
    "blocks": {
        "type": "text", "model_blocks": True, "default": "20,30",
        "label": "Blocks",
        "hint": "Where the look is kept and applied, e.g. 14,20,30. Each block kept costs about 100 MB per remembered clip.",
        "when": {"enabled": True},
    },
    "lock": {
        "type": "bool", "default": True,
        "label": "Lock harder when it drifts",
        "hint": "KV-Lock: leans harder on steps where the picture is still changing a lot and softer once it has settled.",
        "when": {"enabled": True},
    },
    "remember": {
        "type": "int", "default": 3, "min": 1, "max": 8,
        "label": "Clips remembered",
        "hint": "How many recent clips wait for a rating on this key.",
        "when": {"enabled": True},
    },
}

KIND = "identity_bank"
GATE = (0.35, 0.85)       # sigmas the lean is on for: the early steps set the layout, the last ones the fine detail
LIMIT = 0.6               # never lean more than this: the picture would be the bank's
WINDOW = 5                # steps the lock averages over
SCOPE = "funui_identity_bank"


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunUI Identity bank", message)


def averaged(rows):
    """{name: mean tensor} of the liked rows, per name over the rows that agree in shape, or {}."""
    liked = [r["rows"] for r in rows if r["reward"] > 0 and r["rows"]]
    bank = {}
    for name in {n for r in liked for n in r}:
        same = [r[name] for r in liked if name in r and r[name].shape == next(x[name] for x in liked if name in x).shape]
        bank[name] = torch.stack([t.float() for t in same]).mean(0).half()
    return bank


def lock_scale(history, change):
    """The lean's multiplier for the next step: this step's change over the run's own average, held in 0.5..2.
    1 until there is something to compare with."""
    history.append(change)
    del history[:-WINDOW]
    if len(history) < 2:
        return 1.0
    mean = sum(history[:-1]) / (len(history) - 1)
    return max(0.5, min(2.0, change / mean)) if mean > 1e-9 else 1.0


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    dm = getattr(patcher.model, "diffusion_model", None)
    width = getattr(dm, "inner_dim", None)
    if width is None:
        _say("off: could not read the picture width of this model")
        return None
    blocks, problems = dit_hooks.parse_blocks(values.get("blocks"), dit_hooks.block_count(patcher))
    for problem in problems:
        _say(f"ignored {problem}")
    if not blocks:
        raise ValueError(f"no usable blocks in {values.get('blocks')!r}, so nothing would be kept")
    strength = min(max(float(values.get("strength", 0.1)), 0.0), LIMIT)
    locking = bool(values.get("lock", True))
    live = {"bank": {}, "scale": 1.0, "history": [], "last": None, "used": 0, "skipped": 0, "cast": {}}

    def fresh():
        live.update(bank=averaged(taste.rows(KIND)), scale=1.0, history=[], last=None, used=0, skipped=0, cast={})
        liked = sum(1 for r in taste.rows(KIND) if r["reward"] > 0)
        log.once(f"{ID}:state", log.INFO, "FunUI Identity bank",
                 f"key {taste.key!r}: {liked} liked clip(s) in the bank, " + (f"leaning at {strength:g}" if live["bank"] and strength > 0 else "only keeping"))

    captured = taste.collect(patcher, key, KIND, keep=int(values.get("remember", 3)), only="liked", fresh=fresh)

    def scoped(block):
        def hook(args, extra):
            to = dict(args.get("transformer_options") or {})
            to[SCOPE] = block
            return extra["original_block"]({**args, "transformer_options": to})
        return hook

    for b in blocks:
        dit_hooks.add_block_hook(patcher, key, b, scoped(b))

    def override(func, q, k, v, *args, **kwargs):
        to = kwargs.get("transformer_options") or {}
        block = to.get(SCOPE)
        if block is None or dit_hooks.probing(to) or q.ndim != 3 or q.shape != k.shape or q.shape[-1] != width:
            return func(q, k, v, *args, **kwargs)          # not the picture's own attention in a kept block
        if dit_hooks.last_step(to):
            captured[f"k{block}"], captured[f"v{block}"] = k.mean(0, dtype=torch.float32).half().detach(), v.mean(0, dtype=torch.float32).half().detach()
        bank_k, bank_v = live["bank"].get(f"k{block}"), live["bank"].get(f"v{block}")
        sigma = to.get("sigmas")
        if bank_k is None or strength <= 0.0 or sigma is None or not GATE[0] <= float(sigma.reshape(-1)[0]) <= GATE[1]:
            return func(q, k, v, *args, **kwargs)
        if bank_k.shape != k.shape[1:]:
            live["skipped"] += 1
            return func(q, k, v, *args, **kwargs)
        cast = live["cast"].setdefault((block, k.device, k.dtype), (bank_k.to(k.device, k.dtype), bank_v.to(k.device, k.dtype)))
        alpha = min(strength * live["scale"], LIMIT)
        live["used"] += 1
        return func(q, k + alpha * (cast[0] - k), v + alpha * (cast[1] - v), *args, **kwargs)

    dit_hooks.add_attention_override(patcher, key, override)

    def apply_model(executor, x, t, *args, **kwargs):
        out = executor(x, t, *args, **kwargs)
        if not locking:
            return out
        split = streams.video_of(out, streams.model_args(args, kwargs))
        if split is None:
            return out
        video = split[0].detach().float()
        if live["last"] is not None and live["last"].shape == video.shape:
            change = float(torch.linalg.vector_norm(video - live["last"]) / torch.linalg.vector_norm(video).clamp_min(1e-9))
            live["scale"] = lock_scale(live["history"], change)
        live["last"] = video
        return out

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)

    def finish(executor, *args, **kwargs):
        out = executor(*args, **kwargs)
        if live["bank"] and strength > 0:
            if live["skipped"] and not live["used"]:
                _say("the bank is from a clip of another size, so it was not used this run")
            elif not live["used"]:
                _say("no attention call reached the bank's blocks this run (another feature may have replaced attention)")
        return out

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key + ".report", finish)
    return f"keeps blocks {','.join(map(str, blocks))}; leans {strength:g}" + (" (locks harder when it drifts)" if locking else "") + ", read fresh every run"


PROVIDES = {"modifier": install}
