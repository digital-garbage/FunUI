"""Attention explorer: each run tries a small attention change, and your ratings pick which ones stay.

The chosen blocks are split by depth into a few groups. Every run, each group draws
one of three softmax temperatures -- sharper, as is, looser -- from how that group's
earlier runs were rated (Thompson draws, like Shot memory). A like makes the change
that run used more likely for that group, a dislike less. Groups drift toward what
you liked and away from what you did not, and the prompt still leads: temperature
only re-weighs what is already in context (see Randomizer).

"As is" starts with a head start so a fresh key stays close to the plain model.
How far a change goes learns too: a liked run that changed something pulls the size
toward the size it used, a disliked one pushes it away (as Shot memory does for its
amount), with a little exploration. Or type your own size.
Zero extra model calls; the speed cost is a division on the query.

Three more levers, all learned from the same ratings:
- Prompt focus: each group also draws a gain on the prompt tokens' keys, so attention to
  the prompt can be sharpened or loosened separately from everything else.
- Prompt weighting: a rating counts more for runs of a similar prompt (pooled prompt
  embedding, steep cosine) and less for other prompts, so what you tune on one prompt
  shows up on that prompt first.
- Clip regions: the picture's four time bands and the sound each draw their own sharper or
  looser setting, so a part you liked can stay while another changes.
- Recipes: when this prompt has a combination you liked, it is reused, with some groups
  re-drawn so it keeps exploring.

Not proven: with a handful of ratings the per-group counts are noisy, and whether
depth groups are the right unit is a guess. Experimental until it has been rated on
a real model.
"""

import math
import random

import torch
import torch.nn.functional as F
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, registry

ID = "attention_bandit"
TITLE = "Attention explorer"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["block_scope"]
USES = ["taste_store"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Attention explorer",
        "hint": "Each run tries a small attention change; your ratings decide which ones stay.",
    },
    "mode": {
        "type": "enum", "default": "learned", "options": [
            {"value": "learned", "label": "Learn from ratings"},
            {"value": "manual", "label": "My value"},
        ],
        "label": "How far", "ui": "segmented",
        "hint": "Learn from ratings: each rating nudges the size of a change toward what you liked. My value: you set it below.",
        "when": {"enabled": True},
    },
    "range": {
        "type": "float", "default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05,
        "label": "Change size", "ui": "slider",
        "hint": "How much sharper or looser a change is. Keep it small.",
        "when": {"enabled": True, "mode": "manual"},
    },
    "prompt_focus": {
        "type": "bool", "default": True,
        "label": "Prompt focus",
        "hint": "Also learns how strongly each group attends to your prompt, apart from everything else.",
        "when": {"enabled": True},
    },
    "regions": {
        "type": "bool", "default": True,
        "label": "Clip regions",
        "hint": "Also learns the clip's four time bands and its sound separately, so a good part can stay while another changes.",
        "when": {"enabled": True},
    },
    "blocks": {
        "type": "text", "model_blocks": True, "default": "2-10,29-46",
        "label": "Blocks",
        "hint": "e.g. 2-10,29-46. Split by depth into groups that learn separately.",
        "when": {"enabled": True},
    },
}

KIND = "attention_bandit"
GROUPS = 4
# Each arm is a factor as 1 + size * ARMS[i]: lower, as is, higher.
ARMS = (-1.0, 0.0, 1.0)
STAY = 1                    # index of "as is"
STAY_HEAD_START = 2.0       # extra likes "as is" starts with, so a new key is stable
NEW_HEAD_START = 4.0        # the same for prompt focus and clip regions, which change less often until they have data
# A factor on the prompt's keys only is not the same size of change as a temperature: it moves the prompt's
# logits against everyone else's, and at the full size it wrecked a trained key. A fraction of the size.
PROMPT_SCALE = 0.3
MAX_ROWS = 10 ** 9          # no cap: a rated run is a few KB at most
START_SPREAD = 0.2
SPREAD_MIN, SPREAD_MAX = 0.05, 0.5
LEARN_MAX = 0.35            # what ratings can grow the size to; 0.5x/1.5x on many blocks wrecked a picture on H3
LR = 0.5
EXPLORE = 0.05
# ponytail: pooled-prompt cosine sits high for most prompts, so the curve is steep and the
# constants are guesses. The log says how many rated runs count as this prompt.
SHARP = 100.0               # similarity = exp(SHARP * (cos - 1)); cos 0.99 -> 0.37
FLOOR = 0.1                 # what a different prompt's rating still counts for
UNKNOWN = 0.5               # a rating stored without its prompt
SAME = 0.5                  # similarity from which a run counts as this prompt
RECIPE_BAR = 0.65           # a liked combination is reused when its draw beats this
REGIONS = ("start of the clip", "second quarter", "third quarter", "end of the clip", "sound")
BANDS = 4                   # equal time bands of the picture; the sound is the last region
RECIPE_EXPLORE = 0.25       # per group: chance a reused recipe re-draws that group


def _say(message, level=log.ALERT):
    log.once(f"{ID}:{message}", level, "FunUI Attention explorer", message)


def split(blocks, groups=GROUPS):
    """Sorted block indices -> up to `groups` lists, by depth, sizes differing by at most one."""
    n = min(groups, len(blocks))
    size, extra = divmod(len(blocks), n)
    out, at = [], 0
    for i in range(n):
        step = size + (1 if i < extra else 0)
        out.append(list(blocks[at:at + step]))
        at += step
    return out


def similarity(cond, stored):
    """1 = the same prompt, toward 0 = a different one; None when either is unknown."""
    if cond is None or not torch.is_tensor(stored) or stored.numel() != cond.numel():
        return None
    cos = float(F.cosine_similarity(cond.float().flatten(), stored.float().flatten(), dim=0))
    return math.exp(SHARP * (min(cos, 1.0) - 1.0))


def weights_of(rows, cond):
    """([how much each rated run counts now], [its similarity or None])."""
    sims = [similarity(cond, r["rows"].get("cond")) for r in rows]
    return [UNKNOWN if s is None else FLOOR + (1.0 - FLOOR) * s for s in sims], sims


def _same_signature(p, signature):
    return "sig" in p and [int(x) for x in p["sig"].tolist()] == signature


def counts(rows, signature, n_groups, field="arms", weights=None):
    """[group][arm] -> [likes, dislikes] over the rated runs made with this block layout,
    each counting its weight (1 when none given)."""
    table = [[[0.0, 0.0] for _ in ARMS] for _ in range(n_groups)]
    for i, r in enumerate(rows):
        p = r["rows"]
        if field not in p or r["reward"] == 0 or not _same_signature(p, signature):
            continue
        w = 1.0 if weights is None else weights[i]
        for g, arm in enumerate(int(a) for a in p[field].tolist()):
            table[g][arm][0 if r["reward"] > 0 else 1] += w
    return table


def _sign(x):
    return (x > 0) - (x < 0)


def learned_spread(rows, signature):
    """The size of a change, replayed from the rated runs in order: a liked run that changed
    something pulls it toward the size it used, a disliked one pushes it away."""
    m = START_SPREAD
    for r in rows:
        p = r["rows"]
        if "spread" not in p or "arms" not in p or not r["reward"] or not _same_signature(p, signature):
            continue
        if all(int(a) == STAY for f in ("arms", "tarms", "rarms") if f in p for a in p[f].tolist()):
            continue                                  # nothing changed, so nothing to blame on the size
        used = float(p["spread"])
        # A like pulls toward the size it used. A dislike can only say "too much": it pushes the
        # size down from a bigger one, never up from a smaller one (that would ratchet to the cap).
        m += LR * (used - m) if r["reward"] > 0 else LR * min(0.0, m - used)
        m = min(max(m, SPREAD_MIN), LEARN_MAX)
    return m


def tally(rows):
    """'N rated runs: L liked, D disliked' over everything this key holds for this kind."""
    liked = sum(1 for r in rows if r["reward"] > 0)
    disliked = sum(1 for r in rows if r["reward"] < 0)
    return f"{len(rows)} rated runs ({liked} liked, {disliked} disliked)"


def ranges(blocks):
    out, start = [], None
    for i, b in enumerate(blocks):
        if start is None:
            start = b
        if i + 1 == len(blocks) or blocks[i + 1] != b + 1:
            out.append(str(start) if start == b else f"{start}-{b}")
            start = None
    return ",".join(out)


def describe(groups, picks, spread, words=("sharper", "looser"), what="", blocks=True):
    """Which blocks (or regions) went down or up, exactly; groups left as is are not mentioned."""
    moved = {}
    for g, a in zip(groups, picks):
        if a != STAY:
            moved.setdefault(a, []).extend(g)
    parts = [f"{what}{words[0] if ARMS[a] < 0 else words[1]} {1.0 + spread * ARMS[a]:g}x "
             + (f"at blocks {ranges(sorted(bs))}" if blocks else f"in {', '.join(bs)}")
             for a, bs in sorted(moved.items())]
    return "; ".join(parts)


def draw(table, rng=random, head=STAY_HEAD_START):
    """One arm per group: the arm whose Beta draw (likes + 1, dislikes + 1) is highest."""
    picks = []
    for group in table:
        best, best_theta = STAY, -1.0
        for arm, (liked, disliked) in enumerate(group):
            theta = rng.betavariate(liked + (head if arm == STAY else 0.0) + 1.0, disliked + 1.0)
            if theta > best_theta:
                best, best_theta = arm, theta
        picks.append(best)
    return picks


def pick_recipe(rows, sims, signature, focus, regions=False, rng=random):
    """The arms of a combination liked on THIS prompt, or None. Each combination seen on this
    prompt gets a Beta draw from its likes and dislikes; the best one is reused when it beats the bar."""
    combos = {}
    for r, s in zip(rows, sims):
        p = r["rows"]
        if s is None or s < SAME or not r["reward"] or "arms" not in p or "spread" not in p \
                or not _same_signature(p, signature) or (focus and "tarms" not in p) \
                or (regions and "rarms" not in p):
            continue
        key = (tuple(int(a) for a in p["arms"].tolist()),
               tuple(int(a) for a in p["tarms"].tolist()) if focus else None,
               tuple(int(a) for a in p["rarms"].tolist()) if regions else None)
        c = combos.setdefault(key, [0, 0, START_SPREAD])
        c[0 if r["reward"] > 0 else 1] += 1
        if r["reward"] > 0:
            c[2] = float(p["spread"])
    best, best_theta = None, -1.0
    for key, (liked, disliked, spread) in combos.items():
        if not liked:
            continue
        theta = rng.betavariate(1.0 + liked, 1.0 + disliked)
        if theta > best_theta:
            best, best_theta = (key, spread, liked, disliked), theta
    return best if best_theta > RECIPE_BAR else None


def make_plan(rows, cond, signature, n_groups, focus, manual_spread=None, regions=False, rng=random):
    """What this run does: {arms, tarms | None, rarms | None, spread, recipe | None, same, unknown}."""
    weights, sims = weights_of(rows, cond)
    arms = draw(counts(rows, signature, n_groups, "arms", weights), rng)
    tarms = draw(counts(rows, signature, n_groups, "tarms", weights), rng, head=NEW_HEAD_START) if focus else None
    rarms = draw(counts(rows, signature, len(REGIONS), "rarms", weights), rng, head=NEW_HEAD_START) if regions else None
    spread = learned_spread(rows, signature)
    recipe = pick_recipe(rows, sims, signature, focus, regions, rng)
    if recipe is not None:
        (r_arms, r_tarms, r_rarms), spread, liked, disliked = recipe
        for mine, theirs in ((arms, r_arms), (tarms, r_tarms), (rarms, r_rarms)):
            for i in range(len(mine or ())):
                if rng.random() >= RECIPE_EXPLORE:
                    mine[i] = theirs[i]
        recipe = (liked, disliked)
    spread = manual_spread if manual_spread is not None else min(max(
        spread + rng.gauss(0.0, EXPLORE), SPREAD_MIN), LEARN_MAX)
    return {"arms": arms, "tarms": tarms, "rarms": rarms, "spread": spread, "recipe": recipe,
            "same": sum(1 for s in sims if s is not None and s >= SAME),
            "unknown": sum(1 for s in sims if s is None)}


def _prompt_of(guider):
    try:
        p = guider.conds["positive"][0]["cross_attn"]
    except (AttributeError, KeyError, IndexError, TypeError):
        return None
    if not torch.is_tensor(p):
        return None
    p = p.detach().float()
    while p.dim() > 1:
        p = p.mean(dim=0)
    return p.cpu()


def _segments(options):
    """The packed sequence's {kind: (start, stop)}, or {} when this model does not say."""
    try:
        return {kind: (int(a), int(b)) for a, b, kind in options["minimax_h3_layout"].segments}
    except (KeyError, AttributeError, TypeError, ValueError):
        return {}


def _text_rows(options):
    """How many leading rows of the packed sequence are the prompt, or 0."""
    a, b = _segments(options).get("text", (0, 0))
    return b - a if a == 0 else 0


def region_spans(options):
    """[(start, stop)] per REGIONS entry: the target picture in equal time bands (its rows run
    frame by frame), then the target sound. [] when the layout is unknown."""
    seg = _segments(options)
    if "video" not in seg or "audio" not in seg:
        return []
    a, b = seg["video"]
    edges = [a + (b - a) * i // BANDS for i in range(BANDS + 1)]
    return [(edges[i], edges[i + 1]) for i in range(BANDS)] + [seg["audio"]]


def paired_regions(q, factors, video_width, sound_width):
    """q with each clip region's rows asking differently, for a model whose picture and sound attend in
    calls of their own (rows run frame by frame): the picture's self-attention is cut into BANDS equal
    time bands, the sound's self-attention is the last region. Any other call comes back as it was."""
    width = int(q.shape[-1])
    if width == video_width:
        edges = [int(q.shape[1]) * i // BANDS for i in range(BANDS + 1)]
        spans = list(zip(edges[:-1], edges[1:]))
        mine = factors[:BANDS]
    elif width == sound_width:
        spans, mine = [(0, int(q.shape[1]))], factors[BANDS:BANDS + 1]
    else:
        return q
    if all(f == 1.0 for f in mine):
        return q
    q = q.clone()
    for (a, b), factor in zip(spans, mine):
        if factor != 1.0:
            q[:, a:b] /= factor
    return q


# Carried on the per-block copy of transformer_options, as Randomizer does, so an
# attention call knows its block's factors without a shared flag.
_SCOPE = "funui_attention_bandit"


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    manual = values.get("mode") == "manual"
    manual_spread = min(max(float(values.get("range", START_SPREAD)), SPREAD_MIN), SPREAD_MAX)
    focus = bool(values.get("prompt_focus", True))
    regions = bool(values.get("regions", True))
    blocks, problems = dit_hooks.parse_blocks(values.get("blocks"), dit_hooks.block_count(patcher))
    for problem in problems:
        _say(f"ignored {problem}")
    if not blocks:
        raise ValueError(f"no usable blocks in {values.get('blocks')!r}, so nothing would change")
    groups = split(blocks)
    signature = [len(groups), blocks[0], blocks[-1]]
    live = {"factors": {}, "regions": [1.0] * len(REGIONS), "scaled": 0, "calls": 0,
            "text": 0, "text_wanted": False, "region_hits": 0, "region_wanted": False}

    captured = taste.collect(patcher, key, KIND, keep=MAX_ROWS)
    dm = getattr(patcher.model, "diffusion_model", None)
    paired = hasattr(dm, "transformer_blocks")                 # picture and sound attend in separate calls, one stream each
    video_width, sound_width = getattr(dm, "inner_dim", None), getattr(dm, "audio_inner_dim", None)
    if focus and paired:
        focus = False                  # the prompt's rows are not one run of the sequence here: that part is H3's

    def scoped_for(index):
        def scoped(args, extra):
            f = live["factors"].get(index)
            if f is None or (f == (1.0, 1.0) and not live["region_wanted"]):
                return extra["original_block"](args)
            to = dict(args.get("transformer_options") or {})
            to[_SCOPE] = f
            return extra["original_block"]({**args, "transformer_options": to})
        return scoped

    for index in blocks:
        dit_hooks.add_block_hook(patcher, key, index, scoped_for(index))

    def override(func, q, k, v, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        scope = options.get(_SCOPE)
        live["calls"] += 1
        if scope is None:
            return func(q, k, v, *args, **kwargs)
        temperature, prompt = scope
        live["scaled"] += 1
        if temperature != 1.0:
            q = q / temperature
        if live["region_wanted"] and paired and q.ndim == 3 and q.shape == k.shape:
            q = paired_regions(q, live["regions"], video_width, sound_width)
            live["region_hits"] += 1
        elif live["region_wanted"] and q.ndim == 4:
            spans = region_spans(options)
            if spans and spans[-1][1] <= q.shape[2]:
                q = q.clone()
                for (a, b), factor in zip(spans, live["regions"]):
                    if factor != 1.0:
                        q[:, :, a:b] /= factor         # only those rows ask differently: the rest of the clip is untouched
                live["region_hits"] += 1
        if prompt != 1.0 and kwargs.get("skip_reshape") and k.ndim == 4:
            n = _text_rows(options)
            if 0 < n < k.shape[2]:
                k = k.clone()
                k[:, :, :n] *= prompt                  # the prompt's keys only: its share of attention sharpens or loosens
                live["text"] += 1
        return func(q, k, v, *args, **kwargs)

    dit_hooks.add_attention_override(patcher, key, override)

    def run(executor, *args, **kwargs):
        rows = taste.rows(KIND)
        cond = _prompt_of(getattr(executor, "class_obj", None))
        plan = make_plan(rows, cond, signature, len(groups), focus, manual_spread if manual else None, regions)
        arms, tarms, rarms, spread = plan["arms"], plan["tarms"], plan["rarms"], plan["spread"]
        tfac = [1.0 + PROMPT_SCALE * spread * ARMS[a] for a in tarms] if focus else [1.0] * len(groups)
        rfac = [1.0 + spread * ARMS[a] for a in rarms] if regions else [1.0] * len(REGIONS)
        live.update(scaled=0, calls=0, text=0, region_hits=0, regions=rfac,
                    region_wanted=any(f != 1.0 for f in rfac), text_wanted=any(t != 1.0 for t in tfac),
                    factors={b: (1.0 + spread * ARMS[a], tf)
                             for g, a, tf in zip(groups, arms, tfac) for b in g})
        captured.update({"arms": torch.tensor(arms), "sig": torch.tensor(signature),
                         "spread": torch.tensor(float(spread))})
        if focus:
            captured["tarms"] = torch.tensor(tarms)
        if regions:
            captured["rarms"] = torch.tensor(rarms)
        if cond is not None:
            captured["cond"] = cond.half()
        known = "prompt unknown" if cond is None else f"{plan['same']} of them this prompt"
        how = f"recipe liked {plan['recipe'][0]}x, disliked {plan['recipe'][1]}x" if plan["recipe"] else "exploring"
        changes = "; ".join(x for x in (
            describe(groups, arms, spread),
            describe(groups, tarms, PROMPT_SCALE * spread, ("weaker", "stronger"), "prompt ") if focus else "",
            describe([[n] for n in REGIONS], rarms, spread, what="clip regions ", blocks=False) if regions else "") if x)
        log.once(f"{ID}:run", log.INFO, "FunUI Attention explorer",
                 f"key {taste.key!r}, {tally(rows)}, {known}, {how}: {changes or 'no change this run'}")
        out = executor(*args, **kwargs)
        wanted = bool(changes)
        if wanted and not live["scaled"]:
            _say("changed nothing this run: no attention call reached the changed blocks "
                 f"({live['calls']} calls seen). Another feature may have replaced attention.", log.WARNING)
        elif live["text_wanted"] and not live["text"]:
            _say("prompt focus changed nothing this run: this model does not say where the prompt sits in "
                 "its sequence, so only the plain attention changes were applied.", log.WARNING)
        elif live["region_wanted"] and not live["region_hits"]:
            _say("clip regions changed nothing this run: this model does not say where the picture and "
                 "sound sit in its sequence.", log.WARNING)
        elif wanted:
            log.once(f"{ID}:applied", log.INFO, "FunUI Attention explorer",
                     f"applied to {live['scaled']} of {live['calls']} attention calls"
                     + (f", prompt keys scaled in {live['text']}" if live["text_wanted"] else "")
                     + (f", clip regions in {live['region_hits']}" if live["region_wanted"] else ""))
        return out

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, f"{key}.run", run)
    size = f"size {manual_spread:g}" if manual else "size learned from ratings"
    return (f"{len(groups)} depth groups over {len(blocks)} blocks, {size}"
            + (", prompt focus on" if focus else "") + (", clip regions on" if regions else ""))


PROVIDES = {"modifier": install}
