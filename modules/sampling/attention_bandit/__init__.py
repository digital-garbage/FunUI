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

Not proven: with a handful of ratings the per-group counts are noisy, and whether
depth groups are the right unit is a guess. Experimental until it has been rated on
a real model.
"""

import random

from ..._core import dit_hooks, log, registry

ID = "attention_bandit"
TITLE = "Attention explorer"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["dit_block_hooks"]
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
    "blocks": {
        "type": "text", "default": "2-10,29-46",
        "label": "Blocks",
        "hint": "e.g. 2-10,29-46. Split by depth into groups that learn separately.",
        "when": {"enabled": True},
    },
}

KIND = "attention_bandit"
GROUPS = 4
# Each arm is a temperature as 1 + RANGE * ARMS[i]: sharper, as is, looser.
ARMS = (-1.0, 0.0, 1.0)
STAY = 1                    # index of "as is"
STAY_HEAD_START = 2.0       # extra likes "as is" starts with, so a new key is stable
MAX_ROWS = 10 ** 9      # no cap: a rated run is under 1 KB
START_SPREAD = 0.2
SPREAD_MIN, SPREAD_MAX = 0.05, 0.5
LR = 0.5
EXPLORE = 0.05


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


def counts(rows, signature, n_groups):
    """[group][arm] -> [likes, dislikes] over the rated runs made with this same block layout."""
    table = [[[0.0, 0.0] for _ in ARMS] for _ in range(n_groups)]
    for r in rows:
        p = r["rows"]
        if "arms" not in p or "sig" not in p or r["reward"] == 0:
            continue
        if [int(x) for x in p["sig"].tolist()] != signature:
            continue
        for g, arm in enumerate(int(a) for a in p["arms"].tolist()):
            table[g][arm][0 if r["reward"] > 0 else 1] += 1.0
    return table


def _sign(x):
    return (x > 0) - (x < 0)


def learned_spread(rows, signature):
    """The size of a change, replayed from the rated runs in order: a liked run that changed
    something pulls it toward the size it used, a disliked one pushes it away."""
    m = START_SPREAD
    for r in rows:
        p = r["rows"]
        if "spread" not in p or "sig" not in p or "arms" not in p or not r["reward"]:
            continue
        if [int(x) for x in p["sig"].tolist()] != signature:
            continue
        if all(int(a) == STAY for a in p["arms"].tolist()):
            continue                                  # nothing changed, so nothing to blame on the size
        m += LR * _sign(r["reward"]) * (float(p["spread"]) - m)
        m = min(max(m, SPREAD_MIN), SPREAD_MAX)
    return m


def describe(groups, picks, spread):
    """Which blocks went sharper or looser, exactly; groups left as is are not mentioned."""
    moved = {}
    for g, a in zip(groups, picks):
        if a != STAY:
            moved.setdefault(a, []).extend(g)
    parts = [f"{'sharper' if ARMS[a] < 0 else 'looser'} {1.0 + spread * ARMS[a]:g}x at blocks {ranges(sorted(bs))}"
             for a, bs in sorted(moved.items())]
    return "; ".join(parts) or "no change this run"


def ranges(blocks):
    out, start = [], None
    for i, b in enumerate(blocks):
        if start is None:
            start = b
        if i + 1 == len(blocks) or blocks[i + 1] != b + 1:
            out.append(str(start) if start == b else f"{start}-{b}")
            start = None
    return ",".join(out)


def draw(table, rng=random):
    """One arm per group: the arm whose Beta draw (likes + 1, dislikes + 1) is highest."""
    picks = []
    for group in table:
        best, best_theta = STAY, -1.0
        for arm, (liked, disliked) in enumerate(group):
            head = STAY_HEAD_START if arm == STAY else 0.0
            theta = rng.betavariate(liked + head + 1.0, disliked + 1.0)
            if theta > best_theta:
                best, best_theta = arm, theta
        picks.append(best)
    return picks


# Carried on the per-block copy of transformer_options, as Randomizer does, so an
# attention call knows its block's temperature without a shared flag.
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
    blocks, problems = dit_hooks.parse_blocks(values.get("blocks"), dit_hooks.block_count(patcher))
    for problem in problems:
        _say(f"ignored {problem}")
    if not blocks:
        raise ValueError(f"no usable blocks in {values.get('blocks')!r}, so nothing would change")
    groups = split(blocks)
    signature = [len(groups), blocks[0], blocks[-1]]
    live = {"temps": {}}

    def fresh():
        import torch
        rows = taste.rows(KIND)
        picks = draw(counts(rows, signature, len(groups)))
        spread = manual_spread if manual else min(max(
            learned_spread(rows, signature) + random.gauss(0.0, EXPLORE), SPREAD_MIN), SPREAD_MAX)
        live["temps"] = {b: 1.0 + spread * ARMS[a] for g, a in zip(groups, picks) for b in g}
        captured.update({"arms": torch.tensor(picks), "sig": torch.tensor(signature),
                         "spread": torch.tensor(float(spread))})
        log.once(f"{ID}:run", log.INFO, "FunUI Attention explorer",
                 f"key {taste.key!r}: {describe(groups, picks, spread)}")

    captured = taste.collect(patcher, key, KIND, keep=MAX_ROWS, fresh=fresh)

    def scoped_for(index):
        def scoped(args, extra):
            t = live["temps"].get(index)
            if t is None or t == 1.0:
                return extra["original_block"](args)
            to = dict(args.get("transformer_options") or {})
            to[_SCOPE] = t
            return extra["original_block"]({**args, "transformer_options": to})
        return scoped

    for index in blocks:
        dit_hooks.add_block_hook(patcher, key, index, scoped_for(index))

    def override(func, q, k, v, *args, **kwargs):
        t = (kwargs.get("transformer_options") or {}).get(_SCOPE)
        return func(q if t is None else q / t, k, v, *args, **kwargs)

    dit_hooks.add_attention_override(patcher, key, override)
    size = f"size {manual_spread:g}" if manual else "size learned from ratings"
    return f"{len(groups)} depth groups over {len(blocks)} blocks, {size}"


PROVIDES = {"modifier": install}
