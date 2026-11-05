"""The block map's maths and its file: what a profile run measures, and how a feature reads it.

A profile run nudges one group of blocks at a time, in one of two ways, and measures how far the model's
picture guess moves:

* leverage: the size of that move, relative to the picture and to the size of the nudge. A block with a
  big leverage changes the picture a lot for a small nudge.
* fragility: the share of the move that sits in the finest detail (what is left after a 3x3 blur). That
  is the share that looks like noise or damaged edges rather than a change of content.

Two kinds of nudge, because blocks do not react alike to both: "attention" (the attention query scaled,
what Attention explorer and the temperature features do) and "residual" (the block's own push scaled,
what REINS, Block repeat and steering do in effect).

A feature asks `multiplier(kind, block, fraction_of_run, blocks)` and gets a number around 1: how much
more or less than its set strength to use in that block at that point of the run. The effect share a
block gets is its quality, (1 - fragility)^2, and the nudge that produces that share is the share over
the leverage, so the picture change goes where it is clean and the nudge is small where leverage is big.
Averaged over the blocks the feature uses it is 1, so a strength slider keeps its meaning.
"""

import json
import math
import time

import torch
import torch.nn.functional as F

from ..._core import config

SWITCH = config.ROOT / "block_map.enabled"
FOLDER = config.ROOT / "block_map"
KINDS = ("attention", "residual")
DELTA = 0.1                      # the size of a probe nudge
GROUP_SIZE = 5                   # blocks nudged together
POINTS = (0.1, 0.45, 0.8)        # where in the run the profile is taken (fraction of the steps)
LOW, HIGH = 0.25, 3.0            # a multiplier stays inside these
REFERENCE = 0.5                  # a nudge the typical block is taken to handle without harm ...
HEADROOM = 3.0                   # ... times this is the most picture change any block is let produce
FLAT = 0.10                      # leverage spread under this, and fragility spread under 0.05: nothing to aim at

problem = None                   # why nothing was measured, for the panel


def enabled() -> bool:
    try:
        return SWITCH.read_text().strip() == "1"
    except OSError:
        return False


def set_enabled(on: bool) -> bool:
    SWITCH.parent.mkdir(parents=True, exist_ok=True)
    SWITCH.write_text("1" if on else "0")
    return bool(on)


def groups(n):
    """[(first, stop)] blocks per probe group."""
    return [(a, min(n, a + GROUP_SIZE)) for a in range(0, n, GROUP_SIZE)]


def point_of(step, steps):
    """Which profile point this step is, or None. The first step at or after each point's place in the run."""
    if steps < 2:
        return None
    for i, p in enumerate(POINTS):
        if step == min(steps - 1, int(round(p * (steps - 1)))):
            return i
    return None


def path_of(n):
    return FOLDER / f"{int(n)}_blocks.json"


def response(base, probed, delta=DELTA):
    """-> (leverage, fragility) of one probe. Both 5-D pictures [B, C, T, H, W]."""
    d = (probed.float() - base.float())
    ref = base.float()
    size = float(d.norm())
    lev = size / (max(float(ref.norm()), 1e-8) * delta)
    if size <= 1e-12 or d.dim() != 5:
        return lev, 0.0
    b, c, t, h, w = d.shape
    x = d.reshape(b * c * t, 1, h, w)
    low = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
    frag = min(1.0, max(0.0, float((x - low).norm() / x.norm().clamp(min=1e-12)) ** 2))
    return lev, frag


class Probe:
    """What the hooks read: which nudge is on right now, and which block is running."""

    def __init__(self, n):
        self.n = n
        self.kind = None          # "attention" / "residual" while a probe pass runs
        self.group = None         # (first, stop)
        self.block = 0
        self.done = set()
        self.cells = {}           # (kind, group index, point) -> (leverage, fragility)
        self.size = None          # (H, W) of the picture latent, for the record

    def active(self):
        return self.kind is not None and self.group is not None and self.group[0] <= self.block < self.group[1]


def attention_override(state):
    def override(func, q, k, v, heads, *args, **kw):
        if state.kind == "attention" and state.active() and torch.is_tensor(q):
            q = q * (1.0 + DELTA)
        return func(q, k, v, heads, *args, **kw)
    return override


def block_hook(state, block):
    def hook(args, extra):
        if state.kind is None:
            return extra["original_block"](args)
        state.block = block
        before = args.get("img")
        if isinstance(before, (tuple, list)) and before:
            before = before[0]                   # an audio+video block is handed (video, audio): the video is the picture
        if state.kind == "residual" and state.active() and torch.is_tensor(before):
            before = before.clone()              # a block may add to its input in place: keep the input as it was
        out = extra["original_block"](args)
        if state.kind == "residual" and state.active() and torch.is_tensor(before):
            after = out.get("img") if isinstance(out, dict) else None
            video = after[0] if isinstance(after, (tuple, list)) and after else after       # an audio+video block hands back (video, audio)
            if torch.is_tensor(video) and video.shape == before.shape:
                nudged = before + (1.0 + DELTA) * (video - before)
                out = {**out, "img": type(after)([nudged, *after[1:]]) if isinstance(after, (tuple, list)) else nudged}
        return out
    return hook


def result(state):
    """The finished profile as a plain dict, or None when no cell was measured."""
    if not state.cells:
        return None
    grid = groups(state.n)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "blocks": state.n, "groups": grid,
           "points": list(POINTS), "size": state.size, "delta": DELTA}
    for kind in KINDS:
        cells = [[state.cells.get((kind, g, p)) for p in range(len(POINTS))] for g in range(len(grid))]
        if any(c is not None for row in cells for c in row):
            out[kind] = {"leverage": [[None if c is None else c[0] for c in row] for row in cells],
                         "fragility": [[None if c is None else c[1] for c in row] for row in cells]}
    return out


def save(res):
    FOLDER.mkdir(parents=True, exist_ok=True)
    path = path_of(res["blocks"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(path)


def load(n):
    try:
        return json.loads(path_of(n).read_text())
    except (OSError, ValueError):
        return None


def clear(n=None):
    paths = [path_of(n)] if n is not None else list(FOLDER.glob("*_blocks.json")) if FOLDER.exists() else []
    for p in paths:
        try:
            p.unlink()
        except OSError:
            pass


def _interp(xs, ys, x):
    """Piecewise linear through the points that have a value; flat outside them."""
    pts = [(a, b) for a, b in zip(xs, ys) if b is not None]
    if not pts:
        return None
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / max(1e-9, x1 - x0)
    return pts[-1][1]


class MapView:
    """A saved profile, asked for multipliers."""

    def __init__(self, data):
        self.data = data
        self.n = int(data["blocks"])
        self.centres = [(a + b - 1) / 2.0 for a, b in data["groups"]]
        self._cache = {}
        self._memo = {}

    def has(self, kind):
        return kind in self.data

    def _fields(self, kind, frac):
        """(leverage, fragility) for every block at this fraction of the run."""
        key = (kind, round(float(frac) * 20))
        if key in self._cache:
            return self._cache[key]
        d, pts = self.data[kind], self.data["points"]
        lev_g, frg_g = [], []
        for row_l, row_f in zip(d["leverage"], d["fragility"]):
            lev_g.append(_interp(pts, row_l, frac))
            frg_g.append(_interp(pts, row_f, frac))
        lev = [_interp(self.centres, lev_g, float(b)) for b in range(self.n)]
        frg = [_interp(self.centres, frg_g, float(b)) for b in range(self.n)]
        self._cache[key] = (lev, frg)
        return lev, frg

    def flat(self, kind):
        lev, frg = self._fields(kind, 0.5)
        lev = [v for v in lev if v is not None]
        frg = [v for v in frg if v is not None]
        if not lev or not frg:
            return True
        mean = sum(lev) / len(lev)
        return (max(lev) - min(lev)) < FLAT * max(mean, 1e-9) and (max(frg) - min(frg)) < 0.05

    def multipliers(self, kind, frac, blocks):
        """{block: multiplier} for the blocks a feature uses, mean 1; all 1 when the profile is flat or missing."""
        blocks = [int(b) for b in blocks]
        memo = (kind, round(float(frac) * 20), tuple(blocks))
        if memo in self._memo:
            return self._memo[memo]
        self._memo[memo] = self._multipliers(kind, frac, blocks)
        return self._memo[memo]

    def _multipliers(self, kind, frac, blocks):
        if not blocks or not self.has(kind) or len(blocks) == 1 or self.flat(kind):
            return {b: 1.0 for b in blocks}
        lev, frg = self._fields(kind, frac)
        raw = {}
        for b in blocks:
            if b >= self.n or lev[b] is None or frg[b] is None or lev[b] <= 1e-9:      # a block the probe saw no effect from is unmeasured, not infinitely cheap
                raw[b] = None
                continue
            raw[b] = max(0.05, 1.0 - frg[b]) ** 2 / max(lev[b], 1e-6)
        known = [v for v in raw.values() if v is not None]
        if not known:
            return {b: 1.0 for b in blocks}
        mean = sum(known) / len(known)
        return {b: (1.0 if v is None else min(HIGH, max(LOW, v / mean))) for b, v in raw.items()}

    def adjust(self, kind, block, frac, blocks, value):
        """The feature's own `value` (its change from neutral: a temperature minus 1, a strength) for this block:
        spread by `multipliers`, then softly limited so no block is pushed far past what its profile says it can
        take. The limit is the picture change a typical block makes from REFERENCE times HEADROOM, divided by this
        block's leverage and its fragility, so a setting of 10 is respected up to that limit and not beyond it.
        A flat or missing profile returns `value` as it came."""
        value = float(value)
        if value == 0.0 or not self.has(kind) or self.flat(kind):
            return value
        m = self.multipliers(kind, frac, blocks).get(int(block), 1.0)
        lev, frg = self._fields(kind, frac)
        b = int(block)
        known = sorted(v for v in lev if v is not None and v > 0)
        if b >= self.n or lev[b] is None or frg[b] is None or lev[b] <= 1e-9 or not known:
            return value * m
        cap = REFERENCE * HEADROOM * known[len(known) // 2] / (max(lev[b], 1e-6) * (1.0 + 2.0 * frg[b]))
        x = abs(value) * m
        return math.copysign(cap * math.tanh(x / cap), value)

    def multiplier(self, kind, block, frac, blocks):
        return self.multipliers(kind, frac, blocks).get(int(block), 1.0)
