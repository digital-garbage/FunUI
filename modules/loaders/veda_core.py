"""Veda sparse attention, the torch half: tiles, plans, predictor, selection, one layer's call.

Veda (Han, Yang, Hu, Mei, Jiang, Qi; MIT; arXiv 2605.30325) lets a small distilled predictor say
which 128-token 3D tiles of the attention map matter, and computes only the top ~10% of them.
Written here from Veda-on-ComfyUI's core (itself a rewrite of Miowtion's miowtion/veda, MIT), in
this codebase's style; see VEDA-NOTICE.txt. The released predictors were trained against these
exact rules (tile order, pooling, budget, own-tile, global rows), so none of them may drift.

Pure torch, no ComfyUI import: it runs and is tested on any device. The kernel that attends the
kept tiles is a callable handed in (veda_kernel.attend on a GPU, `reference_attend` in tests).
Every tensor made here is built on, or moved to, the device of the tensor it meets.
"""
import contextlib
import functools
import json
import math
import os
import time
from types import SimpleNamespace

import torch

TILE = 128
FORMAT = "miowtion-veda-predictor-v1"
_STORAGE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn}
FPS = 24
_CHUNK_BYTES = 256 * 2**20          # one tile-ordered q/k/v/out copy of a head chunk; four are alive at once


class VedaError(ValueError):
    """The file or layout cannot be used; the message is shown to the person."""


# ── budgets: (ratio, tiles), exactly one set. ratio = share of key tiles KEPT at equal kernel cost ─────────────

def sparsity_budget(share):
    """Skip `share` (0..1) of the key tiles."""
    return (1.0 - float(share), None)


def keeps_all(budget):
    return budget[1] is None and budget[0] >= 1.0


def describe_budget(budget):
    return f"{budget[1]:g} tiles" if budget[1] is not None else f"{100.0 * (1.0 - min(budget[0], 1.0)):g}% sparse"


def _per_row(budget, real_tokens, n_cols):
    if budget[1] is not None:
        return float(budget[1])
    ideal = math.ceil(real_tokens / TILE)       # equal cost: a padded tiling never gets a bigger budget
    return budget[0] * ideal * ideal / n_cols


def _split(budget, n_cols):
    """(k_lo, k_hi, frac) with 1 <= k_lo <= k_hi <= n_cols."""
    if budget >= n_cols:
        return n_cols, n_cols, 0.0
    lo = min(max(1, math.floor(budget)), n_cols)
    return lo, min(lo + 1, n_cols), round(min(max(budget - lo, 0.0), 1.0), 12)


@functools.lru_cache(maxsize=256)
def _bresenham(n_rows, frac):
    ramp = torch.floor(torch.arange(n_rows + 1, dtype=torch.float64) * frac)
    return ramp[1:] > ramp[:-1]


# ── tiles ──────────────────────────────────────────────────────────────────────────────────────────────────────

def shapes():
    """Every power-of-two (t, h, w) with t * h * w = TILE."""
    return sorted((2 ** i, 2 ** j, TILE >> (i + j)) for i in range(8) for j in range(8 - i))


def _padded(grid, shape):
    return tuple(-(-g // s) * s for g, s in zip(grid, shape))


def _count(grid, shape):
    p = _padded(grid, shape)
    return p[0] * p[1] * p[2] // TILE


def least_padding_shape(grid):
    """Shape that pads `grid` least (ties: the most cubic, then the first); how reference spans are tiled."""
    fit = [s for s in shapes() if all(a <= b for a, b in zip(s, grid))]
    return min(fit or shapes(), key=lambda s: (_count(grid, s), max(s) / min(s), s))


def span_tiles(start, grid, shape):
    """[n_tiles, 128] packed row ids of one span, -1 on padding. Box order is (h, w, t) outer to inner, rows
    inside a box are t, h, w row-major, real rows compacted to the front (so a tile's real rows are a prefix)."""
    t, h, w = grid
    tp, hp, wp = _padded(grid, shape)
    cells = torch.full((tp, hp, wp), -1, dtype=torch.long)
    cells[:t, :h, :w] = start + torch.arange(t * h * w).view(t, h, w)
    st, sh, sw = shape
    tiles = cells.view(tp // st, st, hp // sh, sh, wp // sw, sw).permute(2, 4, 0, 1, 3, 5).reshape(-1, TILE)
    return torch.gather(tiles, 1, torch.argsort((tiles < 0).to(torch.int8), dim=1, stable=True))


def build_layout(spans, seq_len, device="cpu"):
    """One permutation of the packed sequence. `spans` = [(start, grid, shape)] in packed order, the target last;
    every other row is global (text, audio, untiled conditions) and is chunked into tiles after the spans."""
    if not spans:
        raise VedaError("the target span must be tiled")
    covered, tiles, stop = torch.zeros(seq_len, dtype=torch.bool), [], 0
    for start, grid, shape in spans:
        rows = grid[0] * grid[1] * grid[2]
        if start < stop or start + rows > seq_len:
            raise VedaError(f"span at {start} overlaps another or runs past {seq_len}")
        stop = start + rows
        covered[start:stop] = True
        tiles.append(span_tiles(start, grid, shape))
    video = torch.cat(tiles)
    rest = torch.nonzero(~covered).view(-1)
    n_global = -(-rest.numel() // TILE)
    tail = torch.full((n_global * TILE,), -1, dtype=torch.long)
    tail[:rest.numel()] = rest
    perm = torch.cat([video.view(-1), tail])
    count = (perm.view(-1, TILE) >= 0).sum(1).to(torch.int32)
    partial = torch.nonzero((count > 0) & (count < TILE)).view(-1)
    n_video = video.shape[0]
    on = lambda x: x.to(device)
    return SimpleNamespace(
        n_video=n_video, n_ref=sum(t.shape[0] for t in tiles[:-1]), n_global=n_global, n_tiles=n_video + n_global,
        ref_tokens=sum(g[0] * g[1] * g[2] for _, g, _ in spans[:-1]),
        target_tokens=spans[-1][1][0] * spans[-1][1][1] * spans[-1][1][2], seq_len=seq_len,
        widest=int((count > 0).sum()),      # an upper bound on the kept tiles of any row (a global row sees every tile): no device read needed
        bres={}, valid_count=on(count), gather_index=on(perm.clamp(min=0)), scatter_index=on(torch.where(perm < 0, seq_len, perm)),
        pad_slots=on(torch.nonzero(perm < 0).view(-1)), partial_video=on(partial[partial < n_video]),
        kv_ok=on(count > 0), slot_valid=on((perm >= 0).to(torch.int32)))


def gather(x, lay, heads):
    """[S, H, D] -> [N, H', D] in tile order, padding slots zero."""
    out = x[lay.gather_index[:, None], heads[None, :]]
    if lay.pad_slots.numel():
        out.index_fill_(0, lay.pad_slots, 0)
    return out


def scatter_(out, tiled, lay, heads):
    """Tile-ordered rows back into out[S + 1, H, D]; padding lands in the spare row S."""
    out[lay.scatter_index[:, None], heads[None, :]] = tiled


# ── predictor: pool each video tile to [mean | max | min], score tile pairs ────────────────────────────────────

@torch.no_grad()
def pool(x, lay):
    """[N, H', D] tile order -> [H', n_video, 3D] fp32 (mean, max, min over each tile's real rows; 0 for empty)."""
    n, (heads, dim) = lay.n_video, x.shape[1:]
    tiles = x[:n * TILE].view(n, TILE, heads, dim)
    mean = tiles.sum(dim=1, dtype=torch.float32) / lay.valid_count[:n].clamp(min=1).to(torch.float32)[:, None, None]
    lo, hi = torch.aminmax(tiles, dim=1)
    part = lay.partial_video
    if part.numel():           # padding rows are 0, a legal value: the partial tiles' extremes are redone masked
        sub = tiles.index_select(0, part)
        live = (torch.arange(TILE, device=x.device)[None, :] < lay.valid_count.index_select(0, part)[:, None])[:, :, None, None]
        hi.index_copy_(0, part, sub.masked_fill(~live, float("-inf")).amax(dim=1))
        lo.index_copy_(0, part, sub.masked_fill(~live, float("inf")).amin(dim=1))
    feats = torch.cat([mean, hi.float(), lo.float()], dim=-1)
    return torch.where(lay.kv_ok[:n, None, None], feats, 0.0).permute(1, 0, 2).contiguous()


@torch.no_grad()
def logits(fq, fk, pq, pk):
    """[H', n_q, n_k] fp32 block logits: x_hat = pool @ P + mean, q_hat . k_hat / sqrt(D)."""
    dim = pq.shape[-1]
    qh = torch.bmm(fq, pq.to(device=fq.device, dtype=torch.float32)) + fq[..., :dim]
    kh = torch.bmm(fk, pk.to(device=fk.device, dtype=torch.float32)) + fk[..., :dim]
    return torch.bmm(qh, kh.transpose(1, 2)) / math.sqrt(dim)


# ── selection: top-k per query tile over the video x video quadrant ────────────────────────────────────────────

def column_blocks(lay, generated, reference):
    """[(start, stop, budget, real tokens)]: the reference tiles (when tiled) and the target, each with its own top-k
    (one pooled top-k would starve the references, the target's spatial neighbours always win)."""
    target = (lay.n_ref, lay.n_video, generated, lay.target_tokens)
    return [target] if lay.n_ref == 0 else [(0, lay.n_ref, reference, lay.ref_tokens), target]


@torch.no_grad()
def select(scores, lay, blocks):
    """(index [H', n_video, K] int64 key tiles, keep [H', n_video, K] bool). Rules: a query tile always keeps its own
    tile and that uses budget, empty key tiles are never kept, the fractional budget is spread by a Bresenham pattern."""
    heads, nv, dev = scores.shape[0], lay.n_video, scores.device
    rows = torch.arange(nv, device=dev)
    idxs, keeps = [], []
    for start, stop, budget, real in blocks:
        n = stop - start
        if keeps_all(budget):
            idxs.append(torch.arange(start, stop, device=dev).expand(heads, nv, n))
            keeps.append(lay.kv_ok[start:stop].expand(heads, nv, n))
            continue
        s = scores[:, :, start:stop].float().clone()
        s.masked_fill_(~lay.kv_ok[None, None, start:stop], float("-inf"))
        own = ((rows >= start) & (rows < stop))[None, :, None]
        col = (rows - start).clamp(0, n - 1)[None, :, None].expand(heads, nv, 1)
        s.scatter_(-1, col, torch.where(own, float("inf"), s.gather(-1, col)))
        lo, hi, frac = _split(_per_row(budget, real, n), n)
        vals, idx = torch.topk(s, hi, dim=-1, sorted=True)
        extra = lay.bres.get((nv, float(frac), str(dev)))          # built on the host once, then kept on the device with the layout
        if extra is None:
            extra = lay.bres[(nv, float(frac), str(dev))] = _bresenham(nv, float(frac)).to(device=dev, dtype=torch.long)
        allowed = lo + extra
        idxs.append(idx + start)
        keeps.append((torch.arange(hi, device=dev)[None, None, :] < allowed[None, :, None]) & (vals > float("-inf")))
    return torch.cat(idxs, -1), torch.cat(keeps, -1)


@torch.no_grad()
def block_mask(index, keep, lay):
    """[H', n_tiles, n_tiles] bool, rows are query tiles. Global tiles are dense both ways."""
    n, nv = lay.n_tiles, lay.n_video
    mask = torch.zeros(index.shape[0], n, n, dtype=torch.bool, device=index.device)
    mask[:, :nv].scatter_(2, index, keep)
    mask[:, :, nv:] = lay.kv_ok[nv:]
    mask[:, nv:, :] = lay.kv_ok
    return mask


def tile_index_list(mask, widest=None):
    """A mask as the per-row list of kept tiles: (index [H', n, widest] int32 ascending, count [H', n] int32). Entries past a row's
    count are never read, so `widest` may be any upper bound (a layout knows one); None reads the exact widest from the device."""
    count = mask.sum(-1, dtype=torch.int32)
    if widest is None:
        widest = int(count.max().item()) if count.numel() else 0
    order = torch.argsort(mask.to(torch.int8), dim=-1, descending=True, stable=True)
    return order[..., :widest].to(torch.int32).contiguous(), count


def reference_attend(q, k, v, mask, lay):
    """Exact block-sparse attention in fp32 on tile-ordered [N, H', D]: the ground truth and the test stand-in
    for the kernel. O(N^2) memory, never for a real sequence."""
    allowed = mask.repeat_interleave(TILE, 1).repeat_interleave(TILE, 2) & lay.slot_valid.bool()[None, None, :]
    probs = torch.softmax((torch.einsum("qhd,khd->hqk", q.float(), k.float()) * q.shape[-1] ** -0.5)
                          .masked_fill(~allowed, float("-inf")), dim=-1).nan_to_num(0.0)
    out = torch.einsum("hqk,khd->qhd", probs, v.float()) * lay.slot_valid.to(torch.float32)[:, None, None]
    return out.to(q.dtype)


# ── what the packed layout holds ────────────────────────────────────────────────────────────────────────────────

def _grid_of(position_ids, a, b):
    """(T, H, W) when rows a:b are T, H, W row-major (checked against the positions), else None."""
    pos = position_ids[a:b]
    axes = [torch.unique(pos[:, i]) for i in range(3)]
    t, h, w = (x.numel() for x in axes)
    if t * h * w != b - a:
        return None
    want = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    return (t, h, w) if torch.equal(want, pos.to(want.dtype)) else None


def describe_layout(layout):
    """ComfyUI's H3 PackedLayout -> a hashable (seq_len, (target start, grid), ((ref start, grid), ...), skipped notes).
    References (cond, ref_img) whose rows are not a verified row-major grid stay global, which is always correct."""
    try:
        segments = [(int(a), int(b), kind) for a, b, kind in layout.segments]
        _, lt, lh, lw, _ = layout.signature
        seq_len = int(layout.seq_len)
    except (AttributeError, TypeError, ValueError) as exc:
        raise VedaError(f"not a MiniMax H3 packed layout ({exc})") from exc
    video = [(a, b) for a, b, kind in segments if kind == "video"]
    if len(video) != 1:
        raise VedaError(f"expected one target video segment, found {len(video)}")
    grid = (int(lt), int(lh) // 2, int(lw) // 2)
    if grid[0] * grid[1] * grid[2] != video[0][1] - video[0][0]:
        raise VedaError(f"the video segment does not match the latent grid {grid}")
    pos, refs, skipped = getattr(layout, "position_ids", None), [], []
    for a, b, kind in segments:
        if kind in ("cond", "ref_img"):
            g = None if pos is None else _grid_of(pos, a, b)
            (refs.append((a, g)) if g else skipped.append(f"{kind}[{a}:{b}]"))
    return seq_len, (video[0][0], grid), tuple(refs), tuple(skipped)


# ── tile plans and the predictor file ──────────────────────────────────────────────────────────────────────────

def _plan(name, grid, shape_list, head_shape):
    return SimpleNamespace(name=name, grid=tuple(grid), shapes=[tuple(s) for s in shape_list], head_shape=head_shape, groups={})


def _aspect(grid):
    return grid[2] / grid[1]


def describe_grid(grid):
    frames = 17 * ((grid[0] - 2) // 5) + 5 if grid[0] >= 2 else 1       # H3: 17n + 5 frames for a latent length
    return f"{grid[2] * 32}x{grid[1] * 32}, {frames / FPS:.1f} s"


def choose_plan(plans, grid):
    """(plan, exact, words): the plan with the nearest aspect, then nearest duration, then least padding. The H<->W
    mirrored plans are candidates too. Never fails: a plan of another size still tiles any grid, it is only
    outside what the predictor was trained on, and `words` says so."""
    grid = tuple(int(g) for g in grid)
    mirrored = [_plan(p.name + "_T", (p.grid[0], p.grid[2], p.grid[1]), [(t, w, h) for t, h, w in p.shapes], p.head_shape) for p in plans]
    target = math.log(_aspect(grid))
    best = min(plans + mirrored, key=lambda p: (round(abs(math.log(_aspect(p.grid)) - target), 3), abs(p.grid[0] - grid[0]),
                                                sum(_count(grid, s) for s in p.shapes)))
    if best.grid == grid:
        return best, True, f"trained for this size ({describe_grid(grid)})"
    return best, False, f"nearest trained size: {describe_grid(best.grid)} (this video: {describe_grid(grid)})"


def head_groups(plan, layer, device="cpu"):
    """[(shape, heads int64 on `device`)] of one layer: heads that share a tile shape. Cached on the plan, per device."""
    key = (layer, str(device))
    if key not in plan.groups:
        row = torch.tensor(plan.head_shape[layer])
        plan.groups[key] = [(plan.shapes[i], torch.nonzero(row == i).view(-1).to(device)) for i in sorted(set(plan.head_shape[layer]))]
    return plan.groups[key]


def _budget_of(meta, prefix, fallback):
    kind = meta.get(f"{prefix}_budget_kind")
    if kind is None:
        return fallback
    try:
        value = float(meta[f"{prefix}_budget_value"])
    except (KeyError, ValueError) as exc:
        raise VedaError(f"{prefix}_budget_value: {exc}") from exc
    if kind not in ("ratio", "tiles"):
        raise VedaError(f"{prefix}_budget_kind: unknown kind {kind!r}")
    return (value, None) if kind == "ratio" else (None, value)


def read_metadata(path):
    from safetensors import safe_open
    name = os.path.basename(path)
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = dict(f.metadata() or {})
    except Exception as exc:  # noqa: BLE001 - safetensors raises several types
        raise VedaError(f"{name} is not a readable safetensors file ({exc})") from exc
    if meta.get("format") != FORMAT:
        raise VedaError(f"{name} is not a Veda predictor (format {meta.get('format')!r}). Put only Veda predictor files in models/veda.")
    return meta


def load_bundle(path):
    """The predictor file, read strictly: every tensor must match its declared shape. Projections are held in bf16 on the
    host and follow q's device at call time, so the predictor does not pin VRAM while ComfyUI offloads the model."""
    from safetensors import safe_open
    meta, name = read_metadata(path), os.path.basename(path)
    try:
        layers, heads, dim = int(meta["num_layers"]), int(meta["num_heads"]), int(meta["head_dim"])
        keep = meta.get("keep_ratio")
        generated = _budget_of(meta, "target", None if keep is None else (float(keep), None))
        if generated is None:
            raise KeyError("target_budget_kind")
        reference = _budget_of(meta, "ref", generated)
        plan_json = json.loads(meta["plans"])
    except (KeyError, ValueError) as exc:
        raise VedaError(f"{name}: incomplete metadata ({exc})") from exc
    stored = meta.get("dtype", "float32")
    if stored not in _STORAGE:
        raise VedaError(f"{name}: unknown storage dtype {stored!r}")
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    shape, proj = (heads, 3 * dim, dim), {"proj_q": [], "proj_k": []}
    for layer in range(layers):
        for kind, out in proj.items():
            key = f"layers.{layer}.{kind}"
            value = tensors.pop(key, None)
            if value is None or tuple(value.shape) != shape:
                raise VedaError(f"{name}: {key} missing or not {list(shape)}")
            if stored == "float8_e4m3fn":
                scale = tensors.pop(key + ".__scale", None)
                if scale is None or tuple(scale.shape) != (heads,):
                    raise VedaError(f"{name}: {key} has no per-head fp8 scale")
                value = value.float() * scale.float().view(-1, 1, 1)
            out.append(value.to(torch.bfloat16).contiguous())
    if tensors:
        raise VedaError(f"{name}: unexpected tensors {sorted(tensors)[:4]}")
    plans = [_plan(p["geometry"], p["grid"], [tuple(int(x) for x in s.split("x")) for s in p["shapes"]], [list(r) for r in p["head_shape"]])
             for p in plan_json.values()]
    if not plans:
        raise VedaError(f"{name}: no tile plans")
    for p in plans:
        if len(p.head_shape) != layers or len(p.head_shape[0]) != heads:
            raise VedaError(f"{name}: plan {p.name} does not fit {layers} layers x {heads} heads")
    return SimpleNamespace(path=path, layers=layers, heads=heads, dim=dim, generated=generated, reference=reference,
                           plans=plans, proj_q=proj["proj_q"], proj_k=proj["proj_k"], meta=meta)


def describe_bundle(b):
    trained = describe_budget(b.generated) + ("" if b.reference == b.generated else f" / reference {describe_budget(b.reference)}")
    return f"{os.path.basename(b.path)}, {b.layers} layers x {b.heads} heads, trained {trained}"


class PhaseTimer:
    """Opt-in time per phase, summed over a run and read once at its end. CUDA events on a GPU (the work is asynchronous there);
    plain clock time elsewhere. `with timer("score"): ...`"""

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.spans = {}

    @contextlib.contextmanager
    def __call__(self, name):
        if self.cuda:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            yield
            b.record()
            self.spans.setdefault(name, []).append((a, b))
        else:
            t = time.perf_counter()
            yield
            self.spans.setdefault(name, []).append(time.perf_counter() - t)

    def totals(self):
        """{phase: milliseconds}; reads the device once."""
        if self.cuda:
            torch.cuda.synchronize()
            return {k: sum(a.elapsed_time(b) for a, b in v) for k, v in self.spans.items()}
        return {k: 1000.0 * sum(v) for k, v in self.spans.items()}


_NO_TIMER = lambda name: contextlib.nullcontext()


# ── one layer ───────────────────────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def attention(q, k, v, layer, spec, plan, bundle, budgets, attend, cache, stats, timer=_NO_TIMER):
    """One layer of Veda attention. q, k, v: [S, H, D] (post QK-norm and RoPE, any strides, any device).
    For each group of heads sharing a tile shape, in memory-bounded chunks: gather to tile order, pool, score, select,
    build the block mask, `attend(q_t, k_t, v_t, mask, layout)`, scatter back. Returns [S, H, D] in q's dtype.
    `cache` holds the tile layouts (keyed by geometry, shape and device); `stats` accumulates what was computed."""
    seq_len, heads, dim = q.shape
    dev = q.device
    generated, reference = budgets
    tile_refs = not keeps_all(reference)
    out = q.new_zeros(seq_len + 1, heads, dim)
    pq, pk = bundle.proj_q[layer], bundle.proj_k[layer]
    if dev.type == "cuda" and not pq.is_pinned():          # pinned host memory: the copy below does not stop the host
        try:
            pq = bundle.proj_q[layer] = pq.pin_memory()
            pk = bundle.proj_k[layer] = pk.pin_memory()
        except RuntimeError:
            pass
    pq, pk = pq.to(dev, non_blocking=True), pk.to(dev, non_blocking=True)
    for shape, group in head_groups(plan, layer, dev):
        key = (spec, shape, tile_refs, str(dev))
        if key not in cache:
            spans = [(s, g, least_padding_shape(g)) for s, g in spec[2]] if tile_refs else []
            cache[key] = build_layout(spans + [(spec[1][0], spec[1][1], shape)], seq_len, dev)
        lay = cache[key]
        blocks = column_blocks(lay, generated, reference)
        step = max(1, _CHUNK_BYTES // (lay.n_tiles * TILE * dim * q.element_size()))
        for hs in group.split(step):
            with timer("gather"):
                q_t, k_t = gather(q, lay, hs), gather(k, lay, hs)
            with timer("score"):
                scores = logits(pool(q_t, lay), pool(k_t, lay), pq.index_select(0, hs), pk.index_select(0, hs))
            with timer("select"):
                index, keep = select(scores, lay, blocks)
            del scores
            kept = keep.sum()
            stats["kept"] = stats.get("kept", 0) + kept
            stats["pairs"] = stats.get("pairs", 0) + len(hs) * lay.n_video ** 2
            stats["work"] = stats.get("work", 0) + kept + len(hs) * lay.n_global * (lay.n_tiles + lay.n_video)
            stats["dense"] = stats.get("dense", 0) + len(hs) * (seq_len / TILE) ** 2
            stats["global"] = lay.n_global
            with timer("select"):
                mask = block_mask(index, keep, lay)
            with timer("gather"):
                v_t = gather(v, lay, hs)
            with timer("kernel"):
                o_t = attend(q_t, k_t, v_t, mask, lay)
            with timer("scatter"):
                scatter_(out, o_t, lay, hs)
    return out[:seq_len]
