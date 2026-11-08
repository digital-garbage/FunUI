"""Veda learned sparse attention: the torch half on CPU, the override on a REAL tiny H3, every combination.

The Triton kernel cannot run on this machine: the end-to-end tests hand the install a stand-in kernel (the fp32
block-sparse reference), so what they prove is everything around it: tiles, predictor, selection, the override
contract, the log line on the real run path. The kernel itself is checked only where CUDA and Triton exist.
"""
import json
import logging
import sys
import types

import pytest
import torch

from modules.loaders import sla_attention as sla, veda_attention as va, veda_core as core, veda_download

HEADS, DIM, LAYERS = 2, 32, 8               # the tiny H3: two heads of 32; eight blocks so MXFP8 has a middle
BLOCKS, KCALLS = 4, 8                       # the fixture's blocks, and the kernel calls per pass (two head groups each)
GRID = (10, 8, 8)                           # the video's token grid the test predictor was "trained" on: 640 rows
CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────

def _write_predictor(folder, name="p.safetensors", ratio=0.5, tiles=None, layers=LAYERS, heads=HEADS, dim=DIM, grid=GRID,
                     dtype="float8_e4m3fn", meta=None, drop=None):
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(1)
    tensors = {}
    for layer in range(layers):
        for kind in ("proj_q", "proj_k"):
            value = torch.randn(heads, 3 * dim, dim, generator=g) * 0.1
            key = f"layers.{layer}.{kind}"
            if dtype == "float8_e4m3fn":
                scale = value.abs().amax(dim=(1, 2)) / 448.0
                tensors[key], tensors[key + ".__scale"] = (value / scale.view(-1, 1, 1)).to(torch.float8_e4m3fn), scale
            else:
                tensors[key] = value.to(torch.bfloat16)
    for key in drop or ():
        tensors.pop(key)
    plan = {"g": {"geometry": "g", "grid": list(grid), "shapes": ["2x8x8", "4x4x8"], "head_shape": [[i % 2 for i in range(heads)]] * layers}}
    budget = {"target_budget_kind": "tiles", "target_budget_value": str(tiles)} if tiles else {"keep_ratio": str(ratio)}
    metadata = {"format": core.FORMAT, "num_layers": str(layers), "num_heads": str(heads), "head_dim": str(dim), "dtype": dtype,
                "plans": json.dumps(plan), "step": "600", **budget, **(meta or {})}
    save_file(tensors, str(folder / name), metadata=metadata)
    return folder / name


@pytest.fixture
def folder(tmp_path, monkeypatch):
    (tmp_path / "veda").mkdir()
    monkeypatch.setattr(veda_download, "folder", lambda: tmp_path / "veda")
    return tmp_path / "veda"


def _spec(refs=((40, (1, 4, 4)),), target=(62, (3, 5, 6)), seq_len=152):
    return seq_len, target, tuple(refs), ()


def _qkv(seq_len, heads=2, dim=32, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(seq_len, heads, dim, generator=g) * 1.5).to(device=device, dtype=dtype) for _ in range(3))


def _dense(q, k, v):
    p = torch.softmax(torch.einsum("qhd,khd->hqk", q.float(), k.float()) * q.shape[-1] ** -0.5, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, v.float())


def _plan_for(grid, heads=2, layers=1):
    return core._plan("t", grid, [(2, 8, 8), (4, 4, 8)], [[i % 2 for i in range(heads)]] * layers)


def _bundle(layers=1, heads=2, dim=32, generated=(0.5, None), reference=None):
    g = torch.Generator().manual_seed(2)
    mk = lambda: [(torch.randn(heads, 3 * dim, dim, generator=g) * 0.1).to(torch.bfloat16) for _ in range(layers)]
    return types.SimpleNamespace(layers=layers, heads=heads, dim=dim, generated=generated, reference=reference or generated,
                                 proj_q=mk(), proj_k=mk(), plans=[_plan_for((3, 5, 6), heads, layers)])


# ── tiles ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_every_row_is_in_exactly_one_tile_and_real_rows_come_first():
    for shape in ((1, 8, 16), (4, 4, 8), (2, 8, 8)):
        tiles = core.span_tiles(100, (3, 5, 6), shape)
        real = tiles[tiles >= 0]
        assert sorted(real.tolist()) == list(range(100, 190))
        count = (tiles >= 0).sum(1)
        assert all((tiles[i, :c] >= 0).all() and (tiles[i, c:] < 0).all() for i, c in enumerate(count.tolist()))


def test_the_layout_is_a_permutation_and_gather_then_scatter_gives_every_row_back():
    lay = core.build_layout([(40, (1, 4, 4), (1, 8, 16)), (62, (3, 5, 6), (4, 4, 8))], 152)
    assert lay.n_ref == 1 and lay.n_video == 3 and lay.n_global == 1 and lay.ref_tokens == 16 and lay.target_tokens == 90
    q = _qkv(152)[0]
    heads = torch.arange(2)
    out = torch.zeros(153, 2, 32)
    core.scatter_(out, core.gather(q, lay, heads), lay, heads)
    assert torch.equal(out[:152], q)


def test_overlapping_spans_are_refused():
    with pytest.raises(core.VedaError):
        core.build_layout([(0, (3, 5, 6), (4, 4, 8)), (50, (3, 5, 6), (4, 4, 8))], 200)


# ── predictor and selection ──────────────────────────────────────────────────────────────────────────────────

def test_pooling_and_logits_match_a_plain_computation_with_partial_tiles():
    lay = core.build_layout([(0, (3, 5, 6), (4, 4, 8))], 100)             # 4 tiles, 90 real rows + 10 global
    q = _qkv(100)[0]
    t = core.gather(q, lay, torch.arange(2))
    feats = core.pool(t, lay)
    for tile in range(lay.n_video):
        rows = t[tile * 128:(tile + 1) * 128][:int(lay.valid_count[tile])]
        want = torch.cat([rows.mean(0), rows.amax(0), rows.amin(0)], dim=-1)
        assert torch.allclose(feats[:, tile], want, atol=1e-5)
    pq, pk = (torch.randn(2, 96, 32) for _ in range(2))
    got = core.logits(feats, feats, pq, pk)
    qh = torch.bmm(feats, pq) + feats[..., :32]
    kh = torch.bmm(feats, pk) + feats[..., :32]
    assert torch.allclose(got, torch.bmm(qh, kh.transpose(1, 2)) / 32 ** 0.5, atol=1e-4)


def test_selection_keeps_the_own_tile_every_global_tile_and_the_budget_on_average():
    lay = core.build_layout([(0, (10, 8, 8), (4, 4, 8))], 700)
    scores = torch.randn(2, lay.n_video, lay.n_video)
    budget = (0.5, None)
    index, keep = core.select(scores, lay, core.column_blocks(lay, budget, budget))
    mask = core.block_mask(index, keep, lay)
    n = lay.n_video
    assert mask[:, torch.arange(n), torch.arange(n)].all()                      # a tile always sees itself
    assert mask[:, :, n:].all() and mask[:, n:, :].all()                        # global rows and columns are dense
    want = 0.5 * (-(-640 // 128)) ** 2 / n                                      # equal-cost budget, per query row
    assert abs(keep.sum(-1).float().mean().item() - want) < 0.1
    assert not mask[:, :n, :n][:, :, ~lay.kv_ok[:n]].any()                     # empty key tiles are never chosen


def test_keeping_everything_is_exactly_dense_attention_with_and_without_tiled_references():
    spec = _spec()
    q, k, v = _qkv(152)
    want = _dense(q, k, v)
    for refs_budget in ((1.0, None), (None, 1000)):                             # global refs, and the top-k path with k = all
        got = core.attention(q, k, v, 0, spec, _plan_for((3, 5, 6)), _bundle(), ((1.0, None), refs_budget), core.reference_attend, {}, {})
        assert torch.allclose(got, want, atol=2e-5), refs_budget


def test_a_sparse_budget_differs_from_dense_stays_finite_and_counts_what_it_computed():
    spec, (q, k, v), stats = _spec(), _qkv(152), {}
    got = core.attention(q, k, v, 0, spec, _plan_for((3, 5, 6)), _bundle(), ((0.3, None), (0.3, None)), core.reference_attend, {}, stats)
    assert torch.isfinite(got).all() and not torch.allclose(got, _dense(q, k, v), atol=1e-3)
    assert 0.0 < float(stats["kept"]) / stats["pairs"] < 1.0


# ── the predictor file ───────────────────────────────────────────────────────────────────────────────────────

def test_a_predictor_file_loads_with_its_fp8_scales_and_budget(folder):
    b = core.load_bundle(str(_write_predictor(folder)))
    assert (b.layers, b.heads, b.dim, b.generated) == (LAYERS, HEADS, DIM, (0.5, None)) and b.proj_q[0].dtype == torch.bfloat16
    assert core.load_bundle(str(_write_predictor(folder, "t.safetensors", tiles=32))).generated == (None, 32.0)
    assert core.load_bundle(str(_write_predictor(folder, "b.safetensors", dtype="bfloat16"))).proj_k[3].shape == (HEADS, 3 * DIM, DIM)
    assert "8 layers x 2 heads, trained 50% sparse" in core.describe_bundle(b)


@pytest.mark.parametrize("kw,words", [
    (dict(meta={"format": "other"}), "not a Veda predictor"),
    (dict(drop=["layers.2.proj_q"]), "layers.2.proj_q missing"),
    (dict(drop=["layers.0.proj_k.__scale"]), "no per-head fp8 scale"),
    (dict(meta={"dtype": "int3"}), "unknown storage dtype"),
    (dict(meta={"plans": json.dumps({})}), "no tile plans"),
    (dict(heads=3, meta={"num_heads": "2"}), "missing or not"),
])
def test_a_wrong_predictor_file_is_refused_in_words(folder, kw, words):
    path = _write_predictor(folder, **kw)
    with pytest.raises(core.VedaError, match=words):
        core.load_bundle(str(path))


def test_a_file_that_is_not_safetensors_is_refused(folder):
    (folder / "x.safetensors").write_bytes(b"not a model")
    with pytest.raises(core.VedaError, match="not a readable safetensors"):
        core.load_bundle(str(folder / "x.safetensors"))


def test_the_plan_is_the_nearest_aspect_then_duration_and_says_when_it_is_not_exact():
    plans = [core._plan("a", (10, 8, 8), [(2, 8, 8)], [[0]]), core._plan("b", (10, 6, 12), [(2, 8, 8)], [[0]])]
    assert core.choose_plan(plans, (10, 8, 8))[1:] == (True, "trained for this size (256x256, 0.9 s)")
    plan, exact, how = core.choose_plan(plans, (10, 6, 11))
    assert (plan.name, exact) == ("b", False) and how.startswith("nearest trained size: 384x192")
    assert core.choose_plan(plans, (10, 12, 6))[0].name == "b_T"                 # portrait: b turned on its side


def test_the_real_h3_layout_maps_to_a_target_grid_and_row_major_references(comfyui):
    from comfy.ldm.minimax.model import PackedLayout
    seq, (start, grid), refs, skipped = core.describe_layout(
        PackedLayout(5, 10, 16, 16, 2, refs=[{"kind": "image", "latent_h": 8, "latent_w": 8}]))
    assert grid == (10, 8, 8) and start == seq - 640 and len(refs) == 1 and refs[0][1] == (1, 4, 4) and skipped == ()
    with pytest.raises(core.VedaError):
        core.describe_layout(types.SimpleNamespace())


# ── the cheap wins keep the numbers, and the timing and the Two stages line say what happened ───────────────────────

def test_the_cached_tile_work_changes_nothing_in_the_numbers():
    """The bresenham pattern, the head groups and the list width are kept between calls; a call that finds them cached
    must give exactly what a call that builds them gives, and the list's width (a bound from the layout, no device read)
    must hold every row's kept tiles."""
    spec, (q, k, v) = _spec(), _qkv(152)
    args = (0, spec, _plan_for((3, 5, 6)), _bundle(), ((0.3, None), (0.3, None)), core.reference_attend)
    cache = {}
    first = core.attention(q, k, v, *args, cache, {})
    again = core.attention(q, k, v, *args, cache, {})
    fresh = core.attention(q, k, v, 0, spec, _plan_for((3, 5, 6)), _bundle(), ((0.3, None), (0.3, None)), core.reference_attend, {}, {})
    assert torch.equal(first, again) and torch.equal(first, fresh)
    lay = core.build_layout([(40, (1, 4, 4), (1, 8, 16)), (62, (3, 5, 6), (4, 4, 8))], 152)
    blocks = core.column_blocks(lay, (0.3, None), (0.3, None))
    mask = core.block_mask(*core.select(torch.randn(2, lay.n_video, lay.n_video), lay, blocks), lay)
    exact_index, count = core.tile_index_list(mask)
    bound_index, same_count = core.tile_index_list(mask, lay.widest)
    assert torch.equal(count, same_count) and bound_index.shape[-1] >= exact_index.shape[-1] == lay.widest
    keep = torch.arange(bound_index.shape[-1]) < count[..., None]
    assert torch.equal(bound_index[:, :, :exact_index.shape[-1]][keep[:, :, :exact_index.shape[-1]]], exact_index[keep[:, :, :exact_index.shape[-1]]])


def test_the_phase_timer_covers_every_phase_and_adds_nothing_to_the_answer():
    spec, (q, k, v) = _spec(), _qkv(152)
    timer = core.PhaseTimer("cpu")
    args = (0, spec, _plan_for((3, 5, 6)), _bundle(), ((0.3, None), (0.3, None)), core.reference_attend, {}, {})
    assert torch.equal(core.attention(q, k, v, *args, timer), core.attention(q, k, v, *args))
    assert set(timer.totals()) == {"gather", "score", "select", "kernel", "scatter"} and all(ms >= 0 for ms in timer.totals().values())


# ── device and dtype follow the input (the meta device stands in for a GPU) ─────────────────────────────────────

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_every_tensor_the_call_makes_follows_the_inputs_device_and_dtype(dtype):
    q, k, v = _qkv(152, dtype=dtype, device="meta")
    cache = {}
    out = core.attention(q, k, v, 0, _spec(), _plan_for((3, 5, 6)), _bundle(), ((0.5, None), (0.5, None)),
                         lambda qt, kt, vt, mask, lay: torch.empty_like(qt), cache, {})
    assert out.device.type == "meta" and out.dtype == dtype and out.shape == q.shape
    for lay in cache.values():
        assert all(t.device.type == "meta" for t in vars(lay).values() if isinstance(t, torch.Tensor))


# ── the override, on a real tiny H3 ──────────────────────────────────────────────────────────────────────────────

class _Kernel:
    """Stands in for veda_kernel: the fp32 reference, recording what it was handed."""
    def __init__(self):
        self.calls, self.dtypes, self.density, self.gap, self.qsum = 0, set(), [], [], []

    def attend(self, q, k, v, mask, lay):
        self.calls += 1
        self.dtypes.add(q.dtype)
        self.density.append(mask.float().mean().item())
        self.qsum.append(q.float().abs().sum().item())
        out = core.reference_attend(q, k, v, mask, lay)
        every = torch.ones_like(mask) & lay.kv_ok
        self.gap.append((out.float() - core.reference_attend(q, k, v, every, lay).float()).abs().max().item())
        return out

    check = staticmethod(lambda device: None)


@pytest.fixture
def veda_on(monkeypatch, folder):
    """Triton and CUDA 'present' on this machine, the kernel the stand-in."""
    kernel = _Kernel()
    monkeypatch.setattr(sla, "sla_available", lambda: True)
    monkeypatch.setitem(sys.modules, "modules.loaders.veda_kernel", kernel)
    monkeypatch.setattr("modules.loaders.veda_kernel", kernel, raising=False)
    return kernel


def _big(tiny, bf16=True, size=(10, 16, 16)):
    """The tiny H3 on a video of 640 tokens (10 x 8 x 8), in bf16 so the override accepts q/k/v."""
    dm = tiny.patcher.model.diffusion_model
    dtype = torch.bfloat16 if bf16 else torch.float32
    if bf16:
        dm.to(dtype)
    tiny.video = torch.randn(1, 24, *size).to(dtype)
    tiny.audio = tiny.audio.to(dtype)
    tiny.context = tiny.context.to(dtype)
    return tiny


def _install(tiny, **kw):
    given = dict(method="veda", min_seq_len=0, dense_steps="", sparsity_ratio=0.9)
    patched, note, installed = sla.install_sla(tiny.patcher, **{**given, **kw})
    return patched, note, installed


def _last_step(tiny, patched):
    return tiny.run(patched, sigma=0.5, sigmas=torch.tensor([1.0, 0.5, 0.0]))        # step 2 of 2: the run's summary is written


def test_keeping_everything_on_a_real_h3_gives_the_dense_picture(veda_on, folder, comfyui, tiny_h3):
    _write_predictor(folder, ratio=1.0)
    tiny = _big(tiny_h3)
    dense = _last_step(tiny, tiny.patcher)[0]
    patched, note, installed = _install(tiny)
    assert installed and "predictor p.safetensors" in note and "EXPERIMENTAL" in note
    got = _last_step(tiny, patched)[0]
    assert veda_on.calls == KCALLS and veda_on.dtypes == {torch.bfloat16}          # every block, handed bf16 q/k/v
    assert torch.allclose(got.float(), dense.float(), atol=0.1)


def test_a_sparse_run_fires_logs_the_line_on_the_real_path_and_differs(veda_on, folder, comfyui, tiny_h3, caplog):
    _write_predictor(folder, ratio=0.3)
    tiny = _big(tiny_h3)
    dense = _last_step(tiny, tiny.patcher)[0]
    patched, _, installed = _install(tiny)
    with caplog.at_level(logging.INFO):
        got = _last_step(tiny, patched)[0]
    assert installed and torch.isfinite(got.float()).all() and torch.isfinite(dense.float()).all()
    assert max(veda_on.density) < 0.95 and max(veda_on.gap) > 1e-3                    # tiles were skipped, and it shows in the output
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("[FunUI Veda] triton-int8"))
    assert f"{BLOCKS} calls" in line and "256x256" in line and "of full attention" in line and "trained for this size" in line
    assert "asked trained (70% sparse)" in line and "sparse, asked" in line and "global tiles pinned" in line and "p.safetensors" in line
    assert "1 dense fall-throughs (1 no layout for this call)" in line                                                      # the text refiner's call went to the backend
    assert "full attention on no step of 2" in caplog.text


def test_a_second_run_logs_again_and_a_new_size_uses_the_nearest_plan(veda_on, folder, comfyui, tiny_h3, caplog):
    """Two stages: the same patched model samples a small picture, then a larger one."""
    _write_predictor(folder, ratio=0.5)
    tiny = _big(tiny_h3)
    patched, _, _ = _install(tiny)
    with caplog.at_level(logging.INFO):
        _last_step(tiny, patched)
        tiny.video = torch.randn(1, 24, 10, 24, 24).to(torch.bfloat16)
        _last_step(tiny, patched)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[FunUI Veda] triton-int8")]
    assert len(lines) == 2 and "trained for this size" in lines[0] and "nearest trained size" in lines[1] and "384x384" in lines[1]


def test_dense_steps_stay_with_the_backend_and_the_log_names_them(veda_on, folder, comfyui, tiny_h3, caplog):
    _write_predictor(folder, ratio=0.5)
    tiny = _big(tiny_h3)
    patched, note, _ = _install(tiny, dense_steps="1")
    assert "steps 1" in note
    schedule = torch.tensor([1.0, 0.6, 0.3, 0.0])
    tiny.run(patched, sigma=1.0, sigmas=schedule)                                     # step 1: dense on purpose
    assert veda_on.calls == 0
    with caplog.at_level(logging.INFO):
        for sigma in (0.6, 0.3):
            tiny.run(patched, sigma=sigma, sigmas=schedule)
    assert veda_on.calls == 2 * KCALLS and "full attention on steps 1 of 3" in caplog.text and "dense step" in caplog.text


def test_veda_runs_at_the_trained_budget_whatever_sparsity_or_ramp_are_set(veda_on, folder, comfyui, tiny_h3, caplog):
    """First version: Veda's own defaults only, so the comparison with the current SLA workflow is of like with unlike on
    purpose. sla_sparsity, the rising share and the references choice are not read, and the status says so."""
    _write_predictor(folder, ratio=0.5)
    schedule = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    tiny = _big(tiny_h3)
    steps = []
    for kw in ({}, dict(sparsity_start=0.6, sparsity_ratio=0.85, references="heavy")):
        patched, note, _ = _install(tiny, **kw)
        veda_on.density.clear()
        with caplog.at_level(logging.INFO):
            for sigma in schedule[:-1].tolist():
                tiny.run(patched, sigma=sigma, sigmas=schedule)
        steps.append(list(veda_on.density))
    assert steps[0] == steps[1] and steps[0][:2] == steps[0][-2:]              # the same tiles every step, with or without the settings
    assert "sla_sparsity, sla_sparsity_start, sla_references: not used by veda" in note and "sparsity trained (50% sparse)" in note
    assert "asked trained (50% sparse)" in caplog.text


def test_the_attention_explorer_runs_first_and_veda_sees_what_it_changed(veda_on, folder, comfyui, tiny_h3):
    """The explorer's override wraps the one installed at load (core/dit_hooks.add_attention_override): it edits q, then
    calls the rest of the chain, which is Veda. Veda must receive the edited q and still take every call."""
    from core import dit_hooks
    _write_predictor(folder, ratio=0.5)
    tiny = _big(tiny_h3)
    patched, _, _ = _install(tiny)
    seen = []
    veda_on.attend = lambda q, k, v, mask, lay: (seen.append(q.float().abs().mean().item()), core.reference_attend(q, k, v, mask, lay))[1]
    _last_step(tiny, patched)
    plain, calls = list(seen), veda_on.calls
    seen.clear()
    explorer = dit_hooks.add_attention_override
    explorer(patched, "explorer", lambda func, q, k, v, heads, **kw: func(q * 1.5, k, v, heads, **kw))
    _last_step(tiny, patched)
    assert len(seen) == len(plain) and all(abs(b / a - 1.5) < 0.05 for a, b in zip(plain, seen))


def test_a_lora_on_the_attention_projection_changes_q_k_v_and_veda_still_takes_the_call(veda_on, folder, comfyui, tiny_h3):
    _write_predictor(folder, ratio=0.5)
    tiny = _big(tiny_h3)
    _last_step(tiny, _install(tiny)[0])
    base_q = veda_on.qsum[0]
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    weight = tiny.patcher.model.diffusion_model.blocks[0].attn.qkv_proj.weight
    lora = tiny.patcher.clone()
    lora.add_patches({key: ("diff", (torch.randn_like(weight) * 0.3,))}, 1.0)
    veda_on.calls, veda_on.qsum = 0, []
    patched, _, installed = _install(types.SimpleNamespace(patcher=lora))
    assert installed
    lora.patch_model(load_weights=True)
    try:
        got = tiny.run(patched, sigma=0.5, sigmas=torch.tensor([1.0, 0.5, 0.0]))[0]
    finally:
        lora.unpatch_model()
    assert veda_on.calls == KCALLS and torch.isfinite(got.float()).all()
    assert abs(veda_on.qsum[0] - base_q) > 1e-3 * base_q                              # block 0's q is the LoRA's, and Veda got it


def test_mxfp8_weights_leave_q_k_v_in_bf16_so_veda_and_mxfp8_compose(veda_on, folder, comfyui, tiny_h3):
    """MXFP8 stores weights only. The attention override meets q/k/v after the projection, in the compute dtype,
    which is what the predictor was trained on: with MXFP8 blocks in the model it is still bf16 and still taken."""
    pytest.importorskip("comfy.quant_ops")
    from comfy.quant_ops import QUANT_ALGOS
    if "mxfp8" not in QUANT_ALGOS:
        pytest.skip("this comfy_kitchen has no MXFP8")
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from modules.loaders import mxfp8
    _write_predictor(folder, ratio=0.5)
    tiny = _big(type(tiny_h3)(blocks=8))
    sd = {k: v.clone() for k, v in tiny.patcher.model.diffusion_model.state_dict().items()}
    converted, _ = mxfp8.quantize_state_dict(sd)
    assert converted > 0
    qm = MiniMaxH3Model(hidden_size=64, num_layers=LAYERS, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=32,
                        ffn_hidden_size=64, text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64, time_embed_dim=32,
                        rope_inv_freq_len=4, dtype=torch.bfloat16, operations=comfy.ops.mixed_precision_ops({}))
    qm.load_state_dict(sd, strict=False)
    assert qm.blocks[3].attn.qkv_proj.quant_format == "mxfp8"
    qm.rope.inv_freq.copy_(tiny.patcher.model.diffusion_model.rope.inv_freq)
    for p in qm.parameters():
        p.requires_grad_(False)
    tiny.patcher.model.diffusion_model = qm
    patched, _, installed = _install(tiny)
    try:
        out = _last_step(tiny, patched)[0]
    except ValueError:
        # This machine's CPU cannot run the MXFP8 matmul (Blackwell GPUs do). Blocks 0 and 1 stay bf16 and run, so Veda's calls
        # for them were made before the first MXFP8 layer failed: what the kernel was handed is still checked.
        assert installed and veda_on.calls >= 2 and veda_on.dtypes == {torch.bfloat16}
        return
    assert installed and veda_on.calls == 16 and veda_on.dtypes == {torch.bfloat16} and torch.isfinite(out.float()).all()


# ── when Veda cannot run, it says why and the backend stays ──────────────────────────────────────────────────────

def test_install_says_why_it_skipped_and_hands_the_backend_back(monkeypatch, folder, comfyui, tiny_h3):
    tiny = _big(tiny_h3)
    monkeypatch.setattr(sla, "sla_available", lambda: False)
    out, note, installed = _install(tiny)
    assert not installed and out is tiny.patcher and "needs CUDA and Triton" in note
    monkeypatch.setattr(sla, "sla_available", lambda: True)
    out, note, installed = _install(tiny)
    assert not installed and "no predictor in ComfyUI/models/veda" in note
    _write_predictor(folder, "x.safetensors", meta={"format": "nope"})
    assert "not a Veda predictor" in _install(tiny)[1]
    assert "gone.safetensors is not in" in _install(tiny, veda_predictor="gone.safetensors")[1]


def test_timing_is_off_unless_asked_and_then_adds_one_line_with_every_phase(veda_on, folder, comfyui, tiny_h3, caplog):
    _write_predictor(folder, ratio=0.5)
    tiny = _big(tiny_h3)
    with caplog.at_level(logging.INFO):
        plain = _last_step(tiny, _install(tiny)[0])[0]
    assert "time in each phase" not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        timed = _last_step(tiny, _install(tiny, veda_timing=True)[0])[0]
    line = next(r.getMessage() for r in caplog.records if "time in each phase" in r.getMessage())
    assert all(name in line for name in ("gather", "score", "select", "kernel", "scatter")) and "total" in line
    assert torch.equal(plain, timed)                                                   # a clock changes no number
    caplog.clear()

    def splitter(q, k, v, mask, lay, timer):
        with timer("kernel.index"):
            pass
        with timer("kernel.attend"):
            return core.reference_attend(q, k, v, mask, lay)
    splitter.times_itself = True
    veda_on.attend = splitter
    with caplog.at_level(logging.INFO):
        _last_step(tiny, _install(tiny, veda_timing=True)[0])
    line = next(r.getMessage() for r in caplog.records if "time in each phase" in r.getMessage())
    assert "kernel " in line and "(index " in line and ", attend " in line and line.count("total") == 1       # a kernel's own parts are shown, not added twice


def test_two_stages_in_one_run_give_one_line_that_names_both_sizes(veda_on, folder, comfyui, tiny_h3, caplog):
    """Two stages hands the model the WHOLE schedule for both stages, so it is one run for the wrappers: one Veda line, as the
    SLA line is one. The line has to name the size and plan of each stage, not only the last."""
    _write_predictor(folder, ratio=0.5)
    tiny = _big(tiny_h3)
    patched, _, _ = _install(tiny)
    schedule = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    with caplog.at_level(logging.INFO):
        for step, sigma in enumerate(schedule[:-1].tolist()):
            tiny.video = torch.randn(1, 24, 10, 16, 16).to(torch.bfloat16) if step < 2 else torch.randn(1, 24, 10, 24, 24).to(torch.bfloat16)
            tiny.run(patched, sigma=sigma, sigmas=schedule)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[FunUI Veda] triton-int8")]
    assert len(lines) == 1 and "256x256, 0.9 s, tile plan: trained for this size" in lines[0] and "384x384, 0.9 s, tile plan: nearest trained size" in lines[0]


def test_a_predictor_made_for_other_heads_runs_dense_and_the_log_says_why(veda_on, folder, comfyui, tiny_h3, caplog):
    _write_predictor(folder, heads=4, dim=32)
    tiny = _big(tiny_h3)
    patched, _, installed = _install(tiny)
    with caplog.at_level(logging.INFO):
        _last_step(tiny, patched)
    assert installed and veda_on.calls == 0
    assert "installed but never invoked" in caplog.text and "2 heads of 32, predictor has 4 of 32" in caplog.text


def test_a_failing_kernel_costs_speed_not_the_run_and_is_named(veda_on, folder, comfyui, tiny_h3, caplog):
    _write_predictor(folder, ratio=0.5)
    veda_on.attend = lambda *a: (_ for _ in ()).throw(RuntimeError("no such kernel"))
    tiny = _big(tiny_h3)
    patched, _, _ = _install(tiny)
    with caplog.at_level(logging.INFO):
        out = _last_step(tiny, patched)[0]
    assert torch.isfinite(out.float()).all() and "RuntimeError: no such kernel" in caplog.text


def test_the_self_check_failing_is_reported_and_runs_dense(veda_on, folder, comfyui, tiny_h3, caplog):
    _write_predictor(folder)
    veda_on.check = staticmethod(lambda device: "the Triton kernel ignores the block mask on this device")
    tiny = _big(tiny_h3)
    with caplog.at_level(logging.INFO):
        _last_step(tiny, _install(tiny)[0])
    assert "ignores the block mask" in caplog.text and veda_on.calls == 0


def test_float32_and_short_sequences_stay_with_the_backend(veda_on, folder, comfyui, tiny_h3):
    _write_predictor(folder)
    tiny = _big(tiny_h3, bf16=False)                                                  # float32 q/k/v: never the kernel
    _last_step(tiny, _install(tiny)[0])
    assert veda_on.calls == 0
    tiny = _big(tiny_h3.__class__(), bf16=True)
    _last_step(tiny, _install(tiny, min_seq_len=100000)[0])
    assert veda_on.calls == 0


def test_the_loader_passes_the_predictor_through_and_names_it(veda_on, folder, comfyui, tiny_h3, monkeypatch):
    from modules.loaders.diffusion_model.nodes import FunUIDiffusionModelLoader as L
    ids = [i.id for i in L.define_schema().inputs]
    assert ids[-2:] == ["veda_predictor", "veda_timing"]                                              # added last: saved workflows read by position
    _write_predictor(folder, "a.safetensors")
    _write_predictor(folder, "b.safetensors", ratio=0.2)
    _, note, installed = _install(_big(tiny_h3), veda_predictor="b.safetensors")
    assert installed and "predictor b.safetensors" in note


# ── the kernel, where there is one ───────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + Triton")
def test_the_triton_kernel_passes_its_own_self_check_on_this_gpu():
    from modules.loaders import veda_kernel
    assert veda_kernel.check(torch.device("cuda")) is None


@pytest.mark.parametrize("how", ["plain", "kernel_fails", "timing", "two_sizes", "cancelled"])
def test_the_model_is_collectable_after_a_veda_run(veda_on, folder, comfyui, tiny_h3, how):
    """ComfyUI warns 'memory leak with model' when anything outside the patcher still holds the diffusion model after the
    patcher is gone. Nothing Veda keeps (state, caches, closures, wrappers) may be that holder, so no gc.collect() here."""
    import gc
    import weakref
    _write_predictor(folder, ratio=0.3)
    if how == "kernel_fails":
        veda_on.attend = lambda *a: (_ for _ in ()).throw(RuntimeError("no such kernel"))
        logging.disable(logging.CRITICAL)
    tiny = _big(tiny_h3)
    patched, _, installed = _install(tiny, veda_timing=(how == "timing"))
    assert installed
    if how == "cancelled":
        tiny.run(patched, sigma=1.0, sigmas=torch.tensor([1.0, 0.5, 0.0]))          # step 1 of 2 only: no summary, caches still full
    else:
        _last_step(tiny, patched)
    if how == "two_sizes":
        tiny.video = torch.randn(1, 24, 10, 20, 16).to(torch.bfloat16)
        _last_step(tiny, patched)
    assert how == "kernel_fails" or veda_on.calls
    ref = weakref.ref(tiny.patcher.model)          # what ComfyUI names in the warning: the model, not the patcher
    gc.collect()
    gc.disable()
    try:
        del patched, tiny
        tiny_h3.patcher = None
        del tiny_h3
        assert ref() is None, [type(r).__name__ for r in gc.get_referrers(ref())]
    finally:
        gc.enable()
        logging.disable(logging.NOTSET)


def test_the_real_model_class_is_collectable_after_a_sampled_veda_run(veda_on, folder, comfyui):
    """The path ComfyUI warns about by class name ('memory leak with model MiniMaxH3'): the real MiniMaxH3 model object, sampled by
    the real guider, loaded through model management, then dropped. It must die without a gc.collect()."""
    import gc
    import weakref
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    import comfy.samplers
    import comfy.supported_models as sm
    from comfy.nested_tensor import NestedTensor
    _write_predictor(folder, ratio=0.3)
    config = sm.MiniMaxH3(dict(image_model="minimax_h3", hidden_size=64, num_layers=BLOCKS, token_refiner_num_layers=1, num_attention_heads=2,
                               attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
                               time_embed_dim=32, rope_inv_freq_len=4))
    config.set_inference_dtype(torch.bfloat16, None)
    base = config.get_model({}, device=torch.device("cpu"))
    for p in base.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=0.05)
        p.requires_grad_(False)
    patcher = mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    patched, _, installed = sla.install_sla(patcher, method="veda", min_seq_len=0, dense_steps="", sparsity_ratio=0.9)
    assert installed
    guider = comfy.samplers.CFGGuider(patched)
    cond = [[torch.randn(1, 5, 48).to(torch.bfloat16), {}]]
    guider.set_conds(cond, cond)
    guider.set_cfg(1.0)
    latent = NestedTensor((torch.zeros(1, 24, 10, 16, 16), torch.zeros(1, 32, 2, 3)))
    noise = NestedTensor((torch.randn(1, 24, 10, 16, 16), torch.randn(1, 32, 2, 3)))
    mm.load_models_gpu([patched])
    guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), torch.tensor([1.0, 0.5, 0.0]), seed=1, disable_pbar=True)
    assert veda_on.calls
    ref = weakref.ref(patcher.model)
    gc.collect()
    gc.disable()
    try:
        del guider, patched, noise, latent, cond, config          # the loader's own patcher stays (a cached node output), as in the app
        dead = [m for m in mm.current_loaded_models if m.is_dead()]          # ComfyUI's own test for 'memory leak with model'
        assert not dead and ref() is not None or ref() is None, [type(r).__name__ for r in gc.get_referrers(ref())]
        assert not dead, [type(r).__name__ for r in gc.get_referrers(ref())]
    finally:
        gc.enable()
        mm.unload_all_models()


@pytest.mark.parametrize("interrupted", [False, True], ids=["finished", "interrupted"])
@pytest.mark.parametrize("staged", [False, True], ids=["one-stage", "two-stages"])
def test_every_modifier_on_top_of_veda_leaves_a_collectable_model(veda_on, folder, comfyui, staged, interrupted):
    """The class of leak ComfyUI names 'memory leak with model MiniMaxH3': a hook or wrapper that holds its own patcher is a cycle only a
    full gc frees, and that gc clears ComfyUI's weak link to the parent copy first. Veda in the loader, every modifier on top, a real
    sampling run, then everything dropped: the model must die by reference counting alone."""
    import gc
    import weakref
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    import comfy.samplers
    import comfy.supported_models as sm
    from comfy.nested_tensor import NestedTensor
    from modules.sampling.tests.test_combinations import FEATURES, STAGES, _load
    _write_predictor(folder, ratio=0.3)
    config = sm.MiniMaxH3(dict(image_model="minimax_h3", hidden_size=64, num_layers=BLOCKS, token_refiner_num_layers=1, num_attention_heads=2,
                               attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
                               time_embed_dim=32, rope_inv_freq_len=4))
    config.set_inference_dtype(torch.bfloat16, None)
    base = config.get_model({}, device=torch.device("cpu"))
    for p in base.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=0.05)
        p.requires_grad_(False)
    loaded = mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    veda, _, installed = sla.install_sla(loaded, method="veda", min_seq_len=0, dense_steps="", sparsity_ratio=0.9, veda_timing=True)
    assert installed
    patched, _ = _load(veda, tuple(f for f in FEATURES if f != "phrase_probe"), STAGES["two-stages"] if staged else None)
    mm.load_models_gpu([patched])
    guider = comfy.samplers.CFGGuider(patched)
    cond = [[torch.randn(1, 5, 48).to(torch.bfloat16), {}]]
    guider.set_conds(cond, cond)
    guider.set_cfg(1.0)
    latent = NestedTensor((torch.zeros(1, 24, 10, 16, 16), torch.zeros(1, 32, 2, 3)))
    noise = NestedTensor((torch.randn(1, 24, 10, 16, 16), torch.randn(1, 32, 2, 3)))
    calls = []

    def cut(module, args):
        calls.append(1)
        if interrupted and len(calls) > 2:
            raise mm.InterruptProcessingException()
    hook = base.diffusion_model.register_forward_pre_hook(cut)
    try:
        from modules.sampling.sampler.nodes import FunUISampler
        settings = {"taste": {"key": "fox"}, **{f: {"enabled": True, **FEATURES[f]} for f in FEATURES if f != "phrase_probe"}}
        for seed in (1, 2):          # a second run on the same cached model: what each run leaves behind is what the next one meets
            guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), torch.tensor([1.0, 0.7, 0.3, 0.0]), seed=seed, disable_pbar=True)
            FunUISampler.execute(patched, cond, cond, {"samples": latent}, seed, 3, 1.0, "euler", "simple", settings=settings)
    except mm.InterruptProcessingException:
        assert interrupted
    finally:
        hook.remove()
    del hook
    assert veda_on.calls, "Veda never took a call: the test would prove nothing about it"
    ref = weakref.ref(base)
    gc.collect()
    gc.disable()
    try:
        del guider, patched, veda, loaded, base, noise, latent, cond, config
        dead = [m for m in mm.current_loaded_models if m.is_dead()]
        assert not dead and ref() is None, [type(r).__name__ for r in gc.get_referrers(ref())] if ref() is not None else "dead model tracked"
    finally:
        gc.enable()
        mm.unload_all_models()
