"""Veda learned sparse attention for MiniMax H3, as the loader's SLA method "veda".

Veda (Han, Yang, Hu, Mei, Jiang, Qi; MIT; arXiv 2605.30325) ported from Veda-on-ComfyUI: see veda_core.py for
the method and VEDA-NOTICE.txt for the licences. EXPERIMENTAL and UNTESTED on a GPU: the pure-torch half is
tested on CPU, the Triton kernel (veda_kernel.py) only ever ran on the author's machine of origin.

It rides the same hook as SLA (`transformer_options["optimized_attention_override"]`, plus SLA's per-step wrapper
for the step counter, the dense steps and the rising share), so it needs no weight change and meets q/k/v only after
the LoRA-patched projections: LoRA, int8_convrot and MXFP8 (weights only) leave what it sees as bf16/fp16 q/k/v, which
is what the predictor was trained on. Calls it does not take (the text refiner, masked calls, short sequences, dense
steps, a head count the predictor was not made for) go to the chosen attention backend, and the log says how many
and why. When Veda cannot run at all, the loader says why and the backend is installed on its own.
"""
import collections
import logging

from . import sla_attention as sla, veda_core as core, veda_download

_NAMED = ("auto",)


def predictor_choices():
    try:
        return list(_NAMED) + veda_download.files()
    except Exception:  # noqa: BLE001 - no ComfyUI folders here (tests, dev server): a list of nothing is fine
        return list(_NAMED)


def _pick(name):
    """The predictor file's path, or (None, why). "auto": the T2VA file when it is there, else the first."""
    try:
        have = veda_download.files()
    except Exception:  # noqa: BLE001
        have = []
    if not have:
        return None, "no predictor in ComfyUI/models/veda (the Veda predictor row in the app offers the download)"
    if name in have:
        return veda_download.folder() / name, None
    if name not in ("", *_NAMED):
        return None, f"the predictor {name} is not in ComfyUI/models/veda"
    t2va = veda_download.MODELS[0]["file"]
    return veda_download.folder() / (t2va if t2va in have else have[0]), None


def _fold(veda):
    """The step that just ended: its work joins the run's totals (one read of the device per step, not per call), and
    its own kept share is what the summary reports, as SLA reports its last step."""
    stats = veda["stats"]
    if stats.get("dense"):
        veda["run"]["work"] += float(stats["work"])
        veda["run"]["dense"] += stats["dense"]
        veda["last"] = (float(stats["kept"]) / stats["pairs"] if stats.get("pairs") else 1.0, stats.get("global", 0))
    stats.clear()


def make_override(state, cfg, bundle, attend, dense_fn=None, dense_label=None):
    """The `optimized_attention_override`. `attend` is the kernel callable, or None for veda_kernel.attend (checked once
    per device, on the first call, against the fp32 reference). A kernel that fails costs speed, not the run."""
    import torch
    veda = state["veda"]
    ok_dtypes = (torch.bfloat16, torch.float16)

    def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense(why):
            state["dense"] += 1
            veda["why"][why] += 1
            run = (lambda *a, **kw: dense_fn(func, *a, **kw)) if dense_fn is not None else func
            return run(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                       skip_output_reshape=skip_output_reshape, **kwargs)

        if state["backend"] is None:
            state["backend"] = dense_label or getattr(func, "__name__", repr(func))
        to = kwargs.get("transformer_options") or {}
        layout, layer = to.get("minimax_h3_layout"), to.get("block_index")
        if mask is not None:
            return dense("masked call")
        if not skip_reshape or q.ndim != 4 or q.shape[0] != 1 or q.shape != k.shape or q.shape != v.shape:
            return dense("not H3's packed self-attention")
        if q.shape[2] < cfg["min_seq_len"]:
            return dense("shorter than min_seq_len")
        if layout is None or getattr(layout, "seq_len", None) != q.shape[2]:
            return dense("no layout for this call")
        if q.dtype not in ok_dtypes:
            return dense(f"{q.dtype} q/k/v")
        if to.get("_funui_sla_dense", False):
            return dense("dense step")
        if not isinstance(layer, int):
            layer = state["layer"]
        state["layer"] += 1
        if veda["step"] != state["step"]:
            _fold(veda)
            veda["step"] = state["step"]
        if layer >= bundle.layers:
            return dense("layer outside the predictor")
        if q.shape[1] != bundle.heads or q.shape[3] != bundle.dim:
            return dense(f"{q.shape[1]} heads of {q.shape[3]}, predictor has {bundle.heads} of {bundle.dim}")
        if state["failed"] is not None:
            return dense("kernel failed earlier")
        try:
            fn = attend
            if fn is None:
                from . import veda_kernel
                fn = veda_kernel.attend
                if str(q.device) not in veda["checked"]:
                    veda["checked"][str(q.device)] = veda_kernel.check(q.device)
                if veda["checked"][str(q.device)]:
                    raise RuntimeError(veda["checked"][str(q.device)])
            spec = veda["specs"].get(id(layout))
            if spec is None or spec[0] is not layout:
                spec = veda["specs"][id(layout)] = (layout, core.describe_layout(layout))
            spec = spec[1]
            plan = veda["plans"].get(spec[1][1])
            if plan is None:
                plan = veda["plans"][spec[1][1]] = core.choose_plan(bundle.plans, spec[1][1])
            if cfg["timing"] and veda["timer"] is None:
                veda["timer"] = core.PhaseTimer(q.device)
            with sla.pause_malloc_graph():          # the cached tile layouts outlive the call: made outside ComfyUI's compiler graph
                out = core.attention(q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1), layer, spec, plan[0], bundle,
                                     (bundle.generated, bundle.reference), fn, veda["tiles"], veda["stats"], veda["timer"] or core._NO_TIMER)
            veda["geometry"][spec[1][1]] = plan[2]          # every size this run met (Two stages meets two), and the plan each got
            state["calls"] += 1
            state["seq"] = q.shape[2]
            return out.transpose(0, 1).unsqueeze(0) if skip_output_reshape else out.reshape(1, q.shape[2], -1)
        except Exception as exc:  # noqa: BLE001 - a bad kernel must not kill the run
            state["failed"] = f"{exc.__class__.__name__}: {exc}"
            logging.warning("[FunUI Veda] sparse attention failed, dense from here this run: %s", state["failed"], exc_info=True)
            return dense("kernel failed")

    return override


def summarise(state, cfg):
    """One line per sampling run (the wrapper calls it when the run ends), then the per-run caches are let go."""
    veda, stats = state["veda"], state["veda"]["stats"]
    why = ", ".join(f"{n} {w}" for w, n in veda["why"].most_common())
    if state["calls"] == 0:
        logging.warning("[FunUI Veda] installed but never invoked: attention was NOT sparsified (%d dense calls%s). Check that the "
                        "model reaching the sampler is the one this loader produced.", state["dense"], f": {why}" if why else "")
    else:
        # Same fields in the same order as the "[FunUI SLA]" line, so the two read side by side. Tile pairs are the video x
        # video ones, counted over the whole run (SLA counts its last step); the global tiles (text, audio) are always exact.
        _fold(veda)
        sizes = veda["geometry"]
        done = min(1.0, veda["run"]["work"] / veda["run"]["dense"]) if veda["run"]["dense"] else 0.0
        kept, pinned = veda["last"]
        logging.info("[FunUI Veda] %s | %d calls | S=%d | video tile pairs %.1f%% kept in the last step (%.1f%% sparse, asked %s) | %d global tiles pinned "
                     "| %d dense fall-throughs%s | displaced %s | computed %.1f%% of full attention | video %s | %s",
                     veda["kernel"], state["calls"], state["seq"], kept * 100.0, (1.0 - kept) * 100.0, cfg["asked"], pinned,
                     state["dense"], f" ({why})" if why else "", state["backend"] or "?", done * 100.0, "; ".join(f"{core.describe_grid(g)}, tile plan: {how}" for g, how in sizes.items()),
                     cfg["predictor_name"])
        if veda["timer"] is not None:
            phases = veda["timer"].totals()
            logging.info("[FunUI Veda] time in each phase this run (ms): %s | total %.0f", ", ".join(f"{k} {v:.0f}" for k, v in phases.items()), sum(phases.values()))
        logging.info("[FunUI Veda] full attention on %s of %d", f"steps {sla._runs(state['dense_at'])}" if state["dense_at"] else "no step", state["n"])
        if state["failed"] is not None:
            logging.warning("[FunUI Veda] the kernel fell back to dense at least once: %s", state["failed"])
    stats.clear()
    veda["run"].update(work=0.0, dense=0.0)
    veda["step"] = None
    veda["tiles"].clear()
    veda["geometry"].clear()
    veda["timer"] = None
    veda["why"].clear()
    veda["specs"].clear()


def install_veda(model, cfg, dense_fn=None, dense_label=None, attend=None):
    """Give `model` Veda attention. Returns (model, status, installed), the contract of sla.install_sla: `installed` is
    False when Veda did not take, with the reason in the status, so the caller installs the chosen backend alone.
    `attend` replaces the Triton kernel (tests only)."""
    if attend is None and not sla.sla_available():
        return model, "SLA: veda needs CUDA and Triton (its kernel is Triton's), not here", False
    path, why = _pick(cfg.get("veda_predictor"))
    if path is None:
        return model, f"SLA: veda skipped, {why}", False
    try:
        bundle = core.load_bundle(str(path))
    except (core.VedaError, OSError) as exc:
        return model, f"SLA: veda skipped, {exc}", False
    unused = [n for n, on in (("sla_sparsity", float(cfg["sparsity_ratio"]) != sla.SLA_DEFAULTS["sparsity_ratio"]),
                              ("sla_sparsity_start", float(cfg["sparsity_start"]) > 0.0), ("sla_references", cfg["references"] != "off")) if on]
    cfg = {**cfg, "sparsity_ratio": sla.SLA_DEFAULTS["sparsity_ratio"], "sparsity_start": 0.0, "references": "off",
           "min_seq_len": int(cfg["min_seq_len"]), "dense_last_steps": int(cfg["dense_last_steps"]),
           "predictor_name": path.name, "engine": "veda", "summarise": summarise, "timing": bool(cfg.get("veda_timing"))}
    cfg["dense_steps"], bad = sla.parse_steps(cfg["dense_steps"])
    notes = [f"dense_steps: ignored {', '.join(bad)} (steps count from 1: '1' is the first, '1-2' the first two)"] if bad else []
    if unused:
        notes.append(f"{', '.join(unused)}: not used by veda, which runs at the predictor's trained budget")
    state = sla.new_state()
    state["veda"] = {"stats": {}, "tiles": {}, "specs": {}, "plans": {}, "checked": {}, "why": collections.Counter(), "geometry": {}, "timer": None, "step": None, "run": {"work": 0.0, "dense": 0.0}, "last": (1.0, 0), "kernel": "triton-int8" if attend is None else "stand-in kernel"}
    patched = sla.patching.clone(model)
    to = patched.model_options.get("transformer_options", {}).copy()
    to["optimized_attention_override"] = make_override(state, cfg, bundle, attend, dense_fn, dense_label)
    patched.model_options["transformer_options"] = to
    patched.add_wrapper_with_key("diffusion_model", "funui_sla_state", sla.make_wrapper(state, cfg))
    cfg["asked"] = f"trained ({core.describe_budget(bundle.generated)})"
    return patched, "\n".join([
        f"SLA on | veda (EXPERIMENTAL, Triton kernel untested on a GPU) | predictor {core.describe_bundle(bundle)} | sparsity {cfg['asked']} | "
        f"references {core.describe_budget(bundle.reference)} | min_seq_len={cfg['min_seq_len']} | full attention on steps "
        f"{sla._runs(i + 1 for i in cfg['dense_steps']) or 'none'} and the last {cfg['dense_last_steps']} | dense calls -> {dense_label or 'as launched'}",
        *(f"SLA: {n}" for n in notes)]), True
