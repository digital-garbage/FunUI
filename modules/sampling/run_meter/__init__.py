"""Run meter: numbers that tell two runs apart, for finding which setting made a picture worse.

It adds no model call. At every step it reads the model's own answer (the picture it predicts) and takes a few
picture-sized sums: how much fine grain the answer has across the picture and over time, and how much it still
changes from the previous step's answer. The sums stay on the card until the run is over, then one line per step
is logged and a small summary is saved. The next run with the same seed is compared with the saved one, so
flipping one setting and rendering again shows what it did to the grain and to the settling.

Grain is the energy of neighbour differences divided by the energy of the picture: smooth is low, a noise
pattern left in the picture is high. Settling is how far this step's answer moved from the previous one: a
schedule that is long enough has it falling towards zero by the last step. Only the picture is read, never the
sound, and nothing is ever changed.
"""

import json
import time

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import config, dit_hooks, log, streams

ID = "run_meter"
TITLE = "Run meter"
MOUNT = "generation.sampling"
STAGE = "post"
CATEGORY = "sampling"
STATUS = "experimental"

FOLDER = config.ROOT / "run_meter"
SAME = 0.02                                          # a change in a number under this share is not called a change

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Measure the run",
        "hint": "Logs how grainy the picture is and how much it changes each step, and compares with the previous run of the same seed, steps and size. No extra model calls.",
    },
}


def _rms(t):
    """Root mean square in float32, without a float copy of the whole tensor. A 0-d tensor on the same device."""
    return torch.linalg.vector_norm(t, dtype=torch.float32) / (t.numel() ** 0.5)


def measure(video, before=None):
    """(across, over_time, settle) of one picture [B, C, T, H, W] as 0-d tensors on its device.
    `over_time` is NaN for a single frame, `settle` NaN with no previous answer of the same shape."""
    v = video.detach()
    scale = _rms(v).clamp_min(1e-12)
    dx, dy = v[..., 1:] - v[..., :-1], v[..., 1:, :] - v[..., :-1, :]
    across = ((_rms(dx) ** 2 + _rms(dy) ** 2) / 2).sqrt() / scale
    nan = torch.full((), float("nan"), device=v.device)
    over = _rms(v[:, :, 1:] - v[:, :, :-1]) / scale if v.shape[2] > 1 else nan
    settle = _rms(v - before.to(device=v.device, dtype=v.dtype)) / scale if before is not None and before.shape == v.shape else nan
    return across, over, settle


def _value(t):
    x = float(t)
    return None if x != x else x


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001 -- nothing saved yet, or unreadable: nothing to compare with
        return None


def _change(now, then):
    if now is None or then is None or then <= 0:
        return "n/a"
    r = now / then - 1.0
    return "same" if abs(r) < SAME else f"{r:+.0%}"


def report(rows, seed, previous):
    """The log message for a finished run, and its summary. `rows` are the per-step dicts, in step order."""
    lines = []
    for r in rows:
        bits = [f"{r['h']}x{r['w']}", f"grain {r['across']:.3f} across"]
        if r["over_time"] is not None:
            bits.append(f"{r['over_time']:.3f} over time")
        bits.append("settling " + (f"{r['settle']:.3f}" if r["settle"] is not None else "n/a (first call at this size)"))
        lines.append(f"step {r['step'] + 1}/{r['total']}: " + ", ".join(bits))
    last = rows[-1]
    settles = [r["settle"] for r in rows if r["settle"] is not None]
    summary = {"seed": seed, "steps": last["total"], "size": [last["h"], last["w"]], "final": last,
               "settle_mean": sum(settles) / len(settles) if settles else None, "when": time.time()}
    if previous is None:
        lines.append("nothing saved to compare with yet")
    elif previous.get("seed") != seed or previous.get("steps") != summary["steps"] or previous.get("size") != summary["size"]:
        lines.append("previous run had another seed, step count or size: no comparison")
    else:
        was = previous["final"]
        lines.append("against the previous run of this seed: final grain across " + _change(last["across"], was["across"])
                     + ", over time " + _change(last["over_time"], was["over_time"])
                     + ", last settling " + _change(last["settle"], was["settle"]))
    return "\n".join(lines), summary


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    state = {"rows": {}, "which": None, "before": None, "seed": None}

    def reset():
        state.update(rows={}, which=None, before=None, seed=None)

    def outer(executor, *args, **kwargs):
        reset()
        state["seed"] = (list(args) + [None] * 8)[7]
        try:
            result = executor(*args, **kwargs)
        except BaseException:
            reset()                                  # a stopped run saves nothing: half a run is not comparable
            raise
        try:
            finish()
        except Exception as exc:                     # noqa: BLE001 -- the meter never costs the picture
            log.warning(TITLE, f"could not report: {exc}")
        finally:
            reset()                                  # latent-sized tensors must not outlive the run
        return result

    def finish():
        done = [dict(r, **{k: _value(r[k]) for k in ("across", "over_time", "settle")})
                for _s, r in sorted(state["rows"].items())]
        if not done:
            log.warning(TITLE, "no step could be read (the model's answer holds no picture)")
            return
        seed = state["seed"]
        seed = int(seed) if isinstance(seed, (int, float)) else None
        text, summary = report(done, seed, _load(FOLDER / "last.json"))
        log.info(TITLE, text)
        try:
            FOLDER.mkdir(parents=True, exist_ok=True)
            (FOLDER / "last.json").write_text(json.dumps(summary), encoding="utf-8")
        except OSError as exc:
            log.once("run_meter:save", log.WARNING, TITLE, f"could not save this run for the next comparison: {exc}")

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options")
        out = executor(x, t, *args, **kwargs)
        where, split = dit_hooks.current_step(to), streams.video_of(out, named)
        if where is None or split is None or dit_hooks.probing(to) or dit_hooks.weak_branch(to):
            return out
        video = split[0]
        which = tuple(int(v) for v in ((to or {}).get("cond_or_uncond") or ()))
        if state["which"] is None:
            state["which"] = which
        if which != state["which"]:
            return out                               # the other half of a guided step: one answer per step is enough
        if where[0] in state["rows"]:
            return out                               # a second call in the same step (a second-order sampler): the first one stands
        before = state["before"]
        across, over, settle = measure(video, before[1] if before is not None and where[0] >= before[0] else None)
        state["before"] = (where[0], video.detach())
        state["rows"][where[0]] = {"step": where[0], "total": where[1], "h": int(video.shape[-2]), "w": int(video.shape[-1]),
                                   "across": across, "over_time": over, "settle": settle}
        return out

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, f"{key}.run", outer)
    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return "grain and settling per step, compared with the previous run when it used the same seed, steps and size"


PROVIDES = {"modifier": install}
