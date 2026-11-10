"""Alternating prompts: what an `[A|B]` in the prompt (or `A|B` between two shortcuts) means for the run.

`[A|B]` gives the run two prompts, A and B, each holding only its own option. Three things use them:

* the main prompt, read on every step that has no option assigned: "A while B" (default), "A B", only A or only B;
With "Read A and B separately" on (off by default), two more things use them:

* the per-step fields: the steps that read only A and the steps that read only B, one model call per step as usual;
* Step merging: its merged steps answer A and B separately and mix the answers.

A step named in Step merging is answered by both options whatever is set here.
"""

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, streams

ID = "prompt_alternatives"
TITLE = "Alternating prompts"
MOUNT = "settings.general"
STAGE = "conditioning"
CATEGORY = "conditioning"
STATUS = "experimental"

#: What Prompt markup leaves in transformer_options: {"options": [whole conditioning per option], "cond_len"}.
SIMUL = "funui_simultaneous"

SETTINGS = {
    "separate": {
        "type": "bool", "default": False,
        "label": "Read A and B separately",
        "hint": "Lets steps read A or B alone, and Step merging answer both. Off: [A|B] only shapes the main prompt.",
    },
    "main": {
        "type": "enum", "default": "while", "options": [
            {"value": "while", "label": "A while B"},
            {"value": "space", "label": "A B"},
            {"value": "first", "label": "Only A"},
            {"value": "second", "label": "Only B"},
        ],
        "label": "Main prompt reads",
        "hint": "What the prompt says on steps that read neither option alone.",
    },
    "a_steps": {
        "type": "text", "default": "",
        "label": "Steps that read only A",
        "hint": "Step numbers from 1, e.g. 1-2 or 1,4. Needs [A|B] in the prompt and the switch above.",
        "when": {"separate": True},
    },
    "b_steps": {
        "type": "text", "default": "",
        "label": "Steps that read only B",
        "hint": "Step numbers from 1, e.g. 3-5. A step in both fields reads A.",
        "when": {"separate": True},
    },
}


def _say(message, level=log.ALERT):
    log.once(f"{ID}:{message}", level, "FunUI Alternating prompts", message)


def install(patcher, values, key):
    if not values.get("separate"):
        return None
    a, bad_a = dit_hooks.parse_blocks(values.get("a_steps"), 100000)
    b, bad_b = dit_hooks.parse_blocks(values.get("b_steps"), 100000)
    for p in bad_a + bad_b:
        _say(f"ignored step entry: {p}")
    steps = {**{i - 1: 1 for i in b if i >= 1}, **{i - 1: 0 for i in a if i >= 1}}      # step index -> option; A wins an overlap
    if not steps:
        return None
    live = {"cache": {}, "used": 0, "seen": False}

    def start(executor, *args, **kwargs):
        live["cache"], live["used"], live["seen"] = {}, 0, False
        try:
            return executor(*args, **kwargs)
        finally:
            live["cache"].clear()
            if not live["used"]:
                _say("Inactive | the prompt had no [A|B] options (a bare A|B counts only between two shortcuts)"
                     if not live["seen"] else
                     "Inactive | none of the steps named here is a step of this run")

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, start)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options") or {}
        where = dit_hooks.current_step(to)
        simul, context = to.get(SIMUL), named.get("c_crossattn")
        live["seen"] = live["seen"] or bool(simul)
        if (not simul or where is None or where[0] not in steps or to.get(streams.OPTION_PINNED)
                or dit_hooks.probing(to) or any(int(i) != 0 for i in (to.get("cond_or_uncond") or [0]))):
            return executor(x, t, *args, **kwargs)
        k = steps[where[0]]
        if not (isinstance(context, torch.Tensor) and context.dim() == 3
                and int(context.shape[1]) == simul["cond_len"] and k < len(simul["options"])):
            _say("Skipped | another feature changed the prompt's conditioning")
            return executor(x, t, *args, **kwargs)
        live["used"] += 1
        args_k, kwargs_k = streams.with_context(args, kwargs, streams.option_context(
            getattr(executor, "class_obj", None), simul["options"][k], context, live["cache"], k))
        return executor(x, t, *args_k, **kwargs_k)

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return f"A on steps {values.get('a_steps') or '-'}, B on steps {values.get('b_steps') or '-'}"


PROVIDES = {"modifier": install}
