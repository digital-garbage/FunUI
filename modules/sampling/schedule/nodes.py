"""FunUI Schedule: the sigmas a run walks down, by the chosen scheduler and step count.

A distilled model is made for one written-out list of noise levels, so that stays the default ("Distilled (official)").
Other step counts follow the same curve (the official anchors are resampled along their own shape), and any scheduler
ComfyUI knows can replace it, or LTX's own ("LTXV", which shifts the curve by the size of the picture)."""

import torch
from comfy_api.latest import io

import comfy.samplers

DISTILLED = "distilled (official)"
LTXV = "ltxv"
#: The distilled 8-step schedule, from the official ComfyUI LTX 2.x templates.
DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)


def scheduler_names():
    return [DISTILLED, LTXV, *comfy.samplers.SCHEDULER_NAMES]


def distilled(steps):
    """The official anchors for 8 steps as they are; any other count walks the same curve (linear between the anchors)."""
    anchors = torch.tensor(DISTILLED_SIGMAS)
    if steps == len(anchors) - 1:
        return anchors
    at = torch.linspace(0, len(anchors) - 1, steps + 1)
    low = at.floor().long().clamp(max=len(anchors) - 2)
    frac = at - low
    out = anchors[low] * (1 - frac) + anchors[low + 1] * frac
    out[-1] = 0.0
    return out


def ltxv(steps, latent):
    from comfy_extras.nodes_lt import LTXVScheduler
    if latent is not None and getattr(latent["samples"], "is_nested", False):
        latent = {"samples": latent["samples"].unbind()[0]}          # the picture's part: that is what sets the curve's shift
    return LTXVScheduler.execute(steps, 2.05, 0.95, True, 0.1, latent)[0]


def sigmas_for(scheduler, steps, model, latent):
    steps = max(1, int(steps))
    if scheduler == DISTILLED:
        return distilled(steps)
    if scheduler == LTXV:
        return ltxv(steps, latent)
    return comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()


class FunUISchedule(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        names = scheduler_names()
        return io.Schema(
            node_id="FunUISchedule",
            display_name="FunUI Schedule",
            category="FunUI/Sampling",
            description="The noise levels a run walks down: the model's own distilled list, LTX's, or any ComfyUI scheduler, at the chosen step count.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("latent", optional=True, tooltip="LTXV shifts its curve by the size of this latent."),
                io.Int.Input("steps", default=8, min=1, max=1000),
                io.Combo.Input("scheduler", options=names, default=DISTILLED),
            ],
            outputs=[io.Sigmas.Output(display_name="sigmas")],
        )

    @classmethod
    def execute(cls, model, steps: int, scheduler: str, latent=None) -> io.NodeOutput:
        return io.NodeOutput(sigmas_for(scheduler, steps, model, latent))
