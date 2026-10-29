"""The LoRA loader.

Up to eight LoRAs in one node, applied in order. Slot 1 keeps the original input
names (`lora_name`, `strength_model`, `strength_clip`) so every saved pipeline
still loads; slots 2..8 add `lora_name_N`, `strength_model_N`, `strength_clip_N`
as optional inputs, which is backwards-compatible with every saved graph.
Chaining several nodes still works and gives the same result.

The slots are numbered combos rather than ComfyUI's autogrow input, for the same
reason the text encoder loader uses them: autogrow turns a Combo template into a
row of sockets, so the file picker would stop being a picker. v4 used a JSON list
in a string widget and a dict subclass that lied to the validator; neither is
needed, and every strength here is a real, validated input.
"""

import comfy.utils
import folder_paths
from comfy_api.latest import io

from ..._core import log, sysinfo
from .keys import match


NONE = "None"
MAX_LORAS = 8


class FunUILoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        files = [NONE] + folder_paths.get_filename_list("loras")
        inputs = [io.Model.Input("model")]
        for n in range(1, MAX_LORAS + 1):
            sfx = "" if n == 1 else f"_{n}"
            first = n == 1
            inputs += [
                io.Combo.Input(f"lora_name{sfx}", options=files, default=NONE, optional=not first,
                               tooltip="None leaves the model untouched." if first
                               else "Another LoRA, applied after the ones above it. None skips this slot."),
                io.Float.Input(f"strength_model{sfx}", default=1.0, min=-100.0, max=100.0, step=0.01, optional=not first,
                               tooltip="How strongly the LoRA changes the model. 0 = not at all, negative = the opposite."),
            ]
            if first:   # the original order: clip sits between slot 1's two strengths
                inputs.append(io.Clip.Input("clip", optional=True, tooltip="Leave unwired for model-only LoRAs."))
            inputs.append(io.Float.Input(f"strength_clip{sfx}", default=1.0, min=-100.0, max=100.0, step=0.01, optional=True,
                                         tooltip="How strongly the LoRA changes the text encoder. Only used when a clip is wired."))
        return io.Schema(
            node_id="FunUILoraLoader",
            display_name="FunUI LoRA Loader",
            category="FunUI/Loaders",
            description=f"Apply up to {MAX_LORAS} LoRAs to a model, and optionally its text encoder, in order.",
            inputs=inputs,
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name=NONE, strength_model=1.0, clip=None, strength_clip=1.0, **kw) -> io.NodeOutput:
        kw.update(lora_name=lora_name, strength_model=strength_model, strength_clip=strength_clip)
        notes = []
        for n in range(1, MAX_LORAS + 1):
            sfx = "" if n == 1 else f"_{n}"
            name = kw.get(f"lora_name{sfx}", NONE)
            if name in (None, NONE):
                if n == 1 and not any(kw.get(f"lora_name_{k}", NONE) not in (None, NONE) for k in range(2, MAX_LORAS + 1)):
                    notes.append("no LoRA picked")
                continue
            model, clip, note = _apply(model, clip, name, kw.get(f"strength_model{sfx}", 1.0),
                                       kw.get(f"strength_clip{sfx}", 1.0))
            notes.append(note)
        return io.NodeOutput(model, clip, "\n".join(notes))


def _apply(model, clip, lora_name, strength_model, strength_clip):
    """One LoRA onto (model, clip). Returns (model, clip, status line)."""
    if strength_model == 0 and (clip is None or strength_clip == 0):
        # Doing nothing is a valid request, but it must return the SAME objects
        # rather than clones: a clone here would silently drop patches a later
        # loader in the chain applied to the original.
        # Loud, because "applied a LoRA at strength 0" is indistinguishable
        # from "forgot to set the strength" by looking at the picture.
        log.alert("FunUI LoRA Loader", f"{lora_name} has both strengths at 0 and was not applied")
        return model, clip, f"{lora_name}: both strengths are 0, not applied"

    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora = comfy.utils.load_torch_file(path, safe_load=True)
    patches, how, dropped = match(model, clip, lora)
    if not patches:
        log.alert("FunUI LoRA Loader", f"{lora_name} matched NOTHING in this model, so it does nothing: "
                  "it is for another model, or names its layers in a way not known here")
        return model, clip, f"{lora_name}: matched nothing, not applied"
    patched_model = model.clone()
    patched_model.add_patches(patches, strength_model)
    patched_clip = None
    if clip is not None:
        patched_clip = clip.clone()
        patched_clip.add_patches(patches, strength_clip)
    if dropped:
        log.alert("FunUI LoRA Loader", f"{lora_name}: {dropped} of {len(patches) + dropped} weights do not fit this "
                  "model's shapes and were left out (trained on another variant of it)")

    applied = [f"model={strength_model}"]
    applied.append(f"clip={strength_clip}" if clip is not None else "clip=not wired")
    found = f"{len(patches)} weights" + (f", keys {how}" if how != "as-is" else "") + (f", {dropped} dropped" if dropped else "")
    log.info("FunUI LoRA Loader", f"{lora_name} applied ({', '.join(applied)}; {found}). {sysinfo.memory_line()}")
    return patched_model, patched_clip, f"FunUI LoRA Loader | {lora_name} ({', '.join(applied)}; {found})"
