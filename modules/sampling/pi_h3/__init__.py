"""pi-H3: MiniMax H3 in 2 or 4 steps, from Lakonik's distilled adapters, with no CFG.

A pi-H3 file is not a plain LoRA (see adapter.py). Switching this on puts it on the model, sets the video and audio shifts
the file was trained with, and runs Euler over the file's own 2 or 4 noise levels, whatever sampler and steps are chosen.
The picture is decoded by the VAE as usual; a DWD file's last step writes the pixels itself and the VAE decode is skipped.

Ported from Hansheng Chen's pi-H3 and ComfyUI-piFlow (Apache-2.0); see the README Thank you section.
Not yet run on a real H3 here.
"""

import copy
import os
import time
import uuid
from pathlib import Path

import folder_paths
from comfy.patcher_extension import WrappersMP

from ..._core import control, log, patching

from . import adapter as ad
from . import dwd

ID = "pi_h3"
TITLE = "pi-H3 few-step"
MOUNT = "generation.sampling"
STAGE = "load"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["audio_stream", "adaln_modalities"]

if "pi_h3" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("pi_h3", str(Path(folder_paths.models_dir) / "pi_h3"))

CONTROLLER_KEY = "funui_pi_h3"


def _files():
    try:
        return [f for f in folder_paths.get_filename_list("pi_h3") if f.endswith(".safetensors")]
    except Exception:          # noqa: BLE001 -- no folder yet is the same as no files
        return []


def _options():
    return [{"value": f, "label": f} for f in _files()] or [{"value": "", "label": "No file in models/pi_h3"}]


_FOUND = _files()

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Few-step pi-H3",
        "hint": "Renders in the adapter's own 2 or 4 steps without CFG. Use the FL2VA base it was trained on, with SLA and turbo LoRAs off.",
    },
    "adapter": {
        "type": "enum", "default": _FOUND[0] if _FOUND else "",
        "options": _options(),
        "label": "pi-H3 adapter file",
        "hint": "A pi_h3 2 or 4 step adapter (DWD or not) from ComfyUI/models/pi_h3. DWD files skip the video VAE decode. Read again each time Settings opens.",
        "when": {"enabled": True},
    },
    "steps": {
        "type": "enum", "default": "auto",
        "options": [{"value": "auto", "label": "From the file name"}, {"value": "2", "label": "2 steps"}, {"value": "4", "label": "4 steps"}],
        "label": "Steps",
        "hint": "How many steps to run. Use the count the file was trained for; the file name says it.",
        "when": {"enabled": True},
    },
}

# One parsed file is kept between runs, so a new seed or prompt does not read the file again. It holds no model.
_kept = {}


def _adapter(path, label):
    stamp = (str(path), os.path.getmtime(path), os.path.getsize(path))
    if _kept.get("stamp") != stamp:
        _kept.clear()
        _kept.update(stamp=stamp, adapter=ad.parse(path, label=label))
    return _kept["adapter"]


def _swap(steps):
    """Run Euler over the file's own noise levels, in place of the sampler and the schedule that were chosen."""
    import comfy.k_diffusion.sampling as ks
    import comfy.samplers
    euler = comfy.samplers.KSAMPLER(ks.sample_euler)
    grid = ad.schedule(ad.SHIFT_VIDEO, steps)

    def run(executor, guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar):
        if abs(float(getattr(guider, "cfg", 1.0)) - 1.0) > 1e-6:
            # pi-H3 was trained without guidance: guidance on top doubles every step and spoils the picture, so it is not applied
            log.once("pi_h3:cfg", log.INFO, "FunUI pi-H3", f"CFG {guider.cfg:g} is not used: pi-H3 runs without guidance")
            guider.cfg = 1.0
        try:
            import latent_preview
            callback = latent_preview.prepare_callback(guider.model_patcher, steps)
        except Exception:      # noqa: BLE001 -- only the progress bar and previews are lost
            pass
        mine = grid.to(device=sigmas.device, dtype=sigmas.dtype)
        options = extra_args.get("model_options", {}).get("transformer_options")
        if options is not None:
            options["sample_sigmas"] = mine
        log.once(f"pi_h3:swap:{steps}", log.INFO, "FunUI pi-H3",
                 f"sampling {steps} steps from the adapter; the chosen sampler and steps ({len(sigmas) - 1}) are not used")
        started = time.time()
        out = euler.sample(guider, mine, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        log.info("FunUI pi-H3", f"sampled {steps} steps in {time.time() - started:.1f} s ({(time.time() - started) / steps:.1f} s a step)")
        return out

    return run


def _forget(raw):
    """Take an earlier install of this module off the model (a second pass starts from the first one's patches)."""
    record = raw.model_options.pop(CONTROLLER_KEY, None)
    if not record:
        return
    raw.model_options.pop(dwd.MARK_KEY, None)
    if record.get("dwd"):
        raw.remove_additional_models("funui_pi_h3_dwd")
        raw.object_patches.pop("diffusion_model._run_blocks", None)
    for key, entries in list(raw.patches.items()):
        left = [e for e in entries if id(e[1]) not in record["mine"]]
        if len(left) != len(entries):
            if left:
                raw.patches[key] = left
            else:
                raw.patches.pop(key)
    raw.patches_uuid = uuid.uuid4()


def _patch(raw, adapter):
    """The file's whole tensors as "set" patches and its LoRA pairs as LoRA patches, on the clone the caller owns.
    -> the record to find them again."""
    import comfy.lora
    for k in adapter.full:
        raw.patches.pop("diffusion_model." + k, None)           # an earlier patch on a replaced tensor would be wrong now
    sets = {"diffusion_model." + k: ("set", (v,)) for k, v in adapter.full.items()}
    sd, to_load = {}, {}
    for i, (k, (a, b)) in enumerate(adapter.lora.items()):
        sd[f"l{i}.lora_A.weight"], sd[f"l{i}.lora_B.weight"] = a, b
        to_load[f"l{i}"] = "diffusion_model." + k
    loras = comfy.lora.load_lora(sd, to_load, log_missing=False)
    raw.add_patches(sets, 1.0)
    raw.add_patches(loras, 1.0)
    return {"mine": {id(v) for v in (*sets.values(), *loras.values())}, "dwd": bool(adapter.dwd)}


def install(patcher, values, key):
    raw = getattr(patcher, "_patcher", patcher)
    if not values["enabled"]:
        _kept.clear()
        _forget(raw)
        return None
    # Installed on the real patcher, not the guarded one: a dropped hook would run the plain base with these weights, which renders noise.
    name = values["adapter"]
    path = folder_paths.get_full_path("pi_h3", name) if name else None
    if path is None:
        raise control.Unavailable("no pi-H3 adapter file is chosen: put one in ComfyUI/models/pi_h3, restart, and pick it in Settings")
    adapter = _adapter(path, name)
    steps = int(values.get("steps")) if str(values.get("steps", "auto")) in ("2", "4") else adapter.steps
    if steps not in ad.MAX_RAW_T:
        raise control.Unavailable(f"{name}: the steps cannot be told from the file name; choose 2 or 4 in Settings")
    native = raw.get_model_object("diffusion_model")
    ad.verify(adapter, native)
    _forget(raw)                                                  # a second pass: the first one's adapter is replaced, not stacked
    sampling = copy.deepcopy(raw.get_model_object("model_sampling"))
    if not hasattr(sampling, "audio_shift") or not hasattr(sampling, "shift"):
        raise ValueError("pi-H3 needs the native MiniMax-H3 audio/video flow sampling")
    sampling.shift, sampling.audio_shift = ad.SHIFT_VIDEO, ad.SHIFT_AUDIO
    raw.add_object_patch("model_sampling", patching.tag(sampling, key))
    options = raw.model_options["transformer_options"] = dict(raw.model_options.get("transformer_options", {}))
    options["minimax_h3_sigma_shift_video"], options["minimax_h3_sigma_shift_audio"] = ad.SHIFT_VIDEO, ad.SHIFT_AUDIO
    raw.model_options[CONTROLLER_KEY] = _patch(raw, adapter)
    if adapter.dwd:
        # the pixel head: the last step's hidden states decode to pixels, and the decode takes them in place of the VAE
        hook = dwd.install(raw, adapter.dwd, native.final_layer.norm.eps)
        raw.add_object_patch("diffusion_model._run_blocks", patching.tag(hook, key))
        raw.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, patching.tag(dwd.outer_sample, key))
        raw.model_options[dwd.MARK_KEY] = patching.tag(dwd.Holder(), key)
    raw.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, patching.tag(_swap(steps), key))
    return f"{adapter.label}, {steps} steps{', DWD pixels (no video decode)' if adapter.dwd else ''}, {adapter.size / 2**30:.2f} GB"


PROVIDES = {"modifier": install, "setting_options": lambda: {"adapter": _options()}}
