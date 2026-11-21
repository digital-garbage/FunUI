"""H3 corrector: a small trained network that fixes H3's picture prediction at every step.

At each step it reads the noisy picture, H3's prediction and the prompt's text states, and predicts what is
still wrong; that correction is capped at a share of the prediction's size and added to the picture part of
the prediction (the sound is left alone). It is optional: any H3 model runs without it. The corrector file
goes in models/h3_correctors and must come from the same family as the model (Sulphur AI's s3_corrector
works with Sulphur 3 and other H3 models). See the README Thank you section.

Costs one extra network pass per step (about 2B weights, run in bf16 on a GPU, stored in fp32).
"""

import os

import folder_paths

from ..._core import log, patching
from .network import Config, Corrector, cap

ID = "h3_corrector"
TITLE = "H3 corrector"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["audio_stream", "adaln_modalities"]

FOLDER = "h3_correctors"
if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[FOLDER] = ([os.path.join(folder_paths.models_dir, FOLDER)], {".safetensors"})


def _files():
    try:
        return folder_paths.get_filename_list(FOLDER)
    except Exception:          # noqa: BLE001 -- no folder is the same as no files
        return []


def _options():
    return [{"value": f, "label": f} for f in _files()] or [{"value": "", "label": f"No file in models/{FOLDER}"}]


SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Correct each step",
        "hint": "A small trained network adds a fix to the picture at every step. Needs a corrector file; costs one extra pass per step.",
    },
    "file": {
        "type": "enum", "keep_missing": True, "default": (_files() or [""])[0], "options": _options(),
        "label": "Corrector file",
        "hint": "A corrector from models/h3_correctors. The list is read again each time Settings opens.",
        "when": {"enabled": True},
    },
    "strength": {
        "type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
        "label": "Strength", "ui": "slider",
        "hint": "1 = as trained. Lower fades the fix out, higher exaggerates it.",
        "when": {"enabled": True},
    },
    "budget": {
        "type": "float", "default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
        "label": "Largest fix", "ui": "slider",
        "hint": "The fix is never bigger than this share of the prediction. The Sulphur corrector was trained at 0.5.",
        "when": {"enabled": True},
    },
}

_LOADED = {}        # file -> (modified time, ModelPatcher): the 8 GB network is read once, not on every run
FLOOR = 0.001       # the prediction's size is never taken as smaller than this when the cap is worked out


def release():
    """The setup no longer uses a corrector: it leaves memory."""
    if _LOADED:
        _LOADED.clear()
        import gc
        gc.collect()
        try:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
        except Exception:          # noqa: BLE001 -- nothing to free on a CPU-only box
            pass


def load(name):
    """The corrector patcher of a file in models/h3_correctors (kept for the next run)."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    path = folder_paths.get_full_path(FOLDER, name)
    if not path:
        raise FileNotFoundError(f"{name} is not in models/{FOLDER}")
    stamp = os.path.getmtime(path)
    if name in _LOADED and _LOADED[name][0] == stamp:
        return _LOADED[name][1]
    release()
    with safe_open(path, "pt") as handle:
        config = Config.from_metadata(handle.metadata())
    network = Corrector(config)
    network.load_state_dict(load_file(path, device="cpu"), strict=True)
    network.eval().requires_grad_(False)
    import comfy.model_management as mm
    import comfy.model_patcher
    patcher = comfy.model_patcher.ModelPatcher(network, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device())
    _LOADED[name] = (stamp, patcher)
    return patcher


def _interval(sigma, sigmas):
    """The sigma this step ends at, from the schedule (None without one). A second call inside a step belongs to the interval it sits in."""
    if sigmas is None:
        return None
    import torch
    values = sigmas.to(device="cpu", dtype=torch.float64).flatten()
    now = float(sigma.flatten()[0])
    at = min(max(int((values[:-1] >= now - 1e-6).sum()) - 1, 0), len(values) - 2)
    return float(values[at + 1])


def make_hook(network, strength, budget):
    import torch
    import comfy.utils

    def post_cfg(args):
        denoised, x, sigma = args["denoised"], args["input"], args["sigma"]
        sigma_next = _interval(sigma, args["model_options"].get("transformer_options", {}).get("sample_sigmas"))
        if sigma_next is None:
            log.once(f"{ID}:schedule", log.ALERT, "FunUI H3 corrector", "off this run: the sampler gave no step schedule")
            return denoised
        shapes = getattr(args["model"], "latent_shapes", None)
        if not shapes or len(shapes) < 2:
            raise RuntimeError("the corrector works on H3's picture + sound latent; this model does not pack them")
        text = next((c["cross_attn"] for c in (args.get("cond") or []) if c.get("cross_attn") is not None), None)
        if text is None or text.shape[-1] != network.config.text_dim:
            raise RuntimeError(f"the corrector needs the prompt's {network.config.text_dim}-wide text states")
        picture_in, *_ = comfy.utils.unpack_latents(x, shapes)
        picture_out, *_ = comfy.utils.unpack_latents(denoised, shapes)
        device = next(network.parameters()).device
        s = float(sigma.flatten()[0])
        z, guess = picture_in.to(device, torch.float32), picture_out.to(device, torch.float32)
        velocity = (guess - z) / s
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw = network(z, velocity, s, sigma_next, text.to(device, torch.float32))
        applied, _ = cap(raw, velocity, budget, FLOOR)
        fixed = (guess + s * strength * applied).to(denoised.dtype).to(denoised.device).reshape(denoised.shape[0], 1, -1)
        return torch.cat((fixed, denoised[:, :, fixed.shape[-1]:]), dim=-1)

    return post_cfg


def install(patcher, values, key):
    if not values.get("enabled"):
        release()
        return None
    name, strength = str(values.get("file") or ""), float(values.get("strength", 1.0))
    if not name:
        log.once(f"{ID}:nofile", log.WARNING, TITLE, f"is on but no corrector file is picked (models/{FOLDER} is empty), so nothing is corrected")
        return None
    if strength <= 0.0:
        return "strength 0: nothing to do"
    holder = load(name)
    patcher.set_additional_models(ID, [holder])          # ComfyUI loads it with the model and counts its memory
    patcher.set_model_sampler_post_cfg_function(patching.tag(make_hook(holder.model, strength, float(values.get("budget", 0.5))), key))
    return f"{name}, strength {strength:g}, fix capped at {float(values.get('budget', 0.5)):g} of the prediction"


PROVIDES = {"modifier": install, "setting_options": lambda: {"file": _options()}}
