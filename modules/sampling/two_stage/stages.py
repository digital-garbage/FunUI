"""The wrapper around one sampling run that cuts it in two."""

import torch

from ..._core import log

TITLE = "FunUI Two stages"


def small_grid(h, w, size):
    """The first stage's latent grid: scaled, kept even (H3 patches 2x2) and never larger than the full one."""
    def side(n):
        return min(n, max(2, int(round(n * size / 2.0)) * 2))
    return side(h), side(w)


def _why_not(guider, latent_image, denoise_mask, sigmas, latent_shapes, steps):
    if denoise_mask is not None:
        return "the run has a mask"
    if latent_shapes is None or len(latent_shapes) != 2 or len(latent_shapes[0]) != 5:
        return "the latent is not a video with a soundtrack"
    if int(torch.count_nonzero(latent_image)) > 0:
        return "the run starts from a picture, not from noise"
    if len(sigmas) - 1 < 2 or steps >= len(sigmas) - 1:
        return f"the run has {len(sigmas) - 1} step(s), too few to cut after {steps}"
    for conds in guider.original_conds.values():
        for c in conds:
            if "minimax_keyframes" in c.get("model_conds", {}) or "minimax_keyframes" in c:
                return "the shot has first/last picture pins sized to the full picture"
    try:
        from ..ptd.controller import CONTROLLER_KEY
        if CONTROLLER_KEY in guider.model_options or CONTROLLER_KEY in guider.model_patcher.model_options:
            return "PTD sets its own schedule"
    except Exception:                                   # noqa: BLE001 -- no PTD module, nothing to clash with
        pass
    return None


def make_wrapper(size, steps):
    def outer(executor, *args, **kwargs):
        import comfy.nested_tensor
        import comfy.utils
        import latent_preview
        from ...latent.second_pass.nodes import downscale_to

        noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed = (list(args) + [None] * 8)[:8]
        latent_shapes = kwargs.get("latent_shapes")
        guider = executor.class_obj
        why = _why_not(guider, latent_image, denoise_mask, sigmas, latent_shapes, steps)
        if why is None:
            _t, _c, frames, h, w = latent_shapes[0]
            h1, w1 = small_grid(h, w, size)
            if (h1, w1) == (h, w):
                why = "the first stage would be as large as the full picture"
        if why is not None:
            log.once(f"two_stage:{why}", log.INFO, TITLE, f"not used this time: {why}")
            return executor(*args, **kwargs)

        video_noise, audio_noise = comfy.utils.unpack_latents(noise, latent_shapes)
        audio_zero = torch.zeros(latent_shapes[1], dtype=latent_image.dtype, device=latent_image.device)
        # A crop of the full noise keeps its statistics (a resize would smooth them) and is fixed by the seed.
        small_noise, shapes1 = comfy.utils.pack_latents(
            (video_noise[..., :h1, :w1].contiguous(), audio_noise))
        small_image, _ = comfy.utils.pack_latents(
            (torch.zeros(*latent_shapes[0][:3], h1, w1, dtype=latent_image.dtype, device=latent_image.device), audio_zero))
        first, second = sigmas[:steps + 1], sigmas[steps:]
        raw_conds = {k: [c.copy() for c in v] for k, v in guider.conds.items()}

        seen = {}
        preview = None
        try:
            preview = latent_preview.prepare_callback(guider.model_patcher, steps)
        except Exception:                               # noqa: BLE001 -- only the progress bar and previews are lost
            pass

        def first_stage_callback(step, x0, x, total):
            seen["x0"] = x0
            if preview is not None:
                nested = lambda t: comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(t, shapes1))
                preview(step, nested(x0), nested(x), total)

        log.info(TITLE, f"first {steps} step(s) at latent {h1}x{w1} instead of {h}x{w}, then {len(second) - 1} at the full size")
        executor(small_noise, small_image, sampler, first, None, first_stage_callback, disable_pbar, seed,
                 latent_shapes=shapes1)
        if "x0" not in seen:
            raise RuntimeError("the first stage gave no clean guess to hand over")

        video, audio = comfy.utils.unpack_latents(seen["x0"], shapes1)
        done = guider.model_patcher.model.process_latent_out(comfy.nested_tensor.NestedTensor((video, audio)).cpu())
        video, audio = done.unbind()
        handed, _ = comfy.utils.pack_latents(
            (downscale_to(video, h, w).to(latent_image.device, latent_image.dtype), audio.to(latent_image.device, latent_image.dtype)))

        guider.conds = raw_conds                        # the first stage encoded the originals for its own size
        shifted = None
        if callback is not None:
            def shifted(step, x0, x, total):
                return callback(step + steps, x0, x, total + steps)
        return executor(noise, handed, sampler, second, None, shifted, disable_pbar, seed, latent_shapes=latent_shapes)

    return outer
