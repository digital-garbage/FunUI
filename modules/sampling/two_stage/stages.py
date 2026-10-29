"""The wrapper around one sampling run that cuts it in two."""

import torch

from ..._core import log
from ..._core import dit_hooks

TITLE = "FunUI Two stages"


def small_grid(h, w, size):
    """The first stage's latent grid: about `size` of each side, kept even (H3 patches 2x2), never larger than the full
    one, and of the SAME shape as the full picture. Rounding each side on its own bent the shape by up to 6 percent
    (18x31 became 12x22), so the layout was decided in a squashed picture and then stretched."""
    best = None
    for h1 in range(2, h + 1, 2):
        for w1 in range(2, w + 1, 2):
            shape_error = abs(w1 / h1 - w / h) / (w / h)
            cost = 4 * shape_error + abs(h1 / h - size) + abs(w1 / w - size)
            if best is None or cost < best[0]:
                best = (cost, h1, w1)
    return (best[1], best[2]) if best else (h, w)


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


FULL = "funui_full_sigmas"


def keep_whole_schedule(executor, guider, sigmas, extra_args, *args, **kwargs):
    """SAMPLER_SAMPLE wrapper: while a stage runs, the model sees the WHOLE schedule in `sample_sigmas`, so every
    module that reads the step number, the step count or how far along the run is (Decisiveness, SLA's sparsity
    ramp, step windows) sees 8 steps and not two runs of 4."""
    full = getattr(guider, FULL, None)
    if full is not None:
        extra_args["model_options"].setdefault("transformer_options", {})["sample_sigmas"] = \
            full.to(device=sigmas.device, dtype=sigmas.dtype)       # where this stage's own schedule lives (the GPU)
    return executor(guider, sigmas, extra_args, *args, **kwargs)


_CACHE = {}


def _upscaler(name):
    """Built once per file (and per change of the file on disk); the network sits on the offload device between runs."""
    import os
    import folder_paths
    from ...latent.second_pass.nodes import FOLDER, load_upscaler
    path = folder_paths.get_full_path_or_raise(FOLDER, name)
    stamp = (path, os.path.getmtime(path))
    if _CACHE.get("stamp") != stamp:
        _CACHE.clear()
        _CACHE.update(stamp=stamp, model=load_upscaler(name))
    return _CACHE["model"]


METHODS = ("bicubic", "mitchell", "catmull-rom", "lanczos", "bislerp", "bilinear", "trilinear", "area", "nearest-exact")


def _bc_kernel(B, C):
    """Mitchell-Netravali cubic family; Mitchell is B=C=1/3, Catmull-Rom is B=0 C=1/2. Support 2."""
    def k(x):
        x = x.abs()
        near = ((12 - 9 * B - 6 * C) * x ** 3 + (-18 + 12 * B + 6 * C) * x ** 2 + (6 - 2 * B)) / 6
        far = ((-B - 6 * C) * x ** 3 + (6 * B + 30 * C) * x ** 2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6
        return torch.where(x < 1, near, torch.where(x < 2, far, torch.zeros(())))
    return k, 2.0


def _lanczos_kernel(x, a=3):
    return torch.where(x.abs() < a, torch.sinc(x) * torch.sinc(x / a), torch.zeros(()))


KERNELS = {"mitchell": _bc_kernel(1 / 3, 1 / 3), "catmull-rom": _bc_kernel(0.0, 0.5), "lanczos": (_lanczos_kernel, 3.0)}


def _weights(n_in, n_out, like, kernel, support):
    """A separable filter as a matrix [n_out, n_in], rows summing to 1 (the footprint widens when shrinking, as PIL's)."""
    scale = n_in / n_out
    centre = (torch.arange(n_out, dtype=torch.float32) + 0.5) * scale
    taps = torch.arange(n_in, dtype=torch.float32) + 0.5
    d = (taps[None, :] - centre[:, None]) / max(scale, 1.0)
    w = kernel(d) * (d.abs() < support)
    return (w / w.sum(1, keepdim=True)).to(device=like.device, dtype=like.dtype)


def _filtered(video, h, w, name):
    """ComfyUI's own "lanczos" goes through 8-bit pictures, which a 24-channel latent cannot survive; these are
    the same filters done on the numbers themselves."""
    kernel, support = KERNELS[name]
    wh = _weights(video.shape[-2], h, video, kernel, support)
    ww = _weights(video.shape[-1], w, video, kernel, support)
    return torch.einsum("hH,bcfHW,wW->bcfhw", wh, video, ww)


def resize_video(video, h, w, method="bicubic"):
    """Every frame to h x w by `method`, frames independent (time is never resampled)."""
    import torch.nn.functional as F
    if tuple(video.shape[-2:]) == (h, w):
        return video
    if method == "bicubic":
        from ...latent.second_pass.nodes import downscale_to
        return downscale_to(video, h, w)
    if method in KERNELS:
        return _filtered(video.float(), h, w, method).to(video.dtype)
    if method == "trilinear":
        return F.interpolate(video.float(), size=(video.shape[2], h, w), mode="trilinear",
                             align_corners=False).to(video.dtype)
    b, c, f, hh, ww = video.shape
    x = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, hh, ww).float()
    if method == "bislerp":
        import comfy.utils
        x = comfy.utils.bislerp(x, w, h)
    else:
        x = F.interpolate(x, size=(h, w), mode=method, **({"align_corners": False} if method == "bilinear" else {}))
    return x.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4).to(video.dtype)


def release_upscaler():
    """Called when the setup no longer uses the upscaler (another method chosen, or Two stages off): it leaves memory."""
    if _CACHE:
        _CACHE.clear()
        import gc
        gc.collect()
        try:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
        except Exception:                           # noqa: BLE001 -- nothing to free on a CPU-only box
            pass


def enlarge(video, h, w, upscaler=None, method="bicubic"):
    """The hand-over: the first stage's clean guess at the full grid. With an upscaler, its learned resize at the
    ratio that gets nearest (it works in even steps), then a bicubic trim to the exact grid; without, `method`."""
    from ...latent.second_pass.nodes import run_upscaler
    if upscaler is None:
        return resize_video(video, h, w, method)
    ratio = (h / video.shape[-2] + w / video.shape[-1]) / 2.0
    if ratio <= 1.0:
        return video
    return resize_video(run_upscaler(upscaler, video, ratio), h, w)


def make_wrapper(size, steps, resize=None, method="bicubic"):
    def outer(executor, *args, **kwargs):
        # Every OUTER_SAMPLE wrapper behind this one belongs around the WHOLE run, not around each stage (Noise
        # merging's per-run state, Taste, STAS' report ...). They are moved to wrap the staged run as one call.
        from comfy.patcher_extension import WrapperExecutor
        rest = executor.wrappers[executor.idx + 1:]
        staged = lambda *a, **k: _staged(executor.original, executor.class_obj, size, steps, resize, method, *a, **k)
        return WrapperExecutor.new_class_executor(staged, executor.class_obj, rest, 0).execute(*args, **kwargs)

    return outer


def _staged(original, guider, size, steps, resize, method, *args, **kwargs):
    run = original
    import comfy.nested_tensor
    import comfy.utils
    import latent_preview

    noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed = (list(args) + [None] * 8)[:8]
    latent_shapes = kwargs.get("latent_shapes")
    why = _why_not(guider, latent_image, denoise_mask, sigmas, latent_shapes, steps)
    if why is None:
        _t, _c, frames, h, w = latent_shapes[0]
        h1, w1 = small_grid(h, w, size)
        if (h1, w1) == (h, w):
            why = "the first stage would be as large as the full picture"
    if why is not None:
        log.once(f"two_stage:{why}", log.INFO, TITLE, f"not used this time: {why}")
        return run(*args, **kwargs)

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

    upscaler = None
    if resize:
        try:
            upscaler = _upscaler(resize)
        except Exception as e:                      # noqa: BLE001 -- a missing file must not lose the run
            log.warning(TITLE, f"latent upscaler not used, bicubic instead: {e}")
    log.info(TITLE, f"first {steps} step(s) at latent {h1}x{w1} instead of {h}x{w}, then {len(second) - 1} at the full size")
    setattr(guider, FULL, sigmas)
    setattr(guider, dit_hooks.STAGE, 1)
    try:
        run(small_noise, small_image, sampler, first, None, first_stage_callback, disable_pbar, seed,
             latent_shapes=shapes1)
    except BaseException:
        setattr(guider, FULL, None)
        setattr(guider, dit_hooks.STAGE, None)
        raise
    if "x0" not in seen:
        raise RuntimeError("the first stage gave no clean guess to hand over")

    video, audio = comfy.utils.unpack_latents(seen["x0"], shapes1)
    done = guider.model_patcher.model.process_latent_out(comfy.nested_tensor.NestedTensor((video, audio)).cpu())
    video, audio = done.unbind()
    enlarged = enlarge(video, h, w, upscaler, method)
    handed, _ = comfy.utils.pack_latents(
        (enlarged.to(latent_image.device, latent_image.dtype), audio.to(latent_image.device, latent_image.dtype)))

    guider.conds = raw_conds                        # the first stage encoded the originals for its own size
    shifted = None
    if callback is not None:
        def shifted(step, x0, x, total):
            return callback(step + steps, x0, x, total + steps)
    setattr(guider, dit_hooks.STAGE, 2)
    try:
        return run(noise, handed, sampler, second, None, shifted, disable_pbar, seed, latent_shapes=latent_shapes)
    finally:
        setattr(guider, FULL, None)
        setattr(guider, dit_hooks.STAGE, None)
