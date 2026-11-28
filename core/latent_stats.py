"""The video VAE's per-channel statistics, remembered when the VAE is loaded.

LTX's latent upscaler works on the latent as the VAE writes it (un-normalised), while the model samples the normalised one,
so whoever runs the upscaler needs the VAE's two small tables. The sampler never sees the VAE, so the loader leaves them here:
the last video VAE loaded is the one the next run uses. Two small CPU tensors, no reference to the VAE itself.
"""

_LAST = {}


def remember(first_stage_model):
    """Called by the VAE loader with `vae.first_stage_model`; anything that is not an LTX video VAE is ignored."""
    stats = getattr(first_stage_model, "per_channel_statistics", None)
    if stats is None or "video" not in type(first_stage_model).__name__.lower():
        return
    try:
        _LAST["mean"] = stats.get_buffer("mean-of-means").detach().float().cpu().clone()
        _LAST["std"] = stats.get_buffer("std-of-means").detach().float().cpu().clone()
    except Exception:                                   # noqa: BLE001 -- a VAE without the tables: no upscaling by LTX's upscaler
        _LAST.clear()


def available():
    return "mean" in _LAST


def _table(name, x):
    return _LAST[name].view(1, -1, 1, 1, 1).to(device=x.device, dtype=x.dtype)


def un_normalize(x):
    return x * _table("std", x) + _table("mean", x)


def normalize(x):
    return (x - _table("mean", x)) / _table("std", x)
