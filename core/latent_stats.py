"""The video VAE's per-channel statistics, reached through the VAE the loader last made.

LTX's latent upscaler works on the latent as the VAE writes it (un-normalised), while the model samples the normalised one,
so whoever runs the upscaler needs the VAE's two small tables. The sampler never sees the VAE, so the loader leaves a weak
reference to it here (the last LTX video VAE loaded is the one the next run uses). The tables are read through the VAE when
needed, exactly as ComfyUI's own LTXV Latent Upsampler reads them; nothing is kept alive by this.
"""

import weakref

_LAST = []


def remember(vae):
    """Called by the VAE loader with the VAE it just made; anything that is not an LTX video VAE is ignored."""
    model = getattr(vae, "first_stage_model", None)
    stats = getattr(model, "per_channel_statistics", None)
    if stats is None or not hasattr(stats, "un_normalize") or "video" not in (type(model).__module__ + type(model).__name__).lower():
        return
    _LAST[:] = [weakref.ref(vae)]


def _stats():
    vae = _LAST[0]() if _LAST else None
    return vae.first_stage_model.per_channel_statistics if vae is not None else None


def available():
    return _stats() is not None


def un_normalize(x):
    return _stats().un_normalize(x)


def normalize(x):
    return _stats().normalize(x)
