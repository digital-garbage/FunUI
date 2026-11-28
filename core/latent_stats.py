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
    if not _is_video_vae(getattr(vae, "first_stage_model", None)):
        return
    _LAST[:] = [weakref.ref(vae)]


def _is_video_vae(model):
    stats = getattr(model, "per_channel_statistics", None)
    return stats is not None and hasattr(stats, "un_normalize") and "video" in (type(model).__module__ + type(model).__name__).lower()


def _found():
    """The VAE the loader left, or (it was never asked: a cached or older load, another way in) a live LTX video VAE."""
    vae = _LAST[0]() if _LAST else None
    if vae is not None:
        return vae.first_stage_model
    import gc
    for obj in gc.get_objects():
        try:
            if type(obj).__name__ == "VideoVAE" and _is_video_vae(obj):
                return obj
        except Exception:                       # noqa: BLE001  an object that cannot be asked is not it
            continue
    return None


def _stats():
    found = _found()
    return found.per_channel_statistics if found is not None else None


def available():
    return _stats() is not None


def un_normalize(x):
    return _stats().un_normalize(x)


def normalize(x):
    return _stats().normalize(x)
