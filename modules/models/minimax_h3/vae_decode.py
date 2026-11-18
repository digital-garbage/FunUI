"""H3 video decode: spatial tiles, and the X2 Detail VAE's packed output.

Ported from v4 (itself from TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler's
"VAE Decode (fast)"). Two things the stock decode cannot do on H3:

* Honour a tile size. `decode_tiled` ignores the tile arguments on H3, and ComfyUI's
  out-of-memory fallback is a generic 3D tiler that ignores H3's blend rules. Here the
  VAE's OWN tile flags are set for the one call and restored.
* Read an X2 Detail VAE, whose decoder emits packed 12-channel output that has to be
  pixel-shuffled to twice the resolution. Stock decode cannot read it at all.

Unvalidated on a GPU, same as v4's own status.
"""

import math

import torch
import torch.nn.functional as F

TILE_MIN = 256          # smaller tiles show a block grid
SPATIAL_RATIO = 16      # the VAE's spatial compression; the tiler assumes it


def x2_ratio(vae) -> int:
    """PixelShuffle factor of an H3 video VAE whose decoder emits packed RGB:
    1 for the stock VAE, 2 for the X2 Detail VAE. 1 when it cannot be read."""
    try:
        d = vae.first_stage_model.decoder
        rows = int(d.proj_out.weight.shape[0])
        packed = rows // (int(d.patch_size_t) * int(d.patch_size) ** 2) // 3
        r = math.isqrt(packed)
        return r if r >= 1 and r * r == packed and rows % 3 == 0 else 1
    except Exception:                                       # noqa: BLE001
        return 1


def unpack(out, up):
    """[B, 3*up*up, T, H, W] packed phases -> [B, 3, T, H*up, W*up]. Layout is R phases,
    then G, then B (what `repeat_interleave` on the pixel statistics matches)."""
    b, c, t, h, w = out.shape
    out = F.pixel_shuffle(out.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), up)
    return out.reshape(b, t, c // (up * up), h * up, w * up).permute(0, 2, 1, 3, 4)


def tiles(tile_px):
    """(tile, overlap) in pixels: at least TILE_MIN, a multiple of 16, overlap a
    quarter of the tile (bigger tiles with a fixed 64px overlap show a block grid)."""
    tile = max(TILE_MIN, (int(tile_px) // SPATIAL_RATIO) * SPATIAL_RATIO)
    return tile, max(16, (tile * 64 // 256 // 16) * 16)


def decode_fast(vae, video, tile_px):
    """Decode an H3 video latent in `tile_px`-pixel spatial tiles straight through H3's
    decoder. Returns IMAGES (B, T, H, W, C) on the intermediate device. The VAE's flags
    are restored even when the decode raises."""
    import comfy.model_management as mm
    inner = vae.first_stage_model
    up = x2_ratio(vae)
    tile, overlap = tiles(tile_px)
    saved = (inner.tiling, inner.tile_size, inner.tile_overlap_min,
             inner.decoder.out_channels, inner.pixel_mean, inner.pixel_std)
    vae.throw_exception_if_invalid()
    try:
        inner.tiling, inner.tile_size, inner.tile_overlap_min = True, tile, overlap
        if up > 1:
            inner.decoder.out_channels = 3 * up * up
        with mm.cuda_device_context(vae.device):
            mm.load_models_gpu([vae.patcher],
                               memory_required=vae.memory_used_decode(video.shape, vae.vae_dtype),
                               force_full_load=vae.disable_offload)
            if up > 1:
                # After the load: a (dynamic) load can restore the checkpoint's 3-channel stats.
                inner.pixel_mean = inner.pixel_mean.repeat_interleave(up * up, dim=1)
                inner.pixel_std = inner.pixel_std.repeat_interleave(up * up, dim=1)
            out = torch.empty(inner.decode_output_shape(video.shape),
                              device=mm.intermediate_device(), dtype=vae.vae_output_dtype())
            inner.decode(video.to(device=vae.device, dtype=vae.vae_dtype), output_buffer=out)
            vae.process_output(out)
    finally:
        (inner.tiling, inner.tile_size, inner.tile_overlap_min,
         inner.decoder.out_channels, inner.pixel_mean, inner.pixel_std) = saved
    if up > 1:
        out = unpack(out, up)
    return out.movedim(1, -1)


def is_image_vae(vae) -> bool:
    """True for a still-picture VAE (H3-Image): it reads one latent frame at a time."""
    return bool(getattr(vae, "not_video", False))


def frames_for(vae, latents) -> int:
    """How many video frames `latents` latent frames stand for: the VAE's own count, else 4 per latent."""
    try:
        return max(int(vae.upscale_ratio[0](latents)), latents)
    except Exception:                                       # noqa: BLE001
        return latents * 4


def stretch(frames, count):
    """[T, H, W, C] -> [count, H, W, C]: each output frame blends its two nearest decoded neighbours."""
    t = frames.shape[0]
    if count <= t or t < 2:
        return frames
    dev = frames.device
    pos = torch.linspace(0, t - 1, count, device=dev)
    lo = pos.floor().long().clamp(max=t - 2)
    w = (pos - lo).view(-1, 1, 1, 1).to(frames.dtype)
    return frames[lo] * (1 - w) + frames[lo + 1] * w


def decode_stills(vae, video):
    """Decode a video latent with a still-picture VAE: every latent frame becomes one picture, then the pictures are blended in time up to the clip's frame count. Returns IMAGES (T, H, W, C). Experimental."""
    clips = []
    for b in range(video.shape[0]):
        pics = [vae.decode(video[b:b + 1, :, i:i + 1]) for i in range(video.shape[2])]
        pics = [p.reshape(-1, *p.shape[-3:]) for p in pics]
        ref = pics[0]
        pics = torch.cat([p.to(device=ref.device, dtype=ref.dtype) for p in pics], dim=0)
        clips.append(stretch(pics, frames_for(vae, video.shape[2])))
    dev, dt = clips[0].device, clips[0].dtype
    return torch.cat([c.to(device=dev, dtype=dt) for c in clips], dim=0)
