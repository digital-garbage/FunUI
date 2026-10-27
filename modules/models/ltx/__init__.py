"""LTX 2.x (audio+video) support.

The second model-support module, built the way the first was: it teaches the system to recognise
this model, build its latent and decode it, and offers pipelines for it. Nothing outside this
folder mentions LTX.

What it contributes, and what it deliberately does NOT:

* **Traits** -- a joint audio+video model with a compressed time axis. It announces `block_scope` (a
  hook may hand a block changed options: the Randomizer) but NOT `dit_block_hooks`: the hooks that edit
  rows (REINS, block repeat...) are written for H3's packed block layout, and LTX's blocks take
  `(video, audio)` as a pair. Until a module is written for that layout it is absent here, said,
  rather than half-working.
* **An empty latent** -- video 128 channels at /32 and /8, plus an audio latent whose length comes
  from the AUDIO VAE and the frame rate. That is why this provider asks for `audio_vae` and
  `frame_rate`: without them the audio half cannot be sized, and a video-only latent would make a
  silent clip while reporting success, so it refuses instead.
* **`ltx_av`** -- the trait the LTX-only modules require (audio coupling, context windows...).
* **The picture inside a sampling-time latent** (`video_stream`) -- video first, then audio, the same
  arrangement H3 uses, so ALG, DynaShift and the guidance modules find the picture.
* **Decode** -- the picture through the video VAE, the sound through the audio VAE.

Tested on a real LTX 2.3 (2026-10-27): Video to Video direct, Image to Video from References, separate
loaders, LoRAs and Noise merging. Other enhancements are not tested on LTX yet; the shapes are also checked
against fakes and ComfyUI's own stock nodes, and the pipelines against the graph builder.
"""

import comfy.model_management
import torch
from comfy.nested_tensor import NestedTensor

from ..._core import dit_hooks as _dit_hooks, log, streams as _streams, traits as _traits, vae_tiles
from .pipeline import optional_nodes as _optional_nodes, presets as _pipeline_presets, upgrade as _upgrade

has_block = _traits.has_block

ID = "model_ltx"
TITLE = "LTX 2.x"
STAGE = "load"
CATEGORY = "system"
STATUS = "experimental"

# Qualified: a bare class name matches anything built with a class of that name.
MODEL_CLASS = "comfy.ldm.lightricks.av_model.LTXAVModel"

VIDEO_CHANNELS = 128
VIDEO_SPATIAL_RATIO = 32
VIDEO_TEMPORAL_RATIO = 8

# ComfyUI tells the audio+video architecture from the video-only one by this key
# (comfy/model_detection.py: `audio_adaln_single.linear.weight`).
_PREFIXES = ("", "model.diffusion_model.")
_SIGNATURE_KEYS = ("audio_adaln_single.linear.weight",)


def is_ltx(model) -> bool:
    return has_block(model, MODEL_CLASS)


def detect(keys) -> bool:
    """Whether a checkpoint's tensor names are an LTX audio+video model's, from its header alone."""
    keyset = set(keys)
    return any(all(f"{prefix}{key}" in keyset for key in _SIGNATURE_KEYS) for prefix in _PREFIXES)


def probe_traits(keys) -> list:
    """What `traits()` would say, read before anything loads."""
    if not detect(keys):
        return []
    return ["audio_stream", "temporal_latent", "temporal_compression", "ltx_av", "block_scope", "block_rows"]


def probe_blocks(keys):
    """How many transformer blocks the file has, read off its header."""
    return _dit_hooks.count_indexed(keys, tuple(f"{p}transformer_blocks." for p in _PREFIXES))


def traits(model):
    if not is_ltx(model):
        return ()
    # ltx_av: transformer_blocks that take (video, audio) as a pair and have the video<->audio
    # cross-attention submodules. What the LTX-only modules require.
    # block_scope: its forward honours patches_replace["dit"][("double_block", i)] and routes attention
    # through optimized_attention with the block's transformer_options, so a hook that only SCOPES a
    # block (hands it changed transformer_options) works. Not dit_block_hooks: a block here takes and
    # returns (video, audio), where the hooks that edit rows expect one packed tensor.
    # block_rows: the hooks that read or push the video rows of a block's output (REINS, Block repeat)
    # handle the pair too (see those modules).
    return ["audio_stream", "ltx_av", "block_scope", "block_rows"]


def frames_for(length: int) -> int:
    """LTX makes 8k+1 frames; a length off that grid is rounded UP to it."""
    length = max(1, int(length))
    return ((length - 1 + VIDEO_TEMPORAL_RATIO - 1) // VIDEO_TEMPORAL_RATIO) * VIDEO_TEMPORAL_RATIO + 1


def empty_latent(model, width, height, length, batch_size=1, audio_vae=None, frame_rate=25.0):
    """The joint latent, or None when this is not an LTX audio+video model."""
    if not is_ltx(model):
        return None
    if audio_vae is None:
        raise RuntimeError(
            "this model makes sound with the picture and sizes it from the audio VAE: wire the audio "
            "VAE into the empty latent (a video-only latent would give a silent clip)")

    if width % VIDEO_SPATIAL_RATIO or height % VIDEO_SPATIAL_RATIO:
        log.once(f"ltx_size:{width}x{height}", log.ALERT, "FunUI LTX",
                 f"{width}x{height} is not a multiple of {VIDEO_SPATIAL_RATIO}: the video comes out "
                 f"{width // VIDEO_SPATIAL_RATIO * VIDEO_SPATIAL_RATIO}x{height // VIDEO_SPATIAL_RATIO * VIDEO_SPATIAL_RATIO}")
    frames = frames_for(length)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [batch_size, VIDEO_CHANNELS, ((frames - 1) // VIDEO_TEMPORAL_RATIO) + 1,
         max(1, height // VIDEO_SPATIAL_RATIO), max(1, width // VIDEO_SPATIAL_RATIO)],
        device=device)
    z_channels = audio_vae.latent_channels
    bins = audio_vae.first_stage_model.latent_frequency_bins
    steps = audio_vae.first_stage_model.num_of_latents_from_frames(frames, frame_rate)
    audio = torch.zeros((batch_size, z_channels, steps, bins), device=device)
    return {"samples": NestedTensor((video, audio)),
            "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
            "downscale_ratio_temporal": VIDEO_TEMPORAL_RATIO}


def decode(latent, model=None, vae=None, audio_vae=None, tile_size=0):
    """The picture and the sound, each through the VAE that understands it."""
    if model is None or not is_ltx(model):
        return None
    if not getattr(latent, "is_nested", False):
        return None
    parts = latent.unbind()
    if len(parts) != 2:
        raise RuntimeError(f"expected a video and an audio part, got {len(parts)}")
    video_latent, audio_latent = parts
    if audio_vae is None:
        raise RuntimeError("this model generates sound and no audio VAE is wired into the decode")

    if tile_size and tile_size > 0:
        try:
            images = vae_tiles.tiled_decode(vae, video_latent, tile_size)
        except Exception as exc:                             # noqa: BLE001
            log.once(f"ltx_decode_tiles:{type(exc).__name__}", log.ALERT, "FunUI LTX decode",
                     f"the tiled decode failed ({type(exc).__name__}: {exc}); decoded in one piece instead")
            images = vae.decode(video_latent)
    else:
        images = vae.decode(video_latent)
    if len(images.shape) == 5:
        images = images.reshape(-1, *images.shape[-3:])

    waveform = audio_vae.decode(audio_latent).movedim(-1, 1).to(audio_latent.device)
    audio = {"waveform": waveform, "sample_rate": int(audio_vae.first_stage_model.output_sample_rate)}
    return images, audio


TRAITS = traits
PROVIDES = {"empty_latent": empty_latent, "decode": decode, "detect": detect,
            "probe_traits": probe_traits, "probe_blocks": probe_blocks, "pipeline_presets": _pipeline_presets, "optional_nodes": _optional_nodes,
            "video_stream": _streams.av_video_stream, "upgrade_slots": _upgrade}
