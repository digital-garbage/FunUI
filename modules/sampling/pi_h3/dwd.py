"""DWD: the video's pixels come from the denoiser's last step, with no video VAE decode.

At the last step of the schedule the final hidden states of the video rows go through the adapter's pixel head. The pixels are
kept under a fingerprint of the video latent the run returned, and the H3 decode (modules/models/minimax_h3) takes them from
there instead of running the VAE. The head is its own model patcher, so ComfyUI loads and offloads it with the denoiser.

Ported from ComfyUI-piFlow (Copyright (c) 2026 Hansheng Chen, Apache-2.0); see the NOTICE.
"""

import collections
import hashlib
import weakref

import torch
from torch import nn

PIXELS_KEY = "funui_pi_h3_dwd_pixels"
MARK_KEY = "funui_pi_h3_dwd"            # on the model options of a model that carries a DWD head: its latent is not for the VAE
PIXEL_PATCH_SIZE = (32, 32)             # one H3 video token is 1x2x2 latents of the 16x VAE: 32x32 pixels per latent frame
MAX_KEPT = 2

_PIXELS = collections.OrderedDict()


class Holder:
    """The decoded pixels of one sampling run (not a dict, which ComfyUI copies with the model options)."""

    def __init__(self):
        self.pixels = None


def fingerprint(video_latent):
    return hashlib.sha1(video_latent.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def take(video_latent):
    """The DWD pixels (F, H, W, 3) in [0, 1] of the run that returned this video latent, or None."""
    return _PIXELS.get(fingerprint(video_latent))


def carries(model):
    """Whether this model (a patcher) has a DWD head, so its latent can only be shown through the head's pixels."""
    options = getattr(model, "model_options", None) or {}
    return MARK_KEY in options


def outer_sample(executor, noise, latent_image, sampler, sigmas, *args, latent_shapes=None, **kwargs):
    """Give every run its own holder, and file the pixels it decoded under the video latent the run returns."""
    import comfy.utils
    holder = Holder()
    executor.class_obj.model_options.setdefault("transformer_options", {})[PIXELS_KEY] = holder
    output = executor(noise, latent_image, sampler, sigmas, *args, latent_shapes=latent_shapes, **kwargs)
    if holder.pixels is not None:
        video = output if latent_shapes is None or len(latent_shapes) < 2 else comfy.utils.unpack_latents(output, latent_shapes)[0]
        _PIXELS[fingerprint(video)] = holder.pixels
        while len(_PIXELS) > MAX_KEPT:
            _PIXELS.popitem(last=False)
    return output


class NormOut(nn.Module):
    """The DWD output norm: an affine-free LayerNorm modulated by (scale, shift) = linear(silu(t_emb))."""

    def __init__(self, hidden, t_dim, eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.eps = eps
        self.linear = operations.Linear(t_dim, 2 * hidden, dtype=dtype, device=device)

    def forward(self, x, t_emb, row):
        scale, shift = self.linear(nn.functional.silu(t_emb)).to(x.dtype).chunk(2, dim=-1)
        return nn.functional.layer_norm(x, x.shape[-1:], eps=self.eps).mul_(1 + scale[row]).add_(shift[row])


def _assign(module, sd):
    """Load tensors as they are (a dense weight keeps the adapter's dtype)."""
    for name, m in module.named_modules():
        prefix = name + "." if name else ""
        if hasattr(m, "factory_kwargs") and prefix + "weight" in sd and prefix + "comfy_quant" not in sd:
            dtype = sd[prefix + "weight"].dtype
            m.factory_kwargs = dict(m.factory_kwargs, dtype=dtype)
            if m.bias is not None:
                m.bias = nn.Parameter(torch.empty_like(m.bias, dtype=dtype), requires_grad=False)
    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"the DWD tensors do not match the pixel head: missing {missing[:4]}, unexpected {unexpected[:4]}")


class Head(nn.Module):
    """The norm and the pixel refiner of a DWD adapter, built from its tensors."""

    def __init__(self, sd, eps, dtype=None, device=None, operations=None):
        super().__init__()
        from .sol_head import SolH3DWDHead
        hidden, t_dim = sd["dwd_norm_out.linear.weight"].shape
        hidden //= 2
        refiner = {k[len("dwd_pixel_refiner."):]: v for k, v in sd.items() if k.startswith("dwd_pixel_refiner.")}
        if "register_tokens" not in refiner:
            raise ValueError("this DWD adapter has a pixel refiner this port does not know")
        in_channels = refiner["feature_expand.weight"].shape[1]
        self.dwd_norm_out = NormOut(hidden, t_dim, eps, dtype=dtype, device=device, operations=operations)
        self.dwd_pixel_refiner = SolH3DWDHead.from_state_dict(refiner, in_channels, PIXEL_PATCH_SIZE, dtype=dtype, device=device,
                                                              operations=operations)

    def decode(self, h, t_emb, mod_segments, layout):
        """The (F, H, W, 3) pixels in [0, 1], fp16 on the intermediate device, of the final hidden states h."""
        _, latent_t, latent_h, latent_w, _ = layout.signature
        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        row = next(r for a, b, r in mod_segments if (a, b) == (va, vb)) // 3          # block mod rows are timestep * 3 + tag
        features = self.dwd_norm_out(h[va:vb], t_emb, row)[None]
        expanded = self.dwd_pixel_refiner.expand(features, (latent_t, latent_h // 2, latent_w // 2))
        del features
        pixels = self.dwd_pixel_refiner.refine(*expanded)[0]
        return pixels.add_(1).mul_(0.5).permute(0, 2, 3, 1).half()


def install(patcher, sd, eps):
    """Put the head on `patcher` (a clone the caller owns), catch the last step's hidden states, and mark the model."""
    import comfy.model_patcher
    import comfy.ops
    head = Head(sd, eps, dtype=sd["dwd_norm_out.linear.weight"].dtype, device=patcher.offload_device,
                operations=comfy.ops.mixed_precision_ops())
    _assign(head, sd)
    holder_patcher = comfy.model_patcher.CoreModelPatcher(head, load_device=patcher.load_device, offload_device=patcher.offload_device)
    patcher.set_additional_models("funui_pi_h3_dwd", [holder_patcher])
    run_blocks = patcher.get_model_object("diffusion_model._run_blocks")
    ref = weakref.ref(head)          # the head lives as long as its patcher; a patch left on the shared denoiser must not keep it

    def run_blocks_and_decode(h, t_emb, mod_segments, layout, transformer_options):
        h = run_blocks(h, t_emb, mod_segments, layout, transformer_options)
        holder = transformer_options.get(PIXELS_KEY)
        sigmas = transformer_options.get("sigmas")
        last = transformer_options.get("sample_sigmas")
        if holder is not None and last is not None and transformer_options.get("cond_or_uncond") == [0] and torch.isclose(
                sigmas.flatten()[0].float(), last[last > 0][-1].float().to(sigmas.device)):
            holder.pixels = ref().decode(h, t_emb, mod_segments, layout)
        return h

    return run_blocks_and_decode
