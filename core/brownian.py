"""Brownian noise that does not warn about its own start.

torchsde rounds a BrownianTree's start to 6 decimals (its default tolerance) and then compares the first query
against the ROUNDED start. A schedule whose smallest sigma has a 7th decimal of 5 or more (0.4465445876 rounds to
0.446545) is therefore queried "before" the tree starts and torchsde prints
`Should have ta>=t0 but got ta=... and t0=...`. The query is clamped by 4e-7, so the picture is not hurt, but the
log fills with warnings on every SDE-type sampler, and Two stages / Turbo hand such sigmas to the sampler.

ComfyUI builds its trees from `sigmas[sigmas > 0].min()` and `sigmas.max()`. The class below widens that span to
the next 6-decimal value outside it (plus one more step of margin) before the tree rounds, so the start is never
above the first query and the end never below the last. Noise inside the span is unchanged.
"""

import math

import torch

_SCALE = 1e6            # torchsde's default tol=1e-6 rounds to 6 decimals
_installed = False


def widened(lo, hi):
    """(lo, hi) as tensors of the same device and dtype as the inputs, pushed outwards past torchsde's rounding."""
    lo, hi = torch.as_tensor(lo), torch.as_tensor(hi)
    new_lo = math.floor(float(lo) * _SCALE) / _SCALE - 1.0 / _SCALE
    new_hi = math.ceil(float(hi) * _SCALE) / _SCALE + 1.0 / _SCALE
    return (torch.as_tensor(new_lo).to(device=lo.device, dtype=lo.dtype),
            torch.as_tensor(new_hi).to(device=hi.device, dtype=hi.dtype))


def install():
    """Swap ComfyUI's BrownianTreeNoiseSampler for the widened one. Safe to call twice."""
    global _installed
    import comfy.k_diffusion.sampling as ksampling
    if _installed or getattr(ksampling.BrownianTreeNoiseSampler, "_funui_widened", False):
        _installed = True
        return
    original = ksampling.BrownianTreeNoiseSampler

    class BrownianTreeNoiseSampler(original):
        _funui_widened = True

        def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda v: v, cpu=False):
            super().__init__(x, sigma_min, sigma_max, seed=seed, transform=transform, cpu=cpu)
            t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
            lo, hi = (t0, t1) if t0 < t1 else (t1, t0)
            lo, hi = widened(lo, hi)
            self.tree = ksampling.BatchedBrownianTree(x, lo, hi, seed, cpu=cpu)

    ksampling.BrownianTreeNoiseSampler = BrownianTreeNoiseSampler
    _installed = True
