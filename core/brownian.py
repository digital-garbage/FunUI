"""Brownian noise that does not warn about its own start.

torchsde rounds a BrownianTree's start to 6 decimals (its default tolerance) and then compares the first query
against the ROUNDED start. A schedule whose smallest sigma has a 7th decimal of 5 or more (0.4465445876 rounds to
0.446545) is therefore queried "before" the tree starts and torchsde prints
`Should have ta>=t0 but got ta=... and t0=...`, then clamps the query to the start itself. The picture is not hurt,
but the log fills with warnings on every SDE-type sampler, and Two stages / Turbo hand such sigmas to the sampler.

The class below builds the tree exactly as ComfyUI does (so the noise of a seed is the stock noise) and clamps a
query to the tree's own span BEFORE torchsde sees it, which is the very same clamp torchsde would apply, silently.
"""

import torch

_installed = False


def clamped(t, lo, hi):
    """`t` limited to [lo, hi] (python floats) on t's own device; a value already inside comes back untouched.
    The bound is held in float64 so a bf16 query is not rounded back out of the span."""
    v = float(t)
    if lo <= v <= hi:
        return t
    return torch.tensor(min(max(v, lo), hi), dtype=torch.float64, device=t.device)


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

        def __call__(self, sigma, sigma_next):
            t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
            tree = self.tree.trees[0]._interval
            lo, hi = float(tree._start), float(tree._end)
            noise = self.tree(clamped(t0, lo, hi), clamped(t1, lo, hi)).to(dtype=tree.dtype)      # a float64 bound must not widen the noise
            return noise / (t1 - t0).abs().sqrt()

    ksampling.BrownianTreeNoiseSampler = BrownianTreeNoiseSampler
    _installed = True
