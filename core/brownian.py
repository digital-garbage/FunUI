"""Brownian noise that does not warn about its own start.

torchsde rounds a BrownianTree's start to 6 decimals (its default tolerance) and then compares the first query
against the ROUNDED start. A schedule whose smallest sigma has a 7th decimal of 5 or more (0.4465445876 rounds to
0.446545) is therefore queried "before" the tree starts and torchsde prints
`Should have ta>=t0 but got ta=... and t0=...`, then clamps the query to the start itself. The picture is not hurt,
but the log fills with warnings on every SDE-type sampler, and Two stages / Turbo hand such sigmas to the sampler.

The class below builds the tree exactly as ComfyUI does (so the noise of a seed is the stock noise) and limits a
query to the tree's own span BEFORE torchsde sees it, which is the very same limit torchsde would apply, silently.
"""

import torch

_installed = False


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
            """ComfyUI's own call, with one difference: a query outside the tree's span is limited to it HERE, as a plain
            number, instead of inside torchsde (which does the same limiting, after a warning)."""
            t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
            batch = self.tree
            a, b, sign = batch.sort(t0, t1)
            device, dtype = a.device, a.dtype
            span = batch.trees[0]._interval
            lo, hi = float(span._start), float(span._end)

            def query(t):                                  # the same cast ComfyUI applies to the query of a CPU tree
                v = float(t.detach().cpu().float()) if batch.cpu_tree else float(t)
                return min(max(v, lo), hi)
            w = torch.stack([tree(query(a), query(b)) for tree in batch.trees]).to(device=device, dtype=dtype) * (batch.sign * sign)
            return (w if batch.batched else w[0]) / (t1 - t0).abs().sqrt()

    ksampling.BrownianTreeNoiseSampler = BrownianTreeNoiseSampler
    _installed = True
