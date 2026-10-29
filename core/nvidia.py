"""Helpers shared by the NVIDIA RTX video effects (nvidia-vfx): the packed 10-bit layout and the driver-library hint."""

def pack10(x):
    """[3,H,W] float 0-1 -> (H,W) uint32 in the SDK's RGB10A2 layout: R bits 0-9, G 10-19, B 20-29, A 30-31."""
    import torch
    v = (x.clamp(0, 1) * 1023).round().to(torch.int64)
    return (v[0] | (v[1] << 10) | (v[2] << 20) | (3 << 30)).to(torch.uint32)


def unpack10(p):
    """(H,W) uint32 -> [3,H,W] float 0-1."""
    import torch
    v = p.to(torch.int64)
    return torch.stack([v & 1023, (v >> 10) & 1023, (v >> 20) & 1023]).float() / 1023.0


def ngx_hint():
    """Why load() can fail on a rental: the SDK needs the driver's own libnvidia-ngx.so.1, which a container host may not mount."""
    import ctypes
    import sys
    if sys.platform != "linux":
        return ""
    try:
        ctypes.CDLL("libnvidia-ngx.so.1")
        return " The driver's NGX library is present, so check the driver is 570.190+ / 580.82+ / 590.44+ and the card is Turing or newer."
    except OSError:
        return (" This machine has no libnvidia-ngx.so.1 (NVIDIA's driver library that the RTX video effects need; some rented GPUs do not mount it, "
                "like libnvidia-encode). Use an upscale model instead, or rent a host that exposes the full NVIDIA driver (graphics/video capabilities).")
