"""The H3 corrector network: a 55-block space-time transformer that reads the noisy picture and H3's velocity and
predicts a velocity residual. Written for the layout of the corrector files (Sulphur AI's s3_corrector.safetensors:
the weight names and the architecture in the file's metadata are theirs), run here with plain PyTorch.
"""

import dataclasses
import json
import math

import torch
import torch.nn.functional as F
from torch import nn

METADATA_KEY = "h3_corrector"


@dataclasses.dataclass(frozen=True)
class Config:
    video_channels: int = 24
    text_dim: int = 5120
    width: int = 1280
    depth: int = 55
    heads: int = 20
    head_dim: int = 64
    mlp_ratio: int = 4
    time_embed_dim: int = 256
    patch: tuple = (1, 2, 2)
    norm_eps: float = 1e-5

    @classmethod
    def from_metadata(cls, metadata):
        if not metadata or METADATA_KEY not in metadata:
            raise ValueError(f"not an H3 corrector: the file has no {METADATA_KEY!r} entry in its metadata")
        raw = json.loads(metadata[METADATA_KEY])
        config = cls(**{**raw, "patch": tuple(raw["patch"])})
        if config.width != config.heads * config.head_dim:
            raise ValueError("not an H3 corrector: width is not heads * head_dim")
        return config


def sigma_embedding(sigma, width, period=10_000.0):
    half = width // 2
    freq = torch.exp(-math.log(period) * torch.arange(half, device=sigma.device, dtype=torch.float32) / half)
    angle = sigma.float()[:, None] * freq[None] * 1000.0
    return F.pad(torch.cat((angle.cos(), angle.sin()), dim=-1), (0, width - 2 * half))


def rotation(frames, height, width, head_dim, device):
    """(cos, sin) [tokens, head_dim / 2]: the head's channel pairs turn by the token's frame, row and column in turn."""
    t_width = 2 * (head_dim // 8)
    y_width = 2 * ((head_dim - t_width) // 4)
    sizes = (t_width, y_width, head_dim - t_width - y_width)
    index = torch.arange(frames * height * width, device=device)
    where = (index // (height * width), (index // width) % height, index % width)
    angles = []
    for pos, size in zip(where, sizes):
        pairs = size // 2
        freq = torch.exp(-math.log(10_000.0) * torch.arange(pairs, device=device, dtype=torch.float32) / max(1, pairs))
        angles.append(pos.float()[:, None] * freq[None])
    angle = torch.cat(angles, dim=-1)
    return angle.cos(), angle.sin()


def turn(x, cos, sin):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.head_dim
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.out = nn.Linear(c.width, c.width)

    def forward(self, rows, rope):
        b, n, d = rows.shape
        q, k, v = (p.transpose(1, 2) for p in self.qkv(rows).reshape(b, n, 3, self.heads, self.head_dim).unbind(2))
        q, k = turn(q, *rope), turn(k, *rope)
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))


class CrossAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.head_dim
        self.q, self.k, self.v, self.out = (nn.Linear(c.width, c.width) for _ in range(4))

    def forward(self, rows, text):
        b, n, d = rows.shape
        split = lambda x: x.reshape(b, x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        return self.out(F.scaled_dot_product_attention(split(self.q(rows)), split(self.k(text)), split(self.v(text))).transpose(1, 2).reshape(b, n, d))


class FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(c.width, c.width * c.mlp_ratio), nn.Linear(c.width * c.mlp_ratio, c.width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm_self, self.norm_cross, self.norm_mlp = (nn.LayerNorm(c.width, eps=c.norm_eps) for _ in range(3))
        self.self_attention, self.cross_attention, self.mlp = SelfAttention(c), CrossAttention(c), FeedForward(c)
        self.modulation = nn.Linear(c.width, 6 * c.width)

    def forward(self, rows, time, text, rope):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (m[:, None] for m in self.modulation(time).chunk(6, dim=-1))
        rows = rows + (1.0 + gate_a) * self.self_attention(self.norm_self(rows) * (1.0 + scale_a) + shift_a, rope)
        rows = rows + self.cross_attention(self.norm_cross(rows), text)
        return rows + (1.0 + gate_f) * self.mlp(self.norm_mlp(rows) * (1.0 + scale_f) + shift_f)


class Corrector(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.patch_in = nn.Conv3d(2 * c.video_channels, c.width, kernel_size=c.patch, stride=c.patch)
        self.text_norm = nn.LayerNorm(c.text_dim, eps=c.norm_eps)
        self.text_projection = nn.Linear(c.text_dim, c.width)
        self.time_mlp = nn.Sequential(nn.Linear(3 * c.time_embed_dim, c.width), nn.SiLU(), nn.Linear(c.width, c.width))
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.depth))
        self.final_norm = nn.LayerNorm(c.width, eps=c.norm_eps)
        self.final_modulation = nn.Linear(c.width, 2 * c.width)
        self.patch_out = nn.Linear(c.width, c.video_channels * math.prod(c.patch))

    def forward(self, latent, velocity, sigma, sigma_next, text):
        """The velocity residual for the step sigma -> sigma_next. latent, velocity [B, C, T, H, W]; text [B, L, text_dim]."""
        c = self.config
        b, ch, frames, height, width = latent.shape
        pt, ph, pw = c.patch
        if velocity.shape != latent.shape or ch != c.video_channels or frames % pt or height % ph or width % pw:
            raise ValueError(f"a picture of {tuple(latent.shape)} does not fit this corrector")
        s0 = torch.as_tensor(sigma, device=latent.device, dtype=torch.float32).reshape(-1).expand(b)
        s1 = torch.as_tensor(sigma_next, device=latent.device, dtype=torch.float32).reshape(-1).expand(b)
        grid = self.patch_in(torch.cat((latent, velocity), dim=1)).permute(0, 2, 3, 4, 1)
        rows = grid.reshape(b, -1, c.width)
        text = self.text_projection(self.text_norm(text))
        time = self.time_mlp(torch.cat([sigma_embedding(s, c.time_embed_dim) for s in (s0, s1, s0 - s1)], dim=-1).to(rows.dtype))
        rope = rotation(frames // pt, height // ph, width // pw, c.head_dim, latent.device)
        for block in self.blocks:
            rows = block(rows, time, text, rope)
        shift, scale = (m[:, None] for m in self.final_modulation(time).chunk(2, dim=-1))
        out = self.patch_out(self.final_norm(rows) * (1 + scale) + shift)
        out = out.reshape(b, frames // pt, height // ph, width // pw, ch, pt, ph, pw)
        return out.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, ch, frames, height, width)


def cap(raw, velocity, fraction, floor):
    """The residual with its RMS softly capped at fraction * max(RMS of the velocity, floor), per clip. -> (residual, its RMS / the velocity's)."""
    raw, velocity = raw.float(), velocity.float()
    dims = tuple(range(1, raw.ndim))
    base = velocity.square().mean(dim=dims).sqrt()
    budget = fraction * base.clamp_min(floor)
    scale = torch.rsqrt(1.0 + raw.square().mean(dim=dims) / budget.square().clamp_min(1e-12))
    applied = raw * scale.reshape(-1, *([1] * (raw.ndim - 1)))
    return applied, applied.square().mean(dim=dims).sqrt() / base.clamp_min(1e-12)
