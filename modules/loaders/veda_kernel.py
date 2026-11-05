"""Block-sparse INT8 attention in Triton for Veda: walks only the kept 128-token tiles.

UNTESTED on a GPU: written on a machine with no Triton or CUDA. `check` runs the kernel on a small problem
against the fp32 reference before any real use, and the caller falls back to dense attention with the reason
when it fails.

Derived from SageAttention 1.0.6 (BSD-3-Clause, Copyright (c) 2024, Jintao Zhang, Haofeng Huang),
quant_per_block.py and attn_qk_int8_per_block.py, through Veda-on-ComfyUI's fork of them (MIT); the licence
texts are in VEDA-NOTICE.txt. Upstream's arithmetic is kept on purpose: INT8 Q and K with one scale per
block, the softmax scale folded into Q's quantisation so the loop can use exp2, P and V in fp16.

What differs from upstream, as in Veda's fork:
  * the key loop walks a per (head, query tile) list of kept tiles instead of the whole sequence;
  * padding slots are masked by each tile's real-row count (real rows are a prefix of every tile), where
    upstream loads out-of-range keys as zeros and lets them into the softmax;
  * a tile is two key blocks of 64 (upstream's block), expanded on the host so the loop stays flat;
  * tile-ordered [N, H, D] tensors throughout. Veda's optional TMA path (off by default there) is not ported.
"""
import contextlib

import torch
import triton
import triton.language as tl

from .veda_core import TILE, block_mask, build_layout, column_blocks, gather, reference_attend, select, tile_index_list

KEY_BLOCK = 64
WARPS = 8 if torch.version.hip else 4      # measured by Veda on an RTX 5070 (4); 8 on ROCm
STAGES = 3
LOG2E = 1.4426950408889634


@triton.jit
def _quantize(X, Xq, Scale, stride_n, stride_h, stride_s, pre_scale, D: tl.constexpr, BLK: tl.constexpr):
    """One INT8 scale per (block of BLK slots, head), upstream's rounding."""
    blk, head = tl.program_id(0), tl.program_id(1)
    offs = head * stride_h + (blk * BLK + tl.arange(0, BLK))[:, None] * stride_n + tl.arange(0, D)[None, :]
    x = tl.load(X + offs).to(tl.float32) * pre_scale
    scale = tl.max(tl.abs(x)) / 127.0
    scale = tl.where(scale > 0.0, scale, 1.0)           # an all-padding block has no scale; 1.0 keeps zeros zero
    q = x / scale
    q += 0.5 * tl.where(q >= 0, 1, -1)
    tl.store(Xq + offs, q.to(tl.int8))
    tl.store(Scale + head * stride_s + blk, scale)


@triton.jit
def _attend(Q, K, V, Q_scale, K_scale, Index, Count, Valid, Out, stride_n, stride_h, stride_vn, stride_vh,
            stride_qs, stride_ks, stride_ih, stride_iq, n_q_tiles, D: tl.constexpr, BLK: tl.constexpr, BLOCK_M: tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n, offs_d = tl.arange(0, BLK), tl.arange(0, D)
    q = tl.load(Q + head * stride_h + offs_m[:, None] * stride_n + offs_d[None, :])
    q_scale = tl.load(Q_scale + head * stride_qs + tile)
    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)      # finite: an all-padding block must not make -inf - -inf
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    count = tl.load(Count + head * n_q_tiles + tile)
    row = Index + head * stride_ih + tile * stride_iq
    for i in range(0, count):
        block = tl.load(row + i)
        start = block * BLK
        k = tl.load(K + head * stride_h + (start + offs_n)[None, :] * stride_n + offs_d[:, None])
        qk = tl.dot(q, k).to(tl.float32) * q_scale * tl.load(K_scale + head * stride_ks + block)
        qk = tl.where(offs_n[None, :] < tl.load(Valid + block), qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + head * stride_vh + (start + offs_n)[:, None] * stride_vn + offs_d[None, :])
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
        m_i = m_ij
    l_i = tl.where(l_i > 0.0, l_i, 1.0)         # an empty row would divide by zero; selection always keeps the diagonal
    tl.store(Out + head * stride_h + offs_m[:, None] * stride_n + offs_d[None, :], (acc / l_i[:, None]).to(Out.type.element_ty))


def _quant(x, block, pre_scale=1.0):
    slots, heads, dim = x.shape
    out = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((heads, slots // block), dtype=torch.float32, device=x.device)
    _quantize[(slots // block, heads)](x, out, scale, x.stride(0), x.stride(1), scale.stride(0), pre_scale=pre_scale, D=dim, BLK=block, num_warps=4)
    return out, scale


def attend(q, k, v, mask, lay, timer=None):
    """The block-sparse attention callable veda_core.attention takes. q, k, v: [N, H', D] contiguous, bf16/fp16, tile order;
    mask: [H', n_tiles, n_tiles] bool. Returns [N, H', D] in q's dtype. `timer` (veda_timing) splits the call into
    kernel.index (the lists the kernel walks), kernel.quantize (INT8 q and k) and kernel.attend (the kernel itself)."""
    t = timer or (lambda name: contextlib.nullcontext())
    slots, heads, dim = q.shape
    with t("kernel.index"):
        index, count = tile_index_list(mask & lay.kv_ok, lay.widest)         # no read of the device to size the list
        sub = TILE // KEY_BLOCK                          # kept tiles -> kept key blocks, the kernel's unit
        blocks = ((index.to(torch.int32) * sub)[..., None] + torch.arange(sub, device=q.device, dtype=torch.int32)).flatten(-2).contiguous()
        valid = (lay.valid_count[:, None] - torch.arange(sub, device=q.device) * KEY_BLOCK).clamp_(0, KEY_BLOCK).flatten().to(torch.int32).contiguous()
    with t("kernel.quantize"):
        q8, q_scale = _quant(q, TILE, dim ** -0.5 * LOG2E)
        k8, k_scale = _quant(k, KEY_BLOCK)
        v16, out = v.to(torch.float16), torch.empty_like(q)
    with t("kernel.attend"):
        _attend[(slots // TILE, heads)](
            q8, k8, v16, q_scale, k_scale, blocks, (count * sub).contiguous(), valid, out, q.stride(0), q.stride(1), v16.stride(0),
            v16.stride(1), q_scale.stride(0), k_scale.stride(0), blocks.stride(0), blocks.stride(1), n_q_tiles=slots // TILE, D=dim,
            BLK=KEY_BLOCK, BLOCK_M=TILE, num_warps=WARPS, num_stages=STAGES)
    return out


attend.times_itself = True


def check(device, dtype=torch.bfloat16):
    """Run the kernel on a small problem (history span, target span with partial tiles, global rows, per-row budgets)
    and compare it with the fp32 reference. Returns None when it holds, else why not (a kernel that ignores the mask
    fails too: an earlier FlashAttention build accepted one and ignored it)."""
    spans = [(70, (1, 8, 16), (1, 8, 16)), (238, (5, 8, 8), (4, 4, 8))]
    lay = build_layout(spans, 238 + 320, device)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(3, lay.seq_len, 2, 128, generator=gen) * 1.5          # peaky softmax: skipped blocks matter
    heads = torch.arange(2, device=device)
    q, k, v = (gather(t.to(device=device, dtype=dtype), lay, heads) for t in x)
    scores = torch.randn(2, lay.n_video, lay.n_video, generator=gen).to(device)
    mask = block_mask(*select(scores, lay, column_blocks(lay, (None, 2), (None, 1))), lay)
    try:
        got = attend(q, k, v, mask, lay)
    except Exception as exc:  # noqa: BLE001 - kernels fail in many ways; report, do not crash
        return f"the Triton kernel failed its self-test: {type(exc).__name__}: {exc}"
    real = lay.slot_valid.bool()
    want = reference_attend(q, k, v, mask, lay)[real].float()
    dense = reference_attend(q, k, v, torch.ones_like(mask) & lay.kv_ok, lay)[real].float()
    got = got[real].float()
    if (got - dense).abs().max() < 0.5 * (dense - want).abs().max():
        return "the Triton kernel ignores the block mask on this device (computes dense attention)"
    limit = 0.05 * max(1.0, want.abs().max().item())           # INT8 with per-block scales: ~1-3% is the format's floor
    err = (got - want).abs().max().item()
    return None if err <= limit else f"the Triton kernel gives wrong results on this device (max error {err:.3g}, allowed {limit:.3g})"
