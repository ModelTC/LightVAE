"""Fused normalization and temporal history kernels for Wan LightVAE."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_bf16(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.bf16 $0, $1, $2;", constraints="=h,h,h", args=[a, b],
        dtype=tl.bfloat16, is_pure=True, pack=1,
    )


@triton.jit
def _norm(X, G, B, Y, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
          W: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
          S2: tl.constexpr, S3: tl.constexpr, S4: tl.constexpr,
          SCALE: tl.constexpr, SILU: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr):
    batch = tl.program_id(1)
    s = tl.program_id(0) * BS + tl.arange(0, BS)
    ch = tl.arange(0, BC)
    spatial = (s // (H * W)) * S2 + ((s // W) % H) * S3 + (s % W) * S4
    offsets = batch * S0 + ch[:, None] * S1 + spatial[None, :]
    mask = (ch[:, None] < C) & (s[None, :] < T * H * W)
    x_low = tl.load(X + offsets, mask, other=0)
    squares = tl.inline_asm_elementwise(
        "mul.rn.bf16 $0, $1, $2;", constraints="=h,h,h", args=[x_low, x_low],
        dtype=tl.bfloat16, is_pure=True, pack=1,
    )
    squared_length = tl.reduce(squares, 0, _add_bf16).to(tl.float32)
    dtype = X.dtype.element_ty
    length = tl.sqrt(squared_length).to(dtype).to(tl.float32)
    eps = tl.full((), 1.e-12, tl.float32).to(dtype).to(tl.float32)
    denominator = tl.maximum(length, eps)
    value = (x_low.to(tl.float32) / denominator[None, :]).to(dtype).to(tl.float32)
    value = (value * SCALE).to(dtype).to(tl.float32)
    gamma = tl.load(G + ch, ch < C, other=0).to(tl.float32)
    bias = tl.load(B + ch, ch < C, other=0).to(tl.float32)
    value = (value * gamma[:, None]).to(dtype).to(tl.float32)
    value = (value + bias[:, None]).to(dtype).to(tl.float32)
    if SILU:
        value = value / (1. + tl.exp(-value))
    out = (batch * C + ch[:, None]) * (T * H * W) + s[None, :]
    tl.store(Y + out, value, mask)


def rms_silu(x, gamma, bias, scale, silu):
    n, c = x.shape[:2]
    t, h, w = (1, *x.shape[2:]) if x.ndim == 4 else x.shape[2:]
    strides = (x.stride(0), x.stride(1), 0, *x.stride()[2:]) if x.ndim == 4 else x.stride()
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        # Spatial tiles use grid.x; grid.y's 65535 limit is too small for 720p.
        _norm[(triton.cdiv(t * h * w, 32), n)](
            x, gamma.contiguous(), bias.contiguous(), out, c, t, h, w,
            *strides, scale, silu, triton.next_power_of_2(c), 32,
            num_warps=8, enable_fp_fusion=False,
        )
    return out


@triton.jit
def _history_up(P, A, CACHE, OUT, CO: tl.constexpr, R: tl.constexpr,
                T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                P3: tl.constexpr, P4: tl.constexpr,
                C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
                C3: tl.constexpr, C4: tl.constexpr,
                A0: tl.constexpr, A1: tl.constexpr, HAS_CACHE: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    co = tl.program_id(0) * BM + tl.arange(0, BM)
    spatial = tl.program_id(1) * BN + tl.arange(0, BN)
    bt = tl.program_id(2)
    batch, time = bt // T, bt % T
    pp = (spatial // W) * P3 + (spatial % W) * P4
    cp = (spatial // W) * C3 + (spatial % W) * C4
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(R, BK)):
        k = start * BK + rk
        mask = (k[:, None] < R) & (spatial[None, :] < H * W)
        if time == 0:
            if HAS_CACHE:
                v1 = tl.load(CACHE + batch * C0 + C1 + k[:, None] * C2 + cp[None, :],
                             mask, other=0)
            else:
                v1 = tl.full((BK, BN), 0, tl.bfloat16)
        else:
            v1 = tl.load(P + batch * P0 + (time - 1) * P1 + (CO + k[:, None]) * P2 + pp[None, :],
                         mask, other=0)
        if time < 2:
            if HAS_CACHE:
                v2 = tl.load(CACHE + batch * C0 + time * C1 + (R + k[:, None]) * C2 + cp[None, :],
                             mask, other=0)
            else:
                v2 = tl.full((BK, BN), 0, tl.bfloat16)
        else:
            v2 = tl.load(P + batch * P0 + (time - 2) * P1 + (CO + R + k[:, None]) * P2 + pp[None, :],
                         mask, other=0)
        q = (v1.to(tl.float32) + v2.to(tl.float32)).to(tl.bfloat16)
        weight = tl.load(A + co[:, None] * A0 + k[None, :] * A1,
                         (co[:, None] < CO) & (k[None, :] < R), other=0)
        acc = tl.dot(weight, q, acc)
    mask = (co[:, None] < CO) & (spatial[None, :] < H * W)
    base = tl.load(P + batch * P0 + time * P1 + co[:, None] * P2 + pp[None, :], mask, other=0)
    out = acc.to(tl.bfloat16).to(tl.float32) + base.to(tl.float32)
    tl.store(OUT + (bt * CO + co[:, None]) * (H * W) + spatial[None, :], out, mask)


def fused_history_up(p, weight, cache):
    batch, time, _, height, width = p.shape
    co, rank = weight.shape[:2]
    out = torch.empty((batch, time, co, height, width), device=p.device, dtype=p.dtype)
    old = p if cache is None else cache
    with torch.cuda.device(p.device):
        _history_up[(triton.cdiv(co, 32), triton.cdiv(height * width, 64), batch * time)](
            p, weight, old, out, co, rank, time, height, width, *p.stride(), *old.stride(),
            *weight.stride()[:2], cache is not None, 32, 64, 32,
            num_warps=4, enable_fp_fusion=False,
        )
    return out
