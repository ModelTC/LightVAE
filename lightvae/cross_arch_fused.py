"""Inference-only normalization/activation fusion for Hunyuan and LTX VAEs."""

import torch
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def torch_norm_act(x, kind, gamma=None, bias=None, eps=1e-12, scale=1.0):
    """Native operation order, including each low-precision rounding step."""
    if kind == "hunyuan":
        value = F.normalize(x, dim=1, eps=eps) * scale
        if gamma is not None:
            value = value * gamma.reshape(1, -1, *([1] * (x.ndim - 2)))
        if bias is not None:
            value = value + (bias.reshape(1, -1, *([1] * (x.ndim - 2)))
                             if isinstance(bias, torch.Tensor) else bias)
        return value * torch.sigmoid(value)
    if kind == "ltx":
        rms = (x.square().mean(dim=1, keepdim=True) + eps).sqrt()
        return F.silu(x / rms)
    raise ValueError(f"Unknown normalization: {kind}")


if triton is not None:
    @triton.jit
    def _norm_act(X, G, B, Y, C: tl.constexpr, T: tl.constexpr,
                  H: tl.constexpr, W: tl.constexpr,
                  S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                  S3: tl.constexpr, S4: tl.constexpr,
                  HUNYUAN: tl.constexpr, HAS_GAMMA: tl.constexpr,
                  TENSOR_BIAS: tl.constexpr, BIAS: tl.constexpr,
                  EPS: tl.constexpr, SCALE: tl.constexpr,
                  BC: tl.constexpr, BS: tl.constexpr):
        # Full-resolution video activations can have strides beyond int32.
        batch = tl.program_id(1).to(tl.int64)
        position = tl.program_id(0) * BS + tl.arange(0, BS)
        channel = tl.arange(0, BC)
        position = position.to(tl.int64)
        channel = channel.to(tl.int64)
        spatial = (position // (H * W)) * S2 + ((position // W) % H) * S3 + (position % W) * S4
        mask = (channel[:, None] < C) & (position[None, :] < T * H * W)
        x = tl.load(X + batch * S0 + channel[:, None] * S1 + spatial[None, :],
                    mask, other=0).to(tl.float32)
        dtype = X.dtype.element_ty
        if HUNYUAN:
            length = tl.sqrt(tl.sum(x * x, axis=0)).to(dtype).to(tl.float32)
            eps = tl.full((), EPS, tl.float32).to(dtype).to(tl.float32)
            value = (x / tl.maximum(length[None, :], eps)).to(dtype).to(tl.float32)
            value = (value * SCALE).to(dtype).to(tl.float32)
            if HAS_GAMMA:
                gamma = tl.load(G + channel, channel < C, other=0).to(tl.float32)
                value = (value * gamma[:, None]).to(dtype).to(tl.float32)
            if TENSOR_BIAS:
                bias = tl.load(B + channel, channel < C, other=0).to(tl.float32)
                value = (value + bias[:, None]).to(dtype).to(tl.float32)
            elif BIAS != 0:
                value = (value + BIAS).to(dtype).to(tl.float32)
            sigmoid = (1. / (1. + tl.exp(-value))).to(dtype).to(tl.float32)
            output = value * sigmoid
        else:
            squared = (x * x).to(dtype).to(tl.float32)
            mean = (tl.sum(squared, axis=0) / C).to(dtype).to(tl.float32)
            variance = (mean + EPS).to(dtype).to(tl.float32)
            rms = tl.sqrt(variance).to(dtype).to(tl.float32)
            value = (x / rms[None, :]).to(dtype).to(tl.float32)
            output = value / (1. + tl.exp(-value))
        offset = (batch * C + channel[:, None]) * (T * H * W) + position[None, :]
        tl.store(Y + offset, output, mask)


def fused_norm_act(x, kind, gamma=None, bias=None, eps=1e-12, scale=1.0):
    """Fuse channel normalization and activation; preserve native FP32 behavior.

    ``hunyuan`` is L2 normalization, affine, and explicit sigmoid/multiply.
    ``ltx`` is PixelNorm followed by SiLU; its caller supplies PixelNorm's eps.
    Four- and five-dimensional inputs may have arbitrary strides.
    """
    if kind not in ("hunyuan", "ltx"):
        raise ValueError(f"Unknown normalization: {kind}")
    supported = (triton is not None and x.is_cuda and torch.version.hip is None
                 and x.dtype == torch.bfloat16 and x.ndim in (4, 5)
                 and not (torch.is_grad_enabled() and (x.requires_grad
                     or (gamma is not None and gamma.requires_grad)
                     or (isinstance(bias, torch.Tensor) and bias.requires_grad)))
                 and torch.cuda.get_device_capability(x.device)[0] >= 8)
    if not supported:
        return torch_norm_act(x, kind, gamma, bias, eps, scale)
    batch, channels = x.shape[:2]
    time, height, width = (1, *x.shape[2:]) if x.ndim == 4 else x.shape[2:]
    strides = ((x.stride(0), x.stride(1), 0, *x.stride()[2:])
               if x.ndim == 4 else x.stride())
    gamma_arg = x if gamma is None else gamma.contiguous()
    tensor_bias = isinstance(bias, torch.Tensor)
    bias_arg = bias.contiguous() if tensor_bias else x
    scalar_bias = 0.0 if bias is None or tensor_bias else float(bias)
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block_spatial = 16 if channels >= 512 else 32
    with torch.cuda.device(x.device):
        _norm_act[(triton.cdiv(time * height * width, block_spatial), batch)](
            x, gamma_arg, bias_arg, output, channels, time, height, width,
            *strides, kind == "hunyuan", gamma is not None, tensor_bias,
            scalar_bias, eps, scale, triton.next_power_of_2(channels), block_spatial,
            num_warps=8, enable_fp_fusion=False,
        )
    return output
