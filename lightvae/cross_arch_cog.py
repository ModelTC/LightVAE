"""CogVideoX inference adapters for converted TLR decoders only."""

import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl

from .cross_arch_tlr import TLRConv3d


@triton.jit
def _offset(i, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
            W: tl.constexpr, S: tl.constexpr):
    w = i % W
    h = i // W % H
    t = i // (W * H) % T
    c = i // (W * H * T) % C
    b = i // (W * H * T * C)
    return b * S[0] + c * S[1] + t * S[2] + h * S[3] + w * S[4]


@triton.jit
def _pack_history(X, HISTORY, Y, C: tl.constexpr, T: tl.constexpr,
                  H: tl.constexpr, W: tl.constexpr, XS: tl.constexpr,
                  HS: tl.constexpr, HAS_HISTORY: tl.constexpr,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    w = i % W
    h = i // W % H
    c = i // (W * H) % C
    t = i // (W * H * C) % (T + 2)
    b = i // (W * H * C * (T + 2))
    xt = tl.maximum(t - 2, 0)
    value = tl.load(X + b * XS[0] + c * XS[1] + xt * XS[2]
                    + h * XS[3] + w * XS[4], i < N, 0)
    if HAS_HISTORY:
        past = tl.load(HISTORY + b * HS[0] + c * HS[1] + t * HS[2]
                       + h * HS[3] + w * HS[4], (i < N) & (t < 2), 0)
        value = tl.where(t < 2, past, value)
    tl.store(Y + i, value, i < N)


def packed_causal_history(x, history):
    """Prepend two cached/replicated frames and pack BTCHW in one pass."""
    b, c, t, h, w = x.shape
    out = torch.empty((b, t + 2, c, h, w), device=x.device, dtype=x.dtype)
    past = x if history is None else history
    _pack_history[(triton.cdiv(out.numel(), 1024),)](
        x, past, out, c, t, h, w, x.stride(), past.stride(),
        history is not None, out.numel(), 1024)
    return out.permute(0, 2, 1, 3, 4)


class CogTLRPadding(nn.Module):
    """Keep the upstream cache contract while fusing its inference layout copy."""

    def __init__(self, source):
        super().__init__()
        self.conv = source.conv
        self.pad_mode = source.pad_mode
        self.time_causal_padding = source.time_causal_padding

    def forward(self, inputs, conv_cache=None):
        if self.pad_mode == 'replicate':
            inputs = F.pad(inputs, self.time_causal_padding, mode='replicate')
            conv_cache = None
        else:
            if inputs.is_cuda and inputs.dtype == torch.bfloat16 and self.conv.folded:
                inputs = packed_causal_history(inputs, conv_cache)
            else:
                past = [conv_cache] if conv_cache is not None else [inputs[:, :, :1]] * 2
                inputs = torch.cat(past + [inputs], dim=2)
            conv_cache = inputs[:, :, -2:].clone()
        return self.conv(inputs), conv_cache


@triton.jit
def _condition_silu(X, SCALE, BIAS, Y, C: tl.constexpr, T: tl.constexpr,
                    H: tl.constexpr, W: tl.constexpr, XS: tl.constexpr,
                    SS: tl.constexpr, BS: tl.constexpr,
                    N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    x = tl.load(X + _offset(i, C, T, H, W, XS), i < N, 0).to(tl.float32)
    scale = tl.load(SCALE + _offset(i, C, T, H, W, SS), i < N, 0).to(tl.float32)
    bias = tl.load(BIAS + _offset(i, C, T, H, W, BS), i < N, 0).to(tl.float32)
    # Match the source's separate BF16 multiplication, addition, and SiLU.
    product = (x * scale).to(tl.bfloat16).to(tl.float32)
    value = (product + bias).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, value * tl.sigmoid(value), i < N)


def conditioned_silu(x, scale, bias):
    if not x.is_cuda or x.dtype != torch.bfloat16 or torch.is_grad_enabled():
        return F.silu(x * scale + bias)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _, c, t, h, w = x.shape
    _condition_silu[(triton.cdiv(x.numel(), 256),)](
        x, scale, bias, out, c, t, h, w, x.stride(), scale.stride(), bias.stride(),
        x.numel(), 256, enable_fp_fusion=False)
    return out


class CogSpatialNormAct(nn.Module):
    """Retain native GroupNorm and conditioning, fuse affine plus following SiLU."""

    def __init__(self, source):
        super().__init__()
        self.norm_layer = source.norm_layer
        self.conv_y = source.conv_y
        self.conv_b = source.conv_b

    def forward(self, f, zq, conv_cache=None):
        conv_cache = conv_cache or {}
        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            first = F.interpolate(zq[:, :, :1], size=f[:, :, :1].shape[-3:])
            rest = F.interpolate(zq[:, :, 1:], size=f[:, :, 1:].shape[-3:])
            zq = torch.cat([first, rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])
        scale, scale_cache = self.conv_y(zq, conv_cache=conv_cache.get('conv_y'))
        bias, bias_cache = self.conv_b(zq, conv_cache=conv_cache.get('conv_b'))
        output = conditioned_silu(self.norm_layer(f), scale, bias)
        return output, {'conv_y': scale_cache, 'conv_b': bias_cache}


def optimize_cog(vae, *, padding=True, norms=True):
    if padding:
        for name, module in list(vae.decoder.named_modules()):
            if isinstance(getattr(module, 'conv', None), TLRConv3d):
                vae.decoder.set_submodule(name, CogTLRPadding(module))
    if norms:
        for module in vae.decoder.modules():
            if type(module).__name__ == 'CogVideoXResnetBlock3D':
                if (type(module.norm1).__name__ == 'CogVideoXSpatialNorm3D'
                        and isinstance(module.nonlinearity, nn.SiLU)
                        and getattr(module, 'temb_proj', None) is None):
                    module.norm1 = CogSpatialNormAct(module.norm1)
                    module.norm2 = CogSpatialNormAct(module.norm2)
                    module.nonlinearity = nn.Identity()
        if type(vae.decoder.norm_out).__name__ == 'CogVideoXSpatialNorm3D':
            vae.decoder.norm_out = CogSpatialNormAct(vae.decoder.norm_out)
            vae.decoder.conv_act = nn.Identity()
