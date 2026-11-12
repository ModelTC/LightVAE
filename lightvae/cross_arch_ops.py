"""Inference-only adapters for converted decoders; original VAEs never use these."""

import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl

from .cross_arch_tlr import TLRConv3d


@triton.jit
def _pack_pad(X, Y, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
              W: tl.constexpr, OT: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
              S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
              S3: tl.constexpr, S4: tl.constexpr, FRONT: tl.constexpr,
              TOP: tl.constexpr, LEFT: tl.constexpr, N: tl.constexpr,
              BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    w = tl.minimum(tl.maximum(i % OW - LEFT, 0), W - 1)
    h = tl.minimum(tl.maximum(i // OW % OH - TOP, 0), H - 1)
    c = i // (OW * OH) % C
    t = tl.minimum(tl.maximum(i // (OW * OH * C) % OT - FRONT, 0), T - 1)
    b = i // (OW * OH * C * OT)
    value = tl.load(X + b * S0 + c * S1 + t * S2 + h * S3 + w * S4, i < N, 0)
    tl.store(Y + i, value, i < N)


def packed_replicate_pad(x, padding):
    """Replicate padding and BCTHW→BTCHW packing in one memory pass."""
    left, right, top, bottom, front, back = padding
    b, c, t, h, w = x.shape
    out = torch.empty((b, t + front + back, c, h + top + bottom, w + left + right),
                      dtype=x.dtype, device=x.device)
    _pack_pad[(triton.cdiv(out.numel(), 1024),)](
        x, out, c, t, h, w, out.shape[1], out.shape[3], out.shape[4],
        *x.stride(), front, top, left, out.numel(), 1024)
    return out.permute(0, 2, 1, 3, 4)


class TLRPadding(nn.Module):
    """Preserve the source convolution's boundaries without its padded-layout copy."""

    def __init__(self, source, family):
        super().__init__()
        self.conv = source.conv
        self.family = family
        self.padding = source.time_causal_padding if family == 'hunyuanvideo15' else None
        self.mode = source.pad_mode if family == 'hunyuanvideo15' else 'replicate'

    def forward(self, x, causal=True):
        pad = self.padding
        if pad is None:
            pad = (0, 0, 0, 0, 2 if causal else 1, 0 if causal else 1)
        if x.is_cuda and x.dtype == torch.bfloat16 and self.conv.folded and self.mode == 'replicate':
            x = packed_replicate_pad(x, pad)
        else:
            x = F.pad(x, pad, mode=self.mode)
        return self.conv(x)


class NormAct(nn.Module):
    def __init__(self, source, kind):
        super().__init__()
        self.kind = kind
        self.eps = getattr(source, 'eps', 1e-12)
        self.scale = getattr(source, 'scale', 1.0)
        self.gamma = getattr(source, 'gamma', None)
        self.bias = getattr(source, 'bias', None)

    def forward(self, x):
        from .cross_arch_fused import fused_norm_act
        return fused_norm_act(x, self.kind, gamma=self.gamma, bias=self.bias,
                              eps=self.eps, scale=self.scale)


class HunyuanResidual(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1 = NormAct(source.norm1, 'hunyuan')
        self.norm2 = NormAct(source.norm2, 'hunyuan')
        self.conv1, self.conv2 = source.conv1, source.conv2
        self.nin_shortcut = getattr(source, 'nin_shortcut', nn.Identity())

    def forward(self, x):
        h = self.conv1(self.norm1(x))
        h = self.conv2(self.norm2(h))
        return self.nin_shortcut(x) + h


class HunyuanDecoder(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.conv_in, self.mid, self.up = source.conv_in, source.mid, source.up
        self.norm_out = NormAct(source.norm_out, 'hunyuan')
        self.conv_out = source.conv_out
        self.repeats = source.block_out_channels[0] // source.z_channels

    def forward(self, z):
        h = self.conv_in(z) + z.repeat_interleave(self.repeats, dim=1)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for stage in self.up:
            for block in stage.block:
                h = block(h)
            if hasattr(stage, 'upsample'):
                h = stage.upsample(h)
        return self.conv_out(self.norm_out(h))


def optimize_padding(vae):
    for name, module in list(vae.decoder.named_modules()):
        if isinstance(getattr(module, 'conv', None), TLRConv3d):
            vae.decoder.set_submodule(name, TLRPadding(module, vae.family))


def optimize_norms(vae):
    if vae.family == 'hunyuanvideo15':
        for name, module in list(vae.decoder.named_modules()):
            if type(module).__name__ == 'ResnetBlock':
                vae.decoder.set_submodule(name, HunyuanResidual(module))
        vae.model.decoder = HunyuanDecoder(vae.decoder)
    elif vae.family == 'ltx25':
        for module in vae.decoder.modules():
            if type(module).__name__ == 'ResnetBlock3D' and not module.inject_noise and not module.timestep_conditioning:
                if type(module.norm1).__name__ == type(module.norm2).__name__ == 'PixelNorm':
                    module.norm1, module.norm2 = NormAct(module.norm1, 'ltx'), NormAct(module.norm2, 'ltx')
                    module.non_linearity = nn.Identity()
        if not vae.decoder.timestep_conditioning:
            vae.decoder.conv_norm_out = NormAct(vae.decoder.conv_norm_out, 'ltx')
            vae.decoder.conv_act = nn.Identity()
