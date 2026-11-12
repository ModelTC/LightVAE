"""Inference kernels shared by non-Wan video VAE adapters."""

import torch
from torch import nn
from torch.nn import functional as F


class HunyuanFrameCausalAttention(nn.Module):
    """Hunyuan VAE attention without its per-token dense causal mask."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.in_channels = source.in_channels
        self.norm = source.norm
        self.q = source.q
        self.k = source.k
        self.v = source.v
        self.proj_out = source.proj_out

    def forward(self, x):
        batch, _, frames, height, width = x.shape
        spatial_tokens = height * width
        h = self.norm(x)
        q, k, v = (layer(h).permute(0, 2, 3, 4, 1)
                   .reshape(batch, 1, frames * spatial_tokens, self.in_channels)
                   .contiguous()
                   for layer in (self.q, self.k, self.v))
        output = torch.cat([
            F.scaled_dot_product_attention(
                q[:, :, frame * spatial_tokens:(frame + 1) * spatial_tokens],
                k[:, :, :(frame + 1) * spatial_tokens],
                v[:, :, :(frame + 1) * spatial_tokens],
            )
            for frame in range(frames)
        ], dim=2)
        output = output.reshape(batch, frames, height, width, self.in_channels)
        output = output.permute(0, 4, 1, 2, 3)
        return x + self.proj_out(output)
