# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""Original Wan2.1 video encoder; output is the normalized posterior mean."""

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from ..utils import load_component

from . import wan21_vae_layers as native
from .wan21_vae_layers import (CACHE_T, AttentionBlock, CausalConv3d, RMS_norm, ResidualBlock, Resample)


class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout))

        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Encoder(nn.Module):
    family = "wan21"
    latent_channels = 16
    spatial_scale = 8
    temporal_scale = 4

    def __init__(self, config: dict | None = None, checkpoint: str | Path | None = None):
        super().__init__()
        architecture = dict(dim=96, z_dim=32, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                            attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
        architecture.update(config or {})
        self.config = architecture
        with torch.device("meta") if checkpoint else nullcontext():
            self.encoder = Encoder3d(**architecture)
            self.conv1 = native.CausalConv3d(32, 32, 1)
        if checkpoint is not None:
            load_component(self, checkpoint, ("encoder.", "conv1."))
        self.cache_slots = native.count_conv3d(self.encoder)
        self.register_buffer("mean", torch.tensor(native.WAN_LATENTS_MEAN).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(native.WAN_LATENTS_STD).view(1, -1, 1, 1, 1), persistent=False)

    def forward(self, pixels):
        if (pixels.shape[2] - 1) % self.temporal_scale:
            raise ValueError("Wan2.1 input must contain 4n+1 frames; pad before encoding.")
        cache = [None] * self.cache_slots
        chunks = [self.encoder(pixels[:, :, :1], feat_cache=cache, feat_idx=[0])]
        for start in range(1, pixels.shape[2], 4):
            chunks.append(self.encoder(pixels[:, :, start:start + 4], feat_cache=cache, feat_idx=[0]))
        mean = self.conv1(torch.cat(chunks, dim=2)).chunk(2, dim=1)[0]
        return (mean - self.mean) * self.std.reciprocal()
