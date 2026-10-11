# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""Original Wan2.1 video decoder; input is a normalized 16-channel latent."""

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from utils import load_component

from . import wan21_vae_layers as native
from .wan21_vae_layers import (CACHE_T, AttentionBlock, CausalConv3d, RMS_norm, ResidualBlock, Resample)


class Decoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_upsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout))

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

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

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None:
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


class Decoder(nn.Module):
    family = "wan21"
    latent_channels = 16
    spatial_scale = 8
    temporal_scale = 4

    def __init__(self, config: dict | None = None, checkpoint: str | Path | None = None):
        super().__init__()
        architecture = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                            attn_scales=[], temperal_upsample=[True, True, False], dropout=0.0)
        architecture.update(config or {})
        self.config = architecture
        with torch.device("meta") if checkpoint else nullcontext():
            self.conv2 = native.CausalConv3d(16, 16, 1)
            self.decoder = Decoder3d(**architecture)
        if checkpoint is not None:
            load_component(self, checkpoint, ("conv2.", "decoder."))
        self.cache_slots = native.count_conv3d(self.decoder)
        self.register_buffer("mean", torch.tensor(native.WAN_LATENTS_MEAN).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(native.WAN_LATENTS_STD).view(1, -1, 1, 1, 1), persistent=False)

    def forward(self, z):
        z = z / self.std.reciprocal() + self.mean
        projected = self.conv2(z)
        cache = [None] * self.cache_slots
        return torch.cat([
            self.decoder(projected[:, :, i:i + 1], feat_cache=cache, feat_idx=[0])
            for i in range(projected.shape[2])
        ], dim=2)
