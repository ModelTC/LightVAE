# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""Original Wan2.2 TI2V video encoder; output is the raw posterior mean."""

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from utils import load_component

from . import wan22_vae_layers as native
from .wan22_vae_layers import (CACHE_T, AttentionBlock, CausalConv3d, RMS_norm, ResidualBlock, Down_ResidualBlock)


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]

        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
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
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Encoder(nn.Module):
    family = "wan22"
    latent_channels = 48
    spatial_scale = 16
    temporal_scale = 4

    def __init__(self, config: dict | None = None, checkpoint: str | Path | None = None):
        super().__init__()
        architecture = dict(dim=160, z_dim=96, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                            attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
        architecture.update(config or {})
        self.config = architecture
        with torch.device("meta") if checkpoint else nullcontext():
            self.encoder = Encoder3d(**architecture)
            self.conv1 = native.CausalConv3d(96, 96, 1)
        if checkpoint is not None:
            load_component(self, checkpoint, ("encoder.", "conv1."))
        self.cache_slots = native.count_conv3d(self.encoder)

    def forward(self, pixels):
        if (pixels.shape[2] - 1) % self.temporal_scale:
            raise ValueError("Wan2.2 input must contain 4n+1 frames; pad before encoding.")
        pixels = native.patchify(pixels, patch_size=2)
        cache = [None] * self.cache_slots
        chunks = [self.encoder(pixels[:, :, :1], feat_cache=cache, feat_idx=[0])]
        for start in range(1, pixels.shape[2], 4):
            chunks.append(self.encoder(pixels[:, :, start:start + 4], feat_cache=cache, feat_idx=[0]))
        return self.conv1(torch.cat(chunks, dim=2)).chunk(2, dim=1)[0]
