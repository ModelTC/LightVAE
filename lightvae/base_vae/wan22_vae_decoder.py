# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""Original Wan2.2 TI2V video decoder; input is the raw 48-channel posterior mean."""

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from ..utils import load_component

from . import wan22_vae_layers as native
from .wan22_vae_layers import (CACHE_T, AttentionBlock, CausalConv3d, RMS_norm, ResidualBlock, Up_ResidualBlock)


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
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

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
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


class Decoder(nn.Module):
    family = "wan22"
    latent_channels = 48
    spatial_scale = 16
    temporal_scale = 4

    def __init__(self, config: dict | None = None, checkpoint: str | Path | None = None):
        super().__init__()
        architecture = dict(dim=256, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                            attn_scales=[], temperal_upsample=[True, True, False], dropout=0.0)
        architecture.update(config or {})
        self.config = architecture
        with torch.device("meta") if checkpoint else nullcontext():
            self.conv2 = native.CausalConv3d(48, 48, 1)
            self.decoder = Decoder3d(**architecture)
        if checkpoint is not None:
            load_component(self, checkpoint, ("conv2.", "decoder."))
        self.cache_slots = native.count_conv3d(self.decoder)

    def forward(self, z):
        projected = self.conv2(z)
        cache = [None] * self.cache_slots
        pixels = torch.cat([
            self.decoder(projected[:, :, i:i + 1], feat_cache=cache, feat_idx=[0], first_chunk=i == 0)
            for i in range(projected.shape[2])
        ], dim=2)
        return native.unpatchify(pixels, patch_size=2)
