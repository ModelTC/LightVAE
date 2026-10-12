# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Original MiniMax-H3 decoder, including normalization and tiled inference."""

from contextlib import nullcontext
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import H3Module, blend, h3_decoded_frames, h3_model_config, load_h3_component, split_tiles, stitch_tiles

# Modified for standalone inference: pipeline and training machinery removed.
# State-dict names follow Diffusers' MiniMax-H3 implementation.


class _SwiGLU(nn.Module):
    def __init__(self, dim, inner_dim, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, bias=True):
        super().__init__()
        self.net = nn.ModuleList([
            _SwiGLU(dim, dim * mult, bias=bias),
            nn.Dropout(0.0), nn.Linear(dim * mult, dim, bias=bias),
        ])

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    r"""
    3-axis rotary embedding for the ViT decoder. Coordinates are length-normalized to `[-1, 1)` per axis and scaled by
    `2 * pi`, and the resulting `(t, h, w)` angles are concatenated and then duplicated, so the first
    `rope_dim_ratio * attention_head_dim` channels of every head are rotated.
    """

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"`dim` {dim} must be divisible by `2 * num_axes` {2 * num_axes}.")
        self.dim, self.theta, self.num_axes = dim, theta, num_axes
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def materialize_buffer(self, device):
        # Nonpersistent buffers are absent from load_state_dict(assign=True).
        self.inv_freq = 1.0 / self.theta ** torch.arange(
            0, 1, 2 * self.num_axes / self.dim, dtype=torch.float32, device=device)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class MiniMaxH3VideoAttnProcessor:
    def __call__(
        self,
        attn: "MiniMaxH3VideoAttention",
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))

        # The reference normalizes Q/K in float32 regardless of the compute dtype.
        query = attn.norm_q(query.float()).to(query.dtype)
        key = attn.norm_k(key.float()).to(key.dtype)

        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            rotary_dim = cos.shape[-1]
            query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
            key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
            query_first, query_second = query_rotary.chunk(2, dim=-1)
            key_first, key_second = key_rotary.chunk(2, dim=-1)
            query_rotated = torch.cat([-query_second, query_first], dim=-1)
            key_rotated = torch.cat([-key_second, key_first], dim=-1)
            query = torch.cat([query_rotary * cos + query_rotated * sin, query_pass], dim=-1)
            key = torch.cat([key_rotary * cos + key_rotated * sin, key_pass], dim=-1)

        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            dropout_p=0.0,
        ).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


class MiniMaxH3VideoAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.use_bias = bias
        inner_dim = heads * dim_head

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(0.0)])

        self.processor = MiniMaxH3VideoAttnProcessor()

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb)


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3VideoAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = FeedForward(dim, mult=ffn_mult, bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        # The reference normalizes in float32 regardless of the compute dtype.
        norm_hidden_states = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(norm_hidden_states, rotary_emb) * self.scale1
        norm_hidden_states = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = hidden_states + self.ff(norm_hidden_states) * self.scale2
        return hidden_states


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    r"""
    Non-causal ViT decoder. Every latent voxel becomes one token; `num_register_tokens` learned register tokens plus a
    single all-zero token are appended (all at position `0`), attended over with full self-attention, and dropped
    again before the patch projection expands each token into a `patch_size_t x patch_size x patch_size` pixel block.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


class Decoder(H3Module):
    """Normalized H3 latents -> RGB [-1,1], both in BCTHW layout."""

    family = "h3"
    spatial_scale = 16
    temporal_scale = 4

    def __init__(self, config=None, checkpoint=None):
        super().__init__()
        self.config, weights = h3_model_config(config, checkpoint, "decoder")
        cfg = self.config
        self.latent_channels = cfg["latent_channels"]
        if math.prod(cfg["spatial_downsample_factors"]) != 16 or math.prod(cfg["temporal_downsample_factors"]) != 4:
            raise ValueError("H3 stitching requires spatial/temporal patch sizes 16/4")
        depth = self._decoder_depth(cfg)
        self.tile_size = cfg["tile_size"]
        self.tile_overlap = cfg["tile_overlap"]
        split_tiles(32, self.tile_size, self.tile_overlap)
        with torch.device("meta") if weights is not None else nullcontext():
            self.post_quant_conv = nn.Conv3d(self.latent_channels, self.latent_channels, 1)
            self.decoder = MiniMaxH3VideoViTDecoder3d(
                in_channels=self.latent_channels,
                out_channels=cfg["out_channels"],
                patch_size=16,
                patch_size_t=4,
                num_layers=depth,
                num_attention_heads=cfg["decoder_num_attention_heads"],
                attention_head_dim=cfg["decoder_attention_head_dim"],
                num_register_tokens=cfg["decoder_num_register_tokens"],
                ffn_mult=cfg["decoder_ffn_mult"],
                rope_theta=cfg["decoder_rope_theta"],
                rope_dim_ratio=cfg["decoder_rope_dim_ratio"],
                norm_eps=cfg["decoder_norm_eps"],
            )
        self.register_buffer("latents_mean", torch.tensor(cfg["latents_mean"]))
        self.register_buffer("latents_std", torch.tensor(cfg["latents_std"]))
        if weights is not None:
            load_h3_component(self, weights)
            self.decoder.rope.materialize_buffer("cpu")

    def _decoder_depth(self, cfg):
        if cfg.get("kept_layers") is not None:
            raise ValueError("Pruned H3 decoder weights require the LightVAE decoder")
        return cfg["decoder_num_layers"]

    def _decode_clip(self, latents):
        mean = self.latents_mean.float().view(1, -1, 1, 1, 1)
        std = self.latents_std.float().view(1, -1, 1, 1, 1)
        latents = (latents.float() * std + mean).to(self.post_quant_conv.weight.dtype)
        return self.decoder(self.post_quant_conv(latents))

    def _decode_spatial(self, latents):
        height, width = latents.shape[-2:]
        heights, height_overlaps = split_tiles(height * 16, self.tile_size, self.tile_overlap)
        widths, width_overlaps = split_tiles(width * 16, self.tile_size, self.tile_overlap)
        rows = [[self._decode_clip(latents[..., h, w]) for w in widths] for h in heights]
        return stitch_tiles(rows, height_overlaps, width_overlaps)

    def forward(self, latents):
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError(f"H3 latents must have shape [B,{self.latent_channels},T,H,W]")
        frames = h3_decoded_frames(latents.shape[2])
        if latents.shape[2] == 2:
            video = self._decode_spatial(latents)[:, :, 3:8]
        else:
            output, previous = [], None
            for begin in range(0, latents.shape[2] - 2, 5):
                raw = self._decode_spatial(latents[:, :, begin:begin + 7])
                first, tail = raw[:, :, 3:20], raw[:, :, 23:28]
                output.append(first if previous is None else blend(previous, first, 5, -3))
                previous = tail
            video = torch.cat((*output, previous), dim=2)
        if video.shape[2] != frames:
            raise RuntimeError("H3 temporal stitching produced the wrong frame count")
        mean = video.new_tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = video.new_tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1, 1)
        return (video.float() * std + mean) * 2 - 1
