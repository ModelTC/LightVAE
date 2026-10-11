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

"""Original MiniMax-H3 encoder, including normalization and tiled inference."""

from contextlib import nullcontext
import math

import torch
from torch import nn
import torch.nn.functional as F

from utils import H3Module, h3_model_config, latent_frames, load_h3_component, padded_frames, split_tiles, stitch_tiles

# Modified for standalone inference: pipeline and training machinery removed.
# State-dict names follow Diffusers' MiniMax-H3 implementation.


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    r"""
    3D convolution used throughout the MiniMax-H3 video encoder.

    Spatial padding is symmetric and uses `spatial_padding_mode` (`"reflect"` in the released checkpoint); temporal
    padding is causal, i.e. `kernel_size_t - 1` zero frames are prepended and nothing is appended.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int | tuple[int, int, int, int] = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode
        # Large 768p feature maps exceed CUDA reflection_pad3d's int32 indexing.
        # Bound individual convolution windows; spatial context is never tiled.
        self.max_temporal_chunk_elements = 1 << 28

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        frame_elements = (hidden_states.shape[0] * max(self.in_channels, self.out_channels)
                          * hidden_states.shape[-2] * hidden_states.shape[-1])
        if frame_elements * hidden_states.shape[2] > self.max_temporal_chunk_elements:
            return self._temporal_chunks(hidden_states, frame_elements)
        return self._convolve(hidden_states, self.temporal_padding)

    def _convolve(self, hidden_states, temporal_padding):
        padding = ((self.spatial_padding,) * 4 if isinstance(self.spatial_padding, int)
                   else self.spatial_padding)
        if any(padding):
            hidden_states = F.pad(
                hidden_states, (*padding, 0, 0), mode=self.spatial_padding_mode
            )
        if temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, temporal_padding, 0), mode="constant")
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)

    def _temporal_chunks(self, hidden_states, frame_elements):
        extent = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        stride = self.stride[0]
        count = (hidden_states.shape[2] + self.temporal_padding - extent) // stride + 1
        chunk = max(1, self.max_temporal_chunk_elements // frame_elements // stride)
        outputs = []
        for first in range(0, count, chunk):
            last = min(first + chunk, count)
            begin = first * stride - self.temporal_padding
            end = (last - 1) * stride - self.temporal_padding + extent
            clip = hidden_states[:, :, max(0, begin):end]

            def run(value, pad=max(0, -begin)):
                # A compact copy also bounds the strides used by CUDA padding.
                return self._convolve(value.contiguous(), pad)

            outputs.append(run(clip))
        return torch.cat(outputs, dim=2)


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
    r"""
    Group normalization applied to each latent frame in isolation (`use_t_isolated_gn` in the original config): the
    temporal axis is folded into the batch axis so statistics never mix across frames.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        # CUDA autocast otherwise promotes the entire 768p activation to FP32.
        # Native GroupNorm accumulates low-precision statistics in FP32, while
        # preserving the input/output dtype (as in BF16 H3 inference).
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            weight = self.weight.to(hidden_states.dtype) if self.weight is not None else None
            bias = self.bias.to(hidden_states.dtype) if self.bias is not None else None
            hidden_states = F.group_norm(hidden_states, self.num_groups, weight, bias, self.eps)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    r"""
    Strided 3x3x3 downsampling convolution. A spatial stride of 2 is preceded by an asymmetric bottom/right pad of 1
    (the convolution itself carries no spatial padding), so the output is exactly `ceil(size / 2)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=(0, 1, 0, 1) if spatial_stride == 2 else 0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply the asymmetric pad INSIDE each bounded convolution window.
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3VideoDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    r"""
    Causal 3D CNN encoder. `block_out_channels` gives the channel count of every level; the per-level
    `spatial_downsample_factors` / `temporal_downsample_factors` multiply out to the total compression ratios.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(len(block_out_channels))
            ]
        )

        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


class Encoder(H3Module):
    """RGB [-1,1] -> normalized posterior-mean latents, in BCTHW layout."""

    family = "h3"
    spatial_scale = 16
    temporal_scale = 4

    def __init__(self, config=None, checkpoint=None):
        super().__init__()
        self.config, weights = h3_model_config(config, checkpoint, "encoder")
        cfg = self.config
        self.latent_channels = cfg["latent_channels"]
        if math.prod(cfg["spatial_downsample_factors"]) != 16 or math.prod(cfg["temporal_downsample_factors"]) != 4:
            raise ValueError("H3 stitching requires spatial/temporal compression 16/4")
        self.tile_size = cfg["tile_size"]
        self.tile_overlap = cfg["tile_overlap"]
        split_tiles(32, self.tile_size, self.tile_overlap)
        keys = (
            "in_channels", "block_out_channels", "layers_per_block", "spatial_downsample_factors",
            "temporal_downsample_factors", "norm_num_groups", "norm_eps", "spatial_padding_mode",
        )
        with torch.device("meta") if weights is not None else nullcontext():
            self.encoder = MiniMaxH3VideoEncoder3d(
                **{key: cfg[key] for key in keys}, out_channels=self.latent_channels * 2,
            )
            self._configure_encoder(cfg)
            self.quant_conv = nn.Conv3d(self.latent_channels * 2, self.latent_channels * 2, 1)
        self.register_buffer("latents_mean", torch.tensor(cfg["latents_mean"]))
        self.register_buffer("latents_std", torch.tensor(cfg["latents_std"]))
        if weights is not None:
            load_h3_component(self, weights)

    def _configure_encoder(self, cfg):
        if cfg.get("kept_residual_indices") is not None:
            raise ValueError("Pruned H3 encoder weights require the LightVAE encoder")

    def _encode_clip(self, video):
        mean = video.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = video.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        video = ((video + 1) / 2 - mean) / std
        return self.quant_conv(self.encoder(video.to(self.quant_conv.weight.dtype)))

    def forward(self, video):
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[-2] % 16 or video.shape[-1] % 16:
            raise ValueError("H3 input must be RGB [B,3,T,H,W] with spatial dimensions divisible by 16")
        frames = padded_frames("h3", video.shape[2])
        window_frames = math.ceil(frames / 17) * 17
        video = torch.cat((video, video[:, :, -1:].expand(-1, -1, window_frames - video.shape[2], -1, -1)), dim=2)
        heights, height_overlaps = split_tiles(video.shape[-2], self.tile_size, self.tile_overlap)
        widths, width_overlaps = split_tiles(video.shape[-1], self.tile_size, self.tile_overlap)
        output = []
        for begin in range(0, window_frames, 17):
            rows = []
            for h in heights:
                row = []
                for w in widths:
                    clip = video[:, :, begin:begin + 17, h.start * 16:h.stop * 16, w.start * 16:w.stop * 16]
                    row.append(self._encode_clip(clip))
                rows.append(row)
            output.append(stitch_tiles(rows, [v // 16 for v in height_overlaps], [v // 16 for v in width_overlaps]))
        moments = torch.cat(output, dim=2)[:, :, :-3]
        if moments.shape[2] != latent_frames("h3", frames):
            raise RuntimeError("H3 encoder produced the wrong latent frame count")
        posterior_mean = moments[:, :self.latent_channels].float()
        mean = self.latents_mean.float().view(1, -1, 1, 1, 1)
        std = self.latents_std.float().view(1, -1, 1, 1, 1)
        return (posterior_mean - mean) / std
