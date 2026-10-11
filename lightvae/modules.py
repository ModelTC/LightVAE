"""Shared modules for the Wan2.1 and Wan2.2 LightVAE decoders."""

from pathlib import Path

import torch
from torch import nn

from base_vae import wan21_vae_layers as wan21
from base_vae.wan21_vae_decoder import Decoder3d as Wan21Decoder3d
from base_vae import wan22_vae_layers as wan22
from base_vae.wan22_vae_decoder import Decoder3d as Wan22Decoder3d
from utils import architecture, frames, load_lightvae_checkpoint, replace_norms_and_upsamples, unframes

from .tlr import TLR
from .triton_kernels import rms_silu


class FusedNorm(nn.Module):
    def __init__(self, source, silu):
        super().__init__()
        self.gamma = source.gamma
        bias = source.bias
        self.register_buffer(
            "bias", bias.detach() if isinstance(bias, torch.Tensor)
            else torch.full_like(source.gamma, bias),
        )
        self.scale, self.silu = source.scale, silu

    def forward(self, x):
        return rms_silu(x, self.gamma, self.bias, self.scale, self.silu)


class Shortcut(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.shortcut = block.shortcut

    def forward(self, x):
        return self.shortcut(x)


class _Component(nn.Module):
    def __init__(self, config):
        super().__init__()
        family = config["family"]
        if family not in ("wan21", "wan22"):
            raise ValueError(f"Unsupported LightVAE family: {family}")
        native = wan21 if family == "wan21" else wan22
        channels, dim = (16, 96) if family == "wan21" else (48, 256)
        self.conv2 = native.CausalConv3d(channels, channels, 1)
        decoder_cls = Wan21Decoder3d if family == "wan21" else Wan22Decoder3d
        self.decoder = decoder_cls(dim=dim, z_dim=channels,
                                   temperal_upsample=[True, True, False])
        if family == "wan21":
            self.register_buffer("latents_mean", torch.tensor(wan21.WAN_LATENTS_MEAN))
            self.register_buffer("latents_std", torch.tensor(wan21.WAN_LATENTS_STD))
        blocks = [(name, block) for name, block in self.decoder.named_modules()
                  if isinstance(block, native.ResidualBlock)]
        kept = config["kept"]
        if kept != sorted(set(kept)) or any(index < 0 or index >= len(blocks) for index in kept):
            raise ValueError("kept must contain sorted, unique original residual block indices")
        for index, (name, block) in enumerate(blocks):
            if index not in kept:
                self.decoder.set_submodule(name, Shortcut(block))
        convolutions = {name: layer for name, layer in self.decoder.named_modules()
                        if isinstance(layer, native.CausalConv3d) and layer.kernel_size[0] == 3}
        records = config["packed_records"]
        paths = [record["path"] for record in records]
        if len(paths) != len(set(paths)) or set(paths) != set(convolutions):
            raise ValueError("packed_records must match every retained temporal convolution exactly")
        for record in records:
            layer = convolutions[record["path"]]
            if (record["shape"] != list(layer.weight.shape)
                    or record["padding"] != layer._padding[0]
                    or record["bias"] != (layer.bias is not None)):
                raise ValueError(f"TLR configuration differs from native geometry: {record['path']}")
            self.decoder.set_submodule(record["path"], TLR(
                record["shape"], record["padding"], record["rank"], record["bias"],
            ))

    @torch.no_grad()
    def initialize(self, family):
        self.to_empty(device="cpu")
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
                layer.reset_parameters()
            elif isinstance(layer, (wan21.RMS_norm, wan22.RMS_norm)):
                layer.gamma.fill_(1)
                if isinstance(layer.bias, torch.Tensor):
                    layer.bias.zero_()
            elif isinstance(layer, TLR) and layer.bias is not None:
                layer.bias.zero_()
        if family == "wan21":
            self.latents_mean.copy_(torch.tensor(wan21.WAN_LATENTS_MEAN))
            self.latents_std.copy_(torch.tensor(wan21.WAN_LATENTS_STD))


class Decoder(nn.Module):
    """Accept normalized Wan2.1 latents or raw Wan2.2 posterior means.

    Loads safetensors exports with embedded architecture metadata.
    The supplied config must match family, kept blocks and packed TLR records.
    """

    temporal_scale = 4

    def __init__(self, config: dict, checkpoint: str | Path | None = None):
        super().__init__()
        self.config = architecture(config)
        self.family = config["family"]
        self.latent_channels, self.spatial_scale = (16, 8) if self.family == "wan21" else (48, 16)
        saved = None
        if checkpoint is not None:
            saved = load_lightvae_checkpoint(checkpoint)
            if architecture(saved) != architecture(config):
                raise ValueError("Checkpoint family, residual depth or TLR ranks differ from config")
        with torch.device("meta"):
            self.component = _Component(config)
        if saved is None:
            self.component.initialize(self.family)
        else:
            self.component.load_state_dict(saved["model"], strict=True, assign=True)
            self.component.float()
        for layer in self.component.modules():
            if isinstance(layer, TLR):
                layer.fold()
        native = wan21 if self.family == "wan21" else wan22
        replace_norms_and_upsamples(self.component, native, FusedNorm)
        self.slots = sum(isinstance(layer, TLR) for layer in self.component.modules())
        self.eval().requires_grad_(False)

    def _apply_layer(self, layer, x, history, slot):
        if isinstance(layer, TLR):
            index = slot[0]
            x, history[index] = layer(x, history[index])
            slot[0] += 1
            return x
        if isinstance(layer, (wan21.ResidualBlock, wan22.ResidualBlock)):
            shortcut = layer.shortcut(x)
            for child in layer.residual:
                x = self._apply_layer(child, x, history, slot)
            return shortcut + x
        if isinstance(layer, (wan21.Resample, wan22.Resample)):
            batch, channels, time, height, width = x.shape
            if layer.mode == "upsample3d":
                index = slot[0]
                old = history[index]
                if old is None:
                    history[index] = "startup"
                else:
                    x, new = layer.time_conv(x, None if isinstance(old, str) else old)
                    history[index] = new if new is not None else "spatial"
                    x = x.reshape(batch, 2, channels, time, height, width)
                    x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch, channels, time * 2, height, width)
                slot[0] += 1
            return unframes(layer.resample(frames(x)), batch, x.shape[2])
        return layer(x)

    def forward(self, z):
        if (not z.is_cuda or z.dtype != torch.bfloat16
                or torch.cuda.get_device_capability(z.device)[0] < 9):
            raise ValueError("LightVAE inference requires CUDA BF16 on Hopper/Blackwell (SM90+)")
        if z.ndim != 5 or z.shape[1] != self.latent_channels or min(z.shape) < 1:
            raise ValueError(f"Expected nonempty B,{self.latent_channels},T,H,W latent")
        component, decoder = self.component, self.component.decoder
        if self.family == "wan21":
            mean = component.latents_mean.view(1, -1, 1, 1, 1)
            std = component.latents_std.view(1, -1, 1, 1, 1)
            z = z / std.reciprocal() + mean
        projected = component.conv2(z)
        history, outputs = [None] * self.slots, []
        for time in range(projected.shape[2]):
            slot = [0]
            x = self._apply_layer(decoder.conv1, projected[:, :, time:time + 1], history, slot)
            for layer in decoder.middle:
                x = self._apply_layer(layer, x, history, slot)
            if self.family == "wan21":
                for layer in decoder.upsamples:
                    x = self._apply_layer(layer, x, history, slot)
            else:
                for stage in decoder.upsamples:
                    shortcut = x
                    for layer in stage.upsamples:
                        x = self._apply_layer(layer, x, history, slot)
                    if stage.avg_shortcut is not None:
                        x = x + stage.avg_shortcut(shortcut, first_chunk=(time == 0))
            for layer in decoder.head:
                x = self._apply_layer(layer, x, history, slot)
            outputs.append(x if self.family == "wan21" else wan22.unpatchify(x, 2))
        return torch.cat(outputs, dim=2)
