"""Official LTX-2.5 convolutional video VAE for decoder TLR experiments."""

from dataclasses import dataclass
from pathlib import Path
import os
import sys

import torch
from torch import nn
from safetensors import safe_open


MODEL_ROOT = Path(os.environ.get(
    'LIGHTVAE_CROSS_ARCH_MODELS',
    Path(__file__).resolve().parents[1] / 'weights/cross_arch'))


def _official_ltx(root):
    core = Path(os.environ.get(
        'LIGHTVAE_LTX_CORE_SRC', Path(root) / 'src/ltx2/packages/ltx-core/src'))
    deps = Path(os.environ.get('LIGHTVAE_LTX_DEPS', Path(root) / 'deps'))
    for path in (core, deps):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from ltx_core.loader.checkpoint import CheckpointReader
    from ltx_core.model.video_vae.convolution import CausalConv3d
    from ltx_core.model.video_vae.model_configurator import (
        VAE_DECODER_COMFY_KEYS_FILTER,
        VAE_ENCODER_COMFY_KEYS_FILTER,
        VideoDecoderConfigurator,
        VideoEncoderConfigurator,
    )
    return (CheckpointReader, CausalConv3d, VideoEncoderConfigurator,
            VideoDecoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER,
            VAE_DECODER_COMFY_KEYS_FILTER)


def _load_component(handle, component: nn.Module, key_filter) -> None:
    weights = {}
    for source_key in handle.keys():
        target_key = key_filter.apply_to_key(source_key)
        if target_key is not None:
            weights[target_key] = handle.get_tensor(source_key)
    component.load_state_dict(weights, strict=True, assign=True)


@dataclass
class LTXVideoVAE:
    model: nn.Module
    weights: Path
    causal_conv_class: type[nn.Module]
    family: str = 'ltx25'

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    @property
    def latent_channels(self):
        return 128

    @property
    def spatial_ratio(self):
        return 32

    @property
    def temporal_ratio(self):
        return 8

    def latent_shape(self, frames: int, height: int, width: int, batch: int = 1):
        if (frames - 1) % 8 or height % 32 or width % 32:
            raise ValueError('LTX-2.5 requires T=8n+1 and H/W divisible by 32')
        return (batch, 128, 1 + (frames - 1) // 8, height // 32, width // 32)

    @torch.no_grad()
    def encode_mean(self, video: torch.Tensor):
        """Return the official encoder's normalized posterior mean for RGB in [-1, 1]."""
        return self.encoder(video)

    def decode(self, latents: torch.Tensor, frames: int | None = None):
        """Decode normalized latents to RGB in [-1, 1], retaining gradients."""
        video = self.decoder(latents)
        return video if frames is None else video[:, :, :frames]

    @property
    def eligible_decoder_convs(self):
        """Residual 3D convolutions; the last two stages can use spatial-only 2D."""
        selected = []
        for name, module in self.decoder.named_modules():
            if not isinstance(module, self.causal_conv_class):
                continue
            if not name.endswith(('.conv1', '.conv2')):
                continue
            path = f'decoder.{name}.conv'
            conv = module.conv
            if not isinstance(conv, nn.Conv3d):
                raise TypeError(f'Expected Conv3d at {path}')
            if (conv.kernel_size != (3, 3, 3) or conv.stride != (1, 1, 1)
                    or conv.dilation != (1, 1, 1) or conv.groups != 1
                    or conv.padding != (0, 1, 1)):
                raise ValueError(f'Unsupported residual convolution at {path}')
            spatial_only = path.startswith(('decoder.up_blocks.6.', 'decoder.up_blocks.8.'))
            selected.append((path, conv, spatial_only))
        if len(selected) != 36 or sum(flag for _, _, flag in selected) != 20:
            raise ValueError(f'Expected 36 residual Conv3ds and 20 spatial-only tail convolutions, got {len(selected)}')
        return selected


def load_ltx_vae(*, root: Path = MODEL_ROOT,
                 device: str | torch.device = 'cpu', dtype: torch.dtype = torch.float32):
    """Load the standalone official LTX-2.5 Conv VAE checkpoint strictly."""
    (CheckpointReader, CausalConv3d, VideoEncoderConfigurator,
     VideoDecoderConfigurator, encoder_filter, decoder_filter) = _official_ltx(root)
    weights = Path(root) / 'ltx25/vae/ltx-2.5-video-vae-conv-bf16.safetensors'
    with CheckpointReader.open((weights,)) as reader:
        metadata = reader.read_checkpoint_metadata()
    if metadata['config']['vae']['_class_name'] != 'CausalVideoAutoencoder':
        raise ValueError(f'{weights} is not the LTX-2.5 Conv VAE')
    with torch.device('meta'):
        encoder = VideoEncoderConfigurator.from_metadata(metadata)
        decoder = VideoDecoderConfigurator.from_metadata(metadata)
    with safe_open(weights, framework='pt', device='cpu') as handle:
        _load_component(handle, encoder, encoder_filter)
        _load_component(handle, decoder, decoder_filter)
    model = nn.Module()
    model.add_module('encoder', encoder)
    model.add_module('decoder', decoder)
    model.to(device=device, dtype=dtype).eval()
    encoder.requires_grad_(False)
    adapter = LTXVideoVAE(model, weights, CausalConv3d)
    adapter.eligible_decoder_convs
    return adapter


load_vae = load_ltx_vae
