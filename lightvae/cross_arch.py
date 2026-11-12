"""Original video VAE adapters for cross-architecture decoder experiments.

Reconstruction uses the original encoder posterior mean in native latent units.
Only residual 3x3x3 decoder convolutions are eligible for TLR; resampling,
input/output projections, and the encoder remain as published.
"""

from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from safetensors.torch import load_file


MODEL_ROOT = Path(os.environ.get(
    'LIGHTVAE_CROSS_ARCH_MODELS',
    Path(__file__).resolve().parents[1] / 'weights/cross_arch'))


@dataclass(frozen=True)
class FamilySpec:
    folder: str
    spatial_scale: int
    temporal_scale: int
    residual_convolutions: int
    spatial_only_prefixes: tuple[str, ...]


SPECS = {
    'hunyuanvideo15': FamilySpec(
        'hunyuanvideo15', 16, 4, 34,
        ('decoder.up.3.', 'decoder.up.4.')),
    'cogvideox15': FamilySpec(
        'cogvideox15', 8, 4, 36,
        ('decoder.up_blocks.2.', 'decoder.up_blocks.3.')),
    'ltx25': FamilySpec(
        'ltx25', 32, 8, 36,
        ('decoder.up_blocks.6.', 'decoder.up_blocks.8.')),
}


@lru_cache(maxsize=1)
def _hunyuan_class():
    source = os.environ.get('LIGHTVAE_LIGHTX2V_ROOT')
    if source:
        sys.path.insert(0, source)
    module = importlib.import_module(
        'lightx2v.models.video_encoders.hf.hunyuanvideo15.hunyuanvideo_15_vae')
    return module.AutoencoderKLConv3D


def _model_class(family):
    if family == 'hunyuanvideo15':
        return _hunyuan_class()
    if family == 'cogvideox15':
        from diffusers import AutoencoderKLCogVideoX
        return AutoencoderKLCogVideoX
    raise ValueError(f'Unknown video VAE: {family}')


@dataclass
class VideoVAE:
    family: str
    model: nn.Module
    weights: Path

    @property
    def spec(self):
        return SPECS[self.family]

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    @property
    def latent_channels(self):
        return self.model.config.latent_channels

    @property
    def spatial_ratio(self):
        return self.spec.spatial_scale

    @property
    def temporal_ratio(self):
        return self.spec.temporal_scale

    def latent_shape(self, frames: int, height: int, width: int, batch: int = 1):
        scale = self.spec.spatial_scale
        rate = self.spec.temporal_scale
        if (frames - 1) % rate or height % scale or width % scale:
            raise ValueError(f'{self.family} requires T=4n+1 and H/W divisible by {scale}')
        return (batch, self.model.config.latent_channels,
                1 + (frames - 1) // rate, height // scale, width // scale)

    @torch.no_grad()
    def encode_mean(self, video: torch.Tensor):
        """Return unscaled original-encoder posterior mean for RGB in [-1,1]."""
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError('Expected RGB video [B,3,T,H,W]')
        return self.model.encode(video).latent_dist.mode()

    def decode(self, latents: torch.Tensor, frames: int | None = None):
        """Decode with gradients for recovery; return RGB in [-1,1]."""
        if self.family == 'hunyuanvideo15':
            # The upstream model.decode method is decorated with no_grad.
            result = self.model.decoder(latents)
        else:
            result = self.model.decode(latents).sample
        requested = frames if frames is not None else 1 + self.temporal_ratio * (latents.shape[2] - 1)
        if result.shape[2] < requested:
            raise ValueError(f'{self.family} decoded {result.shape[2]} frames, expected {requested}')
        return result[:, :, :requested]

    def temporal_convolutions(self):
        """List (full module path, inner Conv3d, spatial-only tail flag)."""
        selected = []
        for name, module in self.model.decoder.named_modules():
            path = f'decoder.{name}'
            if not path.endswith(('conv1.conv', 'conv2.conv')):
                continue
            if not isinstance(module, nn.Conv3d):
                raise TypeError(f'Expected published Conv3d at {path}')
            if (module.kernel_size != (3, 3, 3) or module.stride != (1, 1, 1)
                    or module.dilation != (1, 1, 1) or module.groups != 1
                    or module.padding[0] != 0):
                raise ValueError(f'Unsupported temporal residual convolution at {path}')
            spatial_only = path.startswith(self.spec.spatial_only_prefixes)
            selected.append((path, module, spatial_only))
        if len(selected) != self.spec.residual_convolutions:
            raise ValueError(f'{self.family}: expected {self.spec.residual_convolutions} '
                             f'residual Conv3ds, found {len(selected)}')
        return selected

    @property
    def eligible_decoder_convs(self):
        return self.temporal_convolutions()


def load_video_vae(family: str, *, root: Path = MODEL_ROOT,
                   device: str | torch.device = 'cpu', dtype: torch.dtype = torch.float32):
    """Load an official VAE with exact state-dict keys and native geometry."""
    if family == 'ltx25':
        from .cross_arch_ltx import load_ltx_vae
        return load_ltx_vae(root=root, device=device, dtype=dtype)
    spec = SPECS[family]
    folder = Path(root) / spec.folder / 'vae'
    config = json.loads((folder / 'config.json').read_text())
    weights = folder / 'diffusion_pytorch_model.safetensors'
    model_class = _model_class(family)
    with torch.device('meta'):
        model = model_class.from_config(config)
    state = load_file(str(weights), device='cpu')
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.to(device=device, dtype=dtype).eval()
    model.encoder.requires_grad_(False)
    if family == 'hunyuanvideo15':
        model.cpu_offload = False
    adapter = VideoVAE(family, model, weights)
    adapter.temporal_convolutions()
    return adapter


load_vae = load_video_vae
