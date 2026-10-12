"""LightVAE components at ComfyUI's latent boundary (Wan raw, H3 normalized)."""

import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open

from lightvae.modules import Decoder as WanDecoder, FusedNorm
from lightvae.tlr import TLR
from lightvae.h3_lightvae_decoder import Decoder as H3Decoder
from lightvae.h3_lightvae_encoder import Encoder as H3Encoder
from lightvae.utils import split_tiles, stitch_tiles


def read_info(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if "lightvae_comfy" not in metadata:
        raise ValueError("Select a LightVAE *-comfy.safetensors export, not a dense third-party VAE.")
    info = json.loads(metadata["lightvae_comfy"])
    if info["format"] != "lightvae-comfy-v1":
        raise ValueError(f"Unsupported LightVAE format: {info['format']}")
    return info, metadata


def load_component(path, component, backend="auto"):
    path = Path(path)
    info, metadata = read_info(path)
    if info["component"] != component:
        raise ValueError(f"Expected {component}, got {info['component']}: {path.name}")
    if info["family"].startswith("wan"):
        if component != "decoder":
            raise ValueError("Wan LightVAE only compresses the decoder; connect the original encoder VAE.")
        model = WanDecoder(json.loads(metadata["checkpoint_metadata"]), path)
        # Native ComfyUI samplers already undo Wan's diffusion normalization.
        # Skip the standalone decoder's normalization inverse, avoiding a
        # lossy raw -> normalized -> raw BF16 round trip.
        model.input_is_normalized = False
        for layer in model.modules():
            if isinstance(layer, (TLR, FusedNorm)):
                layer.backend = backend
    elif info["family"] == "h3":
        model = (H3Decoder if component == "decoder" else H3Encoder)(checkpoint=path)
    else:
        raise ValueError(f"Unknown LightVAE family: {info['family']}")
    return model.eval().requires_grad_(False), info


class Components(nn.Module):
    def __init__(self, family, decoder=None, encoder=None):
        super().__init__()
        self.family, self.decoder, self.encoder = family, decoder, encoder

    def decode(self, z):
        if self.decoder is None:
            raise ValueError("This VAE has no decoder; select a decoder or connect a base VAE.")
        if z.ndim != 5:
            raise ValueError("Video latents must be [B,C,T,H,W].")
        if self.family == "h3" and z.shape[2] == 1:
            # Match ComfyUI H3 still-image semantics, not a fake 5-frame video.
            video = self.decoder._decode_spatial(z.repeat(1, 1, 5, 1, 1))[:, :, 3:4]
            mean = video.new_tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1, 1)
            std = video.new_tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1, 1)
            return (video.float() * std + mean) * 2 - 1
        return self.decoder(z)

    def encode(self, x):
        if self.encoder is None:
            raise ValueError("Connect the original VAE to base_vae to encode reference images/video.")
        if self.family == "h3":
            encoder = self.encoder
            hs, ho = split_tiles(x.shape[-2], encoder.tile_size, encoder.tile_overlap)
            ws, wo = split_tiles(x.shape[-1], encoder.tile_size, encoder.tile_overlap)
            clips = []
            for begin in range(0, x.shape[2], 17):
                clip = x[:, :, begin:begin + 17]
                # Match native ComfyUI: pad each last video clip to 17, without
                # first extending the whole video to the standalone recon grid.
                if x.shape[2] > 1 and clip.shape[2] < 17:
                    clip = torch.cat((clip, clip[:, :, -1:].expand(-1, -1, 17 - clip.shape[2], -1, -1)), dim=2)
                rows = [[encoder._encode_clip(clip[..., h.start * 16:h.stop * 16, w.start * 16:w.stop * 16])
                         for w in ws] for h in hs]
                clips.append(stitch_tiles(rows, [v // 16 for v in ho], [v // 16 for v in wo]))
            moments = torch.cat(clips, dim=2)
            moments = moments[:, :, -1:] if x.shape[2] == 1 else moments[:, :, :-3]
            mean = moments[:, :encoder.latent_channels].float()
            return ((mean - encoder.latents_mean.float().view(1, -1, 1, 1, 1))
                    / encoder.latents_std.float().view(1, -1, 1, 1, 1))
        return self.encoder(x)

    def decode_tiled(self, z, **kwargs):
        # H3 has its own spatial and non-integer temporal stitching. Generic
        # 4x temporal tiles do not preserve the 5k+2 -> 17k+5 frame alignment.
        return self.decode(z)

    def encode_tiled(self, x, **kwargs):
        return self.encode(x)
