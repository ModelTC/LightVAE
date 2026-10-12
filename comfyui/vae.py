"""Standard ComfyUI VAE object backed by the packed LightVAE runtime."""

import logging
import math

import torch
import comfy.sd
from comfy import model_management
from comfy.model_patcher import ModelPatcher

from .components import Components, load_component


def select_dtype(device, precision):
    if precision != "auto":
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    if device.type == "cpu":
        return torch.float32
    return model_management.vae_dtype(device, [torch.bfloat16, torch.float32])


class LightVAE(comfy.sd.VAE):
    def __init__(self, decoder_path=None, encoder_path=None, base_vae=None,
                 precision="auto", backend="auto", device=None):
        device = device or model_management.vae_device()
        decoder, encoder, infos = None, None, []
        if decoder_path is not None:
            decoder, info = load_component(decoder_path, "decoder", backend)
            infos.append(info)
        if encoder_path is not None:
            encoder, info = load_component(encoder_path, "encoder", backend)
            infos.append(info)
        if not infos or len({info['family'] for info in infos}) != 1:
            raise ValueError("Select at least one LightVAE component; encoder and decoder must share a family.")
        self.family = infos[0]["family"]
        self.latent_channels = infos[0]["latent_channels"]
        spatial = 8 if self.family == "wan21" else 16
        if base_vae is not None and (base_vae.latent_channels != self.latent_channels
                                    or base_vae.spacial_compression_decode() != spatial):
            raise ValueError("base_vae has the wrong latent space (Wan 16ch / Wan 48ch / H3 24ch).")
        self.base_vae = base_vae
        self.first_stage_model = Components(self.family, decoder, encoder).eval().requires_grad_(False)
        self.device, self.vae_dtype = device, select_dtype(device, precision)
        self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        self.output_device = model_management.intermediate_device()
        self.first_stage_model.to(dtype=self.vae_dtype)
        # Native nn modules and TLR use their actual tensor weights, not Comfy's
        # per-layer manual-cast ops: load/offload the component as a whole.
        self.disable_offload = True
        self.patcher = ModelPatcher(self.first_stage_model, load_device=device,
                                    offload_device=model_management.vae_offload_device())
        self.latent_dim, self.output_channels = 3, 3
        self.not_video, self.crop_input = False, True
        self.size, self.pad_channel_value, self.extra_1d_channel = None, None, None
        self.format_encoded = None
        self.handles_tiling = self.family == "h3"
        self.audio_sample_rate = 44100
        self.process_input = lambda image: image * 2 - 1
        self.process_output = lambda image: image.add_(1).mul_(0.5).clamp_(0, 1)
        if self.family == "h3":
            self.upscale_ratio = (lambda t: 1 if t == 1 else (t - 2) // 5 * 17 + 5, spatial, spatial)
            self.downscale_ratio = (lambda t: 1 if t == 1 else math.ceil(t / 17) * 5 - 3,
                                    spatial, spatial)
        else:
            self.upscale_ratio = (lambda t: max(0, t * 4 - 3), spatial, spatial)
            self.downscale_ratio = (lambda t: max(0, (t + 3) // 4), spatial, spatial)
        self.upscale_index_formula = self.downscale_index_formula = (4, spatial, spatial)
        self.memory_used_decode = lambda shape, dtype: (
            (8000 * spatial * spatial * shape[-2] * shape[-1]
             + math.prod(shape) * spatial * spatial * 4) * model_management.dtype_size(dtype))
        self.memory_used_encode = lambda shape, dtype: (
            (6000 * shape[-2] * shape[-1] + math.prod(shape) * 4) * model_management.dtype_size(dtype))
        logging.info("LightVAE %s: device=%s, dtype=%s, kernels=%s; packed TLR retained for Wan",
                     self.family, self.device, self.vae_dtype, backend)

    def _base(self, operation):
        if self.base_vae is None:
            raise ValueError(f"No {operation} selected. Connect the matching original VAE to base_vae.")
        return self.base_vae

    def encode(self, pixel_samples):
        if self.first_stage_model.encoder is None:
            return self._base("encoder").encode(pixel_samples)
        return super().encode(pixel_samples)

    def encode_tiled(self, pixel_samples, **kwargs):
        if self.first_stage_model.encoder is None:
            return self._base("encoder").encode_tiled(pixel_samples, **kwargs)
        return super().encode_tiled(pixel_samples, **kwargs)

    def decode(self, samples_in, vae_options=None):
        if self.first_stage_model.decoder is None:
            return self._base("decoder").decode(samples_in, vae_options or {})
        return super().decode(samples_in, vae_options or {})

    def decode_tiled(self, samples, **kwargs):
        if self.first_stage_model.decoder is None:
            return self._base("decoder").decode_tiled(samples, **kwargs)
        return super().decode_tiled(samples, **kwargs)
