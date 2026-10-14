"""Load block-pruned LightVAE weights into ComfyUI's native H3 video VAE."""

import json

import torch
from torch import nn
from safetensors import safe_open

import comfy.sd
import comfy.ops
import comfy.utils
from comfy import model_management
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE


class ResidualShortcut(nn.Module):
    """A removed encoder block still needs its channel-changing projection."""

    def __init__(self, block):
        super().__init__()
        self.nin_shortcut = getattr(block, "nin_shortcut", nn.Identity())

    def forward(self, x):
        return self.nin_shortcut(x)


def native_state_dict(state, component, heads=32):
    """Rename Diffusers layers and pack Q/K/V in ComfyUI's per-head order."""
    converted = {}
    for name, tensor in state.items():
        if component == "encoder":
            name = (name.replace(".down_blocks.", ".down.")
                    .replace(".resnets.", ".block.")
                    .replace(".downsamplers.0.", ".downsample.")
                    .replace(".conv_shortcut.", ".nin_shortcut."))
        else:
            if ".ff.net.0.proj." in name:
                # Diffusers stores (up, gate); native ComfyUI uses (gate, up).
                up, gate = tensor.chunk(2, dim=0)
                tensor = torch.cat((gate, up), dim=0)
            name = (name.replace(".attn.to_out.0.", ".attn.to_out.")
                    .replace("decoder.proj_in.", "decoder.x_embedder.")
                    .replace(".ff.net.0.proj.", ".ff.w1.")
                    .replace(".ff.net.2.", ".ff.w2."))
        converted[name] = tensor
    if component == "decoder":
        for name in list(converted):
            if ".attn.to_q." not in name:
                continue
            keys = [name.replace(".to_q.", f".to_{projection}.") for projection in ("q", "k", "v")]
            tensors = [converted.pop(key) for key in keys]
            shape = tensors[0].shape
            packed = torch.stack([tensor.reshape(heads, -1, *shape[1:]) for tensor in tensors], dim=1)
            converted[name.replace(".to_q.", ".to_qkv.")] = packed.reshape(3 * shape[0], *shape[1:])
        if "decoder.register_tokens" in converted:
            # Native keeps this unused training buffer in its state dict.
            converted.setdefault("decoder.mask_token", torch.zeros_like(converted["decoder.register_tokens"][:, :1]))
    return converted


def load_component(path, component):
    # Standard and previous *-comfy exports both embed this architecture.
    # No offline export, sidecar JSON, lightvae package or TLR is needed.
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        architecture = json.loads(handle.metadata()["architecture"])
        config = architecture.get("teacher_config", architecture)
        prefixes = ("decoder.", "post_quant_conv.") if component == "decoder" else ("encoder.", "quant_conv.")
        state = {name: handle.get_tensor(name) for name in handle.keys() if name.startswith(prefixes)}
    kept_key = "kept_layers" if component == "decoder" else "kept_residual_indices"
    return native_state_dict(state, component, config.get("decoder_num_attention_heads", 32)), architecture[kept_key]


class LightVAE(comfy.sd.VAE):
    """Native H3 encode/decode; unselected components use the original VAE."""

    def __init__(self, base_vae, decoder_path=None, encoder_path=None):
        if not isinstance(base_vae.first_stage_model, MiniMaxH3VideoVAE):
            raise ValueError("base_vae must be the original MiniMax-H3 VIDEO VAE, not an audio or Wan VAE.")
        self.__dict__.update(vars(base_vae))
        self.base_vae = base_vae
        self.use_decoder = decoder_path is not None
        self.use_encoder = encoder_path is not None
        state = {}
        num_layers = 36
        if self.use_decoder:
            weights, kept = load_component(decoder_path, "decoder")
            num_layers = len(kept)
            state.update(weights)
        quantization = comfy.utils.detect_layer_quantization(state, "")
        operations = (comfy.ops.mixed_precision_ops(quantization, self.vae_dtype)
                      if quantization is not None else comfy.ops.disable_weight_init)
        model = MiniMaxH3VideoVAE(operations=operations, num_layers=num_layers)
        if not self.use_decoder:
            model.decoder = model.post_quant_conv = None
        if self.use_encoder:
            weights, kept = load_component(encoder_path, "encoder")
            index = 0
            for stage in model.encoder.down:
                for local, block in enumerate(stage.block):
                    if index not in kept:
                        stage.block[local] = ResidualShortcut(block)
                    index += 1
            state.update(weights)
        else:
            model.encoder = model.quant_conv = None
        state.update(latents_mean=model.latents_mean, latents_std=model.latents_std)
        model.load_state_dict(state, strict=True, assign=True)
        self.first_stage_model = model.eval().requires_grad_(False).to(dtype=self.vae_dtype)
        model_management.archive_model_dtypes(self.first_stage_model)
        self.patcher = type(base_vae.patcher)(
            self.first_stage_model, load_device=self.device,
            offload_device=base_vae.patcher.offload_device,
        )
        self.size = None
        self.model_size()

    def encode(self, pixel_samples):
        return super().encode(pixel_samples) if self.use_encoder else self.base_vae.encode(pixel_samples)

    def encode_tiled(self, pixel_samples, **kwargs):
        return super().encode_tiled(pixel_samples, **kwargs) if self.use_encoder else self.base_vae.encode_tiled(pixel_samples, **kwargs)

    def decode(self, samples_in, vae_options=None):
        return super().decode(samples_in, vae_options or {}) if self.use_decoder else self.base_vae.decode(samples_in, vae_options or {})

    def decode_tiled(self, samples, **kwargs):
        return super().decode_tiled(samples, **kwargs) if self.use_decoder else self.base_vae.decode_tiled(samples, **kwargs)
