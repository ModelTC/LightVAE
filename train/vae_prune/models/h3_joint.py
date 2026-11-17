"""Pure-pruned H3 E+D; differentiable released temporal/spatial protocol."""

import json
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file
from .h3_native.minimax_h3.pruned_encoder import MiniMaxH3PrunedVideoEncoder
from .h3_native.minimax_h3.pruned_vae import MiniMaxH3PrunedVideoVAE
from .h3_native.minimax_h3.vae_geometry import align_num_frames, spatial_tile_layout
from .h3_native.minimax_h3.video_vae import imagenet_preprocess


def load_component(path, cls, kind):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    if meta.get("model_type") != kind or "architecture" not in meta:
        raise ValueError(f"Expected self-describing {kind} safetensors: {path}")
    arch = json.loads(meta["architecture"])
    if arch.get("search") is not None:
        raise ValueError("Only fixed pruned checkpoints are supported")
    model = cls(**arch)
    model.load_state_dict(load_file(str(path)), strict=True)
    model.enable_gradient_checkpointing()
    return model, arch


class H3Joint(nn.Module):
    @classmethod
    def from_components(cls, encoder, decoder):
        """Share frozen full-teacher components using the same RGB protocol."""
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.encoder, model.decoder = encoder, decoder
        return model.eval()

    def __init__(self, encoder_path, decoder_path):
        super().__init__()
        self.encoder, self.encoder_arch = load_component(
            encoder_path, MiniMaxH3PrunedVideoEncoder, "minimax_h3_pruned_encoder"
        )
        self.decoder, self.decoder_arch = load_component(decoder_path, MiniMaxH3PrunedVideoVAE, "minimax_h3_pruned_vae")
        if self.encoder_arch["teacher_config"] != self.decoder_arch["teacher_config"]:
            raise ValueError("H3 encoder/decoder teacher configurations differ")
        self.requires_grad_(True)

    def set_train_component(self, component):
        if component not in ("joint", "encoder", "decoder"):
            raise ValueError("Unknown train component")
        self.train_component = component
        self.encoder.requires_grad_(component in ("joint", "encoder"))
        self.decoder.requires_grad_(component in ("joint", "decoder"))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        component = getattr(self, "train_component", "joint")
        if component == "encoder":
            self.decoder.eval()
        elif component == "decoder":
            self.encoder.eval()
        return self

    def encode_clip(self, x):
        e = self.encoder
        if not e.use_tiling:
            return e(x)
        layout = spatial_tile_layout(
            x.shape[-2] // 16,
            x.shape[-1] // 16,
            tile_height=e.tile_sample_min_height,
            tile_width=e.tile_sample_min_width,
            overlap_height=e.tile_sample_min_overlap_height,
            overlap_width=e.tile_sample_min_overlap_width,
        )
        rows = [
            [e(x[..., h.start * 16 : h.stop * 16, w.start * 16 : w.stop * 16]) for w in layout.width_slices]
            for h in layout.height_slices
        ]
        return e.stitch_tiles(
            rows, tuple(v // 16 for v in layout.height_overlaps), tuple(v // 16 for v in layout.width_overlaps)
        )

    def forward(self, x, anchor=None, return_training=False, encoder_supervision=False):
        # Public input/output are [-1,1]; loss uses only real frames, never tail padding.
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Expected B,3,T,H,W with spatial multiples of 16")
        real = x.shape[2]
        aligned = align_num_frames(max(96, real))
        padded = torch.cat((x, x[:, :, -1:].expand(-1, -1, aligned - real, -1, -1)), 2)
        pixels = imagenet_preprocess((padded.float() + 1) / 2)
        extra = (-aligned) % 17
        pixels = torch.cat((pixels, pixels[:, :, -1:].expand(-1, -1, extra, -1, -1)), 2)
        moments = torch.cat([self.encode_clip(c) for c in pixels.split(17, dim=2)], 2)[:, :, :-3]
        mean = moments.chunk(2, dim=1)[0]  # posterior mode, no detach: gradients reach E
        z = self.encoder.normalize_latents(mean)
        raw = self.decoder.decode_raw(z)[:, :, :real].float()
        # Avoid the inference-only RGB clamp while optimizing.
        rgb = raw * self.decoder.pixel_std.view(1, 3, 1, 1, 1) + self.decoder.pixel_mean.view(1, 3, 1, 1, 1)
        y = rgb * 2 - 1
        encoder_aux = None
        if encoder_supervision:
            # Same preprocessed real-frame window for student and frozen teacher.
            encoder_pixels = pixels[:, :, : min(real, 17), :256, :256]
            encoder_moments, encoder_features = self.encoder(encoder_pixels, return_features=True)
            encoder_aux = (encoder_pixels, encoder_moments, encoder_features)
        if anchor is None:
            if return_training and encoder_supervision:
                return y, z, None, None, encoder_aux
            return (y, z, None, None) if return_training else y
        # One H3 raw decode window and spatial tile; includes register tokens.
        window = z[:, :, :7, :16, :16]
        _, _, feature = self.decoder(window, auxiliary_feature_index=anchor)
        if encoder_supervision:
            return y, z, feature, window, encoder_aux
        return y, z, feature, window
