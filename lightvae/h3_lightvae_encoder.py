"""MiniMax-H3 LightVAE-Lite encoder with residual-block pruning."""

from torch import nn

from base_vae.h3_vae_encoder import Encoder as BaseEncoder


class ResidualShortcut(nn.Module):
    """Retain the projection of a pruned residual block."""

    def __init__(self, block):
        super().__init__()
        self.conv_shortcut = block.conv_shortcut

    def forward(self, value):
        return value if self.conv_shortcut is None else self.conv_shortcut(value)


class Encoder(BaseEncoder):
    """Use the native H3 latent interface with the exported Lite architecture."""

    def _configure_encoder(self, cfg):
        kept = cfg.get("kept_residual_indices")
        if kept is None:
            raise ValueError("H3 Lite encoder requires kept_residual_indices in checkpoint metadata")
        depth = sum(len(stage.resnets) for stage in self.encoder.down_blocks)
        if kept != sorted(set(kept)) or any(i < 0 or i >= depth for i in kept):
            raise ValueError("Invalid H3 encoder kept_residual_indices")
        index = 0
        for stage in self.encoder.down_blocks:
            for local, block in enumerate(stage.resnets):
                if index not in kept:
                    stage.resnets[local] = ResidualShortcut(block)
                index += 1
