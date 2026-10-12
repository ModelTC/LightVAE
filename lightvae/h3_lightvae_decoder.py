"""MiniMax-H3 LightVAE-Lite decoder with transformer-block pruning."""

from .base_vae.h3_vae_decoder import Decoder as BaseDecoder


class Decoder(BaseDecoder):
    """Retain native H3 normalization and stitching with the Lite depth."""

    def _decoder_depth(self, cfg):
        kept = cfg.get("kept_layers")
        if kept is None:
            raise ValueError("H3 Lite decoder requires kept_layers in its config or checkpoint")
        if not kept or kept != sorted(set(kept)) or kept[0] < 0 or kept[-1] >= cfg["decoder_num_layers"]:
            raise ValueError("H3 kept_layers must be nonempty ordered indices within the original decoder")
        return len(kept)
