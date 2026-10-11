"""Wan2.1 LightVAE-Pro and LightVAE-Lite decoder."""

from .modules import Decoder as WanDecoder


class Decoder(WanDecoder):
    """Decode normalized 16-channel Wan2.1 latents."""

    def __init__(self, config, checkpoint=None):
        if config.get("family") != "wan21":
            raise ValueError("Wan2.1 LightVAE decoder requires a wan21 config")
        super().__init__(config=config, checkpoint=checkpoint)
