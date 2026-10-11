"""Wan2.2 LightVAE-Pro and LightVAE-Lite decoder."""

from .modules import Decoder as WanDecoder


class Decoder(WanDecoder):
    """Decode raw 48-channel Wan2.2 posterior-mean latents."""

    def __init__(self, config, checkpoint=None):
        if config.get("family") != "wan22":
            raise ValueError("Wan2.2 LightVAE decoder requires a wan22 config")
        super().__init__(config=config, checkpoint=checkpoint)
