"""One H3-only node: independently replace the video encoder and decoder."""

from pathlib import Path

import folder_paths
from .h3_vae import LightVAE


DIRECTORY = Path(folder_paths.models_dir) / "vae" / "LightVAE"
folder_paths.add_model_folder_path("lightvae", str(DIRECTORY))
folder_paths.folder_names_and_paths["lightvae"][1].add(".safetensors")


class LightVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = [name for name in folder_paths.get_filename_list("lightvae") if "h3" in name.lower()]
        return {"required": {
            "base_vae": ("VAE",),
            "decoder": (["none"] + [name for name in files if "decoder" in name],),
            "encoder": (["none"] + [name for name in files if "encoder" in name],),
        }}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "LightVAE"
    DESCRIPTION = "Block-pruned H3 video VAE. 'none' keeps that original component. The audio VAE is unchanged."

    def load(self, base_vae, decoder, encoder):
        if decoder == "none" and encoder == "none":
            return (base_vae,)
        decoder_path = None if decoder == "none" else folder_paths.get_full_path_or_raise("lightvae", decoder)
        encoder_path = None if encoder == "none" else folder_paths.get_full_path_or_raise("lightvae", encoder)
        vae = LightVAE(base_vae, decoder_path, encoder_path)
        return (vae,)
