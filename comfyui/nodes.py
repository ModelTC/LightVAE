"""Load accelerated LightVAE components into existing standard VAE workflows."""

from pathlib import Path

import folder_paths
import torch

from .vae import LightVAE


DIRECTORY = Path(folder_paths.models_dir) / "vae" / "LightVAE"
folder_paths.add_model_folder_path("lightvae", str(DIRECTORY))
folder_paths.folder_names_and_paths["lightvae"][1].add(".safetensors")


class LightVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("lightvae")
        return {"required": {
            "decoder": (["none"] + [name for name in files if "decoder" in name],),
            "encoder": (["none"] + [name for name in files if "encoder" in name],),
            "precision": (["auto", "bf16", "fp32", "fp16"], {"default": "auto"}),
            "backend": (["auto", "torch"], {"default": "auto"}),
            "device": (["auto", "cpu"], {"default": "auto"}),
        }, "optional": {"base_vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "LightVAE"
    DESCRIPTION = ("Packed TLR / pruned H3 VAE. Connect base_vae for unmodified encoder/decoder. "
                   "Wan2.2 14B uses wan21 weights; 5B TI2V uses wan22.")

    def load(self, decoder, encoder, precision="auto", backend="auto", device="auto", base_vae=None):
        if decoder == "none" and encoder == "none":
            if base_vae is None:
                raise ValueError("Select a LightVAE component or connect the original VAE to base_vae.")
            return (base_vae,)
        decoder_path = None if decoder == "none" else folder_paths.get_full_path_or_raise("lightvae", decoder)
        encoder_path = None if encoder == "none" else folder_paths.get_full_path_or_raise("lightvae", encoder)
        vae = LightVAE(decoder_path, encoder_path, base_vae, precision, backend,
                       torch.device("cpu") if device == "cpu" else None)
        return (vae,)
