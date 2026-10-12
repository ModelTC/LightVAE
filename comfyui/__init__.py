"""ComfyUI custom nodes for packed LightVAE inference."""

from .nodes import LightVAELoader

NODE_CLASS_MAPPINGS = {"LightVAELoader": LightVAELoader}
NODE_DISPLAY_NAME_MAPPINGS = {"LightVAELoader": "Load LightVAE (TLR)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
