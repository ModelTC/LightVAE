"""ComfyUI custom node for block-pruned MiniMax-H3 video VAEs."""

from .nodes import LightVAELoader

NODE_CLASS_MAPPINGS = {"LightVAELoader": LightVAELoader}
NODE_DISPLAY_NAME_MAPPINGS = {"LightVAELoader": "Load H3 LightVAE"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
