"""Convert an H3 Lite decoder to ComfyUI-native ConvRot-INT8 components.

Run from the LightVAE checkout with ComfyUI on PYTHONPATH:
    python -m comfyui.convert_h3_int8 --input decoder.safetensors --output decoder-int8.safetensors
"""

import argparse
import json
from pathlib import Path

import torch
from comfy_kitchen.tensor import TensorWiseINT8Layout
from safetensors import safe_open
from safetensors.torch import save_file

from .h3_vae import load_component


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with safe_open(str(args.input), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata())
    state, kept = load_component(args.input, "decoder")
    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    config_tensor = torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)
    count = 0
    with torch.inference_mode():
        for name in list(state):
            weight = state[name]
            # Native H3 injects quantized ops into the transformer blocks only.
            if not name.startswith("decoder.transformer_blocks.") or not name.endswith(".weight") or weight.ndim != 2:
                continue
            quantized, params = TensorWiseINT8Layout.quantize(
                weight.to(args.device), is_weight=True, per_channel=True,
                convrot=True, convrot_groupsize=256, stochastic_rounding=0,
            )
            prefix = name.removesuffix("weight")
            state[name] = quantized.cpu().contiguous()
            state[prefix + "weight_scale"] = params.scale.cpu().contiguous()
            state[prefix + "comfy_quant"] = config_tensor.clone()
            count += 1
            print(f"[{count}/{len(kept) * 4}] {name}", flush=True)
    if count != len(kept) * 4:
        raise ValueError(f"Expected four quantized linears per block, got {count} for {len(kept)} blocks")
    metadata["weight_format"] = "comfyui-native-int8-convrot"
    metadata["quantization"] = json.dumps(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.contiguous() for name, tensor in state.items()}, str(args.output), metadata=metadata)
    print(f"Saved {args.output} ({args.output.stat().st_size / 1e9:.3f} GB)", flush=True)


if __name__ == "__main__":
    main()
