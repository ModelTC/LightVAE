# ⚡ H3 LightVAE for ComfyUI

**One node. Two selections. Native H3 inference.**

This integration supports MiniMax-H3 only. Our H3 weights use **block pruning**,
not TLR. The node loads the retained blocks into ComfyUI's native H3 video VAE;
encoding, decoding, normalization, temporal stitching and model offload stay native.
No `pip install lightvae`, weight export, sidecar JSON or custom Triton kernels are needed.

## Install

Update ComfyUI to a version with native MiniMax-H3 support. Run with the Python
environment that starts ComfyUI:

```bash
cd /path/to/ComfyUI
git clone https://github.com/ModelTC/LightVAE.git LightVAE
python LightVAE/comfyui/install.py --comfy-root .

python -m pip install huggingface_hub
hf download lightx2v/LightVAE \
  lightvae-lite-h3-decoder.safetensors \
  lightvae-lite-h3-encoder.safetensors \
  --local-dir models/vae/LightVAE
```

Already cloned? Run `git -C LightVAE pull`, then rerun the installer.
It creates `custom_nodes/ComfyUI-LightVAE/` automatically; **do not clone the
entire repository directly into `custom_nodes`**.

Restart ComfyUI and search for **Load H3 LightVAE**. The original H3 video VAE
must be connected to `base_vae`. Select Lite components in this node, not the
standard **Load VAE** node. If you do not have the original video VAE yet:

```bash
hf download Comfy-Org/MiniMax-H3 \
  vae/minimax_h3_video_vae_fp16.safetensors \
  --local-dir models
```

The standard H3 files above and our previous `*-h3-*-comfy.safetensors` exports
are both accepted. You do not need to download both formats.
Rerun the installer after updating this checkout; it replaces the old adapter
and removes its obsolete Wan workflow files. Other installed nodes and weights are untouched.

## Choose a workflow

| Workflow | What changes |
| :-- | :-- |
| [FL2V: first + last frame](workflows/h3_fl2v_lightvae.json) | **Decoder only.** Original VAE encodes both keyframes. |
| [Ref: image references](workflows/h3_ref_image_lightvae.json) | Encoder and decoder can be replaced independently. |
| [Ref: video + soundtrack](workflows/h3_ref_video_lightvae.json) | Encoder and decoder can be replaced independently; audio stays original. |

Drag the JSON into ComfyUI, select the model files, upload your images/video and run.
Defaults: **1344×768 · 124 frames · 24 FPS · seed 42 · 20 base-model steps**.
These are not 4-/8-step turbo-LoRA workflows.
See [workflow notes and required models](workflows/README.md).

## Encoder / decoder selection

Connect **Load VAE → Load H3 LightVAE → VAE Decode**.
`none` means “use that component of the connected original VAE”.

| Encoder | Decoder | Result |
| :-- | :-- | :-- |
| `none` | H3 Lite | Original encoding + Lite decoding |
| H3 Lite | H3 Lite | Lite encoding + Lite decoding |
| H3 Lite | `none` | Lite encoding + original decoding |
| `none` | `none` | Original VAE unchanged |

For **FL2V**, leave `encoder = none`. Its workflow connects the original VAE
directly to `MiniMaxH3ImageToVideo`, so keyframe encoding cannot change accidentally.
For **Ref**, the selected VAE feeds both `MiniMaxH3ReferenceToVideo` and `VAE Decode`.
The separate **audio VAE is never replaced**.

## ConvRot-INT8 decoder

An optional quantized Lite decoder uses ComfyUI's native ConvRot-INT8 operations.
Update this checkout and rerun the installer before using it.

```bash
hf download lightx2v/LightVAE \
  lightvae-lite-h3-decoder-int8-convrot.safetensors \
  --local-dir /path/to/ComfyUI/models/vae/LightVAE
```

Restart ComfyUI and select that file in **Load H3 LightVAE → decoder**.
Keep `base_vae` connected; encoder selection, audio VAE and workflow connections
stay unchanged. This INT8 file is not supported by the standalone `infer.py` or
`benchmark.py` scripts.

For native NVIDIA CUDA acceleration, use a **CUDA 13.0+ PyTorch build** and a
compatible driver. Older builds may use fallback operations; performance depends
on the GPU and backend.

<details>
<summary>Convert a standard Lite decoder yourself</summary>

Run from the LightVAE repository directory in your ComfyUI environment:

```bash
PYTHONPATH=/path/to/ComfyUI python -m comfyui.convert_h3_int8 \
  --input /path/to/lightvae-lite-h3-decoder.safetensors \
  --output /path/to/ComfyUI/models/vae/LightVAE/lightvae-lite-h3-decoder-int8-convrot.safetensors
```

The converter writes a new checkpoint and leaves the original weights unchanged.

</details>
