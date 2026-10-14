# ⚡ H3 LightVAE for ComfyUI

**One node. Two selections. Native H3 inference.**

This integration supports MiniMax-H3 only. Our H3 weights use **block pruning**,
not TLR. The node loads the retained blocks into ComfyUI's native H3 video VAE;
encoding, decoding, normalization, temporal stitching and model offload stay native.
No `pip install lightvae`, weight export, sidecar JSON or custom Triton kernels are needed.

## Install

From your LightVAE checkout, run with the Python that starts ComfyUI:

```bash
python comfyui/install.py --comfy-root /path/to/ComfyUI

hf download lightx2v/LightVAE \
  lightvae-lite-h3-decoder.safetensors \
  lightvae-lite-h3-encoder.safetensors \
  --local-dir /path/to/ComfyUI/models/vae/LightVAE
```

Restart ComfyUI and search for **Load H3 LightVAE**. The original H3 video VAE
must be connected to `base_vae`. If you do not have it yet:

```bash
hf download Comfy-Org/MiniMax-H3 \
  vae/minimax_h3_video_vae_fp16.safetensors \
  --local-dir /path/to/ComfyUI/models
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
