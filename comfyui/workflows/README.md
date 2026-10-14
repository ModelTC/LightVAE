# 🎬 MiniMax-H3 workflows

Install [Load H3 LightVAE](../README.md#install), restart ComfyUI, then drag in a JSON.
Only **ComfyUI core + this node** are needed. Outputs go to `ComfyUI/output/LightVAE/`.

| Workflow | Inputs | VAE wiring |
| :-- | :-- | :-- |
| [h3_fl2v_lightvae.json](h3_fl2v_lightvae.json) | First frame + last frame + prompt | Original encoder; LightVAE decoder only |
| [h3_ref_image_lightvae.json](h3_ref_image_lightvae.json) | Reference image(s) + prompt | Encoder / decoder independently selectable |
| [h3_ref_video_lightvae.json](h3_ref_video_lightvae.json) | Reference video + its soundtrack + prompt | Encoder / decoder independently selectable |

All three default to **1344×768, 124 frames, 24 FPS**, with native stereo audio.
Sampling: **20 steps, `res_multistep`, `simple`, no CFG; video shift 12 / audio shift 3**.
No turbo LoRA is required. Keep seed 42 when comparing VAE choices.

## Models

Place these ComfyUI-compatible checkpoints under `ComfyUI/models/`:

| File | Directory | Needed for |
| :-- | :-- | :-- |
| `minimax_h3_fl2va_bf16.safetensors` | `diffusion_models/` | FL2V |
| `minimax_h3_ref2va_bf16.safetensors` | `diffusion_models/` | Ref |
| `qwen3vl_32b_minimax_h3_bf16.safetensors` | `text_encoders/` | All |
| `minimax_h3_video_vae_fp16.safetensors` | `vae/` | Original video VAE / fallback |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/` | Original audio VAE |
| `lightvae-lite-h3-decoder.safetensors` | `vae/LightVAE/` | Lite decoding |
| `lightvae-lite-h3-decoder-int8-convrot.safetensors` | `vae/LightVAE/` | Optional ConvRot-INT8 Lite decoding |
| `lightvae-lite-h3-encoder.safetensors` | `vae/LightVAE/` | Optional Lite reference encoding |

Get the base models from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
and our components from [lightx2v/LightVAE](https://huggingface.co/lightx2v/LightVAE).
Download only the DiT for the task you use; the BF16 DiT and 32B text encoder are large.
Existing compatible quantized DiT/text-encoder files can be selected instead.

To use the INT8 Lite decoder, select its file in **Load H3 LightVAE → decoder**;
the same workflows and encoder choices apply. See [INT8 setup](../README.md#convrot-int8-decoder).

## FL2V

Upload **first_frame.png** and **last_frame.png**. The original video VAE feeds
`MiniMaxH3ImageToVideo.vae`; LightVAE feeds only the final `VAE Decode` node.
Keep `encoder = none` and select the Lite decoder. Set `decoder = none` for an original-VAE baseline.
Disconnect `last_frame` for I2V, or both frame inputs for T2V.

## Ref

Upload your reference image or a short **24 FPS** reference video.
Set either component to `none` to use its original version; select both Lite files to replace both.
`base_vae` is already connected in each workflow.

Use **`<Picture 1>`**, **`<Video 1>`**, **`<Audio 1>`** in the prompt; socket suffixes start at 0.
Additional images can be connected to the growing reference inputs.
The video workflow connects its soundtrack to `ref_video_audios.ref_video_audio_0`,
not to the standalone `ref_audios` input. A connected video soundtrack takes an Audio index.
The native ref node does not resample an arbitrary-FPS video; prepare a 24 FPS clip before uploading.

These workflows change only the video VAE, not the diffusion model, sampler or audio codec.
