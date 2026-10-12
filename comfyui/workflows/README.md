# 🎬 Video generation with LightVAE

Drag one of these JSON files into **current ComfyUI** after installing [Load LightVAE (TLR)](../README.md).
They are flat, editable workflows: no hidden subgraphs, paid API calls, turbo LoRAs or extra custom nodes.
Only **ComfyUI core + this LightVAE node** are required. Outputs are MP4 files under `ComfyUI/output/LightVAE/`.

## Choose a workflow

| Workflow | Task | Default output | LightVAE components |
| :-- | :-- | :-- | :-- |
| [Wan2.1 1.3B T2V](wan21_t2v_lightvae.json) | Prompt → video | 832×480, 81 frames, 16 FPS | **wan21** Pro / Lite decoder; original encoder |
| [Wan2.2 5B T2V](wan22_5b_t2v_lightvae.json) | Prompt → video | 1280×704, 121 frames, 24 FPS | **wan22** Pro / Lite decoder |
| [Wan2.2 5B I2V](wan22_5b_i2v_lightvae.json) | First image + prompt → video | 1280×704, 121 frames, 24 FPS | **wan22** Pro / Lite decoder; original encoder |
| [Wan2.2 14B I2V](wan22_14b_i2v_lightvae.json) | First image + prompt → video, two experts | 640×640, 81 frames, 16 FPS | **wan21** Pro / Lite decoder; original encoder |
| [H3 reference image](h3_ref_image_lightvae.json) | Reference image(s) + prompt → video + audio | 1344×768, 124 frames, 24 FPS | H3 Lite video encoder + decoder |
| [H3 reference video](h3_ref_video_lightvae.json) | Reference video + its soundtrack + prompt → video + audio | 1344×768, 124 frames, 24 FPS | H3 Lite video encoder + decoder |

**Wan2.2 14B uses the Wan2.1 16-channel latent space.** Do not select a wan22/48-channel decoder for it.
The 5B TI2V model is the one that uses the Wan2.2 VAE. Wan lengths should be `4k+1`; H3 lengths are `17k+5`.

For a first run, start with Wan2.1 1.3B T2V. The H3 BF16 DiT and 32B text encoder are very large;
LightVAE reduces VAE cost, not their weight or attention memory. CPU offload and sufficient host RAM may still be needed.

## Get started

Follow the [node installation guide](../README.md#installation), place the `*-comfy.safetensors`
weights in `ComfyUI/models/vae/LightVAE/`, and restart ComfyUI.
Drag a workflow JSON into the browser. For I2V/R2V, upload
your own image or video; the `example.png` / `reference_24fps.mp4` filenames are placeholders.

## Select encoder and decoder

### Wan

The original **Load VAE** node is already connected to **Load LightVAE → base_vae**.
Select `lightvae-pro-wan21-decoder-comfy.safetensors` or `lightvae-lite-wan21-decoder-comfy.safetensors`
for Wan2.1 / Wan2.2 14B; use the corresponding wan22 files for 5B. Leave `encoder = none`.
I2V conditioning continues to use the original encoder; generated frames use the selected decoder.

Set **decoder = none, encoder = none** to use the original VAE for both operations.
Keep the same seed when comparing outputs. The original `.pth` Wan VAE files also work:
select your actual filename if it differs from the default `.safetensors` name.

### H3 ref tasks

The default selects **both** H3 LightVAE files. The same `VAE` output goes to
`MiniMaxH3ReferenceToVideo.vae` for reference encoding and `VAEDecode.vae` for generated video.
The original video VAE is **not required** for this default.

To compare combinations, add an original **Load VAE** node with
`minimax_h3_video_vae_fp16.safetensors` and connect it to the LightVAE node's `base_vae`:

| `encoder` selection | `decoder` selection | Execution |
| :-- | :-- | :-- |
| H3 Lite encoder | H3 Lite decoder | Both compressed; `base_vae` optional |
| `none` | H3 Lite decoder | Original encoder + Lite decoder |
| H3 Lite encoder | `none` | Lite encoder + original decoder |
| `none` | `none` | Original encoder + original decoder |

Always use an H3 **video** VAE for `base_vae`. The separate **audio** VAE stays original
and feeds `VAEDecodeAudio`; LightVAE is not an audio codec.
The H3 graphs use **ref2va**, not the fl2va/T2V checkpoint, and no 4-/8-step distilled LoRA.

### Reference numbering and video input

- Use `<Picture 1>`, `<Picture 2>`, `<Video 1>` and `<Audio 1>` in the prompt.
  Input socket `ref_image_0` means **Picture 1**, not Picture 0.
- Additional images can be connected to the growing image-reference inputs. The default is `match` preprocessing.
- The video example connects frames to `ref_videos.ref_video_0` and its soundtrack to
  **`ref_video_audios.ref_video_audio_0`**, not the standalone `ref_audios` input.
- With one video carrying a connected soundtrack, that soundtrack is **Audio 1**;
  a subsequently connected standalone audio becomes **Audio 2**. A silent video supplies no soundtrack.
- The native H3 ref node expects **24 FPS** frames. `Get Video Components` does not resample them.
  Use a short reference clip, ideally no longer than the 124-frame output: the current native node
  trims longer refs to the output length and then to its `17k+5` frame grid.

Prepare a 24 FPS reference, preserving any soundtrack (replace the paths):

```bash
ffmpeg -n -i input.mp4 -t 5.166666667 \
  -map 0:v:0 -map '0:a:0?' -vf fps=24 \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k \
  reference_24fps.mp4
```

This creates a new file; `-n` refuses to overwrite one. Upload it through the workflow's **Load Video** node.
H3 samples a joint video/audio latent; native `VAEDecode` takes its video component,
`VAEDecodeAudio` takes its audio component, and `CreateVideo` muxes the generated audio into the MP4.

## Required model files

Download only the files for the workflow you use. These are **ComfyUI-compatible single-file**
DiT/text-encoder/VAE checkpoints; a Diffusers directory or arbitrary LightX2V training checkpoint
cannot be substituted just by renaming it. Click the filenames below and place them in the specified directories.
The workflow loader nodes also contain their download URLs in `properties.models`.
When ComfyUI runs on a remote server, a browser download may save to your own computer;
download or copy the files into the server's model directories instead.

| Use | File | Destination under `ComfyUI/models/` |
| :-- | :-- | :-- |
| All Wan | [umt5_xxl_fp8_e4m3fn_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) | `text_encoders/` |
| Wan2.1 1.3B | [wan2.1_t2v_1.3B_fp16.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors) | `diffusion_models/` |
| Wan2.1 / Wan2.2 14B | [wan_2.1_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) | `vae/` |
| Wan2.2 5B | [wan2.2_ti2v_5B_fp16.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors) | `diffusion_models/` |
| Wan2.2 5B | [wan2.2_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors) | `vae/` |
| Wan2.2 14B I2V | [wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors) | `diffusion_models/` |
| Wan2.2 14B I2V | [wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors) | `diffusion_models/` |
| H3 ref | [minimax_h3_ref2va_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors) | `diffusion_models/` |
| H3 ref | [qwen3vl_32b_minimax_h3_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors) | `text_encoders/` |
| H3 audio | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `vae/` |
| H3 original-video baseline, optional | [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) | `vae/` |
| LightVAE | Selected `*-comfy.safetensors` exports from this project | `vae/LightVAE/` |

The H3 defaults use BF16 to avoid making NVFP4/AWQ kernels a requirement. The Wan templates retain
the official FP8-scaled encoder/expert checkpoint choices; their compatibility is a **DiT/text-encoder**
issue, separate from the LightVAE backend. Choose compatible alternative ComfyUI exports if needed.

## Sampling defaults

| Workflow | Sampling | CFG | Shift |
| :-- | :-- | --: | :-- |
| Wan2.1 | 30 steps, `uni_pc`, `simple` | 6 | 8 |
| Wan2.2 5B | 20 steps, `uni_pc`, `simple` | 5 | 8 |
| Wan2.2 14B I2V | 20 steps, `euler`, `simple`; high expert `[0,10)`, low `[10,20)` | 3.5 | 5 for both experts |
| H3 ref | 20 steps, `res_multistep`, `simple`, `BasicGuider` | No CFG | Video 12, audio 3 |

For Wan2.2 14B, the first sampler keeps leftover noise; the second continues **without adding noise**.
If changing its total steps, update both nodes' total and split settings consistently.
These are base-model examples, not the earlier DMD/PDMD training inference schedules.

## Validation and provenance

The layouts follow the official [Wan2.1](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/text_to_video_wan.json),
[Wan2.2 5B](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_5B_ti2v.json),
[Wan2.2 14B](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_14B_i2v.json),
and [H3 ref](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json)
node recipes, inspected on 2026-10-12. They expose the VAE connections, omit optional turbo branches,
and use the defaults above. H3 checkpoint precision is explicitly changed to BF16, not claimed to match
the current official template's quantized default. See the [native H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native).

All six graphs passed native node-schema and API prompt validation against ComfyUI
`e8665ca8e4089ab917567cdb491308e1397663d4`. This checks graph structure, not installed model files.

**Wan2.1 T2V** and **H3 image-reference generation** have also produced complete MP4s.
The H3 test used 1344×768, 362 frames at 24 FPS (~15.08 s) with audio; the supplied workflow
defaults to 124 frames. The remaining generation workflows have not had full end-to-end tests.
See [test results and VAE decoding speed](../README.md#test-results).

