<div align="center">

# ⚡ LightVAE

**Towards Compact and Efficient Video Autoencoders**

Wan2.1 · Wan2.2 · MiniMax-H3

[🤗 Models](https://huggingface.co/lightx2v/LightVAE) · [📄 Paper](https://arxiv.org/abs/2610.14726) · [⚡ LightX2V](https://github.com/ModelTC/LightX2V)

</div>


## ✨ Overview

Video VAE decoding is a computational bottleneck in video generation.
**LightVAE** compresses pretrained video VAE decoders while preserving their
original latent interface, so the original encoder can be used directly.

- **LightVAE-Pro** — temporal low-rank convolutions with the original decoder
  depth, available for Wan2.1 and Wan2.2.
- **LightVAE-Lite** — additional block pruning for Wan; a pruned decoder and
  a separate Lite encoder for MiniMax-H3.
- **Original latent interface** — Wan models reuse their original,
  uncompressed encoder.

This repository provides **video reconstruction** and **decoder speed benchmarks**.
To use LightVAE in a video generation pipeline, see
[LightX2V](https://github.com/ModelTC/LightX2V).

For **ComfyUI**, see the [custom node, wiring examples and measured results](comfyui/README.md)
and the [ready-to-use video generation workflows](comfyui/workflows/README.md).
The node keeps Wan's temporal low-rank factors and supports the standard `VAE`
encode/decode interface. Triton is optional: compatible NVIDIA GPUs use fused
kernels; other supported PyTorch devices use a factored PyTorch path.

![Architecture comparison of Wan2.1 VAE, Turbo-VAED, Flash-VAED, and LightVAE](assets/architecture.png)

## 📊 Results

Decoder throughput and reconstruction PSNR reported in Table 1 of the paper.

![Table 1: decoder throughput and reconstruction quality comparison](assets/table1.png)

## 🎬 Visual Comparisons

Selected reconstruction examples. Labels show decoder FPS and PSNR; Wan models
share the original encoder, while H3 also reports encoder FPS.

> **🖥️ Test setup** · NVIDIA H200 · PyTorch 2.11.0 · CUDA 12.8 · BF16  
> **🎞️ Video format** · 3 clips × 124 frames · 768p (768 × 1344, portrait) · 24 FPS

Each clip is reconstructed separately, then concatenated into a 372-frame
comparison. Timings use eager execution with one warmup run and the median
of three timed runs. PSNR is measured before MP4 compression.


### Wan2.1

**Left → right:** Original VAE · LightVAE-Pro · LightVAE-Lite

<video src="https://github.com/user-attachments/assets/94f0175c-a9bf-4f08-a178-45989785e087" controls width="100%"></video>

### Wan2.2

**Left → right:** Original VAE · LightVAE-Pro · LightVAE-Lite

<video src="https://github.com/user-attachments/assets/5084f558-c200-4764-98cd-8011d95b915f" controls width="100%"></video>

### MiniMax-H3

**Left → right:** Original VAE · Lite decoder · Lite encoder + decoder

<video src="https://github.com/user-attachments/assets/023473e8-8a91-4ffc-91b5-e7e42f0abf4b" controls width="100%"></video>

---

## 🛠️ Installation

Tested with **Python 3.11** and **PyTorch 2.11.0 (CUDA 12.8)**.
Run the following command from this repository directory:

```bash
git clone https://github.com/ModelTC/LightVAE.git
cd LightVAE
python -m pip install -r requirements.txt
```

This installs the shared **`lightvae`** package and the reconstruction/download dependencies.
For model-only or ComfyUI usage, use `python -m pip install .`; for local development,
use `python -m pip install -e ".[cli]"`. The CLI scripts and ComfyUI node share the same
model definitions and kernels—there is no separate ComfyUI runtime copy.

```python
from lightvae.wan21_lightvae_decoder import Decoder
from lightvae.h3_lightvae_encoder import Encoder
```

Triton is optional (usually included with NVIDIA Linux PyTorch). For a compatible
Linux CUDA environment, `python -m pip install "triton>=3.4"` enables fused Wan
kernels. H3 uses PyTorch operations. See the [GPU support matrix](comfyui/README.md#gpu-support)
for backend selection, tested hardware and limitations.


## 🤗 Model Weights

### LightVAE

Download all LightVAE checkpoints with one command:

```bash
hf download lightx2v/LightVAE --local-dir weights/LightVAE
```

### Original VAEs

Download the backbone you need for encoding and baseline comparison:

```bash
# Wan2.1: 16-channel VAE
hf download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth --local-dir weights/Wan2.1

# Wan2.2: 48-channel TI2V VAE
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir weights/Wan2.2

# MiniMax-H3: VAE config and all weight shards
hf download MiniMaxAI/MiniMax-H3 --include "vae/*" --local-dir weights/MiniMax-H3
```

Already have the weights? Pass their paths directly. Benchmark weights default
to `weights/LightVAE`; override with `--weights-dir` or `LIGHTVAE_MODEL_DIR`.

## 🎥 Reconstruction

`infer.py` accepts **independent encoder and decoder paths** through `--encoder`
and `--decoder`. `--model` selects the backbone; `--variant` selects the decoder
architecture (`base`, `pro`, or `lite`; H3 supports Base and Lite).

Wan's original VAE checkpoint contains both components: when passed to
`--encoder`, only its encoder is loaded. LightVAE decoder exports do not require
an additional original decoder checkpoint.

### Wan2.1

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model wan21 --variant lite \
  --encoder weights/Wan2.1/Wan2.1_VAE.pth \
  --decoder weights/LightVAE/lightvae-lite-wan21-decoder.safetensors \
  --input demo.mp4 --frames 81 --height 480 --width 832 \
  --output outputs/wan21_lite_recon.mp4
```

For **Pro**, use `--variant pro` and
`--decoder weights/LightVAE/lightvae-pro-wan21-decoder.safetensors`, with a new
output filename. Keep the same encoder.

### Wan2.2

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model wan22 --variant lite \
  --encoder weights/Wan2.2/Wan2.2_VAE.pth \
  --decoder weights/LightVAE/lightvae-lite-wan22-decoder.safetensors \
  --input demo.mp4 --frames 81 --height 704 --width 1280 \
  --output outputs/wan22_lite_recon.mp4
```

For **Pro**, use `--variant pro` and
`--decoder weights/LightVAE/lightvae-pro-wan22-decoder.safetensors`, with a new
output filename. Keep the same encoder.

### MiniMax-H3

**Original encoder → Lite decoder**

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model h3 --variant lite \
  --encoder weights/MiniMax-H3/vae \
  --decoder weights/LightVAE/lightvae-lite-h3-decoder.safetensors \
  --input demo.mp4 --frames 124 --height 768 --width 1344 \
  --output outputs/h3_lite_recon.mp4
```

**Lite encoder → Lite decoder** · No original VAE weights needed.

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model h3 --variant lite --encoder-variant lite \
  --encoder weights/LightVAE/lightvae-lite-h3-encoder.safetensors \
  --decoder weights/LightVAE/lightvae-lite-h3-decoder.safetensors \
  --input demo.mp4 --frames 124 --height 768 --width 1344 \
  --output outputs/h3_lite_encdec_recon.mp4
```

For an **original VAE baseline**, use `--variant base` and pass the original VAE
path to both `--encoder` and `--decoder` (leave `--encoder-variant` at `base`).

**Outputs:** an MP4 and a matching JSON with PSNR, timings, and checkpoint paths.
Existing files are never overwritten.

Inference uses BF16; PSNR is measured before MP4 compression. `--frames` limits
the input length. Omit both `--height` and `--width` to retain the input
resolution, or set both for resize-to-cover followed by a center crop.

## ⏱️ Speed Benchmark

Compare Base / Pro / Lite decoder speed for Wan, or Base / Lite for H3.
Each decoder runs in a separate process with the same seeded latent input.
Use a new output directory for each run. Here, `--base-weights` specifies the
original VAE **comparison baseline**; it is not an `infer.py` argument.

### Wan2.1

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark.py \
  --weights-dir weights/LightVAE \
  --model wan21 --base-weights weights/Wan2.1/Wan2.1_VAE.pth \
  --frames 81 --height 480 --width 832 \
  --warmup 5 --repeats 30 --output outputs/wan21_speed
```

### Wan2.2

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark.py \
  --weights-dir weights/LightVAE \
  --model wan22 --base-weights weights/Wan2.2/Wan2.2_VAE.pth \
  --frames 81 --height 704 --width 1280 \
  --warmup 5 --repeats 30 --output outputs/wan22_speed
```

### MiniMax-H3

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark.py \
  --weights-dir weights/LightVAE \
  --model h3 --base-weights weights/MiniMax-H3/vae \
  --frames 124 --height 768 --width 1344 \
  --warmup 5 --repeats 30 --output outputs/h3_speed
```

**Outputs:** `summary.csv`, `summary.json`, and per-decoder logs.
