# Cross-architecture video VAE inference

`cross_arch_infer.py` reconstructs a video with the original encoder and either
the original decoder or a recovered TLR decoder. `cross_arch_benchmark.py`
measures decoder throughput on one GPU. Both support HunyuanVideo-1.5,
LTX-2.5 Conv VAE, and CogVideoX1.5-5B; the model names are
`hunyuanvideo15`, `ltx25`, and `cogvideox15`.

## Weights and source dependencies

The `--model-root` directory has one subdirectory per model. These commands
download only the VAE weights:

```bash
MODEL_ROOT=weights/cross_arch

hf download tencent/HunyuanVideo-1.5 \
  vae/config.json vae/diffusion_pytorch_model.safetensors \
  --local-dir "$MODEL_ROOT/hunyuanvideo15"

hf download zai-org/CogVideoX1.5-5B \
  vae/config.json vae/diffusion_pytorch_model.safetensors \
  --local-dir "$MODEL_ROOT/cogvideox15"

hf download Lightricks/LTX-2.5 \
  vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --local-dir "$MODEL_ROOT/ltx25"
```

The LTX-2.5 repository may require accepting its Hugging Face access terms.
LTX's Conv VAE uses the [official LTX-2 source](https://github.com/Lightricks/LTX-2).
The tested source revision is `4f521619ae0029e8fbb64d555030c2b121a33b4f`.
Clone it under `$MODEL_ROOT/src/ltx2`, or set `LIGHTVAE_LTX_CORE_SRC` to its
`packages/ltx-core/src` directory. Its dependencies include `colour-science`;
the loader also reads `LIGHTVAE_LTX_DEPS` if those Python packages are installed
in a separate directory. The Hunyuan adapter uses the `AutoencoderKLConv3D`
implementation in [LightX2V](https://github.com/ModelTC/LightX2V); install
LightX2V in the Python environment or set `LIGHTVAE_LIGHTX2V_ROOT` to its
checkout. CogVideoX1.5 uses `diffusers.AutoencoderKLCogVideoX`.

```bash
mkdir -p "$MODEL_ROOT/src"
git clone https://github.com/Lightricks/LTX-2.git "$MODEL_ROOT/src/ltx2"
git -C "$MODEL_ROOT/src/ltx2" checkout 4f521619ae0029e8fbb64d555030c2b121a33b4f
python -m pip install diffusers av numpy colour-science
```

In addition to LightVAE's normal requirements, the tested environment has
`diffusers`, `av`, `numpy`, and the official LTX package dependencies. Model
weights are loaded with strict state-dict checks. LTX loads the standalone
**Conv** VAE checkpoint, not the spatial or framewise VAE variant.

## Reconstruct a video

```bash
cd LightVAE
MODEL_ROOT=weights/cross_arch

CUDA_VISIBLE_DEVICES=0 python cross_arch_infer.py \
  --model ltx25 --model-root "$MODEL_ROOT" \
  --input input.mp4 --frames 81 --height 480 --width 832 \
  --dtype fp32 --output outputs/ltx25_original.mp4

CUDA_VISIBLE_DEVICES=0 python cross_arch_infer.py \
  --model ltx25 --model-root "$MODEL_ROOT" \
  --tlr-checkpoint /path/to/ltx25_recovered.pt \
  --input input.mp4 --frames 81 --height 480 --width 832 \
  --dtype fp32 --output outputs/ltx25_tlr.mp4
```

Change `--model` and checkpoint for the other families. The input is resized
to cover the requested dimensions, center-cropped, and padded by repeating its
last row/column/frame when needed for the VAE stride. The reported PSNR uses
the unpadded RGB reconstruction before MP4 encoding. Each run writes an MP4 and
a JSON report; output files must be new paths.

## Unified eager BF16 decoder throughput

```bash
CUDA_VISIBLE_DEVICES=0 python cross_arch_benchmark.py \
  --model ltx25 --model-root "$MODEL_ROOT" \
  --tlr-checkpoint /path/to/ltx25_recovered.pt \
  --frames 33 --height 480 --width 832 \
  --dtype bf16 --warmup 1 --repeats 3 \
  --output outputs/ltx25_speed
```

Use the same `cross_arch_benchmark.py` command for `hunyuanvideo15`, `ltx25`,
and `cogvideox15`, with a new output directory for each family. Without
`--tlr-checkpoint`, the command measures only the original decoder. Original
and TLR variants run in separate processes with the same seeded latent tensor.
The 33-frame setting matches the Figure 7 validation run.
The benchmark defaults to eager BF16; it does not invoke `torch.compile`. The
median timing synchronizes CUDA before and after each
decode, excludes loading, warmup, encoding, transfers, and file I/O, and reports
`frames / median seconds` as FPS. The output directory contains `summary.csv`,
`summary.json`, and per-variant JSON/log files.

The TLR inference path keeps decoder activations and weights in BF16, packs
spatial/history projections into one `Conv2d`, and uses a fused CUDA kernel for
history addition, the low-rank pointwise projection, and the spatial residual
add. The fused kernel is used for folded BF16 TLR checkpoints on NVIDIA SM8+;
FP32 and training continue through the differentiable PyTorch path.

Only the TLR variant uses the inference adaptations below. The base variant
retains the original source decoder, including its normalization, padding,
attention, and Conv3d operators. Both use the same BF16 precision for speed,
as in the Wan benchmark; FP32 reconstruction quality is measured separately.

TLR residual convolutions fuse replicate padding and frame-major layout packing
in one pass. Spatial-only layers convolve just the current frames instead of
also computing and discarding temporal halo frames. Hunyuan RMS normalization
and explicit swish, and LTX PixelNorm and SiLU, have separate fused kernels that
respect their different operation order and BF16 rounding. FP32 uses native
PyTorch operations. These are inference adaptations, not changes to recovery
training.

The TLR Hunyuan decoder attention uses frame-prefix SDPA calls instead of building a
dense token-by-token causal mask. Each frame attends to all earlier frames and
all spatial tokens in its own frame, matching the original mask without its
per-token Python/GPU writes.

The CogVideoX TLR decoder fuses its two-frame causal-cache prepend with
frame-major packing. The original per-layer cache contract and per-video cache
reset remain intact. Its conditioned normalization retains native GroupNorm
and latent interpolation, while fusing the following BF16 affine operations
and SiLU, including the source's intermediate BF16 rounding. The original
decoder uses none of these adaptations.

## Random-weight architecture throughput

For HunyuanVideo-1.5 and LTX-2.5, `--random-weights` compares the native base
decoder with the exact eta 0.4 and eta 0.6 recovered TLR architectures. It needs
no original VAE weights or recovered checkpoint. It still needs the family
source implementation and Python dependencies described above; LTX source can
be selected with `LIGHTVAE_LTX_CORE_SRC` without populating `--model-root`.

```bash
CUDA_VISIBLE_DEVICES=0 python cross_arch_benchmark.py \
  --model ltx25 --random-weights \
  --frames 33 --height 480 --width 832 \
  --dtype bf16 --warmup 5 --repeats 30 \
  --output outputs/ltx25_random_speed
```

Change `--model` to `hunyuanvideo15` and choose a new output directory for HY.
The default structure files are packaged under
`lightvae/config/cross_arch_benchmark/`; `--structure-dir` selects another
directory containing `<family>.json`. These small files preserve the native
model config and the full `tlr_records` from each recovered checkpoint. TLR
ranks come directly from those records; eta is not interpreted as a channel
fraction. The loader constructs the final folded inference modules on the
meta device, then materializes only the decoder with seeded random weights.
It does not load weights, run SVD, or materialize an encoder. Convolution
weights use a normal distribution with standard deviation 0.01, normalization
scales and latent standard-deviation buffers are one, and biases and means are
zero.

The `base`, `eta0.4`, and `eta0.6` variants run in separate processes with the
same seeded random latent tensor. Reports include the structure-file hash,
source checkpoint step, and exact TLR ranks; `summary.json` also contains both
speedups. Base uses native operators, and the TLR variants use the same
inference kernels as recovered checkpoints. Timing remains BF16 eager,
decoder-only, with warmup excluded. These results measure architecture
throughput with synthetic weights and inputs; they do not measure
reconstruction quality or replace measurements with trained weights.

## DAVIS retention-threshold calibration

`cross_arch_eta.py` reuses the verified original-encoder FP32 latents from the
30-video DAVIS calibration set. It compares the original decoder with direct
SVD TLR initialization at eta 0.2, 0.4, 0.6 and 0.8, without recovery or depth
pruning. By default the last two residual stages are spatial-only, matching the
recovery architecture. `--tail temporal` keeps temporal branches in every
residual convolution for a separate ablation.

```bash
PYTHONPATH="$PWD:/path/to/LightVAE_private_ready" python cross_arch_eta.py \
  --model ltx25 --model-root "$MODEL_ROOT" \
  --cache-dir /path/to/eta/ltx25 \
  --manifest /path/to/eta_calibration.jsonl \
  --output outputs/ltx25_eta --tail spatial
```

Each clip uses its first 33 frames at 480x832. PSNR is FP32 eager, computed from
the whole RGB video's MSE and then averaged equally across videos. Timing is
BF16 eager decoder-only, with five global warmups, one warmup per clip, and
five synchronized repeats per clip; FPS is 33 divided by the average of the
per-clip median times. Loading, encoding and I/O are excluded. `--variant 0.4`
allows assigning variants to separate GPUs; `--summarize` combines completed
results. DAVIS is the calibration set here, so these are not independent
held-out quality results.

The original Hunyuan decoder has 17 residual blocks (34 converted Conv3ds),
with the final six blocks spatial-only. LTX-2.5 Conv has 18 blocks (36
convolutions), with the final ten blocks spatial-only; its decoder is
**noncausal**, so TLR uses the center temporal slice. Input/output and
upsampling convolutions remain original. These full-depth variants are not
equivalent to the five-residual-block, depth-pruned Wan Lite architecture.
CogVideoX1.5 also has 18 residual blocks (36 converted convolutions); the final
two upsampling stages contain eight spatial-only residual blocks (16
convolutions). Its temporal branches retain the final, causal kernel slice.

## Recovered checkpoint contract

The loader accepts a `.pt` dictionary with
`schema_version: 1`, `family`, `step`, `eta`, `geometry`, `tlr_records`, and
`decoder_state`. It also accepts a self-contained `.safetensors` export through
the same `--tlr-checkpoint` option. Its tensor keys are the unchanged
`decoder_state` keys; the header contains `model_type: lightvae_decoder` and
`checkpoint_metadata`, a JSON string preserving all other checkpoint fields.
An optional `.metadata.json` sidecar is for inspection; loading requires only
the `.safetensors` file and the original VAE under `--model-root`.

The exported full-depth decoder names are
`lightvae-pro-hunyuanvideo15-decoder.safetensors` and
`lightvae-pro-ltx25-decoder.safetensors`.

Record paths are relative to the original decoder. The loader
builds the exact TLR modules from `tlr_records`, strictly loads the recovered
decoder state, and folds an unfused checkpoint for inference. The original
encoder comes from the official VAE and is not duplicated in the recovery
checkpoint.

The last two decoder residual stages use spatial-only Conv2d when marked in the
records; earlier residual Conv3d layers use the TLR temporal correction. A
checkpoint from one family cannot be loaded into another. These scripts do not
convert or recover a VAE; they consume checkpoints produced by the recovery
run.
