# Search, merge, and recover

This directory contains the training implementation of LightVAE. It is independent
of the inference kernels and does not require another LightX2V checkout.

- **Wan2.1 / Wan2.2:** residual-block search → LoRA merge and physical pruning →
  temporal low-rank initialization → decoder recovery.
- **MiniMax-H3:** encoder, decoder, or joint recovery from fixed pruned weights.
  The source implementation has no H3 search/export entry point; this migration
  does not add a new H3 search recipe.

No model weights, datasets, old experiment outputs, or Flash/Turbo evaluation
baselines are included. All commands below run from the repository root.

## Install

Use a CUDA-compatible PyTorch/torchvision environment, then install:

```bash
python -m pip install -e . -e './train'
# H3 recovery additionally needs Diffusers with MiniMax-H3 support:
python -m pip install -e './train[h3]'
python -c 'from diffusers import AutoencoderKLMiniMaxH3'
```

LPIPS uses pretrained VGG for the training loss and AlexNet for evaluation. Cache
those weights before launching on machines without network access. This package
uses native differentiable model operators; inference-only folded kernels do not
replace the training graph.

## Data

Provide separate training and validation JSONL manifests. Each record has a
media path, dataset label, and stable source-video group:

```json
{"path": "/path/to/video.mp4", "dataset": "videos", "group": "video-00001"}
```

`path` can also be a directory of JPG/PNG frames. ZIP-backed frame sequences use
`path` for the archive and an ordered `zip_members` list for that sequence.
Different sequences can share an archive. Train/validation groups and media must
not overlap. The reader resizes to cover the requested canvas and center-crops;
training samples a temporal start, while validation starts at frame zero.
Use clips long enough for the configured frame count. Paths are resolved against
the current working directory; absolute media paths are convenient for clusters.

Wan2.1 uses normalized 16-channel latents and 8× spatial compression. Wan2.2 uses
raw 48-channel posterior means and 16× compression. Wan inputs require `4n+1`
frames. All recovery examples use posterior means.

## Wan Lite: search → merge → recover

Set the family, original VAE and manifests. A family must match all three stages.
For Wan2.2 TI2V-5B, replace `wan21` with `wan22` and provide `Wan2.2_VAE.pth`.

```bash
export FAMILY=wan21
export TEACHER=/path/to/Wan2.1_VAE.pth
export TRAIN_MANIFEST=/path/to/train.jsonl
export VAL_MANIFEST=/path/to/val.jsonl
export RUN_DIR=outputs/wan21_lite_run
mkdir -p "$RUN_DIR"
```

Search runs on **one GPU**. It optimizes LoRA and stage-constrained Gumbel gates;
base weights are frozen. Each of the five decoder stages retains one residual
block by default. Checkpoints contain live search weights and separate gate EMA.

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m vae_prune.search \
  --config "train/configs/search/$FAMILY.json" \
  --teacher "$TEACHER" \
  --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --output "$RUN_DIR/search.pt" \
  > "$RUN_DIR/search.log" 2>&1
```

Export selects blocks using gate EMA, **merges retained LoRA updates before SVD**,
physically removes residual branches, and applies TLR. Shape-changing shortcuts
remain. The default retention threshold is η=0.6, using the sum of singular values
(not squared energy); the last two high-resolution residual stages use rank zero.
`tail_2d: false` retains their temporal branches.

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m vae_prune.export \
  --config "train/configs/export/${FAMILY}_lite.json" \
  --teacher "$TEACHER" --search-checkpoint "$RUN_DIR/search.pt" \
  --output "$RUN_DIR/merged.pt" --device cuda:0
```

Recover on eight GPUs with DDP. Change `--nproc_per_node` and
`CUDA_VISIBLE_DEVICES` together for a different GPU count.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m vae_prune.recover \
  --config "train/configs/recover/${FAMILY}_lite.json" \
  --teacher "$TEACHER" --student "$RUN_DIR/merged.pt" \
  --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --output "$RUN_DIR/recover" \
  > "$RUN_DIR/recover.log" 2>&1
```

For a single GPU, use `python -m vae_prune.recover` with the same arguments.
Add `--validate-only` for a baseline evaluation. `--steps 1` performs a short
functional check; use a separate output and reduce geometry in a copy of the
config when checking memory fit.

## Wan Pro: skip depth search

Pro keeps all 14 residual blocks. Start directly from the original teacher and
perform TLR initialization, then run recovery using the `*_pro.json` config:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vae_prune.export \
  --config "train/configs/export/${FAMILY}_pro.json" \
  --teacher "$TEACHER" --output "$RUN_DIR/pro_init.pt" --device cuda:0
```

Use `pro_init.pt` as `--student`; do not pass a search checkpoint to Pro export.

## Configuration defaults

| Setting | Wan search | Wan recovery | H3 recovery |
|---|---|---|---|
| Steps | 1,000 | 5,000 | 5,000 |
| Train geometry, T×H×W | 17×256×256 | 81×480×832 | 17×256×256 |
| Validation geometry | Same as search | 17×480×832 | 17×256×256 |
| Precision | BF16 autocast | FP32 | FP32 |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | LoRA 1e-4; gates 1e-3 | 1e-5 → 1e-6, cosine | 1e-5 → 1e-6, cosine |
| LR warmup | None | 50 updates | 50 updates |
| LoRA rank / alpha | 16 / 32 | Merged; all decoder weights train | All selected component weights train |
| Gate temperature | 4.0 → 0.4 | — | — |
| EMA | Gate EMA 0.99 | Model EMA 0.999 | Model EMA 0.999 |
| Accumulation | 1 | 1 | 4 |
| Validation / save | Every 100 updates | Every 100 updates | Every 100 updates |

Configs live in `configs/search/`, `configs/export/`, and `configs/recover/`.
They contain hyperparameters only; model/data/output paths are explicit CLI
arguments. Batch size is one clip per rank; global batch equals GPU count ×
accumulation. Validation uses teacher/student/EMA with equal-video PSNR, SSIM,
and all-frame Alex LPIPS. Wan selects up to 16 fixed validation videos; H3 uses
the supplied validation manifest.

Recovery loss defaults are **L1=1, VGG LPIPS=0.8, probe-guided feature=0.05,
GAN=0.05**. Probe gradients form detached feature-matching weights; the probe
objective is not itself added to the student's loss. GAN starts at optimizer
update 500, with 100 discriminator-only warmup updates and a 300-update generator
ramp. Set `gan.enabled` to `false` to disable it. These are runnable starting recipes,
not a promise to reproduce a particular published checkpoint without matching
its data, training duration and hyperparameters.

## H3 recovery

Supply fixed pruned encoder and decoder safetensors with their embedded
architecture metadata, plus the original teacher's `vae/` directory. Select
`h3_encoder.json`, `h3_decoder.json`, or `h3_joint.json` to control which component
is trainable. Both pruned components are supplied; the untrained component remains
frozen.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m vae_prune.recover_h3 \
  --config train/configs/recover/h3_joint.json \
  --encoder /path/to/lightvae-lite-h3-encoder.safetensors \
  --decoder /path/to/lightvae-lite-h3-decoder.safetensors \
  --teacher /path/to/MiniMax-H3/vae \
  --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --output outputs/h3_joint_recover
```

The original H3 normalization, 17-frame encoder chunks and temporal stitching
remain intact. Short clips are padded internally to the released protocol's
minimum length; padding is excluded from RGB loss and metrics. Encoder KL and
feature losses are independently configurable and disabled by default.

## Checkpoints and inference export

Wan recovery atomically replaces `student.pt`, `student_ema.pt`, and
`training_state.pt`. H3 saves `checkpoint-XXXXXX/{student,ema}/{encoder,decoder}.safetensors`
and retains two checkpoints by default. Each run writes `config.json`,
`train.jsonl`, the validation selection and per-video validation reports.
Existing output directories are rejected. These entry points support
**weights-only warm starts**, not complete optimizer/RNG/data-position resume.

Convert a Wan student/EMA to the inference package's safetensors format:

```bash
python -m vae_prune.publish \
  --checkpoint "$RUN_DIR/recover/student_ema.pt" \
  --output "$RUN_DIR/lightvae-lite-${FAMILY}-decoder.safetensors"
```

This also writes an architecture JSON with the same stem. New search selections
and ranks can differ from the repository's released model configs; pass that JSON
when reconstructing with the new model:

```bash
python infer.py --model "$FAMILY" --variant lite \
  --encoder "$TEACHER" \
  --decoder "$RUN_DIR/lightvae-lite-${FAMILY}-decoder.safetensors" \
  --decoder-config "$RUN_DIR/lightvae-lite-${FAMILY}-decoder.json" \
  --input /path/to/test.mp4 --frames 81 --height 480 --width 832 \
  --output "$RUN_DIR/reconstruction.mp4"
```

H3 recovery already writes inference-compatible safetensors; no Wan conversion is
needed for them. Use the encoder and decoder from the same saved recovery step.

## Tests

```bash
python -m pip install -e './train[test]'
python -m pytest -q train/tests
```

H3 tests require the optional H3 dependencies. Tests cover TLR streaming gradients,
singular-value retention, LoRA merge equivalence, EMA gate selection, probe
weights, GAN scheduling, component freezing and unpadded distributed evaluation.
