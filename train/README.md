# LightVAE Training

Structure search, LoRA merging, and recovery training for pretrained video VAEs.

| Model | Supported workflow |
| :--- | :--- |
| Wan2.1 / Wan2.2 | Block search → LoRA merge → TLR initialization → decoder recovery |
| MiniMax-H3 | Encoder, decoder, or joint recovery from fixed pruned weights |

Training code lives in [`src/`](src/); recipes live in [`configs/`](configs/).
Run all commands below from the **repository root**.

## Installation

Use Python ≥3.10 with CUDA-compatible PyTorch ≥2.8 and torchvision.

```bash
python -m pip install -e . -e './train'
```

LPIPS downloads pretrained VGG and AlexNet weights on first use. Cache them in
advance when training without network access.

## Dataset

Create separate training and validation JSONL manifests:

```json
{"path": "/path/to/video.mp4", "dataset": "videos", "group": "video-00001"}
```

- `path`: video file or directory of JPG/PNG frames. For ZIP archives, add an ordered `zip_members` list.
- `dataset` and `group`: identify the source video; keep sources separate across training and validation.
- Clips must cover the configured frame count. Frames are resized and center-cropped; training samples a random start, validation starts at frame zero.

Use absolute media paths. Wan clips require `4n+1` frames; the supplied recipes
encode posterior means with the frozen teacher.

## Wan Lite

Choose `wan21` or `wan22`, its original VAE checkpoint, and your manifests:

```bash
export FAMILY=wan21
export TEACHER=/path/to/Wan2.1_VAE.pth
export TRAIN_MANIFEST=/path/to/train.jsonl
export VAL_MANIFEST=/path/to/val.jsonl
export RUN_DIR="outputs/${FAMILY}_lite"
mkdir -p "$RUN_DIR"
```

**1. Search — one GPU.** Optimize LoRA and Gumbel gates with frozen base weights.
The default keeps one residual block per decoder stage.

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.search \
  --config "train/configs/search/${FAMILY}.json" \
  --teacher "$TEACHER" \
  --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --output "$RUN_DIR/search.pt" > "$RUN_DIR/search.log" 2>&1
```

**2. Merge and initialize TLR.** Select blocks using gate EMA, merge retained
LoRA updates, then apply SVD. The default uses η=0.6 and removes temporal branches
from the final two high-resolution residual stages (`tail_2d: true`).

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.export \
  --config "train/configs/export/${FAMILY}_lite.json" \
  --teacher "$TEACHER" --search-checkpoint "$RUN_DIR/search.pt" \
  --output "$RUN_DIR/merged.pt" --device cuda:0
```

**3. Recover — eight GPUs.** Train the compressed decoder with DDP.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m src.recover \
  --config "train/configs/recover/${FAMILY}_lite.json" \
  --teacher "$TEACHER" --student "$RUN_DIR/merged.pt" \
  --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --output "$RUN_DIR/recover" > "$RUN_DIR/recover.log" 2>&1
```

For one GPU, replace `torchrun ... -m src.recover` with `python -m src.recover`
and set `CUDA_VISIBLE_DEVICES=0`. Use `--validate-only` for evaluation or
`--steps 1` for a short training check.

**4. Export for inference.** Publish the EMA checkpoint as safetensors with a
matching architecture JSON.

```bash
python -m src.publish \
  --checkpoint "$RUN_DIR/recover/student_ema.pt" \
  --output "$RUN_DIR/decoder.safetensors"

python infer.py --model "$FAMILY" --variant lite \
  --encoder "$TEACHER" --decoder "$RUN_DIR/decoder.safetensors" \
  --decoder-config "$RUN_DIR/decoder.json" \
  --input /path/to/test.mp4 --frames 81 --height 480 --width 832 \
  --output "$RUN_DIR/reconstruction.mp4"
```

## Wan Pro

Pro retains all residual blocks and skips depth search. Initialize directly from
the teacher, then run recovery with `configs/recover/${FAMILY}_pro.json` and
`pro_init.pt` as the student:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.export \
  --config "train/configs/export/${FAMILY}_pro.json" \
  --teacher "$TEACHER" --output "$RUN_DIR/pro_init.pt" --device cuda:0
```

Publish using the same command as Lite; select `--variant pro` for inference.

## MiniMax-H3

Install Diffusers with `AutoencoderKLMiniMaxH3` support:

```bash
python -m pip install -e './train[h3]'
```

Supply pruned encoder and decoder safetensors with architecture metadata, plus
the original teacher's `vae/` directory. Choose `h3_encoder.json`,
`h3_decoder.json`, or `h3_joint.json` to select the trainable components.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m src.recover_h3 \
  --config train/configs/recover/h3_joint.json \
  --encoder /path/to/lightvae-lite-h3-encoder.safetensors \
  --decoder /path/to/lightvae-lite-h3-decoder.safetensors \
  --teacher /path/to/MiniMax-H3/vae \
  --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --output outputs/h3_joint_recover
```

H3 preserves the original normalization and chunking protocol. Internal temporal
padding is excluded from RGB loss and metrics. Saved safetensors can be loaded
directly for inference.

## Recipes

| Setting | Wan search | Wan recovery | H3 recovery |
| :--- | :--- | :--- | :--- |
| Configs | [Search](configs/search/) | [Recovery](configs/recover/) | [Recovery](configs/recover/) |
| Updates | 1,000 | 5,000 | 5,000 |
| Training clip · T×H×W | 17×256×256 | 81×480×832 | 17×256×256 |
| Validation clip | 17×256×256 | 17×480×832 | 17×256×256 |
| Precision | BF16 autocast | FP32 | FP32 |
| Learning rate | LoRA 1e-4; gates 1e-3 | 1e-5 → 1e-6 | 1e-5 → 1e-6 |
| Gradient accumulation | 1 | 1 | 4 |
| EMA decay | Gates: 0.99 | Model: 0.999 | Model: 0.999 |
| Save / validate | Every 100 updates | Every 100 updates | Every 100 updates |

All recipes use AdamW. Recovery uses 50 warmup updates and cosine LR decay.
Batch size is one clip per GPU; **global batch = GPUs × gradient accumulation**.
Search LoRA uses rank 16 / alpha 32. TLR settings are in [export configs](configs/export/);
η measures retained singular-value mass.

**Recovery loss weights:** L1 **1.0**, VGG LPIPS **0.8**, probe-guided feature
matching **0.05**, GAN **0.05**. Probe gradients provide detached matching weights.
GAN starts at update 500, followed by 100 discriminator-only warmup updates and
a 300-update generator ramp; disable it with `gan.enabled: false`.

Validation reports per-video PSNR, SSIM, and Alex LPIPS for teacher, student, and
EMA. Wan uses up to 16 fixed validation videos; H3 uses the supplied manifest.

## Outputs

- **Wan:** `recover/student.pt`, `student_ema.pt`, and `training_state.pt`, updated at each save.
- **H3:** `checkpoint-XXXXXX/{student,ema}/{encoder,decoder}.safetensors`; the latest two checkpoints are retained.
- **Logs:** each run records its config, training metrics, and validation results.

Use a fresh output directory for each recovery run. Current entry points support
**weights-only warm starts**; optimizer, RNG, and data-position resume is not implemented.
