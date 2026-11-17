"""Recover a merged Wan decoder with reconstruction and probe-guided losses."""

import argparse
import copy
import json
import math
import os
import random
import sys
import traceback
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoints import atomic_save, load_student, sha256
from .data import assert_disjoint, load_manifest, read_clip
from .evaluation import VideoMetrics, evaluate_wan
from .losses.wan_recovery import WanRecovery
from .models.api import Decoder, Teacher


def validate_config(config):
    if config["family"] not in ("wan21", "wan22"):
        raise ValueError("family must be wan21 or wan22")
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    for key in ("steps", "accumulation_steps", "eval_every", "save_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= config["ema_decay"] < 1 or not 0 <= config["warmup"] <= config["steps"]:
        raise ValueError("Invalid EMA decay or warmup")
    if not 0 < config["min_lr"] <= config["lr"] or config["gradient_clip"] <= 0:
        raise ValueError("Require 0 < min_lr <= lr and positive gradient_clip")
    ratio = 8 if config["family"] == "wan21" else 16
    for name in ("train_geometry", "val_geometry"):
        geometry = config[name]
        if geometry["frames"] < 1 or (geometry["frames"] - 1) % 4:
            raise ValueError("Wan clip length must be 4n+1")
        if geometry["height"] % ratio or geometry["width"] % ratio:
            raise ValueError(f"{name} dimensions must be multiples of {ratio}")
    weights, gan = config["loss"], config["gan"]
    if weights["l1"] != 1 or any(weights[key] < 0 for key in ("perceptual", "feature", "gan")):
        raise ValueError("Use L1=1 and nonnegative loss weights")
    frames = weights["lpips_frames"]
    if frames != "all" and (type(frames) is not int or frames < 1):
        raise ValueError("lpips_frames must be 'all' or a positive integer")
    if weights["guidance"]["epsilon"] <= 0 or weights["guidance"]["max_weight"] <= 0:
        raise ValueError("Probe guidance epsilon and max_weight must be positive")
    if gan["frames"] < 1 or (gan["frames"] - 1) % 4 or gan["crop_size"] % ratio:
        raise ValueError("GAN frames/crop must align to the latent geometry")
    if gan["start_step"] < 1 or config["validation_videos"] < 1:
        raise ValueError("GAN start_step and validation_videos must be positive")


def learning_rate(config, step):
    warmup = config["warmup"]
    if step <= warmup:
        return config["lr"] * step / warmup
    progress = (step - warmup) / max(1, config["steps"] - warmup)
    return config["min_lr"] + (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, value in zip(ema.parameters(), model.parameters()):
        target.lerp_(value, 1 - decay)
    for target, value in zip(ema.buffers(), model.buffers()):
        target.copy_(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True, help="Merged portable student from vae_prune.export")
    parser.add_argument("--train-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--val-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.steps is not None:
        config["steps"] = args.steps
        config["warmup"] = min(config["warmup"], args.steps)
    validate_config(config)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if world > 1 else args.device)
    if device.type != "cuda":
        raise ValueError("Recovery requires CUDA; CPU unit tests exercise the individual components")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    train = [row for path in args.train_manifest for row in load_manifest(path)]
    validation = [row for path in args.val_manifest for row in load_manifest(path)]
    assert_disjoint(train, validation)
    order = list(range(len(validation)))
    random.Random(config["seed"]).shuffle(order)
    validation = [validation[index] for index in sorted(order[: config["validation_videos"]])]
    exists = [args.output.exists() if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(exists, src=0)
    if exists[0]:
        raise FileExistsError(f"Use a new output directory for warm-start recovery: {args.output}")
    if rank == 0:
        args.output.mkdir(parents=True)
    if world > 1:
        dist.barrier()

    component, source = load_student(args.student, device)
    if source["family"] != config["family"]:
        raise ValueError("Student and teacher families differ")
    model = Decoder(component, config["family"]).to(device).float().train()
    teacher = Teacher(config["family"], args.teacher).to(device).float()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    objective = WanRecovery(config["loss"], config["gan"], device, config["family"], config["accumulation_steps"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], **config["optimizer"])
    train_model = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    metric = VideoMetrics().to(device)
    teacher_scores = {}
    torch.manual_seed(config["seed"] + rank)

    def log(event, **values):
        if rank == 0:
            row = json.dumps({"event": event, **values})
            print(row, flush=True)
            with (args.output / "train.jsonl").open("a") as stream:
                stream.write(row + "\n")

    if rank == 0:
        protocol = {
            **config,
            "world_size": world,
            "global_batch": world * config["accumulation_steps"],
            "student": str(args.student.resolve()),
            "teacher": str(args.teacher.resolve()),
            "student_sha256": sha256(args.student),
            "teacher_sha256": sha256(args.teacher),
            "train_manifests": [str(path.resolve()) for path in args.train_manifest],
            "val_manifests": [str(path.resolve()) for path in args.val_manifest],
            "initialization": "student weights only; fresh optimizer, EMA and GAN",
            "evaluation": "disjoint groups; posterior mean; per-video PSNR; all-frame SSIM/Alex LPIPS; equal-video means",
        }
        (args.output / "config.json").write_text(json.dumps(protocol, indent=2) + "\n")
        (args.output / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in validation))

    def evaluate(step):
        log("validation_start", step=step)
        result = evaluate_wan(
            model, ema, teacher, validation, config["val_geometry"], metric, device, config["precision"], teacher_scores
        )
        if rank == 0:
            (args.output / f"validation-{step:06d}.json").write_text(
                json.dumps({"step": step, **result}, indent=2) + "\n"
            )
        log("validation", step=step, summary=result["summary"])

    def save(step):
        if rank != 0:
            return
        payload = {key: value for key, value in source.items() if key != "model"}
        payload.update(step=step, config=config)
        for name, network in (("student", model), ("student_ema", ema)):
            payload["model"] = {key: value.detach().cpu() for key, value in network.component.state_dict().items()}
            atomic_save(payload, args.output / f"{name}.pt")
        atomic_save(
            {"step": step, "optimizer": optimizer.state_dict(), "gan": objective.state_dict()},
            args.output / "training_state.pt",
        )
        log("checkpoint", step=step, student=str(args.output / "student.pt"), ema=str(args.output / "student_ema.pt"))

    evaluate(0)
    if args.validate_only:
        if world > 1:
            dist.destroy_process_group()
        return
    accumulation = config["accumulation_steps"]
    for step in range(1, config["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        stats = {}
        anchor = objective.anchor(step)
        for micro in range(accumulation):
            seed = config["seed"] + ((step - 1) * accumulation + micro) * world + rank
            record = random.Random(seed).choice(train)
            x = read_clip(record, **config["train_geometry"], seed=seed).to(device)
            if x.shape[2] != config["train_geometry"]["frames"]:
                raise ValueError(f"Training video is too short: {record['path']}")
            context = train_model.no_sync() if world > 1 and micro + 1 < accumulation else nullcontext()
            with context:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config["precision"] == "bf16"):
                    z = teacher.encode(x, mode="mean")
                    target, teacher_features = teacher(z, anchor) if anchor is not None else (None, None)
                    prediction, features = train_model(z, anchor)
                    loss, terms = objective(prediction, x, z, step, seed, features, teacher_features, target, teacher)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                (loss / accumulation).backward()
            for key, value in {"loss": loss.detach(), **terms}.items():
                stats[key] = stats.get(key, 0.0) + float(value) / accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        objective.step()
        update_ema(ema, model, config["ema_decay"])
        if world > 1:
            keys = sorted(stats)
            values = torch.tensor([stats[key] for key in keys], device=device)
            dist.all_reduce(values)
            stats = dict(zip(keys, (values / world).tolist()))
        log("train", step=step, lr=lr, gradient_norm=float(gradient_norm), **stats)
        if step % config["eval_every"] == 0 or step == config["steps"]:
            evaluate(step)
        if step % config["save_every"] == 0 or step == config["steps"]:
            save(step)
            if world > 1:
                dist.barrier()
    log("complete", step=config["steps"])
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        if dist.is_initialized():
            os._exit(1)
        raise
