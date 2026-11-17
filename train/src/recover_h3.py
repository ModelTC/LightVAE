"""Recover a fixed H3 encoder, decoder, or both from portable pruned weights."""

import argparse
import copy
import json
import math
import os
import random
import shutil
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoints import atomic_save
from .data import assert_disjoint, load_manifest, read_clip
from .distributed_eval import gather_rows, local_indices, summarize
from .evaluation import VideoMetrics
from .losses.h3_joint import H3JointLoss, load_teacher_decoder, load_teacher_encoder
from .models.h3_joint import H3Joint


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["train_component"] not in ("encoder", "decoder", "joint"):
        raise ValueError("train_component must be encoder, decoder, or joint")
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    for key in ("steps", "accumulation_steps", "eval_every", "save_every", "save_total_limit"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= config["ema_decay"] < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if config["train_component"] == "decoder" and (config["loss"]["encoder_kl"] or config["loss"]["encoder_feature"]):
        raise ValueError("Encoder supervision requires a trainable encoder")
    for geometry in (config["train_geometry"], config["val_geometry"]):
        if geometry["frames"] < 1 or (geometry["frames"] - 1) % 4:
            raise ValueError("The shared video reader requires 4n+1 frames")
        if not geometry.get("native_resolution", False) and (geometry["height"] % 16 or geometry["width"] % 16):
            raise ValueError("H3 spatial dimensions must be multiples of 16")
    return config


def learning_rate(config, step):
    warmup = config["warmup_steps"]
    peak, minimum = config["learning_rate"], config["min_learning_rate"]
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, config["steps"] - warmup)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


def save_components(model, directory):
    directory.mkdir(parents=True, exist_ok=False)
    for component, architecture, kind, name in (
        (model.encoder, model.encoder_arch, "minimax_h3_pruned_encoder", "encoder.safetensors"),
        (model.decoder, model.decoder_arch, "minimax_h3_pruned_vae", "decoder.safetensors"),
    ):
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in component.state_dict().items()},
            str(directory / name),
            metadata={"model_type": kind, "architecture": json.dumps(architecture)},
        )


def run(args, config, device, rank, world):
    train = [row for path in args.train_manifest for row in load_manifest(path)]
    val = [row for path in args.val_manifest for row in load_manifest(path)]
    assert_disjoint(train, val)
    exists = [args.output.exists() if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(exists, src=0)
    if exists[0]:
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    if rank == 0:
        args.output.mkdir(parents=True)
    if world > 1:
        dist.barrier()
    torch.manual_seed(config["seed"])
    model = H3Joint(args.encoder, args.decoder).to(device).float()
    model.set_train_component(config["train_component"])
    model.train()
    teacher = load_teacher_decoder(args.teacher, model.decoder_arch["teacher_config"]).to(device)
    encoder_teacher = load_teacher_encoder(args.teacher, model.encoder_arch["teacher_config"]).to(device)
    teacher_model = H3Joint.from_components(encoder_teacher, teacher)
    objective = H3JointLoss(
        teacher, config["loss"], config["gan"], device, config["accumulation_steps"], encoder_teacher
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], **config["optimizer"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    metrics = VideoMetrics(net=config["eval_lpips_net"]).to(device)
    encoder_supervision = bool(config["loss"]["encoder_kl"] or config["loss"]["encoder_feature"])
    feature_supervision = bool(config["loss"]["guided_feature"])
    if rank == 0:
        resolved = dict(
            config,
            encoder=str(args.encoder.resolve()),
            decoder=str(args.decoder.resolve()),
            teacher=str(args.teacher.resolve()),
            train_manifest=[str(p.resolve()) for p in args.train_manifest],
            val_manifest=[str(p.resolve()) for p in args.val_manifest],
            world_size=world,
            global_batch=world * config["accumulation_steps"],
            initialization="weights-only warm start",
            encoder_arch=model.encoder_arch,
            decoder_arch=model.decoder_arch,
        )
        (args.output / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        (args.output / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in val))

    def log(event, **values):
        if rank == 0:
            line = json.dumps(dict(event=event, **values))
            print(line, flush=True)
            with (args.output / "train.jsonl").open("a") as stream:
                stream.write(line + "\n")

    def read(row, geometry, seed=0, validation=False):
        clip = read_clip(row, **geometry, seed=seed, validation=validation).to(device)
        if clip.shape[2] != geometry["frames"]:
            raise ValueError(f"Clip has fewer than {geometry['frames']} frames: {row['path']}")
        return clip

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=config["precision"] == "bf16")

    teacher_scores = {}

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        rows = []
        for index in local_indices(len(val)):
            row = val[index]
            pixels = read(row, config["val_geometry"], validation=True)
            if index not in teacher_scores:
                with autocast():
                    prediction = teacher_model(pixels)
                teacher_scores[index] = metrics(prediction, pixels)
            scores = {"teacher": teacher_scores[index]}
            for name, network in (("student", model), ("ema", ema)):
                with autocast():
                    prediction = network(pixels)
                scores[name] = metrics(prediction, pixels)
            rows.append(dict(index=index, dataset=row["dataset"], group=row["group"], metrics=scores))
        rows = gather_rows(rows)
        summary = summarize(rows)
        log("validation", step=step, summary=summary)
        if rank == 0:
            (args.output / f"validation-{step:06d}.json").write_text(
                json.dumps(dict(step=step, summary=summary, rows=rows), indent=2) + "\n"
            )
        model.train()

    def save(step):
        if rank == 0:
            directory = args.output / f"checkpoint-{step:06d}"
            temporary = args.output / f".checkpoint-{step:06d}.tmp"
            save_components(model, temporary / "student")
            save_components(ema, temporary / "ema")
            atomic_save(
                dict(
                    step=step,
                    optimizer=optimizer.state_dict(),
                    gan=None if objective.gan is None else objective.gan.state_dict(),
                ),
                temporary / "training_state.pt",
            )
            temporary.rename(directory)
            for old in sorted(args.output.glob("checkpoint-*"))[: -config["save_total_limit"]]:
                shutil.rmtree(old)
            log("checkpoint", step=step, path=str(directory))
        if world > 1:
            dist.barrier()

    evaluate(0)
    if args.validate_only:
        return
    if not feature_supervision:
        teacher.cpu()
    if not encoder_supervision:
        encoder_teacher.cpu()
    torch.cuda.empty_cache()
    training_model = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    torch.manual_seed(config["seed"] + rank)
    accumulation = config["accumulation_steps"]
    for step in range(1, config["steps"] + 1):
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        anchor = (step - 1) % len(model.decoder.kept_layers)
        stats = {}
        for microstep in range(accumulation):
            seed = config["seed"] + ((step - 1) * accumulation + microstep) * world + rank
            row = random.Random(seed).choice(train)
            pixels = read(row, config["train_geometry"], seed=seed)
            sync = training_model.no_sync() if world > 1 and microstep + 1 < accumulation else nullcontext()
            with sync:
                with autocast():
                    outputs = training_model(
                        pixels,
                        anchor if feature_supervision else None,
                        return_training=True,
                        encoder_supervision=encoder_supervision,
                    )
                    loss, terms = objective(outputs, pixels, model.decoder.kept_layers[anchor], step)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                (loss / accumulation).backward()
            for key, value in dict(loss=loss.detach(), **terms).items():
                stats[key] = stats.get(key, 0.0) + float(value) / accumulation
            del outputs, loss, terms, pixels
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        objective.step()
        with torch.no_grad():
            for averaged, parameter in zip(ema.parameters(), model.parameters()):
                averaged.lerp_(parameter, 1 - config["ema_decay"])
        if world > 1:
            keys = sorted(stats)
            values = torch.tensor([stats[key] for key in keys], device=device)
            dist.all_reduce(values)
            stats = dict(zip(keys, (values / world).tolist()))
        log("train", step=step, lr=lr, gradient_norm=float(norm), **stats)
        if step % config["eval_every"] == 0 or step == config["steps"]:
            evaluate(step)
        if step % config["save_every"] == 0 or step == config["steps"]:
            save(step)
    log("complete", step=config["steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--val-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if world > 1 else args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    run(args, config, device, rank, world)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
