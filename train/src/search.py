"""Single-GPU residual-block search with LoRA and straight-through Gumbel gates."""

import argparse
import json
import os
import random
from pathlib import Path

import torch

from .checkpoints import atomic_save
from .data import assert_disjoint, load_manifest, read_clip
from .losses.reconstruction import ReconstructionLoss
from .models import wan22
from .models.api import Decoder, Teacher
from .models.wan21 import WanPrunedDecoder


def build_search_component(teacher, family, search):
    if family == "wan21":
        component = WanPrunedDecoder(search=search)
        component.initialize_from_teacher(teacher.component)
        return component
    return wan22.Component(teacher.component, search=search)


def temperature_at(step, steps, start, end):
    return start * (end / start) ** (step / steps)


@torch.no_grad()
def evaluate(model, teacher, records, geometry, device, precision):
    was_training = model.training
    model.eval()
    scores = {}
    for row in records:
        pixels = read_clip(row, **geometry, validation=True).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            reconstruction, _ = model(teacher.encode(pixels))
        mse = ((reconstruction.float().clamp(-1, 1) - pixels) / 2).square().mean()
        scores.setdefault(row["dataset"], []).append(float(-10 * mse.clamp_min(1e-12).log10()))
    model.train(was_training)
    return {name: {"psnr": sum(values) / len(values), "count": len(values)} for name, values in scores.items()}


def save_search(component, config, step, gate_ema, validation, output):
    atomic_save(
        {
            "format_version": 1,
            "family": config["family"],
            "step": step,
            "config": config,
            "model": {name: value.detach().cpu() for name, value in component.state_dict().items()},
            "gate_ema": gate_ema.detach().cpu(),
            "kept": component.selected_layers(gate_ema),
            "validation": validation,
        },
        output,
    )


def run(config, output, device):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Search is single-GPU; use python, not multi-process torchrun.")
    if output.exists():
        raise FileExistsError(output)
    family, steps = config["family"], config["steps"]
    geometry = config["geometry"]
    alignment = 8 if family == "wan21" else 16
    if family not in ("wan21", "wan22") or steps < 1:
        raise ValueError("Search requires wan21/wan22 and a positive step count.")
    if (
        geometry["frames"] < 1
        or (geometry["frames"] - 1) % 4
        or any(geometry[key] <= 0 or geometry[key] % alignment for key in ("height", "width"))
    ):
        raise ValueError(f"Use 4n+1 frames and spatial dimensions divisible by {alignment}.")
    precision = config["precision"]
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    start, end = config["temperature"]["start"], config["temperature"]["end"]
    decay = config["gate_ema_decay"]
    if min(start, end) <= 0 or not 0 <= decay < 1:
        raise ValueError("Temperatures must be positive; gate_ema_decay must lie in [0,1).")
    train = [row for path in config["train_manifest"] for row in load_manifest(path)]
    val = [row for path in config["val_manifest"] for row in load_manifest(path)]
    assert_disjoint(train, val)
    seed = config["seed"]
    torch.manual_seed(seed)
    teacher = Teacher(family, config["teacher"])
    component = build_search_component(teacher, family, config["search"])
    teacher.to(device)
    model = Decoder(component, family).to(device).train()
    loss_fn = ReconstructionLoss(**config["loss"]).to(device)
    if loss_fn.weights["feature"]:
        raise ValueError("Search uses RGB losses; feature matching belongs to recovery.")
    options = config["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in component.named_parameters() if p.requires_grad and n != "gate_logits"],
                "lr": options["lora_lr"],
            },
            {"params": [component.gate_logits], "lr": options["gate_lr"]},
        ],
        betas=tuple(options["betas"]),
        weight_decay=options["weight_decay"],
    )
    gate_ema = component.gate_logits.detach().clone()
    validation = None
    for step in range(1, steps + 1):
        row = random.Random(seed + step).choice(train)
        pixels = read_clip(row, **geometry, seed=seed + step).to(device)
        component.temperature = temperature_at(step, steps, start, end)
        component.prepare_search_step(pixels.shape[0])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            latents = teacher.encode(pixels)
            target = teacher(latents)[0] if loss_fn.weights["teacher_l1"] else None
            prediction, _ = model(latents)
            loss, _ = loss_fn(prediction, pixels, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite search loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        component.clear_search_step()
        gate_ema.lerp_(component.gate_logits.detach(), 1 - decay)
        print(
            json.dumps(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "temperature": component.temperature,
                    "kept": component.selected_layers(gate_ema),
                }
            ),
            flush=True,
        )
        if step == steps or (config["evaluate_every"] and step % config["evaluate_every"] == 0):
            with torch.no_grad():
                logits = component.gate_logits.detach().clone()
                component.gate_logits.copy_(gate_ema)
            validation = evaluate(model, teacher, val, geometry, device, precision)
            with torch.no_grad():
                component.gate_logits.copy_(logits)
            print(json.dumps({"step": step, "validation": validation}), flush=True)
        if step == steps or (config["save_every"] and step % config["save_every"] == 0):
            save_search(component, config, step, gate_ema, validation, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--train-manifest", type=Path, nargs="+")
    parser.add_argument("--val-manifest", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("teacher", "train_manifest", "val_manifest", "output", "steps"):
        value = getattr(args, key)
        if value is not None:
            config[key] = (
                [str(path) for path in value]
                if isinstance(value, list)
                else str(value)
                if isinstance(value, Path)
                else value
            )
    for key in ("teacher", "train_manifest", "val_manifest", "output"):
        if not config.get(key):
            parser.error(f"Provide --{key.replace('_', '-')} or set {key} in the config.")
    for key in ("train_manifest", "val_manifest"):
        if isinstance(config[key], str):
            config[key] = [config[key]]
    run(config, Path(config["output"]), torch.device(args.device))


if __name__ == "__main__":
    main()
