"""Merge search LoRA, prune residual branches, and initialize temporal compression."""

import argparse
import json
from pathlib import Path

import torch

from .checkpoints import atomic_save, sha256
from .models import temporal, wan22
from .models.api import Teacher


@torch.no_grad()
def compress(component, family, fraction=0.6, tail_2d=True):
    if family == "wan22":
        return wan22.convert(component, fraction=fraction, tail_2d=tail_2d)
    records = temporal.convert(component, fraction=fraction)
    if tail_2d:
        prefixes = tuple(f"upsamples.{index}.residual." for index in (8, 9, 10, 12, 13, 14))
        for record in records:
            if record["path"].startswith(prefixes):
                old = component.decoder.get_submodule(record["path"])
                new = temporal.PackedTemporalConv(record["shape"], record["padding"], 0, record["bias"]).to(
                    old.down.weight.device
                )
                new.down.weight.copy_(old.down.weight[: old.co])
                if old.bias is not None:
                    new.bias.copy_(old.bias)
                component.decoder.set_submodule(record["path"], new)
                record.update(rank=0, pure2d=True)
    return records


def merge_searched(teacher, saved, family):
    if saved["family"] != family:
        raise ValueError("Search checkpoint family differs from export config.")
    if family == "wan21":
        _, component, _ = temporal.merge_search(saved, saved["config"])
    else:
        search = wan22.Component(teacher.component, search=saved["config"]["search"])
        search.load_state_dict(saved["model"], strict=True)
        selected = search.selected_layers(saved["gate_ema"])
        if selected != saved["kept"]:
            raise ValueError("Saved EMA selection differs from kept residual indices.")
        component = wan22.merge(search, teacher.component, saved["gate_ema"])
    return component


def run(config, teacher_path, search_checkpoint, output, device="cpu"):
    if output.exists():
        raise FileExistsError(output)
    family = config["family"]
    if family not in ("wan21", "wan22"):
        raise ValueError("Export family must be wan21 or wan22.")
    if config["variant"] not in ("pro", "lite"):
        raise ValueError("Export variant must be pro or lite.")
    if (config["variant"] == "lite") != (search_checkpoint is not None):
        raise ValueError("Lite export requires a search checkpoint; Pro export starts from the original teacher.")
    teacher = Teacher(family, teacher_path)
    if search_checkpoint is not None:
        saved = torch.load(search_checkpoint, map_location="cpu", weights_only=True)
        component = merge_searched(teacher, saved, family)
        kept, step = saved["kept"], saved["step"]
    else:
        component = teacher.component if family == "wan21" else wan22.Component(teacher.component)
        kept, step = list(range(14)), 0
    del teacher
    component.to(device)
    records = compress(component, family, config["fraction"], config["tail_2d"])
    origin = search_checkpoint or teacher_path
    atomic_save(
        {
            "format_version": 1,
            "family": family,
            "kept": kept,
            "packed_records": records,
            "model": {name: value.detach().cpu() for name, value in component.state_dict().items()},
            "name": output.stem,
            "config": config,
            "provenance": {"source": str(origin.resolve()), "source_sha256": sha256(origin), "source_step": step},
        },
        output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "family": family,
                "kept": kept,
                "source_step": step,
                "temporal_operators": len(records),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--search-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu", help="Device for temporal SVD initialization")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for name in ("teacher", "search_checkpoint", "output"):
        value = getattr(args, name)
        if value is not None:
            config[name] = str(value)
    for name in ("teacher", "output"):
        if not config.get(name):
            parser.error(f"Provide --{name} or set {name} in the config.")
    search = Path(config["search_checkpoint"]) if config.get("search_checkpoint") else None
    run(config, Path(config["teacher"]), search, Path(config["output"]), args.device)


if __name__ == "__main__":
    main()
