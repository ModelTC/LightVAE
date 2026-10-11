"""Compare LightVAE decoder speed only against its original VAE baseline."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from utils import (DECODERS, DEFAULT_WEIGHTS_DIR, DTYPES, FAMILIES, ROOT, VARIANTS,
                   build_model, check_precision, decoder_spec, environment, latent_frames,
                   measure, padded_frames, positive_int, read_config, setup)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=FAMILIES, required=True)
    parser.add_argument("--variants", choices=("pro", "lite"), nargs="+",
                        help="Default: pro lite for Wan; lite for H3. Base always included.")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    for name in ("frames", "height", "width"):
        parser.add_argument(f"--{name}", type=positive_int)
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--repeats", type=positive_int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def build_commands(args):
    variants = args.variants if args.variants is not None else VARIANTS[args.model]
    if len(set(variants)) != len(variants):
        raise ValueError("Do not repeat a variant")
    commands = []
    for variant in ("base", *variants):
        decoder, config, weights = decoder_spec(args.model, variant, args.weights_dir, args.base_weights)
        command = [sys.executable, str(ROOT / "benchmark.py"), "--worker", "--decoder", decoder,
                   "--weights", str(weights.resolve()), "--dtype", "bf16",
                   "--device", args.device, "--batch", str(args.batch),
                   "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                   "--seed", str(args.seed), "--output", str(args.output.resolve() / f"{variant}.json")]
        if config is not None:
            command.extend(("--config", str(config)))
        for name in ("frames", "height", "width"):
            value = getattr(args, name)
            if value is not None:
                command.extend((f"--{name}", str(value)))
        commands.append((variant, command))
    return commands


def summarize(reports):
    baseline = reports["base"]
    same = ("gpu", "torch", "cuda", "cudnn", "dtype", "mode", "tf32", "latent_shape",
            "latent_sha256", "requested_frames", "inferred_frames", "warmup", "repeats", "seed")
    rows = []
    for variant, report in reports.items():
        for field in same:
            if report[field] != baseline[field]:
                raise ValueError(f"Unmatched benchmark protocol for {variant}: {field}")
        for key in ("tile_size", "tile_overlap"):
            if report["config"].get(key) != baseline["config"].get(key):
                raise ValueError(f"Unmatched H3 tiling: {key}")
        rows.append(dict(variant=variant, ms=report["ms"], fps=report["fps"],
                         speedup_vs_base=baseline["ms"] / report["ms"],
                         peak_allocated_gib=report["peak_allocated_gib"],
                         peak_reserved_gib=report["peak_reserved_gib"]))
    return rows


def main():
    args = make_parser().parse_args()
    commands = build_commands(args)  # Check every checkpoint before starting a GPU process.
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for variant, command in commands:
        log = args.output / f"{variant}.log"
        print(f"Benchmarking {args.model}/{variant}; log: {log}", flush=True)
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"{variant} failed (exit {result.returncode}); see {log}. No comparison reported.")
        reports[variant] = json.loads((args.output / f"{variant}.json").read_text())
    rows = summarize(reports)
    with (args.output / "summary.json").open("x") as stream:
        json.dump(dict(model=args.model, rows=rows, reports=reports), stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.output / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("variant         latency(ms)        FPS    speedup vs base    peak allocated(GiB)")
    for row in rows:
        print(f"{row['variant']:10s} {row['ms']:16.3f} {row['fps']:10.2f} "
              f"{row['speedup_vs_base']:17.2f}x {row['peak_allocated_gib']:21.3f}")


BENCHMARK_SHAPES = {
    'wan21': (81, 480, 832),
    'wan22': (81, 704, 1280),
    'h3': (124, 768, 1344),
}


def benchmark_shape(family, frames=None, height=None, width=None):
    return tuple(default if value is None else value
                 for value, default in zip((frames, height, width), BENCHMARK_SHAPES[family]))


def run_decoder(args):
    with torch.inference_mode():
        check_precision(args.decoder, args.dtype)
        device = setup(args.device, args.seed)
        config = read_config(args.config)
        model = build_model('decoder', args.decoder, config, args.weights)
        config = model.config
        args.frames, args.height, args.width = benchmark_shape(
            model.family, args.frames, args.height, args.width)
        scale = model.spatial_scale
        if args.height % scale or args.width % scale:
            raise ValueError(f'Height and width must be divisible by {scale}')
        inferred = padded_frames(model.family, args.frames)
        shape = (args.batch, model.latent_channels, latent_frames(model.family, inferred),
                 args.height//scale, args.width//scale)
        model.to(device=device, dtype=DTYPES[args.dtype])
        # Keep base/Pro/Lite latents independent of model initialization.
        generator = torch.Generator(device=device).manual_seed(args.seed)
        z = torch.randn(shape, device=device, dtype=DTYPES[args.dtype], generator=generator)
        latent_sha256 = hashlib.sha256(z.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        output, timing = measure(model, z, args.warmup, args.repeats)
        expected = (args.batch, 3, inferred, args.height, args.width)
        if tuple(output.shape) != expected:
            raise ValueError(f'Decoder geometry mismatch: {tuple(output.shape)} vs {expected}')
        report = dict(**environment(device), decoder=args.decoder, dtype=args.dtype,
                      config=config, latent_shape=list(shape),
                      output_shape=list(output.shape), requested_frames=args.frames, inferred_frames=inferred,
                      fps=args.batch*args.frames*1000/timing['ms'],
                      inferred_fps=args.batch*inferred*1000/timing['ms'], **timing,
                      warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                      latent_sha256=latent_sha256,
                      protocol='Random latent; eager synchronized wall-clock median. Loading/warmup excluded. FPS uses requested frames; inferred_fps includes temporal padding. Not a quality evaluation.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(report, indent=2, allow_nan=False))


def worker_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder', choices=DECODERS, required=True)
    parser.add_argument('--config', type=Path, help='Decoder JSON; required for LightVAE')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--height', type=positive_int, help='Default: Wan2.1 480, Wan2.2 704, H3 768')
    parser.add_argument('--width', type=positive_int, help='Default: Wan2.1 832, Wan2.2 1280, H3 1344')
    parser.add_argument('--frames', type=positive_int, help='Default: Wan2.1/Wan2.2 81, H3 124')
    parser.add_argument('--batch', type=positive_int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=DTYPES, default='bf16')
    parser.add_argument('--warmup', type=positive_int, default=5)
    parser.add_argument('--repeats', type=positive_int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file')
    run_decoder(parser.parse_args())


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        del sys.argv[1]
        worker_main()
    else:
        main()
