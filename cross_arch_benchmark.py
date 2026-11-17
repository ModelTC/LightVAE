"""Single-GPU decoder throughput for original and recovered TLR video VAEs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from lightvae.cross_arch import MODEL_ROOT
from lightvae.cross_arch_random import STRUCTURE_ROOT, load_random_variant
from lightvae.cross_arch_runtime import load_variant
from utils import DTYPES, environment, measure, positive_int, setup


FAMILIES = ('hunyuanvideo15', 'ltx25', 'cogvideox15')


def worker(args):
    device = setup(args.device, args.seed)
    dtype = DTYPES[args.dtype]
    if args.random_weights:
        variant = args.worker.removeprefix('eta') if args.worker != 'base' else 'base'
        vae, source = load_random_variant(args.model, args.model_root, args.structure_dir,
                                         variant, dtype=dtype, seed=args.seed)
    else:
        vae, source = load_variant(
            args.model, args.model_root,
            args.tlr_checkpoint if args.worker == 'tlr' else None, dtype=dtype)
    shape = vae.latent_shape(args.frames, args.height, args.width)
    vae.decoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    digest = hashlib.sha256(z.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def decode(latents):
        return vae.decode(latents, frames=args.frames)

    prediction, timing = measure(decode, z, args.warmup, args.repeats)
    expected = (1, 3, args.frames, args.height, args.width)
    if tuple(prediction.shape) != expected:
        raise ValueError(f'Wrong decoder output shape {tuple(prediction.shape)}; expected {expected}')
    runtime = environment(device)
    runtime['mode'] = 'eager'
    report = dict(
        **runtime, family=args.model, **source, dtype=args.dtype,
        compile_mode='none', shape=list(expected), latent_shape=list(shape),
        latent_sha256=digest, parameters=sum(p.numel() for p in vae.decoder.parameters()),
        ms=timing['ms'], fps=1000 * args.frames / timing['ms'],
        samples_ms=timing['samples_ms'], warmup=args.warmup, repeats=args.repeats,
        peak_allocated_gib=timing['peak_allocated_gib'],
        peak_reserved_gib=timing['peak_reserved_gib'],
        timing_protocol='Synchronized single-GPU decoder-only wall time; median of repeats; '
                        'load, compilation, warmup, encoding, transfer and I/O excluded',
    )
    target = args.output / f'{args.worker}.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"{args.model}/{args.worker}: {report['fps']:.3f} FPS, "
          f"{report['ms']:.3f} ms, {report['peak_allocated_gib']:.2f} GiB peak", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=FAMILIES, required=True)
    parser.add_argument('--model-root', type=Path, default=MODEL_ROOT)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument('--tlr-checkpoint', type=Path)
    weights.add_argument('--random-weights', action='store_true',
                         help='Compare base and exact eta=0.4/0.6 TLR structures without VAE weights')
    parser.add_argument('--structure-dir', type=Path, default=STRUCTURE_ROOT,
                        help='Directory containing small family config/TLR-record JSON files')
    parser.add_argument('--frames', type=positive_int, default=81)
    parser.add_argument('--height', type=positive_int, default=480)
    parser.add_argument('--width', type=positive_int, default=832)
    parser.add_argument('--dtype', choices=DTYPES, default='bf16')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=positive_int, default=5)
    parser.add_argument('--repeats', type=positive_int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=('base', 'tlr', 'eta0.4', 'eta0.6'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.random_weights and args.model == 'cogvideox15':
        parser.error('--random-weights supports hunyuanvideo15 and ltx25')
    if args.worker and ((args.random_weights and args.worker == 'tlr')
                        or (not args.random_weights and args.worker.startswith('eta'))):
        parser.error('Worker variant does not match the weight mode')
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    variants = (('base', 'eta0.4', 'eta0.6') if args.random_weights
                else ('base', 'tlr') if args.tlr_checkpoint is not None else ('base',))
    reports = {}
    for variant in variants:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                   '--worker', variant]
        with (args.output / f'{variant}.log').open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    check=False)
        if result.returncode:
            raise RuntimeError(f'{variant} benchmark failed; see {args.output / (variant + ".log")}')
        reports[variant] = json.loads((args.output / f'{variant}.json').read_text())
    speedups = {}
    for variant in variants[1:]:
        original, compressed = reports['base'], reports[variant]
        for key in ('gpu', 'torch', 'cuda', 'cudnn', 'dtype', 'shape', 'latent_shape',
                    'latent_sha256', 'warmup', 'repeats', 'compile_mode'):
            if original[key] != compressed[key]:
                raise ValueError(f'Unmatched benchmark setting: {key}')
        speedups[variant] = original['ms'] / compressed['ms']
    summary = dict(family=args.model, reports=reports, tlr_speedup=speedups.get('tlr'))
    if args.random_weights:
        summary.update(random_weights=True, tlr_speedups=speedups)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.output / 'summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('variant', 'ms', 'fps', 'parameters',
                                                     'peak_allocated_gib'))
        writer.writeheader()
        for variant, report in reports.items():
            writer.writerow({key: report[key] if key != 'variant' else variant
                             for key in writer.fieldnames})
    print(f'{args.model}: ' + ', '.join(f'{v} {r["fps"]:.2f} FPS' for v, r in reports.items()))
    for variant, speedup in speedups.items():
        print(f"{'TLR' if variant == 'tlr' else variant} speedup: {speedup:.3f}x")


if __name__ == '__main__':
    main()
