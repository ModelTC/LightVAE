"""Compare Wan2.2 Base, Pro, Lite and Turbo decoder speed with random weights.

No checkpoints or videos are downloaded. Each model runs in a separate process;
an OOM/failure is recorded and the remaining variants still run.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from utils import build_model, environment, measure, positive_int, read_config, setup

ROOT = Path(__file__).resolve().parent


def worker(args):
    device = setup(args.device, args.seed)
    variant = args.worker
    config = (None if variant == 'base' else
              read_config(ROOT / f'lightvae/config/lightvae-{variant}-wan22.json'))
    model = build_model('decoder', 'wan22' if variant == 'base' else 'wan22_lightvae',
                        config, checkpoint=None).to(device=device, dtype=torch.bfloat16)
    shape = (args.batch, 48, (args.frames - 1) // 4 + 1, args.height // 16, args.width // 16)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    digest = hashlib.sha256(z.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    output, timing = measure(model, z, args.warmup, args.repeats)
    expected = (args.batch, 3, args.frames, args.height, args.width)
    if tuple(output.shape) != expected or output.dtype != torch.bfloat16:
        raise ValueError(f'Wrong output: {output.shape}, {output.dtype}; expected {expected}, BF16')
    report = dict(**environment(device), variant=variant, random_weights=True,
                  weights_source='random', dtype='bf16', config=model.config,
                  parameters=sum(p.numel() for p in model.parameters()),
                  latent_shape=list(shape), output_shape=list(output.shape),
                  latent_sha256=digest, requested_frames=args.frames, inferred_frames=args.frames,
                  seed=args.seed, warmup=args.warmup, repeats=args.repeats,
                  fps=args.batch * args.frames * 1000 / timing['ms'], **timing,
                  protocol='Random weights and seeded random latent; BF16 eager synchronized '
                           'decoder wall-time median. Loading, warmup, encoding and I/O excluded. '
                           'Architecture throughput only, not a reconstruction-quality evaluation.')
    (args.output / f'{variant}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{variant}: {report['ms']:.3f} ms, {report['fps']:.3f} FPS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=positive_int, default=17)
    parser.add_argument('--height', type=positive_int, default=704)
    parser.add_argument('--width', type=positive_int, default=1280)
    parser.add_argument('--batch', type=positive_int, default=1)
    parser.add_argument('--warmup', type=positive_int, default=5)
    parser.add_argument('--repeats', type=positive_int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True, help='New result directory')
    parser.add_argument('--variants', choices=('base', 'pro', 'lite', 'turbo'), nargs='+',
                        default=['base', 'pro', 'lite', 'turbo'])
    parser.add_argument('--turbo-source', type=Path, default=ROOT / 'third_party/Turbo-VAED')
    parser.add_argument('--turbo-config', type=Path,
                        default=ROOT / 'configs/benchmarks/turbo_wan22_reconstructed.json')
    parser.add_argument('--worker', choices=('base', 'pro', 'lite'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.frames - 1) % 4 or args.height % 16 or args.width % 16:
        parser.error('frames must be 4*n+1; height and width must be divisible by 16')
    if len(set(args.variants)) != len(args.variants):
        parser.error('Do not repeat variants')
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    common = []
    for name in ['frames', 'height', 'width', 'batch', 'warmup', 'repeats', 'seed', 'device']:
        common += [f'--{name}', str(getattr(args, name))]
    reports, failures, rows = {}, {}, []
    for variant in args.variants:
        if variant == 'turbo':
            command = [sys.executable, str(ROOT / 'benchmark_turbo.py'), '--random-weights',
                       '--source-dir', str(args.turbo_source.resolve()),
                       '--config', str(args.turbo_config.resolve()),
                       '--output', str((args.output / 'turbo.json').resolve()), *common]
        else:
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                       '--output', str(args.output.resolve()), *common]
        log = args.output / f'{variant}.log'
        print(f'Benchmarking {variant}; log: {log}', flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            detail = log.read_text(errors='replace')
            status = 'OOM' if 'out of memory' in detail.lower() else 'FAILED'
            failures[variant] = dict(status=status, returncode=result.returncode, log=str(log))
            print(f'{variant}: {status}; continuing with the remaining variants.', flush=True)
            rows.append(dict(variant=variant, status=status))
            continue
        report = json.loads((args.output / f'{variant}.json').read_text())
        if reports:
            reference = next(iter(reports.values()))
            for key in ['gpu', 'torch', 'cuda', 'cudnn', 'dtype', 'mode', 'tf32',
                        'latent_shape', 'output_shape', 'latent_sha256', 'requested_frames',
                        'warmup', 'repeats', 'seed']:
                if report[key] != reference[key]:
                    raise ValueError(f'Mismatched protocol: {variant}/{key}')
        reports[variant] = report
        rows.append(dict(variant=variant, status='OK',
                         **{key: report[key] for key in ['ms', 'fps', 'peak_allocated_gib']}))
    for row in rows:
        if row['status'] == 'OK' and 'base' in reports:
            row['speedup_vs_base'] = reports['base']['ms'] / row['ms']
    summary = dict(random_weights=True, reports=reports, failures=failures, rows=rows,
                   note='Architecture throughput only. Turbo uses a locally reconstructed Wan '
                        'configuration; its parameter-free settings are not author-confirmed.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.output / 'summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant', 'status', 'ms', 'fps',
                                                    'speedup_vs_base', 'peak_allocated_gib'])
        writer.writeheader()
        writer.writerows(rows)
    print('variant    status      latency(ms)       FPS', flush=True)
    for row in rows:
        values = f"{row['ms']:14.3f} {row['fps']:9.3f}" if row['status'] == 'OK' else ''
        print(f"{row['variant']:10s} {row['status']:7s} {values}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
