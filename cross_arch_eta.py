"""DAVIS eta sweep: FP32 reconstruction and eager BF16 decoder throughput.

Reuses the 30 original-encoder latent caches from Figure 7 calibration.
The last two residual stages are spatial-only by default. No recovery or
torch.compile is involved; fused inference is enabled only for TLR speed.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import torch
from torch import nn

from lightvae.cross_arch import load_vae
from lightvae.cross_arch_tlr import convert_decoder, fold_decoder
from utils import environment, measure, positive_int, setup


ROOT = Path(__file__).resolve().parent
EXPERIMENT = Path('/data/nvme3/gushiqiao/outputs/lightvae_tlr_cross_arch_20261107')
FRAMES, HEIGHT, WIDTH = 33, 480, 832
VARIANTS = ('base', '0.2', '0.4', '0.6', '0.8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def video_psnr(prediction, reference):
    if prediction.shape != reference.shape or not torch.isfinite(prediction).all():
        raise ValueError('Invalid reconstructed video')
    squared = 0.
    for frame in range(FRAMES):
        error = (prediction[:, :, frame].float().clamp(-1, 1)
                 - reference[:, :, frame]) / 2
        squared += error.square().sum(dtype=torch.float64).item()
    return -10 * math.log10(max(squared / prediction.numel(), 1e-12))


def calibration(args, vae):
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if (len(rows) != 30 or len({r['group'] for r in rows}) != 30
            or any(r['dataset'] != 'davis' or len(r['zip_members']) < FRAMES for r in rows)):
        raise ValueError('Expected the 30 distinct Figure 7 DAVIS calibration videos')
    cached_protocol = json.loads((args.cache_dir / 'protocol.json').read_text())
    weights_hash = sha256(vae.weights)
    if (cached_protocol['family'] != args.model
            or cached_protocol['geometry'] != [FRAMES, HEIGHT, WIDTH]
            or cached_protocol['manifest_sha256'] != sha256(args.manifest)
            or cached_protocol['weights_sha256'] != weights_hash
            or cached_protocol['weights_bytes'] != vae.weights.stat().st_size):
        raise ValueError('Cached latents do not match the original VAE and calibration protocol')
    rows = rows[:args.limit]
    latents, latent_hashes = [], []
    for index, row in enumerate(rows):
        path = args.cache_dir / 'latents' / f'{index:02d}.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        latent = saved['latent']
        if (saved['group'] != row['group'] or latent.dtype != torch.float32
                or tuple(latent.shape) != vae.latent_shape(FRAMES, HEIGHT, WIDTH)
                or not torch.isfinite(latent).all()):
            raise ValueError(f'Invalid original-encoder cache: {path}')
        latents.append(latent)
        latent_hashes.append(sha256(path))
    return rows, latents, weights_hash, latent_hashes


def build(args):
    vae = load_vae(args.model, root=args.model_root, device='cpu', dtype=torch.float32)
    rows, latents, weights_hash, latent_hashes = calibration(args, vae)
    vae.model.encoder = nn.Identity()
    records = []
    if args.variant != 'base':
        selected = vae.eligible_decoder_convs
        paths = [path.removeprefix('decoder.') for path, _, _ in selected]
        spatial = ([path.removeprefix('decoder.') for path, _, flag in selected if flag]
                   if args.tail == 'spatial' else [])
        records = convert_decoder(
            vae.decoder, paths, float(args.variant), pure2d_paths=spatial,
            center_slice=1 if args.model == 'ltx25' else 2, svd_device=args.device,
            chunk_frames=args.chunk_frames)
        fold_decoder(vae.decoder, records)
    vae.model.to(device=args.device, dtype=torch.float32).eval().requires_grad_(False)
    return vae, rows, latents, records, weights_hash, latent_hashes


def protocol(args, rows, records, weights_hash, latent_hashes):
    sources = [Path(__file__), ROOT / 'lightvae/kernels.py', ROOT / 'lightvae/triton_kernels.py',
               *sorted((ROOT / 'lightvae').glob('cross_arch*.py'))]
    return dict(
        schema_version=1, model=args.model, variant=args.variant, tail=args.tail,
        eta=None if args.variant == 'base' else float(args.variant),
        geometry=[FRAMES, HEIGHT, WIDTH], groups=[row['group'] for row in rows],
        manifest=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
        cache_dir=str(args.cache_dir.resolve()), latent_sha256=latent_hashes,
        weights_sha256=weights_hash, chunk_frames=args.chunk_frames,
        warmup=args.warmup, repeats=args.repeats, operators=records,
        quality='FP32 eager; original encoder posterior mean; original surrounding operators',
        speed='BF16 eager; original source base, folded and fused TLR',
        timing='Five global warmups (configurable), one warmup per video, synchronized repeats; decoder-only wall time; '
               'mean of per-video median latency; FPS=33000/mean_ms',
        psnr='Clamped reconstruction in RGB [0,1]; video-MSE PSNR, equal-video mean',
        data_use='DAVIS eta calibration, not independent held-out evaluation',
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        **environment(args.device),
    )


def measure_quality(args, vae, rows, latents, target):
    from vae_prune.data import read_clip
    scores = []
    for index, (row, latent) in enumerate(zip(rows, latents), 1):
        original = read_clip(row, frames=FRAMES, height=HEIGHT, width=WIDTH,
                             validation=True, stream_spatial=True).to(args.device)
        prediction = vae.decode(latent.to(args.device), frames=FRAMES)
        score = video_psnr(prediction, original)
        scores.append(dict(group=row['group'], psnr=score))
        write_json(target / 'quality_progress.json', scores)
        print(f'{args.model}/{args.variant} quality {index}/{len(rows)} '
              f'{row["group"]}: {score:.4f} dB', flush=True)
        del original, prediction
    result = dict(psnr=statistics.mean(row['psnr'] for row in scores), rows=scores,
                  videos=len(scores), dtype='fp32', tf32=False, execution='eager')
    write_json(target / 'quality.json', result)
    return result


def measure_speed(args, vae, rows, latents, target):
    if args.variant != 'base':
        from lightvae.cross_arch_runtime import optimize_tlr
        optimize_tlr(vae)
    vae.model.to(dtype=torch.bfloat16).eval()
    torch.cuda.empty_cache()
    samples = []
    for index, (row, latent) in enumerate(zip(rows, latents), 1):
        value = latent.to(device=args.device, dtype=torch.bfloat16)
        warmup = args.warmup + 1 if index == 1 else 1
        prediction, timing = measure(lambda z: vae.decode(z, frames=FRAMES),
                                     value, warmup, args.repeats)
        if tuple(prediction.shape) != (1, 3, FRAMES, HEIGHT, WIDTH):
            raise ValueError('Unexpected decoder output geometry')
        samples.append(dict(group=row['group'], **timing))
        write_json(target / 'speed_progress.json', samples)
        print(f'{args.model}/{args.variant} speed {index}/{len(rows)} '
              f'{row["group"]}: {timing["ms"]:.3f} ms', flush=True)
        del prediction, value
    ms = statistics.mean(row['ms'] for row in samples)
    result = dict(ms=ms, fps=1000 * FRAMES / ms, rows=samples, videos=len(samples),
                  dtype='bf16', tf32=False, execution='eager', compile_mode='none',
                  peak_allocated_gib=max(row['peak_allocated_gib'] for row in samples))
    write_json(target / 'speed.json', result)
    return result


@torch.inference_mode()
def worker(args):
    args.device = setup(args.device, 2026)
    vae, rows, latents, records, weights_hash, latent_hashes = build(args)
    name = 'base' if args.variant == 'base' else f'eta_{round(float(args.variant) * 100):02d}'
    target = args.output / name
    target.mkdir(parents=True, exist_ok=True)
    config = protocol(args, rows, records, weights_hash, latent_hashes)
    saved = target / 'protocol.json'
    if saved.exists() and json.loads(saved.read_text()) != config:
        raise ValueError(f'Existing measurements use a different protocol: {saved}')
    write_json(saved, config)
    if not args.speed_only and not (target / 'quality.json').exists():
        measure_quality(args, vae, rows, latents, target)
    if not args.quality_only and not (target / 'speed.json').exists():
        measure_speed(args, vae, rows, latents, target)


def summarize(output):
    rows = []
    common = None
    matched = ('model', 'tail', 'geometry', 'groups', 'manifest_sha256', 'latent_sha256',
               'weights_sha256', 'chunk_frames', 'warmup', 'repeats', 'source_sha256',
               'gpu', 'torch', 'cuda', 'cudnn', 'tf32')
    for variant in VARIANTS:
        name = 'base' if variant == 'base' else f'eta_{round(float(variant) * 100):02d}'
        folder = output / name
        row = dict(variant=variant)
        for filename, keys in [('quality.json', ('psnr',)), ('speed.json', ('ms', 'fps'))]:
            if (folder / filename).exists():
                measured = json.loads((folder / filename).read_text())
                row.update({key: measured[key] for key in keys})
        if len(row) > 1:
            config = json.loads((folder / 'protocol.json').read_text())
            if config['variant'] != variant:
                raise ValueError(f'Mislabeled variant in {folder}')
            current = {key: config[key] for key in matched}
            if common is not None and common != current:
                raise ValueError(f'Cannot compare different benchmark protocols: {folder}')
            common = current
            rows.append(row)
    baseline = next((row.get('ms') for row in rows if row['variant'] == 'base'), None)
    for row in rows:
        row['speedup'] = baseline / row['ms'] if baseline and 'ms' in row else None
    write_json(output / 'summary.json', rows)
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('variant', 'psnr', 'ms', 'fps', 'speedup'))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('hunyuanvideo15', 'ltx25', 'cogvideox15'), required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--manifest', type=Path, default=EXPERIMENT / 'data/eta_calibration.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--tail', choices=('spatial', 'temporal'), default='spatial')
    parser.add_argument('--chunk-frames', type=positive_int)
    parser.add_argument('--warmup', type=positive_int, default=5)
    parser.add_argument('--repeats', type=positive_int, default=5)
    parser.add_argument('--limit', type=positive_int, default=30)
    parser.add_argument('--device', default='cuda:0')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--speed-only', action='store_true')
    mode.add_argument('--quality-only', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.limit > 30:
        parser.error('--limit must not exceed 30')
    args.cache_dir = args.cache_dir or EXPERIMENT / 'eta' / args.model
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.output)
    elif args.variant:
        worker(args)
    else:
        for variant in VARIANTS:
            with (args.output / f'{variant}.log').open('a') as stream:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                '--variant', variant], stdout=stream, stderr=subprocess.STDOUT,
                               check=True)
        summarize(args.output)


if __name__ == '__main__':
    main()
