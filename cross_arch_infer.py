"""Reconstruct a video with an original or recovered TLR decoder."""

import argparse
import gc
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from lightvae.cross_arch_runtime import load_variant
from utils import (DTYPES, environment, measure, positive_int, psnr, read_video,
                   setup, write_video)


FAMILIES = ('hunyuanvideo15', 'ltx25', 'cogvideox15')


def prepare_video(frames, temporal_ratio, spatial_ratio, height, width):
    rgb = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1
    source_height, source_width = rgb.shape[-2:]
    scale = max(height / source_height, width / source_width)
    new_height = max(height, round(source_height * scale))
    new_width = max(width, round(source_width * scale))
    rgb = F.interpolate(rgb, (new_height, new_width), mode='bilinear',
                        align_corners=False, antialias=True)
    top, left = (new_height - height) // 2, (new_width - width) // 2
    x = rgb[:, :, top:top + height, left:left + width]
    original = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    t, h, w = original.shape[-3:]
    padded = F.pad(original,
                   (0, (-w) % spatial_ratio, 0, (-h) % spatial_ratio,
                    0, (-(t - 1)) % temporal_ratio), mode='replicate')
    return original, padded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=FAMILIES, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--tlr-checkpoint', type=Path)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=positive_int, default=81)
    parser.add_argument('--height', type=positive_int, default=480)
    parser.add_argument('--width', type=positive_int, default=832)
    parser.add_argument('--dtype', choices=DTYPES, default='fp32')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=positive_int, default=1)
    parser.add_argument('--repeats', type=positive_int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.mp4':
        parser.error('--output must end in .mp4')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)

    device = setup(args.device, args.seed)
    dtype = DTYPES[args.dtype]
    vae, source = load_variant(args.model, args.model_root, args.tlr_checkpoint,
                               dtype=dtype)
    frames, fps = read_video(args.input, args.frames)
    original, padded = prepare_video(frames, vae.temporal_ratio, vae.spatial_ratio,
                                     args.height, args.width)
    vae.encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    encoder_input = padded.to(device=device, dtype=dtype)
    latents, encode = measure(vae.encode_mean, encoder_input, args.warmup, args.repeats)
    del encoder_input
    vae.encoder.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    vae.decoder.to(device=device, dtype=dtype).eval().requires_grad_(False)

    def decode(z):
        return vae.decode(z, frames=padded.shape[2])

    reconstructed, decode_time = measure(decode, latents, args.warmup, args.repeats)
    cropped = reconstructed[:, :, :original.shape[2],
                            :original.shape[3], :original.shape[4]].cpu()
    score = psnr(cropped, original)
    count = original.shape[2]
    report = dict(
        **environment(device), family=args.model, **source, dtype=args.dtype,
        source_video=str(args.input.resolve()), source_fps=str(fps),
        shape=list(original.shape), padded_shape=list(padded.shape),
        latent_shape=list(latents.shape), psnr=score,
        encoder_ms=encode['ms'], decoder_ms=decode_time['ms'],
        encoder_fps=count * 1000 / encode['ms'],
        decoder_fps=count * 1000 / decode_time['ms'],
        warmup=args.warmup, repeats=args.repeats,
        psnr_protocol='Unpadded RGB video MSE in [0,1] before MP4 encoding',
        timing_protocol='Eager synchronized wall-clock median; load, I/O and warmup excluded',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video(args.output, cropped, fps)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{args.model} {source['variant']}: PSNR {score:.3f} dB, "
          f"decoder {report['decoder_fps']:.2f} FPS", flush=True)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
