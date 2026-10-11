"""Reconstruct a video with explicit encoder and decoder checkpoints."""

import argparse
import gc
import json
from pathlib import Path

import torch

from utils import (DTYPES, FAMILIES, build_model, check_pair, check_precision,
                   component_spec, environment, measure, positive_int, prepare_video,
                   psnr, read_config, read_video, setup, write_video)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=FAMILIES, required=True)
    parser.add_argument("--variant", choices=("lite", "pro", "base"), default="lite",
                        help="Decoder architecture (default: lite; Pro is Wan-only)")
    parser.add_argument("--encoder", "--encoder-weights", dest="encoder_weights", type=Path, required=True,
                        help="Encoder checkpoint; an original full VAE file or H3 VAE directory also works")
    parser.add_argument("--decoder", "--decoder-weights", dest="decoder_weights", type=Path, required=True,
                        help="Decoder checkpoint matching --model and --variant")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 filename; metrics use the same name with .json")
    parser.add_argument("--frames", type=positive_int, default=81, help="Maximum input frames (default 81)")
    parser.add_argument("--height", type=positive_int)
    parser.add_argument("--width", type=positive_int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--encoder-variant", choices=("base", "lite"), default="base",
                        help="Encoder architecture (default: base); Lite is H3-only")
    return parser


def resolve_args(args):
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must be an MP4 filename, e.g. outputs/wan21_lite.mp4")
    for path in (args.output, args.output.with_suffix(".json")):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
    if (args.height is None) != (args.width is None):
        raise ValueError("Supply both --height and --width")
    encoder, encoder_config, encoder_weights = component_spec(
        args.model, args.encoder_variant, "encoder", args.encoder_weights)
    decoder, config, decoder_weights = component_spec(
        args.model, args.variant, "decoder", args.decoder_weights)
    resolved = dict(vars(args), encoder=encoder, decoder=decoder, dtype="bf16",
                    encoder_config=encoder_config, encoder_weights=encoder_weights,
                    config=config, decoder_weights=decoder_weights)
    return argparse.Namespace(**resolved)


def run(args):
    """Encode, reconstruct, and save an MP4 plus its quality/timing report."""
    with torch.inference_mode():
        report_path = args.output.with_suffix(".json")
        check_precision(args.decoder, args.dtype)
        device = setup(args.device, args.seed)
        dtype = DTYPES[args.dtype]
        config, encoder_config = read_config(args.config), read_config(args.encoder_config)
        encoder = build_model('encoder', args.encoder, encoder_config, args.encoder_weights)
        decoder = build_model('decoder', args.decoder, config, args.decoder_weights)
        check_pair(encoder, decoder)
        config, encoder_config = decoder.config, encoder.config
        frames, fps = read_video(args.input, args.frames)
        original, padded = prepare_video(frames, encoder.family, encoder.spatial_scale,
                                         args.height, args.width)
        del frames
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(f'Input: {tuple(original.shape)}; padded: {tuple(padded.shape)}', flush=True)
        encoder.to(device=device, dtype=dtype)
        encoder_input = padded.to(device=device, dtype=dtype)
        z, encode = measure(encoder, encoder_input, args.warmup, args.repeats)
        del encoder, encoder_input
        gc.collect()
        torch.cuda.empty_cache()
        decoder.to(device=device, dtype=dtype)
        prediction, decode = measure(decoder, z.to(dtype), args.warmup, args.repeats)
        if tuple(prediction.shape) != tuple(padded.shape):
            raise ValueError(f'Decoder geometry mismatch: {tuple(prediction.shape)} vs {tuple(padded.shape)}')
        t, h, w = original.shape[-3:]
        cropped = prediction[:, :, :t, :h, :w].cpu()
        score = psnr(cropped, original)
        encode['fps'] = t*1000/encode['ms']
        decode['fps'] = t*1000/decode['ms']
        encode['inferred_fps'] = padded.shape[2]*1000/encode['ms']
        decode['inferred_fps'] = padded.shape[2]*1000/decode['ms']
        report = dict(**environment(device), encoder=args.encoder, decoder=args.decoder, dtype=args.dtype,
                      config=config, encoder_config=encoder_config, source_fps=str(fps),
                      encoder_weights=str(args.encoder_weights), decoder_weights=str(args.decoder_weights),
                      variant=args.variant, encoder_variant=args.encoder_variant,
                      shape=list(original.shape), padded_shape=list(padded.shape), latent_shape=list(z.shape),
                      scored_frames=t, psnr=score, encoder_timing=encode, decoder_timing=decode,
                      warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                      psnr_protocol='Unpadded RGB video MSE, clamped [0,1], FP64 accumulation, before MP4 encoding',
                      timing_protocol='Eager wall-clock median with CUDA synchronization; load, I/O, warmup, PSNR and MP4 encoding excluded')
        write_video(args.output, cropped, fps)
        with report_path.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(f'PSNR: {score:.6f} dB; encoder: {encode["fps"]:.2f} FPS; decoder: {decode["fps"]:.2f} FPS')
        print(f'Saved {args.output} and {report_path}.')


def main():
    run(resolve_args(make_parser().parse_args()))


if __name__ == "__main__":
    main()
