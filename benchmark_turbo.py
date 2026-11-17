"""Benchmark the official Turbo-VAED Wan2.2 decoder with portable file paths.

The upstream Wan config is absent from its published Git tree. The supplied
configuration is reconstructed locally and strictly checked against the weights;
its parameter-free choices have not been confirmed by the authors.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import diffusers
import torch

from utils import environment, measure, positive_int, setup

ROOT = Path(__file__).resolve().parent
SOURCE_RELATIVE = 'diffusers_vae/src/diffusers/models/autoencoders/autoencoder_kl_turbo_vaed.py'
EXPECTED_CHECKPOINT_SHA = '4b9ba663f2e3f77a087c75657e440a1118ed436ac37f1dc53c6669c66b96163c'
EXPECTED_SOURCE_GIT = '6bd3adf679f140abb7d87fe183ae69e54bf64a7f'
EXPECTED_SOURCE_SHA = '480fead05a8f2305204a04516e138a4f3116f0278e78dbd4a171b7115e6a3469'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(args):
    source_dir = args.source_dir.resolve()
    source = source_dir / SOURCE_RELATIVE
    git_sha = None
    # Vendored source folders need no Git installation or repository metadata.
    if (source_dir / '.git').exists():
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=source_dir, text=True).strip()
        if git_sha != EXPECTED_SOURCE_GIT:
            raise ValueError(f'Turbo source revision must be {EXPECTED_SOURCE_GIT}, got {git_sha}')
    source_sha = sha256(source)
    if source_sha != EXPECTED_SOURCE_SHA:
        raise ValueError('Turbo source differs from the pinned official implementation')
    checkpoint_sha = None
    if not args.random_weights:
        checkpoint_sha = sha256(args.weights)
        if checkpoint_sha != EXPECTED_CHECKPOINT_SHA:
            raise ValueError('Turbo weights differ from the official HF LFS SHA256')
    config = json.loads(args.config.read_text())
    module_name = 'diffusers.models.autoencoders.table1_official_turbo'
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if args.random_weights:
        torch.manual_seed(args.seed)
        with torch.device('cpu'):
            full = module.AutoencoderKLTurboVAED.from_config(config)
        weights_metadata = dict(weights_source='random', initialization_seed=args.seed,
                                loaded_weights='Random initialization; no checkpoint loaded')
    else:
        with torch.device('meta'):
            full = module.AutoencoderKLTurboVAED.from_config(config)
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True, mmap=True)
        strip = lambda state: {key.removeprefix('module.'): value for key, value in state.items()}
        state = strip(checkpoint['state_dict']['gen_model'])
        raw_load = full.load_state_dict(state, strict=True, assign=True)
        ema = strip(checkpoint['ema_state_dict'])
        if not ema or not set(ema).issubset(state):
            raise ValueError('Unexpected Turbo EMA checkpoint keys')
        state.update(ema)
        ema_load = full.load_state_dict(state, strict=True, assign=True)
        weights_metadata = dict(
            weights_source='checkpoint',
            checkpoint_path=str(args.weights.resolve()), checkpoint_sha256=checkpoint_sha,
            checkpoint_repository='https://huggingface.co/zy12123/Turbo-VAED-Wan2.2-TI2V-5B',
            checkpoint_revision='a75c91084250866b3b2de557d3d4409a3f6cb247',
            checkpoint_epoch=checkpoint['epoch'], checkpoint_step=checkpoint['current_step'],
            loaded_weights='state_dict.gen_model then official ema_state_dict overlay',
            ema_tensors=len(ema), raw_state_tensors=len(state),
            strict_load_missing_keys=raw_load.missing_keys + ema_load.missing_keys,
            strict_load_unexpected_keys=raw_load.unexpected_keys + ema_load.unexpected_keys,
        )
    # The official _decode passes raw z directly to decoder with these flags off.
    if not (full.use_tiling is full.use_slicing is full.use_framewise_decoding is False):
        raise ValueError('Benchmark requires tiling, slicing and framewise decoding disabled')
    model = full.decoder.eval().requires_grad_(False)
    metadata = dict(
        config=config, config_sha256=sha256(args.config), config_path=str(args.config.resolve()),
        config_status=('reconstructed locally; no checkpoint checked in random mode; not author-confirmed'
                       if args.random_weights else
                       'reconstructed locally; strict checkpoint geometry match; not author-confirmed'),
        official_missing_config='configs/Turbo-VAED-Wan.json is referenced by wan_decoder_utils.py but absent from the pinned Git tree',
        source_git=git_sha, source_expected_git=EXPECTED_SOURCE_GIT,
        source_sha256=source_sha, source_path=str(source),
        source_repository='https://github.com/hustvl/Turbo-VAED',
        **weights_metadata,
        parameters=sum(p.numel() for p in model.parameters()),
        diffusers=diffusers.__version__, tiling=False, slicing=False, framewise_decoding=False,
        latent_semantics='raw 48-channel Wan2.2 posterior latent; passed unchanged to official decoder',
        input_normalization='none', output_postprocessing='none', compile_mode='none',
    )
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    weights_group = parser.add_mutually_exclusive_group(required=True)
    weights_group.add_argument('--weights', type=Path)
    weights_group.add_argument('--random-weights', action='store_true',
                               help='Initialize the configured architecture without loading a checkpoint')
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'third_party/Turbo-VAED')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/benchmarks/turbo_wan22_reconstructed.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--frames', type=positive_int, default=17)
    parser.add_argument('--height', type=positive_int, default=704)
    parser.add_argument('--width', type=positive_int, default=1280)
    parser.add_argument('--batch', type=positive_int, default=1)
    parser.add_argument('--warmup', type=positive_int, default=5)
    parser.add_argument('--repeats', type=positive_int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--validate-cpu', action='store_true')
    args = parser.parse_args()
    if (args.frames - 1) % 4 or args.height % 16 or args.width % 16:
        parser.error('frames must be 4*n+1; height and width must be divisible by 16')
    if args.output.exists():
        raise FileExistsError(args.output)
    shape = (args.batch, 48, (args.frames - 1) // 4 + 1, args.height // 16, args.width // 16)
    expected = (args.batch, 3, args.frames, args.height, args.width)
    torch.set_num_threads(4)
    model, metadata = build(args)
    if args.validate_cpu:
        # Shape execution only: no CUDA initialization or GPU allocation.
        model.to(device='meta', dtype=torch.bfloat16)
        with torch.inference_mode():
            value = torch.empty(shape, device='meta', dtype=torch.bfloat16)
            prediction = model(value)
        if prediction.shape != expected or prediction.dtype != torch.bfloat16:
            raise ValueError('Turbo decoder output geometry or precision mismatch')
        report = dict(**metadata, validation='CPU construction and meta shape/dtype execution; GPU not run',
                      torch=str(torch.__version__), output_shape=list(prediction.shape),
                      output_dtype=str(prediction.dtype), latent_shape=list(value.shape))
    else:
        device = setup(args.device, args.seed)
        model.to(device=device, dtype=torch.bfloat16)
        if not all(p.dtype == torch.bfloat16 for p in model.parameters()):
            raise ValueError('Turbo decoder weights must all be BF16')
        generator = torch.Generator(device=device).manual_seed(args.seed)
        value = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        latent_sha = hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        prediction, timing = measure(model, value, args.warmup, args.repeats)
        if prediction.shape != expected or prediction.dtype != torch.bfloat16:
            raise ValueError('Turbo decoder output geometry or precision mismatch')
        report = dict(**environment(device), **metadata, decoder='turbo_vaed_wan22', dtype='bf16',
                      output_dtype=str(prediction.dtype), latent_dtype=str(value.dtype),
                      latent_shape=list(value.shape), output_shape=list(prediction.shape),
                      latent_sha256=latent_sha, requested_frames=args.frames, inferred_frames=args.frames,
                      fps=args.batch * args.frames * 1000 / timing['ms'],
                      inferred_fps=args.batch * args.frames * 1000 / timing['ms'], **timing,
                      warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                      protocol='Random latent; eager synchronized wall-clock median. Loading/warmup excluded. Same LightVAE utils.measure. No tiling, frame slicing, normalization, output conversion or compilation. Not a quality evaluation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
