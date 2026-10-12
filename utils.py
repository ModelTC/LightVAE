"""Standalone CLI routing, video I/O and CUDA timing utilities."""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import av
import numpy as np
import torch
from torch.nn import functional as F

from lightvae.utils import CONFIG_DIR, padded_frames

ROOT = Path(__file__).resolve().parent
FAMILIES = ("wan21", "wan22", "h3")
DEFAULT_WEIGHTS_DIR = Path(os.environ.get(
    "LIGHTVAE_MODEL_DIR", ROOT / "weights/LightVAE"))
VARIANTS = {"wan21": ("pro", "lite"), "wan22": ("pro", "lite"), "h3": ("lite",)}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def weight_name(model, variant="lite", component="decoder"):
    if model not in FAMILIES or variant not in VARIANTS[model]:
        raise ValueError(f"Unsupported model/variant: {model}/{variant}")
    if component != "decoder" and (model, variant, component) != ("h3", "lite", "encoder"):
        raise ValueError("Only H3 has a separately exported LightVAE encoder")
    return f"lightvae-{variant}-{model}-{component}.safetensors"


def component_spec(model, variant, component, checkpoint):
    """Resolve an explicit encoder/decoder checkpoint without loading CUDA."""
    if variant != "base":
        weight_name(model, variant, component)  # Validate the architecture combination.
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing {model} {component} checkpoint: {checkpoint}")
    if not checkpoint.is_file() and not (model == "h3" and variant == "base" and checkpoint.is_dir()):
        raise ValueError(f"{model}/{variant} --{component} must be a checkpoint file")
    if variant == "base":
        config = CONFIG_DIR / "h3-base.json" if model == "h3" else None
        return model, config, checkpoint
    if component == "encoder":
        # H3 Lite encoder depth is stored in the checkpoint metadata.
        return "h3_lite", None, checkpoint
    config_name = "h3-lightvae.json" if model == "h3" else f"lightvae-{variant}-{model}.json"
    return ("h3_lite" if model == "h3" else f"{model}_lightvae"), CONFIG_DIR / config_name, checkpoint


def decoder_spec(model, variant, weights_dir, base_weights):
    """Resolve the benchmark's baseline or conventionally named decoder weights."""
    if variant == "base":
        return component_spec(model, variant, "decoder", base_weights)
    checkpoint = Path(weights_dir) / weight_name(model, variant)
    return component_spec(model, variant, "decoder", checkpoint)


MODEL_MODULES = {
    "encoder": {
        "wan21": "lightvae.base_vae.wan21_vae_encoder",
        "wan22": "lightvae.base_vae.wan22_vae_encoder",
        "h3": "lightvae.base_vae.h3_vae_encoder",
        "h3_lite": "lightvae.h3_lightvae_encoder",
    },
    "decoder": {
        "wan21": "lightvae.base_vae.wan21_vae_decoder",
        "wan22": "lightvae.base_vae.wan22_vae_decoder",
        "h3": "lightvae.base_vae.h3_vae_decoder",
        "wan21_lightvae": "lightvae.wan21_lightvae_decoder",
        "wan22_lightvae": "lightvae.wan22_lightvae_decoder",
        "h3_lite": "lightvae.h3_lightvae_decoder",
    },
}
DECODERS = tuple(MODEL_MODULES["decoder"])
WAN_LIGHTVAE_DECODERS = ("wan21_lightvae", "wan22_lightvae")


def read_config(path):
    return json.loads(Path(path).read_text()) if path else None


def build_model(kind, name, config=None, checkpoint=None):
    module = importlib.import_module(MODEL_MODULES[kind][name])
    cls = module.Encoder if kind == 'encoder' else module.Decoder
    return cls(config=config, checkpoint=checkpoint).eval().requires_grad_(False)


def check_pair(encoder, decoder):
    if (encoder.family, encoder.latent_channels, encoder.spatial_scale) != (
            decoder.family, decoder.latent_channels, decoder.spatial_scale):
        raise ValueError(f'Incompatible latent spaces: {encoder.family} encoder / {decoder.family} decoder')


def setup(device, seed):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return device


def check_precision(decoder, dtype):
    if dtype not in DTYPES:
        raise ValueError(f'Unsupported precision: {dtype}')


def measure(model, value, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(value)
            del output
        torch.cuda.synchronize(value.device)
        torch.cuda.reset_peak_memory_stats(value.device)
        samples = []
        for index in range(repeats):
            torch.cuda.synchronize(value.device)
            start = time.perf_counter()
            output = model(value)
            torch.cuda.synchronize(value.device)
            samples.append((time.perf_counter() - start) * 1000)
            if index != repeats - 1:
                del output
        peaks = dict(peak_allocated_gib=torch.cuda.max_memory_allocated(value.device) / 2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved(value.device) / 2**30)
        if not torch.isfinite(output).all():
            raise FloatingPointError('Non-finite model output')
        return output, dict(ms=statistics.median(samples), samples_ms=samples, **peaks)


def environment(device):
    return dict(gpu=torch.cuda.get_device_name(device), torch=str(torch.__version__),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                mode='eager', tf32=False)


def read_video(path, max_frames=None):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = stream.average_rate
        if fps is None or fps <= 0:
            raise ValueError('Input video has no valid frame rate')
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format='rgb24'))
            if max_frames is not None and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError('Input video has no decodable frames')
    return np.stack(frames), fps


def prepare_video(frames, family, spatial_scale, height=None, width=None):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1
    if height is not None:
        h, w = x.shape[-2:]
        scale = max(height / h, width / w)
        resized = (max(height, round(h*scale)), max(width, round(w*scale)))
        x = F.interpolate(x, resized, mode='bilinear', align_corners=False, antialias=True)
        top, left = (resized[0]-height)//2, (resized[1]-width)//2
        x = x[:, :, top:top+height, left:left+width]
    original = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    t, h, w = original.shape[-3:]
    padding = (0, (-w) % spatial_scale, 0, (-h) % spatial_scale,
               0, padded_frames(family, t)-t)
    return original, F.pad(original, padding, mode='replicate')


def psnr(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError('PSNR inputs must have matching unpadded video shapes')
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Non-finite reconstruction')
    squared_error = 0.
    for frame in range(target.shape[2]):
        x = prediction[:, :, frame].float().clamp(-1, 1).double()
        y = target[:, :, frame].float().clamp(-1, 1).double()
        squared_error += float(((x-y)/2).square().sum())
    mse = squared_error / target.numel()
    return -10 * math.log10(max(mse, 1.e-12))


def write_video(path, prediction, fps):
    _, _, frames, height, width = prediction.shape
    with Path(path).open('xb') as output, av.open(output, 'w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.height, stream.width = height, width
        stream.pix_fmt = 'yuv420p' if height % 2 == width % 2 == 0 else 'yuv444p'
        stream.options = {'crf': '18', 'preset': 'fast'}
        for i in range(frames):
            rgb = ((prediction[0, :, i].float().clamp(-1, 1)+1)*127.5).round().byte()
            frame = av.VideoFrame.from_ndarray(rgb.permute(1, 2, 0).numpy(), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
