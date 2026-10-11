"""Shared model helpers, checkpoint loading, video I/O, geometry and CUDA timing utilities."""

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
from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
FAMILIES = ("wan21", "wan22", "h3")
DEFAULT_WEIGHTS_DIR = Path(os.environ.get(
    "LIGHTVAE_MODEL_DIR", ROOT / "weights/LightVAE"))
VARIANTS = {"wan21": ("pro", "lite"), "wan22": ("pro", "lite"), "h3": ("lite",)}
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


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
        config = ROOT / "config/h3-base.json" if model == "h3" else None
        return model, config, checkpoint
    if component == "encoder":
        # H3 Lite encoder depth is stored in the checkpoint metadata.
        return "h3_lite", None, checkpoint
    config_name = "h3-lightvae.json" if model == "h3" else f"lightvae-{variant}-{model}.json"
    return ("h3_lite" if model == "h3" else f"{model}_lightvae"), ROOT / "config" / config_name, checkpoint


def decoder_spec(model, variant, weights_dir, base_weights):
    """Resolve the benchmark's baseline or conventionally named decoder weights."""
    if variant == "base":
        return component_spec(model, variant, "decoder", base_weights)
    checkpoint = Path(weights_dir) / weight_name(model, variant)
    return component_spec(model, variant, "decoder", checkpoint)


MODEL_MODULES = {
    "encoder": {
        "wan21": "base_vae.wan21_vae_encoder",
        "wan22": "base_vae.wan22_vae_encoder",
        "h3": "base_vae.h3_vae_encoder",
        "h3_lite": "lightvae.h3_lightvae_encoder",
    },
    "decoder": {
        "wan21": "base_vae.wan21_vae_decoder",
        "wan22": "base_vae.wan22_vae_decoder",
        "h3": "base_vae.h3_vae_decoder",
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
    if decoder in WAN_LIGHTVAE_DECODERS and dtype != 'bf16':
        raise ValueError('LightVAE requires --dtype bf16')


def padded_frames(family, frames):
    if frames < 1:
        raise ValueError('Frame count must be positive')
    if family == 'h3':
        return max(5, frames + (5 - frames) % 17)
    return frames + (1 - frames) % 4


def latent_frames(family, frames):
    if frames != padded_frames(family, frames):
        raise ValueError('Pixel frame count does not match the encoder temporal layout')
    if family == 'h3':
        return (frames - 5) // 17 * 5 + 2
    return (frames - 1) // 4 + 1


def h3_decoded_frames(frames):
    if frames < 2 or (frames - 2) % 5:
        raise ValueError("H3 latent frames must have the form 5*n+2")
    return (frames - 2) // 5 * 17 + 5


def split_tiles(length, size, overlap):
    """Return H3 latent-axis slices and overlaps measured in pixel space."""
    if size < 32 or size % 32 or overlap < 0 or overlap % 16 or overlap >= size:
        raise ValueError("H3 tiles need size divisible by 32 and smaller overlap divisible by 16")
    if length <= size:
        return [slice(0, length // 16)], []
    count = math.ceil(length / size)
    while count * size - (count - 1) * overlap < length:
        count += 1
    overlaps = [overlap] * (count - 1)
    excess = count * size - sum(overlaps) - length
    for index in range(excess // 16):
        overlaps[index % (count - 1)] += 16
    starts = [0]
    for extent in overlaps:
        starts.append(starts[-1] + size - extent)
    return [slice(start // 16, (start + size) // 16) for start in starts], overlaps


def blend(left, right, extent, dim):
    extent = min(extent, left.shape[dim], right.shape[dim])
    if not extent:
        return right
    shape = [1] * right.ndim
    shape[dim] = extent
    alpha = (torch.arange(extent, device=right.device, dtype=right.dtype) / extent).view(shape)
    mixed = left.narrow(dim, left.shape[dim] - extent, extent) * (1 - alpha)
    mixed = mixed + right.narrow(dim, 0, extent) * alpha
    return torch.cat((mixed, right.narrow(dim, extent, right.shape[dim] - extent)), dim=dim)


def stitch_tiles(rows, height_overlaps, width_overlaps):
    stitched = []
    for row_index, row in enumerate(rows):
        tiles = []
        for column_index, tile in enumerate(row):
            if row_index:
                tile = blend(rows[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2)
            if column_index:
                tile = blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
            if row_index < len(rows) - 1 and height_overlaps[row_index]:
                tile = tile[..., :-height_overlaps[row_index], :]
            if column_index < len(row) - 1 and width_overlaps[column_index]:
                tile = tile[..., :-width_overlaps[column_index]]
            tiles.append(tile)
        stitched.append(torch.cat(tiles, dim=-1))
    return torch.cat(stitched, dim=-2)


class H3Module(nn.Module):
    """Keep H3's diffusion normalization constants in FP32."""

    def _apply(self, fn, recurse=True):
        mean, std = self.latents_mean, self.latents_std
        super()._apply(fn, recurse)
        self.latents_mean = mean.to(device=self.latents_mean.device)
        self.latents_std = std.to(device=self.latents_std.device)
        return self


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


def frames(x):
    return x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[-2:])


def unframes(x, batch, time):
    return x.reshape(batch, time, *x.shape[1:]).permute(0, 2, 1, 3, 4)


def architecture(config):
    """Extract the architecture fields used for checkpoint compatibility."""
    fields = ("path", "shape", "padding", "rank", "bias")
    records = [{key: record[key] for key in fields} for record in config["packed_records"]]
    return {"family": config["family"], "kept": config["kept"],
            "packed_records": sorted(records, key=lambda record: record["path"])}


def replace_norms_and_upsamples(model, native, fused_norm_cls):
    for name, layer in list(model.named_modules()):
        if isinstance(layer, native.RMS_norm):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            silu = (isinstance(parent, nn.Sequential) and child.isdigit()
                    and int(child) + 1 < len(parent)
                    and isinstance(parent[int(child) + 1], nn.SiLU))
            model.set_submodule(name, fused_norm_cls(layer, silu))
            if silu:
                parent[int(child) + 1] = nn.Identity()
        elif isinstance(layer, native.Upsample):
            model.set_submodule(name, nn.Upsample(
                size=layer.size, scale_factor=layer.scale_factor, mode=layer.mode,
                align_corners=layer.align_corners,
                recompute_scale_factor=layer.recompute_scale_factor,
            ))


def load_lightvae_checkpoint(path):
    """Read Wan LightVAE tensors and their embedded architecture metadata."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        saved = json.loads(handle.metadata()["checkpoint_metadata"])
        if not isinstance(saved, dict) or not {"family", "kept", "packed_records"} <= saved.keys():
            raise ValueError("Missing LightVAE architecture: family, kept, packed_records")
        saved["model"] = {key: handle.get_tensor(key) for key in handle.keys()}
    return saved


def load_component(module, checkpoint, prefixes):
    path = Path(checkpoint)
    if path.suffix == ".safetensors":
        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    component = {key: value for key, value in state.items() if key.startswith(prefixes)}
    module.load_state_dict(component, strict=True, assign=True)


def h3_checkpoint_config(path, component):
    """Read a VAE directory config or the architecture embedded in an H3 export."""
    if path.is_dir():
        return json.loads((path / "config.json").read_text())
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        architecture = json.loads(handle.metadata()["architecture"])
    if "teacher_config" not in architecture:
        return architecture
    config = dict(architecture["teacher_config"])
    key = "kept_layers" if component == "decoder" else "kept_residual_indices"
    config[key] = architecture[key]
    config["tile_size"] = architecture["tile_sample_min_height"]
    config["tile_overlap"] = architecture["tile_sample_min_overlap_height"]
    return config


def h3_model_config(config, checkpoint, component):
    result = json.loads((ROOT / "config/h3-base.json").read_text())
    weights = Path(checkpoint) if checkpoint is not None else None
    stored = h3_checkpoint_config(weights, component) if weights is not None else {}
    keys = result.keys() | {"kept_layers", "kept_residual_indices"}
    stored = {key: value for key, value in stored.items() if key in keys}
    supplied = {key: value for key, value in (config or {}).items() if key in keys}
    for key in (supplied.keys() & stored.keys()) - {"tile_size", "tile_overlap"}:
        if supplied[key] != stored[key]:
            raise ValueError(f"H3 config disagrees with checkpoint: {key}")
    result.update(stored)
    result.update(supplied)
    if result["clip_length"] != 17 or result["token_drop"] != 3:
        raise ValueError("H3 inference uses clip_length=17 and token_drop=3")
    if result["latent_channels"] != len(result["latents_mean"]) or len(result["latents_std"]) != result["latent_channels"]:
        raise ValueError("H3 latent normalization must match latent_channels")
    if min(result["latents_std"]) <= 0:
        raise ValueError("H3 latent standard deviations must be positive")
    return result, weights


def load_h3_component(model, path):
    """Read component tensors from a file or a (possibly sharded) VAE directory."""
    if path.is_dir():
        index = path / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            files = {name: path / shard for name, shard in json.loads(index.read_text())["weight_map"].items()}
        else:
            path = path / "diffusion_pytorch_model.safetensors"
    if not path.is_dir():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            files = {name: path for name in handle.keys()}
    state = model.state_dict()
    stats = {"latents_mean", "latents_std"}
    prefixes = ("decoder.", "post_quant_conv.") if "decoder.proj_in.weight" in state else ("encoder.", "quant_conv.")
    loaded = {name: state[name] for name in stats}
    grouped = {}
    for name, shard in files.items():
        if name.startswith(prefixes):
            grouped.setdefault(shard, []).append(name)
    for shard, names in grouped.items():
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in names:
                loaded[name] = handle.get_tensor(name)
    model.load_state_dict(loaded, strict=True, assign=True)


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
