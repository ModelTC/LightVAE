"""Shared checkpoint, geometry and normalization helpers for LightVAE models."""

import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file

CONFIG_DIR = Path(__file__).resolve().parent / "config"


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
    result = json.loads((CONFIG_DIR / "h3-base.json").read_text())
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
