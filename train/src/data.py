"""Manifest-based video/image-folder input with explicit split provenance."""

import json
import random
import io
import zipfile
from pathlib import Path
import av
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_manifest(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    for row in records:
        if not all(k in row for k in ("path", "dataset", "group")):
            raise ValueError("Every record needs path, dataset, group")
        if not Path(row["path"]).exists():
            raise FileNotFoundError(row["path"])
    return records


def assert_disjoint(train, val):
    def groups(rows):
        return {(r["dataset"], r["group"]) for r in rows}

    def paths(rows):
        return {(str(Path(row["path"]).resolve()), member) for row in rows for member in row.get("zip_members", [None])}

    if groups(train) & groups(val) or paths(train) & paths(val):
        raise ValueError("Train/validation overlap: use distinct videos/groups")


def grid_crop_position(h, w, height, width, visit, seed):
    """Visit every overlapping tile once per shuffled cycle, including borders."""
    if h < height or w < width:
        raise ValueError("Native grid crop cannot exceed source image size")

    def positions(size, crop):
        values = list(range(0, size - crop + 1, max(1, crop // 2)))
        return sorted(set(values + [size - crop]))

    tiles = [(y, x) for y in positions(h, height) for x in positions(w, width)]
    cycle, index = divmod(visit, len(tiles))
    random.Random(seed + cycle).shuffle(tiles)
    return tiles[index]


def read_clip(
    record,
    frames,
    height=None,
    width=None,
    seed=0,
    validation=False,
    return_metadata=False,
    native_resolution=False,
    spatial_multiple=16,
    grid_crop=False,
    crop_visit=0,
    crop_seed=2026,
):
    """Resize-to-cover and center crop; temporal random crop only during training."""
    if frames < 1 or (frames - 1) % 4:
        raise ValueError("frames must be 4n+1")
    path = Path(record["path"])
    if "zip_members" in record:
        files = record["zip_members"]
        available = len(files)
        length = min(frames, (available - 1) // 4 * 4 + 1)
        start = 0 if validation else random.Random(seed).randrange(available - length + 1)
        images = []
        with zipfile.ZipFile(path) as archive:
            for member in files[start : start + length]:
                with Image.open(io.BytesIO(archive.read(member))) as image:
                    images.append(np.asarray(image.convert("RGB")).copy())
    elif path.is_dir():
        files = sorted(path.glob("*.jpg")) + sorted(path.glob("*.png"))
        files = sorted(files)
        available = len(files)
        length = min(frames, (available - 1) // 4 * 4 + 1)
        start = 0 if validation else random.Random(seed).randrange(available - length + 1)
        images = []
        for file in files[start : start + length]:
            with Image.open(file) as image:
                images.append(np.asarray(image.convert("RGB")).copy())
    else:
        with av.open(str(path)) as container:
            available = container.streams.video[0].frames
            if not available:
                available = sum(1 for _ in container.decode(video=0))
        length = min(frames, (available - 1) // 4 * 4 + 1)
        start = 0 if validation else random.Random(seed).randrange(available - length + 1)
        images = []
        decoded_count = 0
        with av.open(str(path)) as container:
            container.streams.video[0].codec_context.thread_count = 2
            for i, frame in enumerate(container.decode(video=0)):
                decoded_count = i + 1
                if i >= start:
                    images.append(frame.to_ndarray(format="rgb24"))
                if len(images) == length:
                    break
        if len(images) != length and decoded_count >= length > 0:
            # EOF before the requested endpoint: AVI metadata can overcount.
            # Keep the requested length and resample using actual decoded count.
            start = 0 if validation else random.Random(seed).randrange(decoded_count - length + 1)
            images = []
            with av.open(str(path)) as container:
                container.streams.video[0].codec_context.thread_count = 2
                for i, frame in enumerate(container.decode(video=0)):
                    if i >= start:
                        images.append(frame.to_ndarray(format="rgb24"))
                    if len(images) == length:
                        break
    if not images or len(images) != length:
        raise ValueError(f"Empty/short video: {path}")
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 127.5 - 1
    h, w = x.shape[-2:]
    original_size = [h, w]
    if grid_crop:
        if native_resolution:
            raise ValueError("grid_crop and native_resolution are mutually exclusive")
        top, left = grid_crop_position(h, w, height, width, crop_visit, crop_seed)
        clip = x[:, :, top : top + height, left : left + width].permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        if return_metadata:
            return clip, dict(
                start=start,
                frames=length,
                original_size=original_size,
                resized_size=[h, w],
                top=top,
                left=left,
                height=height,
                width=width,
                seed=seed,
                validation=validation,
            )
        return clip
    if native_resolution:
        if h % spatial_multiple or w % spatial_multiple:
            raise ValueError(f"Native resolution must align to {spatial_multiple}: {path} {h}x{w}")
        clip = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        if return_metadata:
            return clip, dict(
                start=start,
                frames=length,
                original_size=original_size,
                resized_size=[h, w],
                top=0,
                left=0,
                height=h,
                width=w,
                seed=seed,
                validation=validation,
            )
        return clip
    scale = max(height / h, width / w)
    h, w = max(height, round(h * scale)), max(width, round(w * scale))
    x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False, antialias=True)
    top, left = (h - height) // 2, (w - width) // 2
    clip = x[:, :, top : top + height, left : left + width].permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    if return_metadata:
        return clip, dict(
            start=start,
            frames=length,
            original_size=original_size,
            resized_size=[h, w],
            top=top,
            left=left,
            height=height,
            width=width,
            seed=seed,
            validation=validation,
        )
    return clip
