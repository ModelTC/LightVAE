"""Portable inference/warm-start weights; never silently load partial states."""

import hashlib
from pathlib import Path
import torch
from .models import temporal, wan22
from .models.wan21 import WanPrunedDecoder


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_student(family, kept, records=None):
    if family not in ("wan21", "wan22"):
        raise ValueError(f"Unknown family: {family}")
    # No teacher weights or original repository needed to instantiate a student.
    with torch.device("meta"):
        model = WanPrunedDecoder(kept_residual_indices=kept) if family == "wan21" else wan22.Component(kept=kept)
        if records is not None:
            (temporal if family == "wan21" else wan22).convert(model, records=records)
    return model


def load_student(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    family = saved["family"]
    model = build_student(family, saved["kept"], saved["packed_records"])
    model.load_state_dict(saved["model"], strict=True, assign=True)
    return model.to(device), saved


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
