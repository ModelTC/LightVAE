"""Probe-guided matching and common perceptual/adversarial loss components."""

import copy
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .adversarial import VAEAdversarialObjective


def mean_distance(xs, ys, squared=True):
    if len(xs) != len(ys) or any(x.shape != y.shape for x, y in zip(xs, ys)):
        raise ValueError("Mismatching chunk shapes")
    count = sum(x.numel() for x in xs)

    def distance(x, y):
        delta = x.float() - y.float()
        return (delta.square() if squared else delta.abs()).sum() / count

    return sum(
        checkpoint(distance, x, y, use_reentrant=False) if torch.is_grad_enabled() else distance(x, y)
        for x, y in zip(xs, ys)
    )


@torch.no_grad()
def guidance_weights(delta, grad, epsilon, max_weight, feature_elements):
    """Match the probe gradient with detached nonnegative diagonal weights.

    For mean(w * delta²), the feature gradient is 2 * w * delta / N.
    Preserve that scale; do not renormalize weights to mean one.
    """
    delta = delta.float()
    grad = grad.float() * (feature_elements / 2)
    squared = delta.square()
    ridge = epsilon * squared.mean().clamp_min(1e-12)
    weights = (delta * grad).clamp_min(0) / (squared + ridge)
    return weights.clamp_max(max_weight)


class Reconstruction:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.perceptual = None
        if cfg.get("lpips_weight", 1):
            import lpips

            self.perceptual = lpips.LPIPS(net="vgg").eval().requires_grad_(False).to(self.device)
        self.gan = None

    def start_gan(self):
        source = self.cfg["gan_recipe"]
        recipe = copy.deepcopy(source) if isinstance(source, dict) else json.loads(Path(source).read_text())["gan"]
        if self.cfg.get("explicit_gan_geometry", False):
            recipe["crop_specs"] = [
                {"crop_size": self.cfg["gan_crop_size"], "max_frames": self.cfg["gan_frames"], "weight": 1.0}
            ]
        recipe["spatial_compression_ratio"] = self.cfg.get("latent_spatial_ratio", 8)
        self.gan = VAEAdversarialObjective(
            recipe,
            device=self.device,
            latent_channels=self.cfg.get("latent_channels", 16),
            gradient_accumulation_iters=self.cfg.get("accumulation_steps", 1),
        )

    def step(self):
        if self.gan is not None:
            self.gan.step()
