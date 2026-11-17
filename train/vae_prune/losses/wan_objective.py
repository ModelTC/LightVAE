"""Streaming Wan reconstruction and teacher-suffix probe loss."""

import random

import torch
from torch.utils.checkpoint import checkpoint

from ..models.streaming import layers
from ..models.temporal import apply
from .guidance import Reconstruction, guidance_weights, mean_distance


def suffix(component, features, anchor):
    sequence = layers(component)
    cut = next(i for i, (_, index) in enumerate(sequence) if index == anchor)
    cache = [None] * 256
    outputs = []

    def run(x, *history):
        history, slot = list(history), [0]
        for layer, _ in sequence[cut + 1 :]:
            x = apply(layer, x, history, slot)
        return x, *history

    for feature in features:
        output, *cache = checkpoint(run, feature, *cache, use_reentrant=False)
        outputs.append(output)
    return outputs


def guided_feature(student, teacher, teacher_decoder, teacher_rgb, anchor, cfg, suffix_fn=None):
    proxy = [feature.detach().requires_grad_(True) for feature in student]
    outputs = (suffix if suffix_fn is None else suffix_fn)(teacher_decoder, proxy, anchor)
    probe = mean_distance(outputs, teacher_rgb)
    gradients = torch.autograd.grad(probe, proxy)
    probe_value = probe.detach()
    del outputs, probe, proxy
    count = sum(feature.numel() for feature in student)
    terms = []
    statistics = student[0].new_zeros(3, dtype=torch.float32)
    for feature, target, gradient in zip(student, teacher, gradients):
        delta = feature.detach().float() - target.float()
        weight = guidance_weights(delta, gradient, cfg["epsilon"], cfg["max_weight"], count)
        statistics += torch.stack((delta.square().sum(), (delta * gradient > 0).sum(), weight.sum())) / count

        def loss(a, b, weights):
            return (weights * (a.float() - b.float()).square()).sum() / count

        terms.append(checkpoint(loss, feature, target, weight, use_reentrant=False))
    return sum(terms), {
        "feature": statistics[0],
        "probe": probe_value,
        "compatible_fraction": statistics[1],
        "mean_feature_weight": statistics[2],
        "anchor": float(anchor),
    }


class GPUReconstruction(Reconstruction):
    def perceptual_frame(self, prediction, target):
        with torch.autocast(self.device.type, enabled=False):
            return self.perceptual(prediction.float(), target.float()).mean()

    def __call__(self, chunks, pixels, z, step, sample_seed):
        targets = list(pixels.split([chunk.shape[2] for chunk in chunks], dim=2))
        count = pixels.numel()

        def l1(a, b):
            return (a.float() - b.float()).abs().sum() / count

        l1_loss = sum(checkpoint(l1, a, b, use_reentrant=False) for a, b in zip(chunks, targets))
        frames = [chunk[:, :, i] for chunk in chunks for i in range(chunk.shape[2])]
        perceptual = l1_loss.new_zeros(())
        if self.perceptual is not None:
            requested = self.cfg["lpips_frames"]
            indices = (
                range(len(frames))
                if requested == "all"
                else random.Random(sample_seed).sample(range(len(frames)), min(requested, len(frames)))
            )
            perceptual = sum(
                checkpoint(self.perceptual_frame, frames[i], pixels[:, :, i], use_reentrant=False) for i in indices
            ) / len(indices)
        adversarial = l1_loss.new_zeros(())
        stats = {}
        if self.gan is not None:
            rng = random.Random(sample_seed)
            length = min(self.cfg["gan_frames"], len(frames))
            start = rng.randrange(len(frames) - length + 1)
            crop_h = min(self.cfg["gan_crop_size"], pixels.shape[-2])
            crop_w = min(self.cfg["gan_crop_size"], pixels.shape[-1])
            ratio = self.cfg["latent_spatial_ratio"]
            top = ratio * rng.randrange((pixels.shape[-2] - crop_h) // ratio + 1)
            left = ratio * rng.randrange((pixels.shape[-1] - crop_w) // ratio + 1)
            fake = torch.stack(
                [frames[i][..., top : top + crop_h, left : left + crop_w] for i in range(start, start + length)], dim=2
            )
            real = pixels[:, :, start : start + length, top : top + crop_h, left : left + crop_w]
            temporal = [0 if i == 0 else (i + 3) // 4 for i in range(start, start + length)]
            condition = z[
                :, :, temporal, top // ratio : (top + crop_h) // ratio, left // ratio : (left + crop_w) // ratio
            ]
            with torch.autocast(self.device.type, enabled=False):
                adversarial, stats = self.gan(
                    (fake.float() + 1) / 2, (real.float() + 1) / 2, condition, step - self.cfg["gan_start"]
                )
        total = l1_loss + self.cfg["lpips_weight"] * perceptual + self.cfg["gan_weight"] * adversarial
        return total, {"l1": l1_loss.detach(), "lpips": perceptual.detach(), "gan": adversarial.detach(), **stats}
