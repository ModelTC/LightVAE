"""Wan2.1/2.2 probe-guided recovery with an optional conditional GAN."""

import torch

from ..models.wan22 import suffix as wan22_suffix
from .wan_objective import GPUReconstruction, guided_feature


class WanRecovery:
    def __init__(self, weights, gan, device, family, accumulation_steps=1):
        ratio = 8 if family == "wan21" else 16
        self.anchors = [1, 4, 7, 10, 13] if family == "wan21" else [0, 1, 2, 3, 4]
        self.suffix_fn = None if family == "wan21" else wan22_suffix
        self.cfg = {
            "gan_recipe": gan,
            "latent_channels": 16 if family == "wan21" else 48,
            "accumulation_steps": accumulation_steps,
            "latent_spatial_ratio": ratio,
            "lpips_weight": weights["perceptual"],
            "lpips_frames": weights["lpips_frames"],
            "feature_weight": weights["feature"],
            "gan_weight": weights["gan"] if gan["enabled"] else 0,
            "gan_start": gan["start_step"],
            "gan_frames": gan["frames"],
            "gan_crop_size": gan["crop_size"],
            "explicit_gan_geometry": True,
            "guidance": weights["guidance"],
        }
        self.loss = GPUReconstruction(self.cfg, device)

    def anchor(self, step):
        return self.anchors[(step - 1) % len(self.anchors)] if self.cfg["feature_weight"] else None

    def __call__(self, prediction, x, z, step, seed, features, teacher_features, teacher_rgb, teacher):
        if self.cfg["gan_weight"] and self.loss.gan is None and step >= self.cfg["gan_start"]:
            self.loss.start_gan()
        sizes = [1] + [4] * ((prediction.shape[2] - 1) // 4)
        loss, stats = self.loss(list(prediction.split(sizes, dim=2)), x, z, step, sample_seed=seed)
        if self.cfg["feature_weight"]:
            feature, auxiliary = guided_feature(
                features,
                teacher_features,
                teacher.component,
                list(teacher_rgb.split(sizes, dim=2)),
                self.anchor(step),
                self.cfg["guidance"],
                suffix_fn=self.suffix_fn,
            )
            loss = loss + self.cfg["feature_weight"] * feature
            stats.update(auxiliary, guided_feature=feature.detach())
        return loss, {key: value.detach() if torch.is_tensor(value) else value for key, value in stats.items()}

    def step(self):
        self.loss.step()

    def state_dict(self):
        return self.loss.gan.state_dict() if self.loss.gan is not None else None
