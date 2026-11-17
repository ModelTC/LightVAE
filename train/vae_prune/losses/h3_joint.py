"""H3 joint recovery with Wan's diagonal guidance and conditional GAN recipe."""

import json
import torch
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from safetensors import safe_open
from pathlib import Path
from .guidance import Reconstruction, guidance_weights
from ..models.h3_native.minimax_h3.pruned_vae import MiniMaxH3PrunedVideoVAE, decode_teacher_suffix
from ..models.h3_native.minimax_h3.pruned_encoder import MiniMaxH3PrunedVideoEncoder


def load_teacher_encoder(directory, config):
    directory = Path(directory)
    if (directory / "vae").is_dir():
        directory = directory / "vae"
    actual = json.loads((directory / "config.json").read_text())
    for key, value in config.items():
        if not key.startswith("_") and actual.get(key) != value:
            raise ValueError(f"Teacher config mismatch: {key}")
    teacher = MiniMaxH3PrunedVideoEncoder(config)  # All original residual blocks.
    weights = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(("encoder.", "quant_conv.")):
                    weights[key] = f.get_tensor(key)
    teacher.initialize_from_teacher_state(weights)
    return teacher.eval().requires_grad_(False)


def encoder_distillation(student_moments, teacher_moments, student_features, teacher_features):
    """Element-mean KL(q_student || q_teacher), plus equal-weight stage MSE."""
    if student_moments.shape != teacher_moments.shape:
        raise ValueError("Encoder posterior shapes differ")
    sm, sl = student_moments.float().chunk(2, dim=1)
    tm, tl = teacher_moments.detach().float().chunk(2, dim=1)
    sl, tl = sl.clamp(-30, 20), tl.clamp(-30, 20)
    kl = 0.5 * (tl - sl + torch.exp(sl - tl) + (sm - tm).square() * torch.exp(-tl) - 1)
    if not student_features or len(student_features) != len(teacher_features):
        raise ValueError("Encoder stage counts differ or are empty")
    for s, t in zip(student_features, teacher_features):
        if s.shape != t.shape:
            raise ValueError("Encoder stage feature shapes differ")
    feature = sum(
        (s.float() - t.detach().float()).square().mean() for s, t in zip(student_features, teacher_features)
    ) / len(student_features)
    return kl.mean(), feature


def load_teacher_decoder(directory, config):
    directory = Path(directory)
    if (directory / "vae").is_dir():
        directory = directory / "vae"
    actual = json.loads((directory / "config.json").read_text())
    for key, value in config.items():
        if not key.startswith("_") and actual.get(key) != value:
            raise ValueError(f"Teacher config mismatch: {key}")
    teacher = MiniMaxH3PrunedVideoVAE(config)
    weights = {}
    for path in sorted(directory.glob("*.safetensors")):
        weights.update(load_file(str(path)))
    if not weights:
        raise FileNotFoundError(f"No teacher safetensors: {directory}")
    teacher.initialize_from_teacher_state(weights)
    return teacher.eval().requires_grad_(False)


class H3JointLoss(Reconstruction):
    def __init__(self, teacher, loss_config, gan_config, device, accumulation, encoder_teacher=None):
        super().__init__(
            dict(
                gan_recipe=gan_config,
                lpips_weight=loss_config["lpips"],
                latent_spatial_ratio=16,
                latent_channels=teacher.post_quant_conv.in_channels,
                accumulation_steps=accumulation,
            ),
            device,
        )
        self.teacher = teacher
        self.loss_config = loss_config
        self.feature_weight = loss_config["guided_feature"]
        self.encoder_teacher = encoder_teacher
        self.encoder_kl_weight = loss_config["encoder_kl"]
        self.encoder_feature_weight = loss_config["encoder_feature"]
        self.gan_start = gan_config["start_step"]

    def __call__(self, outputs, target, teacher_anchor, step):
        prediction, z, feature, window = outputs[:4]
        l1 = (prediction.float() - target).abs().mean()

        def perceptual(a, b):
            with torch.autocast("cuda", enabled=False):
                return self.perceptual(a.float(), b.float()).mean()

        lp = l1.new_zeros(())
        if self.loss_config["lpips"]:
            lp = (
                sum(
                    checkpoint(perceptual, prediction[:, :, i], target[:, :, i], use_reentrant=False)
                    for i in range(target.shape[2])
                )
                / target.shape[2]
            )
        guided = probe = weight = l1.new_zeros(())
        if self.feature_weight:
            with torch.no_grad():
                teacher_rgb, _, teacher_feature = self.teacher(window.detach(), auxiliary_feature_index=teacher_anchor)
            proxy = feature.detach().requires_grad_(True)
            probe_rgb = decode_teacher_suffix(self.teacher.decoder, proxy, tuple(window.shape[2:]), teacher_anchor)
            probe = (probe_rgb.float() - teacher_rgb.float()).square().mean()
            (gradient,) = torch.autograd.grad(probe, proxy)
            delta = feature.detach().float() - teacher_feature.float()
            weight = guidance_weights(delta, gradient, 0.01, 10.0, feature.numel())
            guided = (weight * (feature.float() - teacher_feature.float()).square()).mean()
        adv = l1.new_zeros(())
        stats = {}
        if self.loss_config["gan"] and step >= self.gan_start:
            if self.gan is None:
                self.start_gan()
            # H3: each five latent tokens map to 1+4+4+4+4 RGB frames.
            tokens = torch.arange(z.shape[2], device=z.device)
            indices = tokens.repeat_interleave(torch.where(tokens % 5 == 0, 1, 4))[: target.shape[2]]
            condition = z.detach().index_select(2, indices).float()
            with torch.autocast("cuda", enabled=False):
                adv, stats = self.gan((prediction.float() + 1) / 2, (target + 1) / 2, condition, step - self.gan_start)
        encoder_kl = encoder_feature = l1.new_zeros(())
        if self.encoder_kl_weight or self.encoder_feature_weight:
            pixels, moments, features = outputs[4]
            with torch.no_grad():
                teacher_moments, teacher_features = self.encoder_teacher(pixels.detach(), return_features=True)
            encoder_kl, encoder_feature = encoder_distillation(moments, teacher_moments, features, teacher_features)
        total = (
            self.loss_config["l1"] * l1
            + self.loss_config["lpips"] * lp
            + self.feature_weight * guided
            + self.loss_config["gan"] * adv
            + self.encoder_kl_weight * encoder_kl
            + self.encoder_feature_weight * encoder_feature
        )
        return total, dict(
            l1=l1.detach(),
            lpips=lp.detach(),
            guided_feature=guided.detach(),
            encoder_kl=encoder_kl.detach(),
            encoder_feature=encoder_feature.detach(),
            encoder_kl_weight=self.encoder_kl_weight,
            encoder_feature_weight=self.encoder_feature_weight,
            probe=probe.detach(),
            mean_feature_weight=weight.mean(),
            gan=adv.detach(),
            **stats,
        )
