"""Explicit, independently switchable GT reconstruction and teacher terms."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ReconstructionLoss(nn.Module):
    def __init__(self, l1=1.0, perceptual=0.3, teacher_l1=0.0, feature=0.0, temporal=0.0, lpips_frames=8):
        super().__init__()
        self.weights = dict(l1=l1, perceptual=perceptual, teacher_l1=teacher_l1, feature=feature, temporal=temporal)
        if any(v < 0 for v in self.weights.values()) or (
            lpips_frames != "all" and (type(lpips_frames) is not int or lpips_frames < 1)
        ):
            raise ValueError("Loss weights must be nonnegative; lpips_frames is all or a positive integer")
        self.lpips_frames = lpips_frames
        self.perceptual = None
        if perceptual:
            import lpips

            self.perceptual = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def forward(self, prediction, target, teacher=None, features=None, teacher_features=None):
        if prediction.shape != target.shape:
            raise ValueError("Prediction and GT geometry differ")
        x, y = prediction.float(), target.float()
        terms = {"l1": (x - y).abs().mean()}
        if self.weights["perceptual"]:
            # Deterministic temporal coverage; inputs are [-1,1], never double-normalized.
            count = x.shape[2] if self.lpips_frames == "all" else min(self.lpips_frames, x.shape[2])
            indices = torch.linspace(0, x.shape[2] - 1, count, device=x.device).round().long()

            def metric(a, b):
                with torch.autocast(device_type=a.device.type, enabled=False):
                    return self.perceptual(a.float(), b.float()).mean()

            terms["perceptual"] = sum(
                checkpoint(metric, x[:, :, i], y[:, :, i], use_reentrant=False) for i in indices
            ) / len(indices)
        if self.weights["teacher_l1"]:
            if teacher is None:
                raise ValueError("teacher_l1 requires teacher output")
            terms["teacher_l1"] = (x - teacher.detach().float()).abs().mean()
        if self.weights["feature"]:
            if not features or not teacher_features or len(features) != len(teacher_features):
                raise ValueError("Matching stage features required")
            if not isinstance(features, dict) or not isinstance(teacher_features, dict):
                raise ValueError("Feature distillation requires a dictionary of complete stage tensors")
            if features.keys() != teacher_features.keys():
                raise ValueError("Student/teacher stage keys differ")
            if any(a.shape != teacher_features[k].shape for k, a in features.items()):
                raise ValueError("Feature shapes differ; projection is not implicit")
            terms["feature"] = sum(
                (a.float() - teacher_features[k].detach().float()).square().mean() for k, a in features.items()
            )
        if self.weights["temporal"]:
            terms["temporal"] = (x.diff(dim=2) - y.diff(dim=2)).abs().mean() if x.shape[2] > 1 else x.sum() * 0
        return sum(self.weights[k] * v for k, v in terms.items()), {k: v.detach() for k, v in terms.items()}
