"""Reconstruction metrics and unpadded distributed teacher/student comparison."""

import torch
from torch import nn
from torch.nn import functional as F

from .data import read_clip
from .distributed_eval import gather_rows, local_indices, summarize


def frame_ssim(x, y):
    """RGB NCHW [0,1], valid 11×11 Gaussian window, population covariance."""
    if min(x.shape[-2:]) < 11:
        raise ValueError("SSIM requires dimensions >= 11")
    kernel = torch.arange(11, device=x.device, dtype=torch.float32) - 5
    kernel = torch.exp(-kernel.square() / (2 * 1.5**2))
    kernel /= kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11).contiguous()

    def average(value):
        return F.conv2d(value, window, groups=3)

    ux, uy = average(x), average(y)
    vx = average(x * x) - ux * ux
    vy = average(y * y) - uy * uy
    covariance = average(x * y) - ux * uy
    return (
        ((2 * ux * uy + 0.01**2) * (2 * covariance + 0.03**2)) / ((ux * ux + uy * uy + 0.01**2) * (vx + vy + 0.03**2))
    ).mean()


class VideoMetrics(nn.Module):
    def __init__(self, net="alex", lpips_enabled=True):
        super().__init__()
        self.lpips_key = "lpips_" + net
        self.lpips = None
        if lpips_enabled:
            import lpips

            self.lpips = lpips.LPIPS(net=net).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(f"Output geometry differs: {prediction.shape} vs {target.shape}")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite evaluation prediction")
        x, y = prediction.float().clamp(-1, 1), target.float().clamp(-1, 1)
        mse = ((x - y) / 2).square().mean()
        scores = {"psnr": float(-10 * mse.clamp_min(1e-12).log10())}
        frames = x.shape[2]
        scores["ssim"] = (
            sum(float(frame_ssim((x[:, :, i] + 1) / 2, (y[:, :, i] + 1) / 2)) for i in range(frames)) / frames
        )
        if self.lpips is not None:
            scores[self.lpips_key] = (
                sum(float(self.lpips(x[:, :, i], y[:, :, i]).mean()) for i in range(frames)) / frames
            )
        return scores


@torch.no_grad()
def evaluate_wan(model, ema, teacher, records, geometry, metric, device, precision, teacher_scores):
    """Cache only frozen teacher scores; evaluate every video once globally."""
    was_training = model.training
    model.eval()
    rows = []
    try:
        for index in local_indices(len(records)):
            record = records[index]
            x = read_clip(record, **geometry, validation=True).to(device)
            if x.shape[2] != geometry["frames"]:
                raise ValueError(f"Validation video is shorter than {geometry['frames']} frames: {record['path']}")
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                z = teacher.encode(x, mode="mean")
                prediction, _ = model(z)
                averaged, _ = ema(z)
                if index not in teacher_scores:
                    original, _ = teacher(z)
            if index not in teacher_scores:
                teacher_scores[index] = metric(original, x)
            scores = {"teacher": teacher_scores[index], "student": metric(prediction, x), "ema": metric(averaged, x)}
            rows.append(
                {
                    "index": index,
                    "dataset": record["dataset"],
                    "group": record["group"],
                    "path": record["path"],
                    "metrics": scores,
                }
            )
    finally:
        model.train(was_training)
    rows = gather_rows(rows)
    return {"summary": summarize(rows), "rows": rows}
