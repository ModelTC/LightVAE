import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from vae_prune.evaluation import VideoMetrics
from vae_prune.losses.guidance import guidance_weights, mean_distance
from vae_prune.losses.wan_objective import guided_feature
from vae_prune.losses.wan_recovery import WanRecovery
from vae_prune.recover import learning_rate, update_ema, validate_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "recover"


def config(family="wan21"):
    return json.loads((CONFIGS / f"{family}_lite.json").read_text())


def test_configs_and_schedule():
    for family in ("wan21", "wan22"):
        for variant in ("pro", "lite"):
            recipe = json.loads((CONFIGS / f"{family}_{variant}.json").read_text())
            validate_config(recipe)
            assert learning_rate(recipe, recipe["warmup"]) == recipe["lr"]
            assert learning_rate(recipe, recipe["steps"]) == recipe["min_lr"]


def test_guidance_suppresses_opposing_directions_and_detaches_probe():
    delta = torch.tensor([1.0, -2.0, 0.0], requires_grad=True)
    gradient = torch.tensor([2.0, 1.0, 4.0], requires_grad=True)
    weight = guidance_weights(delta, gradient, 0.01, 10.0, 3)
    expected = (delta.detach() * gradient.detach() * 1.5).clamp_min(0) / (
        delta.detach().square() + 0.01 * delta.detach().square().mean()
    )
    torch.testing.assert_close(weight, expected.clamp_max(10))
    assert not weight.requires_grad
    assert weight[1] == weight[2] == 0


def test_chunk_distance_weights_elements():
    student = [torch.ones(1, requires_grad=True), torch.full((4,), 2.0, requires_grad=True)]
    target = [torch.zeros_like(value) for value in student]
    loss = mean_distance(student, target)
    torch.testing.assert_close(loss, torch.tensor(3.4))
    loss.backward()
    torch.testing.assert_close(student[1].grad, torch.full((4,), 0.8))


def test_probe_loss_only_trains_student_features():
    student = [torch.tensor([1.0, 2.0], requires_grad=True)]
    teacher = [torch.tensor([0.0, 4.0])]
    decoder = nn.Linear(2, 2, bias=False).requires_grad_(False)
    decoder.weight.data.copy_(torch.eye(2))

    def suffix(module, features, anchor):
        return [module(value) for value in features]

    loss, stats = guided_feature(
        student, teacher, decoder, teacher, 1, {"epsilon": 0.01, "max_weight": 10}, suffix_fn=suffix
    )
    loss.backward()
    assert student[0].grad is not None and torch.isfinite(student[0].grad).all()
    assert decoder.weight.grad is None
    assert stats["probe"] > 0


def test_disabled_perceptual_and_gan_need_no_extra_models():
    recipe = config()
    recipe["loss"].update(perceptual=0, feature=0, gan=0)
    objective = WanRecovery(recipe["loss"], recipe["gan"], "cpu", "wan21")
    assert objective.loss.perceptual is None
    assert objective.loss.gan is None
    prediction = torch.full((1, 3, 5, 16, 16), 0.25, requires_grad=True)
    target = torch.zeros_like(prediction)
    latent = torch.zeros(1, 16, 2, 2, 2)
    loss, stats = objective(prediction, target, latent, 500, 42, None, None, None, None)
    torch.testing.assert_close(loss, torch.tensor(0.25))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert stats["lpips"] == stats["gan"] == 0


def test_gan_starts_on_configured_step_and_preserves_latent_alignment():
    recipe = config()
    recipe["loss"].update(perceptual=0, feature=0)
    recipe["gan"].update(start_step=2, frames=5, crop_size=16)
    objective = WanRecovery(recipe["loss"], recipe["gan"], "cpu", "wan21", accumulation_steps=4)
    captures = []

    def discriminator(fake, real, condition, stage_step):
        captures.append((condition, stage_step))
        return fake.sum() * 0, {}

    def start():
        objective.loss.gan = discriminator

    prediction = torch.zeros(1, 3, 5, 16, 16, requires_grad=True)
    latent = torch.stack([torch.zeros(16, 2, 2), torch.ones(16, 2, 2)], dim=1).unsqueeze(0)
    with patch.object(objective.loss, "start_gan", side_effect=start) as initialize:
        for step in (1, 2, 3):
            objective(prediction, torch.zeros_like(prediction), latent, step, 0, None, None, None, None)
        initialize.assert_called_once()
    assert [stage for _, stage in captures] == [0, 1]
    torch.testing.assert_close(captures[0][0][0, 0, :, 0, 0], torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]))
    assert objective.cfg["accumulation_steps"] == 4


def test_wan_families_have_distinct_feature_anchors():
    for family, anchors, channels in (("wan21", [1, 4, 7, 10, 13], 16), ("wan22", [0, 1, 2, 3, 4], 48)):
        recipe = config(family)
        with patch("vae_prune.losses.wan_recovery.GPUReconstruction", return_value=MagicMock()):
            objective = WanRecovery(recipe["loss"], recipe["gan"], "cpu", family)
        assert [objective.anchor(step) for step in range(1, 7)] == anchors + anchors[:1]
        assert objective.cfg["latent_channels"] == channels


def test_ema_updates_weights_and_copies_buffers():
    model, ema = nn.BatchNorm1d(3), nn.BatchNorm1d(3)
    model.weight.data.fill_(3)
    model.running_mean.fill_(4)
    update_ema(ema, model, 0.75)
    torch.testing.assert_close(ema.weight, torch.full((3,), 1.5))
    torch.testing.assert_close(ema.running_mean, torch.full((3,), 4.0))


def test_video_metrics_without_download():
    metric = VideoMetrics(lpips_enabled=False)
    x = torch.zeros(1, 3, 5, 16, 16)
    scores = metric(x, x)
    assert scores["psnr"] == 120
    assert abs(scores["ssim"] - 1) < 1e-6
