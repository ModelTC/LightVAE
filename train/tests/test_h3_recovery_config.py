"""H3 recovery uses configured loss weights and keeps encoder supervision explicit."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vae_prune.losses.guidance import Reconstruction
from vae_prune.losses.h3_joint import H3JointLoss
from vae_prune.recover_h3 import learning_rate, load_config

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "recover"


@pytest.mark.parametrize("component", ["encoder", "decoder", "joint"])
def test_config_matches_train_component(component):
    config = load_config(CONFIG_ROOT / f"h3_{component}.json")
    assert config["train_component"] == component
    assert learning_rate(config, config["warmup_steps"]) == config["learning_rate"]
    assert learning_rate(config, config["steps"]) == config["min_learning_rate"]


def test_frozen_encoder_cannot_have_supervision(tmp_path):
    config = load_config(CONFIG_ROOT / "h3_decoder.json")
    config["loss"]["encoder_kl"] = 0.1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="trainable encoder"):
        load_config(path)


def test_loss_weights_and_disabled_gan():
    config = load_config(CONFIG_ROOT / "h3_joint.json")
    loss_config = copy.deepcopy(config["loss"])
    loss_config.update(l1=2.0, lpips=0.0, guided_feature=0.0, gan=0.0)
    teacher = nn.Module()
    teacher.post_quant_conv = nn.Conv3d(4, 4, 1)

    def initialize(self, cfg, device):
        self.cfg, self.device, self.gan = cfg, device, None
        self.perceptual = None

    with patch.object(Reconstruction, "__init__", initialize):
        objective = H3JointLoss(teacher, loss_config, config["gan"], "cpu", 1)
    prediction = torch.ones(1, 3, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    latent = torch.zeros(1, 4, 1, 1, 1)
    with patch.object(objective, "start_gan", side_effect=AssertionError("GAN disabled")):
        loss, metrics = objective((prediction, latent, None, None), target, None, 5000)
    assert loss.item() == 2.0
    assert metrics["gan"].item() == 0.0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 2 / prediction.numel()))
