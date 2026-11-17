import unittest
import torch
from torch import nn
from vae_prune.models.h3_joint import H3Joint


class TrainComponentTests(unittest.TestCase):
    def test_freeze_and_gradient_flow(self):
        for mode in ("encoder", "decoder", "joint"):
            model = H3Joint.from_components(nn.Linear(2, 2), nn.Linear(2, 2))
            model.set_train_component(mode)
            model.train()
            model.decoder(model.encoder(torch.ones(1, 2))).sum().backward()
            for name in ("encoder", "decoder"):
                component = getattr(model, name)
                active = mode in ("joint", name)
                self.assertEqual(component.training, active)
                for p in component.parameters():
                    self.assertEqual(p.requires_grad, active)
                    self.assertEqual(p.grad is not None, active)
            model.eval()
            model.train()
            if mode != "joint":
                self.assertFalse(getattr(model, "decoder" if mode == "encoder" else "encoder").training)


if __name__ == "__main__":
    unittest.main()
