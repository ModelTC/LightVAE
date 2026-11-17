import unittest
import pytest
import torch
from vae_prune.models.temporal import PackedTemporalConv, mass_rank
from vae_prune.models.search_masks import build_residual_search_masks
from vae_prune.losses import ReconstructionLoss
from vae_prune.data import assert_disjoint


class CoreTests(unittest.TestCase):
    def test_rank_uses_singular_sum_not_energy(self):
        self.assertEqual(mass_rank(torch.tensor([4.0, 3.0, 2.0, 1.0]), 0.6), 2)
        with self.assertRaises(ValueError):
            mass_rank(torch.tensor([1.0]), 0.0)

    def test_rank_zero_is_spatial(self):
        layer = PackedTemporalConv((3, 2, 3, 3, 3), 1, 0, True)
        x = torch.randn(1, 2, 5, 8, 8, requires_grad=True)
        y, _ = layer(x, None)
        reference = torch.stack([layer.down(x[:, :, i]) + layer.bias[None, :, None, None] for i in range(5)], dim=2)
        torch.testing.assert_close(y, reference)
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_streaming_projection_history(self):
        layer = PackedTemporalConv((3, 2, 3, 3, 3), 1, 2, True)
        x = torch.randn(1, 2, 5, 8, 8, requires_grad=True)
        whole, _ = layer(x, None)
        first, history = layer(x[:, :, :1], None)
        rest, _ = layer(x[:, :, 1:], history)
        torch.testing.assert_close(whole, torch.cat((first, rest), 2))
        rest.sum().backward()
        self.assertGreater(float(x.grad[:, :, :1].abs().sum()), 0)

    def test_stage_masks(self):
        masks = build_residual_search_masks(
            14,
            dict(grouping="stage", keep_per_group=1),
            ((0, 1), (2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13)),
            default_keep=5,
        )
        self.assertEqual(tuple(masks.shape), (162, 14))
        self.assertTrue((masks.sum(1) == 5).all())

    def test_optional_loss_and_teacher_detachment(self):
        x = torch.randn(1, 3, 1, 4, 4, requires_grad=True)
        teacher = torch.randn_like(x, requires_grad=True)
        loss, terms = ReconstructionLoss(perceptual=0, teacher_l1=0.5, temporal=0.1)(x, x.detach(), teacher)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(float(terms["temporal"]), 0)
        self.assertIsNotNone(x.grad)

    def test_group_overlap_rejected(self):
        with self.assertRaises(ValueError):
            assert_disjoint(
                [dict(dataset="ucf101", group="g01", path="/tmp/a")],
                [dict(dataset="ucf101", group="g01", path="/tmp/b")],
            )


if __name__ == "__main__":
    unittest.main()


def test_zip_sequences_can_share_an_archive_without_sharing_frames():
    train = [dict(dataset="davis", group="train-video", path="/tmp/davis.zip", zip_members=["train/0.jpg"])]
    val = [dict(dataset="davis", group="val-video", path="/tmp/davis.zip", zip_members=["val/0.jpg"])]
    assert_disjoint(train, val)
    val[0]["zip_members"] = ["train/0.jpg"]
    with pytest.raises(ValueError):
        assert_disjoint(train, val)
