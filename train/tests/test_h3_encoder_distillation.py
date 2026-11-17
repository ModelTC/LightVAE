import unittest
import torch
from vae_prune.losses.h3_joint import encoder_distillation


class EncoderDistillationTests(unittest.TestCase):
    def test_identical(self):
        moments = torch.zeros(1, 4, 1, 2, 2, requires_grad=True)
        feature = torch.ones(1, 3, 1, 2, 2, requires_grad=True)
        kl, mse = encoder_distillation(moments, moments, [feature], [feature])
        self.assertEqual(kl.item(), 0.0)
        self.assertEqual(mse.item(), 0.0)

    def test_direction_and_teacher_detach(self):
        student = torch.tensor([1.0, 0.0]).reshape(1, 2, 1, 1, 1).requires_grad_()
        teacher = torch.tensor([0.0, 1.38629436112]).reshape(1, 2, 1, 1, 1).requires_grad_()
        sf = torch.ones(1, 2, requires_grad=True)
        tf = torch.zeros(1, 2, requires_grad=True)
        kl, mse = encoder_distillation(student, teacher, [sf], [tf])
        self.assertAlmostEqual(kl.item(), 0.5 * (1.38629436112 + 0.25 + 0.25 - 1), places=6)
        (kl + mse).backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0.0)
        self.assertGreater(sf.grad.abs().sum().item(), 0.0)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(tf.grad)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            encoder_distillation(torch.zeros(1, 4), torch.zeros(1, 4), [torch.zeros(1, 2)], [torch.zeros(1, 3)])


if __name__ == "__main__":
    unittest.main()
