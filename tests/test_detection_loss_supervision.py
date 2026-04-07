import unittest

import torch

from src.loss.detection_loss import UAVDualModalLoss


class DetectionLossSupervisionTestCase(unittest.TestCase):
    def test_dense_cls_and_positive_regression_path_is_shape_safe(self):
        criterion = UAVDualModalLoss(
            num_classes=2,
            use_scale_weight=False,
            cls_loss_type='varifocal',
            angle_enabled=True,
            angle_weight=0.1,
        )
        pred_scores = torch.tensor([
            [[2.0, -1.0], [0.5, -0.8], [-1.5, 1.2], [0.1, -0.2]],
            [[-0.3, 1.1], [1.4, -0.6], [0.0, 0.0], [0.8, -1.2]],
        ], dtype=torch.float32)
        pred_bboxes = torch.tensor([
            [[0.50, 0.50, 0.20, 0.10, 0.10], [0.20, 0.30, 0.12, 0.08, 0.00], [0.70, 0.60, 0.18, 0.09, -0.10], [0.40, 0.20, 0.15, 0.11, 0.05]],
            [[0.35, 0.45, 0.14, 0.09, -0.05], [0.62, 0.55, 0.16, 0.12, 0.12], [0.10, 0.15, 0.08, 0.05, 0.00], [0.82, 0.72, 0.20, 0.16, -0.08]],
        ], dtype=torch.float32)
        target_scores = torch.zeros_like(pred_scores)
        target_scores[0, 0, 0] = 0.9
        target_scores[0, 2, 1] = 0.6
        target_scores[1, 1, 0] = 0.75
        target_bboxes = pred_bboxes.clone()
        target_bboxes[0, 0] = torch.tensor([0.52, 0.48, 0.22, 0.11, 0.08])
        target_bboxes[0, 2] = torch.tensor([0.68, 0.62, 0.16, 0.10, -0.06])
        target_bboxes[1, 1] = torch.tensor([0.60, 0.57, 0.15, 0.10, 0.10])
        fg_mask = torch.tensor([
            [True, False, True, False],
            [False, True, False, False],
        ])

        total_loss, loss_cls, loss_reg, loss_angle = criterion(
            pred_scores,
            pred_bboxes,
            target_scores,
            target_bboxes,
            fg_mask=fg_mask,
            epoch=3,
        )

        self.assertEqual(total_loss.ndim, 0)
        self.assertEqual(loss_cls.ndim, 0)
        self.assertEqual(loss_reg.ndim, 0)
        self.assertEqual(loss_angle.ndim, 0)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(loss_cls))
        self.assertTrue(torch.isfinite(loss_reg))
        self.assertTrue(torch.isfinite(loss_angle))
        self.assertGreater(float(loss_cls.item()), 0.0)
        self.assertGreater(float(loss_reg.item()), 0.0)

    def test_dense_cls_path_keeps_negative_only_batches_trainable(self):
        criterion = UAVDualModalLoss(num_classes=2, use_scale_weight=False, cls_loss_type='varifocal')
        pred_scores = torch.tensor([[[2.5, -0.5], [1.8, 0.9], [-0.3, 1.2]]], dtype=torch.float32)
        pred_bboxes = torch.zeros((1, 3, 5), dtype=torch.float32)
        target_scores = torch.zeros_like(pred_scores)
        target_bboxes = torch.zeros_like(pred_bboxes)
        fg_mask = torch.zeros((1, 3), dtype=torch.bool)

        total_loss, loss_cls, loss_reg, loss_angle = criterion(
            pred_scores,
            pred_bboxes,
            target_scores,
            target_bboxes,
            fg_mask=fg_mask,
            epoch=0,
        )

        self.assertGreater(float(loss_cls.item()), 0.0)
        self.assertEqual(float(loss_reg.item()), 0.0)
        self.assertEqual(float(loss_angle.item()), 0.0)
        self.assertTrue(torch.isfinite(total_loss))


if __name__ == '__main__':
    unittest.main()
