import math
import unittest

import torch

from src.loss.builder import build_loss
from src.loss.detection_loss import UAVDualModalLoss
from src.utils.config import load_config


class AngleLossTestCase(unittest.TestCase):
    def test_default_full_project_keeps_angle_loss_disabled(self):
        cfg = load_config('configs/main/full_project.yaml')
        self.assertFalse(bool(cfg.loss.angle_enabled))
        self.assertEqual(float(cfg.loss.angle_weight), 0.0)

    def test_full_project_angle_loss_config_loads_without_unrelated_features(self):
        cfg = load_config('configs/main/full_project_angle_loss.yaml')
        criterion = build_loss(cfg.loss)

        self.assertTrue(bool(cfg.loss.angle_enabled))
        self.assertGreater(float(cfg.loss.angle_weight), 0.0)
        self.assertEqual(str(cfg.loss.angle_type), 'wrapped_smooth_l1')
        self.assertIsNotNone(criterion)

        self.assertFalse(bool(cfg.infer.tta.enabled))
        self.assertEqual(dict(cfg.infer.classwise_conf_thresholds), {})
        self.assertFalse(bool(cfg.eval.cross_modal_robustness.enabled))
        self.assertFalse(bool(cfg.eval.error_analysis.enabled))
        self.assertFalse(bool(cfg.tracking.modality.use_scene_adaptation))

    def test_angle_weight_zero_is_legacy_equivalent(self):
        pred_cls = torch.tensor([[0.2], [-0.4]], dtype=torch.float32)
        pred_box = torch.tensor([
            [0.50, 0.50, 0.25, 0.20, 0.30],
            [0.40, 0.60, 0.18, 0.22, -0.45],
        ], dtype=torch.float32)
        tgt_cls = torch.tensor([0, 0], dtype=torch.long)
        tgt_box = torch.tensor([
            [0.51, 0.49, 0.24, 0.21, 0.20],
            [0.39, 0.61, 0.19, 0.23, -0.35],
        ], dtype=torch.float32)

        legacy = UAVDualModalLoss(num_classes=1, angle_enabled=False, angle_weight=0.0)
        zero_weight = UAVDualModalLoss(num_classes=1, angle_enabled=True, angle_weight=0.0)

        legacy_total, legacy_cls, legacy_reg, legacy_angle = legacy(pred_cls, pred_box, tgt_cls, tgt_box, epoch=7)
        zero_total, zero_cls, zero_reg, zero_angle = zero_weight(pred_cls, pred_box, tgt_cls, tgt_box, epoch=7)

        self.assertTrue(torch.allclose(legacy_total, zero_total, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(legacy_cls, zero_cls, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(legacy_reg, zero_reg, atol=1e-6, rtol=1e-6))
        self.assertEqual(float(legacy_angle.item()), 0.0)
        self.assertEqual(float(zero_angle.item()), 0.0)

    def test_wrapped_angle_difference_respects_pi_periodicity(self):
        criterion = UAVDualModalLoss(num_classes=1, angle_enabled=True, angle_weight=0.2)

        pred_theta = torch.tensor([math.pi / 2 - 0.01], dtype=torch.float32)
        tgt_theta = torch.tensor([-math.pi / 2 + 0.01], dtype=torch.float32)

        wrapped = criterion.wrapped_angle_difference(pred_theta, tgt_theta)
        self.assertLess(abs(float(wrapped.item())), 0.03)

        angle_loss_raw = criterion._compute_angle_loss_raw(pred_theta, tgt_theta)
        self.assertLess(float(angle_loss_raw.item()), 0.01)

    def test_enabled_angle_loss_returns_visible_auxiliary_component(self):
        pred_cls = torch.tensor([[0.1]], dtype=torch.float32)
        pred_box = torch.tensor([[0.50, 0.50, 0.25, 0.20, 0.35]], dtype=torch.float32)
        tgt_cls = torch.tensor([0], dtype=torch.long)
        tgt_box = torch.tensor([[0.50, 0.50, 0.25, 0.20, -0.25]], dtype=torch.float32)

        legacy = UAVDualModalLoss(num_classes=1, angle_enabled=False, angle_weight=0.0)
        enabled = UAVDualModalLoss(
            num_classes=1,
            angle_enabled=True,
            angle_weight=0.2,
            angle_type='wrapped_smooth_l1',
            angle_beta=0.1,
        )

        legacy_total, _, _, legacy_angle = legacy(pred_cls, pred_box, tgt_cls, tgt_box, epoch=7)
        enabled_total, _, _, enabled_angle = enabled(pred_cls, pred_box, tgt_cls, tgt_box, epoch=7)

        self.assertEqual(float(legacy_angle.item()), 0.0)
        self.assertGreater(float(enabled_angle.item()), 0.0)
        self.assertGreater(float(enabled_total.item()), float(legacy_total.item()))


if __name__ == '__main__':
    unittest.main()
