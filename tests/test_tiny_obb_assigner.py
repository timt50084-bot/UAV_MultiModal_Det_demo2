import unittest

import torch

from src.loss.assigners.target_assigner import DynamicTinyOBBAssigner
from src.loss.builder import build_assigner


class TinyOBBAssignerTestCase(unittest.TestCase):
    def test_registry_builds_improved_assigner(self):
        assigner = build_assigner({
            "type": "DynamicTinyOBBAssigner",
            "num_classes": 2,
            "topk": 5,
            "lambda_theta": 1.5,
            "tiny_area_threshold": 0.01,
            "tiny_topk_boost": 2,
            "elongated_ratio_threshold": 3.0,
            "use_angle_aware_assign": True,
        })
        self.assertEqual(assigner.__class__.__name__, "DynamicTinyOBBAssigner")

    def test_angle_consistency_penalizes_large_delta_and_respects_periodicity(self):
        assigner = DynamicTinyOBBAssigner(num_classes=1, use_angle_aware_assign=True, lambda_theta=2.0)
        pred_scores = torch.tensor([[[8.0], [8.0], [8.0]]])
        pred_bboxes = torch.tensor([[
            [0.5, 0.5, 0.2, 0.1, 0.0],
            [0.5, 0.5, 0.2, 0.1, torch.pi / 2.0],
            [0.5, 0.5, 0.2, 0.1, torch.pi],
        ]])
        anchor_points = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        gt_labels = torch.zeros((1, 1, 1), dtype=torch.long)
        gt_bboxes = torch.tensor([[[0.5, 0.5, 0.2, 0.1, 0.0]]])

        align_metric, _ = assigner.get_box_metrics(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes)
        self.assertGreater(align_metric[0, 0, 0].item(), align_metric[0, 1, 0].item())
        self.assertAlmostEqual(align_metric[0, 0, 0].item(), align_metric[0, 2, 0].item(), places=5)

    def test_candidate_budget_protects_tiny_and_elongated_targets(self):
        assigner = DynamicTinyOBBAssigner(
            num_classes=1,
            topk=5,
            tiny_area_threshold=0.02,
            tiny_topk_boost=3,
            elongated_ratio_threshold=3.0,
        )
        gt_bboxes = torch.tensor([[
            [0.5, 0.5, 0.3, 0.3, 0.0],
            [0.5, 0.5, 0.05, 0.05, 0.0],
            [0.5, 0.5, 0.4, 0.08, 0.0],
        ]])

        budget = assigner._compute_candidate_budget(gt_bboxes, num_anchors=16)
        self.assertEqual(budget[0, 0].item(), 5)
        self.assertGreater(budget[0, 1].item(), budget[0, 0].item())
        self.assertGreater(budget[0, 2].item(), budget[0, 0].item())

    def test_fallback_recovers_protected_targets(self):
        assigner = DynamicTinyOBBAssigner(
            num_classes=1,
            topk=5,
            tiny_area_threshold=0.02,
            tiny_topk_boost=2,
            elongated_ratio_threshold=3.0,
        )
        anchor_points = torch.tensor([
            [0.10, 0.10],
            [0.12, 0.11],
            [0.80, 0.80],
            [0.82, 0.81],
            [0.50, 0.50],
            [0.52, 0.50],
        ])
        gt_bboxes = torch.tensor([[
            [0.10, 0.10, 0.05, 0.05, 0.0],
            [0.80, 0.80, 0.45, 0.10, 0.0],
        ]])
        is_in_centers = torch.zeros((1, anchor_points.shape[0], 2), dtype=torch.bool)
        mask_gt = torch.ones((1, 2, 1), dtype=torch.float32)

        recovered = assigner.apply_tiny_object_fallback(anchor_points, gt_bboxes, is_in_centers, mask_gt)
        self.assertTrue(recovered[:, :, 0].any())
        self.assertTrue(recovered[:, :, 1].any())

    def test_forward_shapes_and_no_nan(self):
        assigner = DynamicTinyOBBAssigner(
            num_classes=2,
            topk=4,
            lambda_theta=1.5,
            tiny_area_threshold=0.02,
            tiny_topk_boost=2,
            elongated_ratio_threshold=3.0,
            use_angle_aware_assign=True,
        )
        pred_scores = torch.tensor([[
            [3.0, -1.0],
            [2.8, -0.8],
            [0.2, 2.5],
            [0.1, 2.3],
            [1.5, 0.2],
            [1.2, 0.1],
        ]])
        pred_bboxes = torch.tensor([[
            [0.10, 0.10, 0.05, 0.05, 0.00],
            [0.12, 0.11, 0.05, 0.05, 0.05],
            [0.80, 0.80, 0.40, 0.10, 0.10],
            [0.78, 0.82, 0.42, 0.10, 0.15],
            [0.50, 0.50, 0.20, 0.20, 0.00],
            [0.52, 0.48, 0.20, 0.18, 0.02],
        ]])
        anchor_points = pred_bboxes[0, :, :2]
        gt_labels = torch.tensor([[[0], [1]]], dtype=torch.long)
        gt_bboxes = torch.tensor([[
            [0.10, 0.10, 0.05, 0.05, 0.00],
            [0.80, 0.80, 0.42, 0.10, 0.12],
        ]])
        mask_gt = torch.ones((1, 2, 1), dtype=torch.float32)

        target_labels, target_bboxes, target_scores, is_pos = assigner(
            pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
        )

        self.assertEqual(target_labels.shape, (1, 6))
        self.assertEqual(target_bboxes.shape, (1, 6, 5))
        self.assertEqual(target_scores.shape, (1, 6, 2))
        self.assertEqual(is_pos.shape, (1, 6))
        self.assertTrue(torch.isfinite(target_bboxes).all())
        self.assertTrue(torch.isfinite(target_scores).all())

    def test_forward_sanitizes_non_finite_inputs(self):
        assigner = DynamicTinyOBBAssigner(num_classes=1, topk=3)
        pred_scores = torch.tensor([[
            [float('nan')],
            [float('inf')],
            [0.5],
        ]])
        pred_bboxes = torch.tensor([[[
            0.10, 0.10, 0.05, 0.05, 0.0
        ], [
            0.12, 0.12, 0.05, 0.05, 0.0
        ], [
            0.14, 0.14, 0.05, 0.05, 0.0
        ]]])
        anchor_points = pred_bboxes[0, :, :2]
        gt_labels = torch.tensor([[[0]]], dtype=torch.long)
        gt_bboxes = torch.tensor([[[0.10, 0.10, 0.05, 0.05, 0.0]]])
        mask_gt = torch.ones((1, 1, 1), dtype=torch.float32)

        target_labels, target_bboxes, target_scores, is_pos = assigner(
            pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
        )

        self.assertEqual(target_labels.shape, (1, 3))
        self.assertEqual(target_bboxes.shape, (1, 3, 5))
        self.assertEqual(target_scores.shape, (1, 3, 1))
        self.assertEqual(is_pos.shape, (1, 3))
        self.assertTrue(torch.isfinite(target_bboxes).all())
        self.assertTrue(torch.isfinite(target_scores).all())


if __name__ == "__main__":
    unittest.main()
