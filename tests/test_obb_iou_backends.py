import unittest

import numpy as np

from src.metrics.obb_iou_backend import (
    OBB_IOU_BACKEND_GPU_PROB,
    build_obb_iou_backend,
    polygon_iou,
    resolve_obb_iou_backend_name,
)
from src.metrics.task_metrics import normalize_eval_metrics_cfg

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
except ImportError:  # pragma: no cover - optional in lightweight test envs
    _ShapelyPolygon = None

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None


HAS_SHAPELY = _ShapelyPolygon is not None
HAS_TORCH = torch is not None
HAS_CUDA = bool(HAS_TORCH and torch.cuda.is_available())


class OBBIoUBackendTestCase(unittest.TestCase):
    def test_eval_cfg_defaults_to_gpu_prob(self):
        eval_cfg = normalize_eval_metrics_cfg({})

        self.assertEqual(eval_cfg['obb_iou_backend'], OBB_IOU_BACKEND_GPU_PROB)
        self.assertEqual(resolve_obb_iou_backend_name(eval_cfg), OBB_IOU_BACKEND_GPU_PROB)

    def test_removed_cpu_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supports eval.obb_iou_backend='gpu_prob'"):
            build_obb_iou_backend({'obb_iou_backend': 'cpu_polygon'})

    @unittest.skipUnless(HAS_SHAPELY, 'requires shapely for exact polygon analysis helper')
    def test_polygon_iou_helper_keeps_exact_analysis_math(self):
        box_a = np.array([10.0, 10.0, 8.0, 4.0, 0.0], dtype=np.float32)
        same_box = np.array([10.0, 10.0, 8.0, 4.0, 0.0], dtype=np.float32)
        far_box = np.array([30.0, 30.0, 8.0, 4.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(float(polygon_iou(box_a, same_box)), 1.0, places=6)
        self.assertAlmostEqual(float(polygon_iou(box_a, far_box)), 0.0, places=6)

    @unittest.skipIf(HAS_SHAPELY, 'only checks the missing-shapely analysis path')
    def test_polygon_iou_helper_fails_clearly_without_shapely(self):
        with self.assertRaisesRegex(RuntimeError, 'requires shapely'):
            polygon_iou(
                np.array([10.0, 10.0, 8.0, 4.0, 0.0], dtype=np.float32),
                np.array([10.0, 10.0, 8.0, 4.0, 0.0], dtype=np.float32),
            )

    @unittest.skipUnless(HAS_TORCH and not HAS_CUDA, 'requires a torch environment without CUDA')
    def test_gpu_prob_backend_fails_clearly_without_cuda(self):
        with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
            build_obb_iou_backend(OBB_IOU_BACKEND_GPU_PROB)

    @unittest.skipUnless(HAS_CUDA, 'requires CUDA for gpu_prob backend')
    def test_gpu_prob_backend_returns_valid_surrogate_scores(self):
        backend = build_obb_iou_backend(OBB_IOU_BACKEND_GPU_PROB)
        boxes_a = np.array([
            [10.0, 10.0, 8.0, 4.0, 0.0],
            [60.0, 60.0, 8.0, 4.0, 0.0],
        ], dtype=np.float32)
        boxes_b = np.array([
            [10.0, 10.0, 8.0, 4.0, 0.0],
            [14.0, 10.0, 8.0, 4.0, 0.0],
        ], dtype=np.float32)

        matrix = backend.pairwise_iou(boxes_a, boxes_b)

        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.all(matrix >= 0.0))
        self.assertTrue(np.all(matrix <= 1.0 + 1e-6))
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0, places=5)
        self.assertGreaterEqual(float(matrix[0, 0]), float(matrix[0, 1]))


if __name__ == '__main__':
    unittest.main()
