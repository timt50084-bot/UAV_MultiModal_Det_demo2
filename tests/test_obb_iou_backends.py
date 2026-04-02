import unittest
import warnings

import numpy as np

from src.metrics.obb_iou_backend import (
    OBB_IOU_BACKEND_CPU_POLYGON,
    OBB_IOU_BACKEND_GPU_PROB,
    build_obb_iou_backend,
    resolve_obb_iou_backend_name,
)
from src.metrics.obb_metrics import OBBMetricsEvaluator
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
    def test_eval_cfg_defaults_to_cpu_polygon(self):
        eval_cfg = normalize_eval_metrics_cfg({})

        self.assertEqual(eval_cfg['obb_iou_backend'], OBB_IOU_BACKEND_CPU_POLYGON)
        self.assertEqual(resolve_obb_iou_backend_name(eval_cfg), OBB_IOU_BACKEND_CPU_POLYGON)

    @unittest.skipUnless(HAS_SHAPELY, 'requires shapely for cpu_polygon reference path')
    def test_cpu_polygon_backend_keeps_exact_reference_path(self):
        backend = build_obb_iou_backend({'obb_iou_backend': 'cpu_polygon'})
        boxes_a = np.array([[10.0, 10.0, 8.0, 4.0, 0.0]], dtype=np.float32)
        boxes_b = np.array([
            [10.0, 10.0, 8.0, 4.0, 0.0],
            [30.0, 30.0, 8.0, 4.0, 0.0],
        ], dtype=np.float32)

        matrix = backend.pairwise_iou(boxes_a, boxes_b)

        self.assertEqual(matrix.shape, (1, 2))
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(matrix[0, 1]), 0.0, places=6)

    @unittest.skipIf(HAS_SHAPELY, 'only checks the fallback error path when shapely is unavailable')
    def test_cpu_polygon_backend_fails_clearly_without_shapely(self):
        backend = build_obb_iou_backend({'obb_iou_backend': 'cpu_polygon'})

        with self.assertRaisesRegex(RuntimeError, 'requires shapely'):
            backend.pairwise_iou(
                np.array([[10.0, 10.0, 8.0, 4.0, 0.0]], dtype=np.float32),
                np.array([[10.0, 10.0, 8.0, 4.0, 0.0]], dtype=np.float32),
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

    def test_cpu_metrics_evaluator_coerces_non_reference_backend(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            metrics_evaluator = OBBMetricsEvaluator(
                num_classes=1,
                extra_metrics_cfg={
                    'obb_iou_backend': OBB_IOU_BACKEND_GPU_PROB,
                    'small_object': {'enabled': False},
                    'temporal_stability': {'enabled': False},
                    'group_eval': {'enabled': False},
                },
            )

        self.assertEqual(metrics_evaluator.obb_iou_backend_name, OBB_IOU_BACKEND_CPU_POLYGON)
        self.assertTrue(any("always uses eval.obb_iou_backend='cpu_polygon'" in str(item.message) for item in caught))


if __name__ == '__main__':
    unittest.main()
