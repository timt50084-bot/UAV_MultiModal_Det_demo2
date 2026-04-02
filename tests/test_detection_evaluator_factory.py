import unittest
import warnings
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

if torch is not None:
    from src.engine.evaluator import Evaluator, GPUDetectionEvaluator
    from src.engine.evaluator_factory import build_detection_evaluator, get_detection_evaluator_backend
    from src.metrics.obb_metrics import GPUOBBMetricsEvaluator, OBBMetricsEvaluator
else:  # pragma: no cover - exercised only when torch is absent
    Evaluator = None
    GPUDetectionEvaluator = None
    OBBMetricsEvaluator = None
    GPUOBBMetricsEvaluator = None
    build_detection_evaluator = None
    get_detection_evaluator_backend = None

HAS_TORCH = torch is not None
HAS_CUDA = bool(HAS_TORCH and torch.cuda.is_available())

@unittest.skipUnless(HAS_TORCH, 'torch is required for detection evaluator factory tests')
class DetectionEvaluatorFactoryTestCase(unittest.TestCase):
    def test_backend_defaults_to_gpu(self):
        self.assertEqual(get_detection_evaluator_backend({}), 'gpu')
        self.assertEqual(get_detection_evaluator_backend({'evaluator': 'cpu'}), 'cpu')
        self.assertEqual(get_detection_evaluator_backend({'evaluator': 'gpu'}), 'gpu')

    def test_builder_defaults_to_cpu_reference_with_warning_on_non_cuda_device(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            evaluator = build_detection_evaluator(
                dataloader=[],
                device=torch.device('cpu'),
                num_classes=1,
                nms_kwargs={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0},
                eval_cfg={},
                infer_cfg={},
            )

        self.assertIsInstance(evaluator, Evaluator)
        self.assertIsInstance(evaluator.metrics_evaluator, OBBMetricsEvaluator)
        self.assertEqual(evaluator.metrics_evaluator.obb_iou_backend_name, 'cpu_polygon')
        self.assertEqual(evaluator.requested_backend, 'gpu')
        self.assertEqual(evaluator.requested_obb_iou_backend, 'gpu_prob')
        self.assertEqual(evaluator.resolved_backend, 'cpu')
        self.assertEqual(evaluator.resolved_obb_iou_backend, 'cpu_polygon')
        self.assertEqual(evaluator.evaluator_role, 'fallback')
        self.assertEqual(evaluator.resolution_reason, 'non_cuda_fallback')
        self.assertTrue(any("Falling back to the CPU reference evaluator" in str(item.message) for item in caught))

    def test_builder_returns_cpu_evaluator_when_explicit_cpu_reference_is_requested(self):
        evaluator = build_detection_evaluator(
            dataloader=[],
            device=torch.device('cpu'),
            num_classes=1,
            nms_kwargs={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0},
            eval_cfg={'evaluator': 'cpu', 'obb_iou_backend': 'cpu_polygon'},
            infer_cfg={},
        )

        self.assertIsInstance(evaluator, Evaluator)
        self.assertIsInstance(evaluator.metrics_evaluator, OBBMetricsEvaluator)
        self.assertEqual(evaluator.requested_backend, 'cpu')
        self.assertEqual(evaluator.resolved_backend, 'cpu')
        self.assertEqual(evaluator.resolved_obb_iou_backend, 'cpu_polygon')
        self.assertEqual(evaluator.evaluator_role, 'reference')
        self.assertEqual(evaluator.resolution_reason, 'explicit_cpu_reference')

    def test_builder_safely_falls_back_when_gpu_is_explicitly_requested_on_cpu_device(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            evaluator = build_detection_evaluator(
                dataloader=[],
                device=torch.device('cpu'),
                num_classes=1,
                nms_kwargs={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0},
                eval_cfg={'evaluator': 'gpu', 'obb_iou_backend': 'gpu_prob'},
                infer_cfg={},
            )

        self.assertIsInstance(evaluator, Evaluator)
        self.assertIsInstance(evaluator.metrics_evaluator, OBBMetricsEvaluator)
        self.assertEqual(evaluator.requested_backend, 'gpu')
        self.assertEqual(evaluator.requested_obb_iou_backend, 'gpu_prob')
        self.assertEqual(evaluator.resolved_backend, 'cpu')
        self.assertEqual(evaluator.resolved_obb_iou_backend, 'cpu_polygon')
        self.assertEqual(evaluator.evaluator_role, 'fallback')
        self.assertEqual(evaluator.resolution_reason, 'non_cuda_fallback')
        self.assertTrue(any("Falling back to the CPU reference evaluator" in str(item.message) for item in caught))

    def test_builder_rejects_gpu_evaluator_with_incompatible_iou_backend(self):
        with self.assertRaisesRegex(ValueError, "requires eval.obb_iou_backend='gpu_prob'"):
            build_detection_evaluator(
                dataloader=[],
                device=torch.device('cuda:0'),
                num_classes=1,
                nms_kwargs={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0},
                eval_cfg={'evaluator': 'gpu', 'obb_iou_backend': 'cpu_polygon'},
                infer_cfg={},
            )

    @unittest.skipUnless(HAS_CUDA, 'requires CUDA for gpu evaluator factory test')
    def test_builder_returns_gpu_evaluator_when_explicitly_requested(self):
        evaluator = build_detection_evaluator(
            dataloader=[],
            device=torch.device('cuda:0'),
            num_classes=1,
            nms_kwargs={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0},
            eval_cfg={
                'evaluator': 'gpu',
                'obb_iou_backend': 'gpu_prob',
                'small_object': {'enabled': False},
                'temporal_stability': {'enabled': False},
                'group_eval': {'enabled': False},
            },
            infer_cfg={},
        )

        self.assertIsInstance(evaluator, GPUDetectionEvaluator)
        self.assertIsInstance(evaluator.metrics_evaluator, GPUOBBMetricsEvaluator)
        self.assertEqual(evaluator.requested_backend, 'gpu')
        self.assertEqual(evaluator.resolved_backend, 'gpu')
        self.assertEqual(evaluator.resolved_obb_iou_backend, 'gpu_prob')
        self.assertEqual(evaluator.evaluator_role, 'mainline')
        self.assertEqual(evaluator.resolution_reason, 'gpu_mainline')

    @unittest.skipUnless(HAS_CUDA, 'requires CUDA for gpu evaluator factory test')
    def test_builder_returns_gpu_evaluator_by_default_on_cuda(self):
        evaluator = build_detection_evaluator(
            dataloader=[],
            device=torch.device('cuda:0'),
            num_classes=1,
            nms_kwargs={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0},
            eval_cfg={},
            infer_cfg={},
        )

        self.assertIsInstance(evaluator, GPUDetectionEvaluator)
        self.assertIsInstance(evaluator.metrics_evaluator, GPUOBBMetricsEvaluator)
        self.assertEqual(evaluator.requested_backend, 'gpu')
        self.assertEqual(evaluator.requested_obb_iou_backend, 'gpu_prob')
        self.assertEqual(evaluator.resolved_backend, 'gpu')
        self.assertEqual(evaluator.resolved_obb_iou_backend, 'gpu_prob')
        self.assertEqual(evaluator.evaluator_role, 'mainline')
        self.assertEqual(evaluator.resolution_reason, 'gpu_mainline')

    def test_cpu_reference_request_coerces_non_reference_iou_backend(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with patch('src.engine.evaluator_factory.OBBMetricsEvaluator') as metrics_cls:
                metrics_instance = metrics_cls.return_value
                evaluator = build_detection_evaluator(
                    dataloader=[],
                    device=torch.device('cpu'),
                    num_classes=1,
                    nms_kwargs={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0},
                    eval_cfg={'evaluator': 'cpu', 'obb_iou_backend': 'gpu_prob'},
                    infer_cfg={},
                )

        self.assertIsInstance(evaluator, Evaluator)
        metrics_cls.assert_called_once()
        self.assertIs(metrics_instance, evaluator.metrics_evaluator)
        self.assertEqual(evaluator.requested_backend, 'cpu')
        self.assertEqual(evaluator.requested_obb_iou_backend, 'gpu_prob')
        self.assertEqual(evaluator.resolved_backend, 'cpu')
        self.assertEqual(evaluator.resolved_obb_iou_backend, 'cpu_polygon')
        self.assertEqual(evaluator.evaluator_role, 'reference')
        self.assertEqual(evaluator.resolution_reason, 'explicit_cpu_reference')
        self.assertTrue(any("always resolves to eval.obb_iou_backend='cpu_polygon'" in str(item.message) for item in caught))


if __name__ == '__main__':
    unittest.main()
