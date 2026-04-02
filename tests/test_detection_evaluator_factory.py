import unittest

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

if torch is not None:
    from src.engine.evaluator import GPUDetectionEvaluator
    from src.engine.evaluator_factory import build_detection_evaluator, get_detection_evaluator_backend
    from src.metrics.obb_metrics import GPUOBBMetricsEvaluator
else:  # pragma: no cover - exercised only when torch is absent
    GPUDetectionEvaluator = None
    GPUOBBMetricsEvaluator = None
    build_detection_evaluator = None
    get_detection_evaluator_backend = None

HAS_TORCH = torch is not None
HAS_CUDA = bool(HAS_TORCH and torch.cuda.is_available())


@unittest.skipUnless(HAS_TORCH, 'torch is required for detection evaluator factory tests')
class DetectionEvaluatorFactoryTestCase(unittest.TestCase):
    def test_backend_defaults_to_gpu(self):
        self.assertEqual(get_detection_evaluator_backend({}), 'gpu')
        self.assertEqual(get_detection_evaluator_backend({'evaluator': 'gpu'}), 'gpu')

    def test_builder_rejects_non_cuda_device(self):
        with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
            build_detection_evaluator(
                dataloader=[],
                device=torch.device('cpu'),
                num_classes=1,
                nms_kwargs={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0},
                eval_cfg={},
                infer_cfg={},
            )

    def test_builder_rejects_removed_cpu_backend_request(self):
        with self.assertRaisesRegex(ValueError, 'no longer supported'):
            build_detection_evaluator(
                dataloader=[],
                device=torch.device('cuda:0'),
                num_classes=1,
                nms_kwargs={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0},
                eval_cfg={'evaluator': 'cpu'},
                infer_cfg={},
            )

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


if __name__ == '__main__':
    unittest.main()
