import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

if torch is not None:
    from src.engine.evaluator import GPUDetectionEvaluator
    from src.metrics.obb_metrics import GPUOBBMetricsEvaluator
else:  # pragma: no cover - exercised only when torch is absent
    GPUDetectionEvaluator = None
    GPUOBBMetricsEvaluator = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires CUDA for GPU evaluator smoke test')
class GPUDetectionEvaluatorSmokeTestCase(unittest.TestCase):
    def test_gpu_evaluator_runs_base_detection_metrics_path(self):
        device = torch.device('cuda:0')
        imgs_rgb = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        imgs_ir = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        prev_rgb = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        prev_ir = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        targets = torch.tensor([[0.0, 0.0, 10.0, 10.0, 8.0, 4.0, 0.0]], dtype=torch.float32)

        dataloader = [(imgs_rgb, imgs_ir, targets, prev_rgb, prev_ir)]
        metrics_evaluator = GPUOBBMetricsEvaluator(
            num_classes=1,
            device=device,
            extra_metrics_cfg={
                'obb_iou_backend': 'gpu_prob',
                'small_object': {'enabled': False},
                'temporal_stability': {'enabled': False},
                'group_eval': {'enabled': False},
                'error_analysis': {'enabled': False},
            },
        )
        evaluator = GPUDetectionEvaluator(
            dataloader=dataloader,
            metrics_evaluator=metrics_evaluator,
            device=device,
            extra_metrics_cfg={
                'obb_iou_backend': 'gpu_prob',
                'small_object': {'enabled': False},
                'temporal_stability': {'enabled': False},
                'group_eval': {'enabled': False},
                'error_analysis': {'enabled': False},
            },
        )

        class DummyModel(torch.nn.Module):
            pass

        model = DummyModel().to(device)
        patched_preds = [torch.tensor([[10.0, 10.0, 8.0, 4.0, 0.0, 0.95, 0.0]], dtype=torch.float32, device=device)]

        with patch.object(evaluator, '_predict_batch', return_value=patched_preds):
            metrics = evaluator.evaluate(model, epoch=1)

        self.assertAlmostEqual(float(metrics['mAP_50']), 1.0, places=5)
        self.assertAlmostEqual(float(metrics['mAP_50_95']), 1.0, places=5)
        self.assertAlmostEqual(float(metrics['Precision']), 1.0, places=5)
        self.assertAlmostEqual(float(metrics['Recall']), 1.0, places=5)
        self.assertNotIn('TemporalStability', metrics)
        self.assertNotIn('GroupedMetrics', metrics)


if __name__ == '__main__':
    unittest.main()
