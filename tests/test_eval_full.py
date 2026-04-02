import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

try:
    from omegaconf import OmegaConf  # noqa: F401
except ImportError:  # pragma: no cover - optional in lightweight test envs
    OmegaConf = None

from src.metrics.task_metrics import normalize_eval_metrics_cfg
if torch is not None:
    from src.engine.evaluator import Evaluator
else:  # pragma: no cover - exercised only when torch is absent
    Evaluator = None

if OmegaConf is not None:
    from src.utils.config import load_config
    from src.utils.postprocess_tuning import normalize_infer_cfg
else:  # pragma: no cover - exercised only when OmegaConf is absent
    load_config = None
    normalize_infer_cfg = None


class DummyMetricsEvaluator:
    def reset(self):
        self.num_batches = 0

    def add_batch(self, image_ids, batch_preds, batch_gts, batch_metadata=None):
        del image_ids, batch_preds, batch_gts, batch_metadata
        self.num_batches += 1

    def get_full_metrics(self):
        return {
            'mAP_50': 0.5,
            'mAP_50_95': 0.3,
            'Precision': 0.4,
            'Recall': 0.6,
            'mAP_S': 0.25,
            'Recall_S': 0.5,
            'Precision_S': 0.5,
            'TemporalStability': None,
            'GroupedMetrics': {},
        }


if torch is not None:
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, rgb, ir, prev_rgb=None, prev_ir=None):
            self.calls.append({
                'rgb': rgb.detach().cpu().clone(),
                'ir': ir.detach().cpu().clone(),
                'prev_rgb': prev_rgb.detach().cpu().clone(),
                'prev_ir': prev_ir.detach().cpu().clone(),
            })
            return torch.zeros((rgb.shape[0], 1, 6), dtype=rgb.dtype, device=rgb.device)
else:  # pragma: no cover - exercised only when torch is absent
    RecordingModel = None


@unittest.skipUnless(torch is not None, 'torch is required for evaluator tests')
class EvalFullTestCase(unittest.TestCase):
    def test_rgbdrop_and_irdrop_do_not_mutate_baseline_batch(self):
        imgs_rgb = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        imgs_ir = torch.full((1, 3, 4, 4), 2.0, dtype=torch.float32)
        prev_rgb = torch.full((1, 3, 4, 4), 3.0, dtype=torch.float32)
        prev_ir = torch.full((1, 3, 4, 4), 4.0, dtype=torch.float32)
        targets = torch.zeros((0, 7), dtype=torch.float32)

        dataloader = [(imgs_rgb, imgs_ir, targets, prev_rgb, prev_ir)]
        model = RecordingModel()
        evaluator = Evaluator(
            dataloader=dataloader,
            metrics_evaluator=DummyMetricsEvaluator(),
            device=torch.device('cpu'),
            extra_metrics_cfg={'cross_modal_robustness': {'enabled': True}},
        )

        with patch('src.engine.evaluator.flatten_predictions', lambda outputs: (outputs, None)), \
                patch('src.engine.evaluator.non_max_suppression_obb', lambda outputs, **kwargs: [
                    torch.zeros((0, 7), dtype=torch.float32) for _ in range(outputs.shape[0])
                ]):
            metrics = evaluator.evaluate(model, epoch=1)

        self.assertEqual(len(model.calls), 3)
        self.assertTrue(torch.allclose(model.calls[0]['rgb'], torch.ones_like(imgs_rgb)))
        self.assertTrue(torch.allclose(model.calls[0]['ir'], torch.full_like(imgs_ir, 2.0)))
        self.assertTrue(torch.allclose(model.calls[1]['rgb'], torch.zeros_like(imgs_rgb)))
        self.assertTrue(torch.allclose(model.calls[1]['ir'], torch.full_like(imgs_ir, 2.0)))
        self.assertTrue(torch.allclose(model.calls[2]['rgb'], torch.ones_like(imgs_rgb)))
        self.assertTrue(torch.allclose(model.calls[2]['ir'], torch.zeros_like(imgs_ir)))
        self.assertTrue(torch.allclose(imgs_rgb, torch.ones_like(imgs_rgb)))
        self.assertTrue(torch.allclose(imgs_ir, torch.full_like(imgs_ir, 2.0)))
        self.assertIn('RGBOnly_mAP50', metrics)
        self.assertIn('IROnly_mAP50', metrics)
        self.assertIn('RGBDrop_mAP50', metrics)
        self.assertIn('IRDrop_mAP50', metrics)

    def test_legacy_eval_config_still_works(self):
        imgs_rgb = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        imgs_ir = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        prev_rgb = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        prev_ir = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        targets = torch.zeros((0, 7), dtype=torch.float32)

        model = RecordingModel()
        evaluator = Evaluator(
            dataloader=[(imgs_rgb, imgs_ir, targets, prev_rgb, prev_ir)],
            metrics_evaluator=DummyMetricsEvaluator(),
            device=torch.device('cpu'),
            extra_metrics_cfg={'extra_metrics': {'small_object': {'enabled': True, 'area_threshold': 24}}},
        )

        with patch('src.engine.evaluator.flatten_predictions', lambda outputs: (outputs, None)), \
                patch('src.engine.evaluator.non_max_suppression_obb', lambda outputs, **kwargs: [
                    torch.zeros((0, 7), dtype=torch.float32) for _ in range(outputs.shape[0])
                ]):
            metrics = evaluator.evaluate(model, epoch=1)

        self.assertEqual(len(model.calls), 1)
        self.assertIn('mAP_50', metrics)
        self.assertIn('mAP_50_95', metrics)

    def test_full_project_default_keeps_cross_modal_robustness_disabled(self):
        if load_config is None:
            self.skipTest('config stack requires OmegaConf')
        cfg = load_config('configs/main/full_project.yaml')
        eval_cfg = normalize_eval_metrics_cfg(cfg.get('eval', {}))

        self.assertFalse(eval_cfg['cross_modal_robustness']['enabled'])

    def test_formal_robustness_config_enables_only_cross_modal_eval(self):
        if load_config is None or normalize_infer_cfg is None:
            self.skipTest('config stack requires OmegaConf')
        cfg = load_config('configs/main/full_project_robustness.yaml')
        eval_cfg = normalize_eval_metrics_cfg(cfg.get('eval', {}))
        infer_cfg = normalize_infer_cfg(cfg.get('infer', {}), default_imgsz=cfg.dataset.imgsz, nms_cfg=cfg.val.nms)

        self.assertEqual(cfg.experiment.name, 'full_project_robustness')
        self.assertTrue(eval_cfg['cross_modal_robustness']['enabled'])
        self.assertEqual(eval_cfg['cross_modal_robustness']['base_metric'], 'mAP_50')
        self.assertFalse(infer_cfg['multi_scale']['enabled'])
        self.assertFalse(infer_cfg['tta']['enabled'])
        self.assertEqual(infer_cfg['classwise_conf_thresholds'], {})

    def test_full_project_default_keeps_detection_error_analysis_disabled(self):
        if load_config is None:
            self.skipTest('config stack requires OmegaConf')
        cfg = load_config('configs/main/full_project.yaml')
        eval_cfg = normalize_eval_metrics_cfg(cfg.get('eval', {}))

        self.assertFalse(eval_cfg['error_analysis']['enabled'])

    def test_formal_error_analysis_config_enables_only_detection_error_analysis(self):
        if load_config is None or normalize_infer_cfg is None:
            self.skipTest('config stack requires OmegaConf')
        cfg = load_config('configs/main/full_project_error_analysis.yaml')
        eval_cfg = normalize_eval_metrics_cfg(cfg.get('eval', {}))
        infer_cfg = normalize_infer_cfg(cfg.get('infer', {}), default_imgsz=cfg.dataset.imgsz, nms_cfg=cfg.val.nms)

        self.assertEqual(cfg.experiment.name, 'full_project_error_analysis')
        self.assertTrue(eval_cfg['error_analysis']['enabled'])
        self.assertFalse(eval_cfg['cross_modal_robustness']['enabled'])
        self.assertFalse(infer_cfg['multi_scale']['enabled'])
        self.assertFalse(infer_cfg['tta']['enabled'])
        self.assertEqual(infer_cfg['classwise_conf_thresholds'], {})

    def test_evaluator_can_export_detection_error_analysis(self):
        output_dir = Path('tests') / 'tmp_eval_error_analysis'
        if output_dir.exists():
            shutil.rmtree(output_dir)

        imgs_rgb = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        imgs_ir = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        prev_rgb = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        prev_ir = torch.ones((1, 3, 4, 4), dtype=torch.float32)
        targets = torch.zeros((0, 7), dtype=torch.float32)

        model = RecordingModel()
        evaluator = Evaluator(
            dataloader=[(imgs_rgb, imgs_ir, targets, prev_rgb, prev_ir)],
            metrics_evaluator=DummyMetricsEvaluator(),
            device=torch.device('cpu'),
            extra_metrics_cfg={
                'error_analysis': {
                    'enabled': True,
                    'output_dir': str(output_dir),
                    'export_json': True,
                    'export_csv': True,
                    'include_per_image': True,
                }
            },
        )

        try:
            with patch('src.engine.evaluator.flatten_predictions', lambda outputs: (outputs, None)), \
                    patch('src.engine.evaluator.non_max_suppression_obb', lambda outputs, **kwargs: [
                        torch.zeros((0, 7), dtype=torch.float32) for _ in range(outputs.shape[0])
                    ]):
                metrics = evaluator.evaluate(model, epoch=1)

            self.assertIn('ErrorAnalysis', metrics)
            self.assertIn('ErrorAnalysisFiles', metrics)
            self.assertTrue((output_dir / 'summary.json').exists())
            self.assertTrue((output_dir / 'per_image_errors.csv').exists())
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)


if __name__ == '__main__':
    unittest.main()
