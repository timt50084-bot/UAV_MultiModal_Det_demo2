import unittest

import torch

from src.utils.config import load_config
from src.utils.postprocess_tuning import apply_classwise_thresholds, normalize_infer_cfg
from src.utils.result_merge import merge_obb_predictions
from src.utils.tta import build_tta_transforms, invert_tta_predictions


class TTAAndTuningTestCase(unittest.TestCase):
    def test_tta_horizontal_flip_invert_obb(self):
        preds = torch.tensor([[20.0, 30.0, 10.0, 6.0, 0.4, 0.9, 0.0]], dtype=torch.float32)
        flipped = preds.clone()
        flipped[:, 0] = 64.0 - flipped[:, 0]
        flipped[:, 4] = -flipped[:, 4]

        restored = invert_tta_predictions(flipped, {'size': 64, 'horizontal_flip': True}, base_size=64)

        self.assertAlmostEqual(restored[0, 0].item(), preds[0, 0].item(), places=5)
        self.assertAlmostEqual(restored[0, 1].item(), preds[0, 1].item(), places=5)
        self.assertAlmostEqual(restored[0, 4].item(), preds[0, 4].item(), places=5)

    def test_multi_scale_predictions_can_merge(self):
        pred_a = torch.tensor([[32.0, 32.0, 20.0, 12.0, 0.1, 0.90, 0.0]], dtype=torch.float32)
        pred_b_scaled = torch.tensor([[48.0, 48.0, 30.0, 18.0, 0.1, 0.85, 0.0]], dtype=torch.float32)
        pred_b = invert_tta_predictions(pred_b_scaled, {'size': 96, 'horizontal_flip': False}, base_size=64)

        merged = merge_obb_predictions([pred_a, pred_b], method='nms', iou_threshold=0.3, max_det=10)

        self.assertEqual(merged.shape[0], 1)
        self.assertAlmostEqual(merged[0, 0].item(), 32.0, places=4)

    def test_classwise_threshold_override(self):
        preds = torch.tensor([
            [10.0, 10.0, 8.0, 8.0, 0.0, 0.25, 0.0],
            [12.0, 12.0, 8.0, 8.0, 0.0, 0.25, 1.0],
        ], dtype=torch.float32)

        filtered = apply_classwise_thresholds(
            preds,
            class_names=['car', 'truck'],
            global_conf_threshold=0.30,
            classwise_conf_thresholds={'car': 0.20, 'truck': 0.30},
        )

        self.assertEqual(filtered.shape[0], 1)
        self.assertEqual(int(filtered[0, 6].item()), 0)

    def test_empty_classwise_thresholds_preserve_legacy_behavior(self):
        preds = torch.tensor([
            [10.0, 10.0, 8.0, 8.0, 0.0, 0.35, 0.0],
            [12.0, 12.0, 8.0, 8.0, 0.0, 0.40, 1.0],
        ], dtype=torch.float32)

        filtered = apply_classwise_thresholds(
            preds,
            class_names=['car', 'truck'],
            global_conf_threshold=0.30,
            classwise_conf_thresholds={},
        )

        self.assertTrue(torch.equal(filtered, preds))

    def test_fast_mode_backward_compatible(self):
        infer_cfg = normalize_infer_cfg({}, default_imgsz=640, nms_cfg={'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 300})

        self.assertEqual(infer_cfg['mode'], 'fast')
        self.assertFalse(infer_cfg['multi_scale']['enabled'])
        self.assertFalse(infer_cfg['tta']['enabled'])
        self.assertAlmostEqual(infer_cfg['conf_threshold'], 0.001, places=6)

    def test_competition_mode_config_loads(self):
        cfg = load_config('configs/exp_competition_infer.yaml')
        infer_cfg = normalize_infer_cfg(cfg.infer, default_imgsz=cfg.dataset.imgsz, nms_cfg=cfg.val.nms)

        self.assertEqual(infer_cfg['mode'], 'competition')
        self.assertTrue(infer_cfg['multi_scale']['enabled'])
        self.assertTrue(infer_cfg['tta']['enabled'])
        self.assertIn(768, infer_cfg['multi_scale']['sizes'])

    def test_merge_method_handles_empty_predictions(self):
        merged = merge_obb_predictions([], method='nms', iou_threshold=0.5, max_det=10)
        self.assertEqual(merged.shape[0], 0)

    def test_build_tta_transforms_for_competition(self):
        infer_cfg = normalize_infer_cfg(
            {
                'mode': 'competition',
                'multi_scale': {'enabled': True, 'sizes': [640, 768]},
                'tta': {'enabled': True, 'horizontal_flip': True},
            },
            default_imgsz=640,
            nms_cfg={'conf_thres': 0.25, 'iou_thres': 0.45, 'max_det': 300},
        )
        transforms = build_tta_transforms(infer_cfg, 640)
        self.assertEqual(len(transforms), 4)

    def test_explicit_tta_can_be_enabled_while_fast_mode_stays_selected(self):
        infer_cfg = normalize_infer_cfg(
            {
                'mode': 'fast',
                'classwise_conf_thresholds': {},
                'multi_scale': {'enabled': True, 'sizes': [640, 768, 896]},
                'tta': {'enabled': True, 'horizontal_flip': True},
                'merge': {'method': 'nms', 'iou_threshold': 0.55, 'max_det': 300},
            },
            default_imgsz=640,
            nms_cfg={'conf_thres': 0.25, 'iou_thres': 0.45, 'max_det': 300},
        )

        self.assertEqual(infer_cfg['mode'], 'fast')
        self.assertTrue(infer_cfg['enabled'])
        self.assertTrue(infer_cfg['multi_scale']['enabled'])
        self.assertTrue(infer_cfg['tta']['enabled'])
        self.assertTrue(infer_cfg['tta']['horizontal_flip'])
        self.assertEqual(infer_cfg['classwise_conf_thresholds'], {})

    def test_results_oriented_mainline_configs_enable_only_classwise_thresholds(self):
        for config_path, experiment_name in [
            ('configs/main/full_project_results.yaml', 'full_project_results'),
            ('configs/main/tracking_final_results.yaml', 'tracking_final_results'),
        ]:
            cfg = load_config(config_path)
            infer_cfg = normalize_infer_cfg(cfg.infer, default_imgsz=cfg.dataset.imgsz, nms_cfg=cfg.val.nms)

            self.assertEqual(cfg.experiment.name, experiment_name)
            self.assertEqual(infer_cfg['mode'], 'fast')
            self.assertFalse(infer_cfg['tta']['enabled'])
            self.assertFalse(infer_cfg['multi_scale']['enabled'])
            self.assertIn('car', infer_cfg['classwise_conf_thresholds'])

    def test_formal_tta_mainline_configs_enable_tta_without_classwise_thresholds(self):
        for config_path, experiment_name in [
            ('configs/main/full_project_tta.yaml', 'full_project_tta'),
            ('configs/main/tracking_final_tta.yaml', 'tracking_final_tta'),
        ]:
            cfg = load_config(config_path)
            infer_cfg = normalize_infer_cfg(cfg.infer, default_imgsz=cfg.dataset.imgsz, nms_cfg=cfg.val.nms)

            self.assertEqual(cfg.experiment.name, experiment_name)
            self.assertEqual(infer_cfg['mode'], 'fast')
            self.assertTrue(infer_cfg['multi_scale']['enabled'])
            self.assertEqual(infer_cfg['multi_scale']['sizes'], [640, 768, 896])
            self.assertTrue(infer_cfg['tta']['enabled'])
            self.assertTrue(infer_cfg['tta']['horizontal_flip'])
            self.assertEqual(infer_cfg['classwise_conf_thresholds'], {})


if __name__ == '__main__':
    unittest.main()
