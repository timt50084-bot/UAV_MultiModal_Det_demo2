import unittest
from pathlib import Path

from src.loss.builder import build_assigner
from src.model.builder import build_model
from src.utils.config import load_config
from src.utils.config_utils import apply_experiment_runtime_overrides


class ExperimentConfigSmokeTestCase(unittest.TestCase):
    def _load_and_build(self, config_path):
        cfg = load_config(config_path)
        cfg, run_name = apply_experiment_runtime_overrides(cfg, config_path=config_path)
        self.assertTrue(run_name)
        self.assertEqual(cfg.model.num_classes, cfg.assigner.num_classes)
        model = build_model(cfg.model)
        assigner = build_assigner(cfg.assigner)
        self.assertIsNotNone(model)
        self.assertIsNotNone(assigner)
        return cfg

    def test_baseline_config_loads(self):
        cfg = self._load_and_build('configs/exp_baseline.yaml')
        self.assertEqual(cfg.experiment.name, 'baseline')
        self.assertEqual(cfg.model.temporal.mode, 'off')

    def test_fusion_main_config_loads(self):
        cfg = self._load_and_build('configs/exp_fusion_main.yaml')
        self.assertEqual(cfg.experiment.name, 'fusion_main')
        self.assertEqual(cfg.model.fusion.type, 'ReliabilityAwareFusion')

    def test_assigner_main_config_loads(self):
        cfg = self._load_and_build('configs/exp_assigner_main.yaml')
        self.assertEqual(cfg.experiment.name, 'assigner_main')
        self.assertTrue(cfg.assigner.use_angle_aware_assign)

    def test_temporal_main_config_loads(self):
        cfg = self._load_and_build('configs/exp_temporal_main.yaml')
        self.assertEqual(cfg.experiment.name, 'temporal_main')
        self.assertEqual(cfg.model.temporal.mode, 'memory')

    def test_full_project_config_loads(self):
        cfg = self._load_and_build('configs/exp_full_project.yaml')
        self.assertEqual(cfg.experiment.name, 'full_project')
        self.assertEqual(cfg.model.temporal.mode, 'memory')
        self.assertTrue(cfg.assigner.use_angle_aware_assign)

    def test_formal_mainline_variants_load(self):
        variant_configs = [
            'configs/main/full_project_results.yaml',
            'configs/main/full_project_tta.yaml',
            'configs/main/full_project_robustness.yaml',
            'configs/main/full_project_misalignment.yaml',
            'configs/main/full_project_sensor_degradation.yaml',
            'configs/main/full_project_error_analysis.yaml',
            'configs/main/full_project_angle_loss.yaml',
            'configs/main/tracking_final_results.yaml',
            'configs/main/tracking_final_tta.yaml',
            'configs/main/tracking_final_scene_adaptive.yaml',
        ]
        for config_path in variant_configs:
            cfg = self._load_and_build(config_path)
            self.assertIsNotNone(cfg)

    def test_legacy_configs_still_load_if_present(self):
        legacy_configs = [
            'configs/exp_reliability_fusion.yaml',
            'configs/exp_angle_aware_assigner.yaml',
            'configs/exp_temporal_memory.yaml',
            'configs/exp_task_metrics.yaml',
        ]
        for config_path in legacy_configs:
            if Path(config_path).exists():
                cfg = self._load_and_build(config_path)
                self.assertIsNotNone(cfg)


if __name__ == '__main__':
    unittest.main()
