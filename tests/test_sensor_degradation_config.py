import unittest

from src.utils.config import load_config


class SensorDegradationConfigTestCase(unittest.TestCase):
    def test_full_project_default_keeps_sensor_degradation_disabled(self):
        cfg = load_config('configs/main/full_project.yaml')
        sensor_cfg = cfg.dataset.aug_cfg.sensor_degradation

        self.assertFalse(sensor_cfg.enabled)
        self.assertAlmostEqual(float(sensor_cfg.prob), 0.0, places=6)

    def test_formal_sensor_degradation_config_enables_only_sensor_degradation(self):
        cfg = load_config('configs/main/full_project_sensor_degradation.yaml')
        aug_cfg = cfg.dataset.aug_cfg

        self.assertEqual(cfg.experiment.name, 'full_project_sensor_degradation')
        self.assertTrue(aug_cfg.sensor_degradation.enabled)
        self.assertAlmostEqual(float(aug_cfg.sensor_degradation.prob), 0.3, places=6)
        self.assertFalse(aug_cfg.cross_modal_misalignment.enabled)
        self.assertTrue(aug_cfg.enable_cmcp)
        self.assertTrue(aug_cfg.enable_mrre)
        self.assertTrue(aug_cfg.enable_weather)
        self.assertTrue(aug_cfg.enable_modality_dropout)


if __name__ == '__main__':
    unittest.main()
