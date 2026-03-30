import unittest

from src.utils.config import load_config


class CrossModalMisalignmentConfigTestCase(unittest.TestCase):
    def test_full_project_default_keeps_misalignment_disabled(self):
        cfg = load_config('configs/main/full_project.yaml')
        misalignment_cfg = cfg.dataset.aug_cfg.cross_modal_misalignment

        self.assertFalse(misalignment_cfg.enabled)
        self.assertAlmostEqual(float(misalignment_cfg.prob), 0.0, places=6)

    def test_formal_misalignment_config_enables_only_misalignment_addition(self):
        cfg = load_config('configs/main/full_project_misalignment.yaml')
        aug_cfg = cfg.dataset.aug_cfg

        self.assertEqual(cfg.experiment.name, 'full_project_misalignment')
        self.assertTrue(aug_cfg.cross_modal_misalignment.enabled)
        self.assertAlmostEqual(float(aug_cfg.cross_modal_misalignment.prob), 0.2, places=6)
        self.assertEqual(aug_cfg.cross_modal_misalignment.apply_to, 'ir')
        self.assertTrue(aug_cfg.enable_cmcp)
        self.assertTrue(aug_cfg.enable_mrre)
        self.assertTrue(aug_cfg.enable_weather)
        self.assertTrue(aug_cfg.enable_modality_dropout)
        self.assertFalse(aug_cfg.sensor_degradation.enabled)


if __name__ == '__main__':
    unittest.main()
