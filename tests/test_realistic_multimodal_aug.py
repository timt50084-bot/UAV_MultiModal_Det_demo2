import unittest

import numpy as np

from src.data.transforms.augmentations import (
    CrossModalMisalignment,
    MultiModalAugmentationPipeline,
    SensorDegradationAug,
)


class RealisticMultimodalAugTestCase(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[0:64, 0:64]
        self.rgb = np.stack([
            (xx * 4).clip(0, 255),
            (yy * 4).clip(0, 255),
            ((xx + yy) * 2).clip(0, 255),
        ], axis=-1).astype(np.uint8)
        self.ir = np.stack([
            ((63 - xx) * 4).clip(0, 255),
            ((63 - yy) * 4).clip(0, 255),
            ((xx + yy) * 2).clip(0, 255),
        ], axis=-1).astype(np.uint8)
        self.labels = np.array([[0, 0.5, 0.5, 0.2, 0.1, 0.0]], dtype=np.float32)

    def test_cross_modal_misalignment_preserves_shape(self):
        aug = CrossModalMisalignment(
            enabled=True,
            prob=1.0,
            apply_to='ir',
            max_translate_ratio=0.02,
            max_rotate_deg=1.5,
            max_scale_delta=0.02,
        )
        out_rgb, out_ir = aug(self.rgb.copy(), self.ir.copy())

        self.assertEqual(out_rgb.shape, self.rgb.shape)
        self.assertEqual(out_ir.shape, self.ir.shape)

    def test_sensor_degradation_aug_preserves_shape_and_dtype(self):
        aug = SensorDegradationAug(
            enabled=True,
            prob=1.0,
            rgb={
                'overexposure_prob': 1.0,
                'flare_prob': 1.0,
                'haze_prob': 1.0,
                'max_exposure_gain': 1.3,
            },
            ir={
                'noise_prob': 1.0,
                'noise_std': 0.02,
                'drift_prob': 1.0,
                'max_drift': 0.05,
                'hotspot_prob': 1.0,
                'stripe_prob': 1.0,
            },
        )
        out_rgb, out_ir = aug(self.rgb.copy(), self.ir.copy())

        self.assertEqual(out_rgb.shape, self.rgb.shape)
        self.assertEqual(out_ir.shape, self.ir.shape)
        self.assertEqual(out_rgb.dtype, self.rgb.dtype)
        self.assertEqual(out_ir.dtype, self.ir.dtype)

    def test_pipeline_accepts_new_aug_config(self):
        pipeline = MultiModalAugmentationPipeline(
            enable_cmcp=False,
            enable_mrre=False,
            enable_weather=False,
            enable_modality_dropout=False,
            cross_modal_misalignment={
                'enabled': True,
                'prob': 1.0,
                'apply_to': 'ir',
                'max_translate_ratio': 0.01,
                'max_rotate_deg': 1.0,
                'max_scale_delta': 0.01,
            },
            sensor_degradation={
                'enabled': True,
                'prob': 1.0,
                'rgb': {'overexposure_prob': 1.0, 'flare_prob': 0.0, 'haze_prob': 0.0, 'max_exposure_gain': 1.2},
                'ir': {'noise_prob': 1.0, 'noise_std': 0.02, 'drift_prob': 1.0, 'max_drift': 0.05, 'hotspot_prob': 0.0},
            },
        )
        out_rgb, out_ir, out_labels = pipeline(self.rgb.copy(), self.ir.copy(), self.labels.copy())

        self.assertEqual(out_rgb.shape, self.rgb.shape)
        self.assertEqual(out_ir.shape, self.ir.shape)
        self.assertEqual(out_labels.shape, self.labels.shape)

    def test_misalignment_does_not_modify_labels(self):
        pipeline = MultiModalAugmentationPipeline(
            enable_cmcp=False,
            enable_mrre=False,
            enable_weather=False,
            enable_modality_dropout=False,
            cross_modal_misalignment={
                'enabled': True,
                'prob': 1.0,
                'apply_to': 'ir',
                'max_translate_ratio': 0.02,
                'max_rotate_deg': 1.5,
                'max_scale_delta': 0.02,
            },
        )
        labels_before = self.labels.copy()
        _, _, labels_after = pipeline(self.rgb.copy(), self.ir.copy(), self.labels.copy())

        np.testing.assert_allclose(labels_after, labels_before)

    def test_temporal_aug_param_sharing_minimal(self):
        pipeline = MultiModalAugmentationPipeline(
            enable_cmcp=False,
            enable_mrre=False,
            enable_weather=False,
            enable_modality_dropout=False,
            sensor_degradation={
                'enabled': True,
                'prob': 1.0,
                'rgb': {'overexposure_prob': 0.0, 'flare_prob': 0.0, 'haze_prob': 1.0, 'max_exposure_gain': 1.1},
                'ir': {'noise_prob': 0.0, 'noise_std': 0.0, 'drift_prob': 1.0, 'max_drift': 0.05, 'hotspot_prob': 0.0},
            },
        )
        prev_rgb = self.rgb.copy()
        prev_ir = self.ir.copy()
        out_rgb, out_ir, out_prev_rgb, out_prev_ir, _ = pipeline.apply_temporal_pair(
            self.rgb.copy(), self.ir.copy(), prev_rgb, prev_ir, self.labels.copy()
        )

        self.assertEqual(out_rgb.shape, self.rgb.shape)
        self.assertEqual(out_prev_rgb.shape, self.rgb.shape)
        self.assertEqual(out_ir.shape, self.ir.shape)
        self.assertEqual(out_prev_ir.shape, self.ir.shape)


if __name__ == '__main__':
    unittest.main()
