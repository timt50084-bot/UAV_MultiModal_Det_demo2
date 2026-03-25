import unittest

import torch

from src.model.builder import _ensure_model_modules_registered, build_model
from src.registry.fusion_registry import FUSIONS


class ReliabilityFusionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_model_modules_registered()

    def _make_features(self):
        return (
            torch.randn(2, 32, 64, 64),
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 256, 8, 8),
        )

    def test_registry_builds_reliability_fusion(self):
        module = FUSIONS.build({
            "type": "ReliabilityAwareFusion",
            "channel_list": [32, 64, 128, 256],
        })
        self.assertEqual(module.__class__.__name__, "ReliabilityAwareFusion")

    def test_forward_shape_and_no_nan(self):
        module = FUSIONS.build({
            "type": "ReliabilityAwareFusion",
            "channel_list": [32, 64, 128, 256],
        })
        rgb_features = self._make_features()
        ir_features = self._make_features()

        outputs = module(rgb_features, ir_features)
        self.assertEqual(len(outputs), 4)

        for output, rgb_feature in zip(outputs, rgb_features):
            self.assertEqual(tuple(output.shape), tuple(rgb_feature.shape))
            self.assertTrue(torch.isfinite(output).all())

    def test_attention_map_path(self):
        module = FUSIONS.build({
            "type": "ReliabilityAwareFusion",
            "channel_list": [32, 64, 128, 256],
        })
        rgb_features = self._make_features()
        ir_features = self._make_features()

        outputs, maps = module(rgb_features, ir_features, return_attention_map=True, target_size=(128, 128))
        self.assertEqual(len(outputs), 4)
        self.assertIn("P2_Reliability_Map", maps)
        self.assertEqual(maps["P2_Reliability_Map"].shape, (2, 1, 128, 128))
        self.assertTrue(torch.isfinite(maps["P2_Reliability_Map"]).all())

    def test_detector_accepts_nested_fusion_config(self):
        model = build_model({
            "type": "YOLODualModalOBB",
            "num_classes": 1,
            "channels": [32, 64, 128, 256],
            "fusion": {"type": "ReliabilityAwareFusion"},
        })
        self.assertEqual(model.fusion.__class__.__name__, "ReliabilityAwareFusion")


if __name__ == "__main__":
    unittest.main()
