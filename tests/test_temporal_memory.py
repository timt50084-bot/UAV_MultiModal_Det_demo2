"""Compatibility tests for the legacy detector temporal-memory route.

The maintained detection mainline uses two-frame temporal. These tests stay to
guard archived configs and compatibility loading only.
"""

import unittest

import torch

from src.model.builder import _ensure_model_modules_registered, build_model
from src.model.temporal.temporal_memory import TemporalMemoryFusion


class TemporalMemoryTestCase(unittest.TestCase):
    """Legacy / compatibility coverage for detector temporal memory."""

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

    def test_memory_module_forward_shape(self):
        module = TemporalMemoryFusion(channels=[32, 64, 128, 256], memory_len=3, aggregator='weighted_avg')
        current_feats = self._make_features()
        memory_feats = [self._make_features(), self._make_features()]

        outputs = module(current_feats, memory_feats)
        self.assertEqual(len(outputs), 4)
        for output, current_feat in zip(outputs, current_feats):
            self.assertEqual(tuple(output.shape), tuple(current_feat.shape))
            self.assertTrue(torch.isfinite(output).all())

    def test_memory_mode_degrades_gracefully_when_empty(self):
        module = TemporalMemoryFusion(channels=[32, 64, 128, 256], memory_len=3)
        current_feats = self._make_features()

        outputs = module(current_feats, memory_feats=None)
        self.assertEqual(len(outputs), 4)
        for output, current_feat in zip(outputs, current_feats):
            self.assertEqual(tuple(output.shape), tuple(current_feat.shape))
            self.assertTrue(torch.isfinite(output).all())

    def test_detector_accepts_temporal_memory_config(self):
        model = build_model({
            "type": "YOLODualModalOBB",
            "num_classes": 1,
            "channels": [32, 64, 128, 256],
            "fusion": {"type": "SimpleConcatFusion"},
            "temporal": {
                "enabled": True,
                "mode": "memory",
                "memory_len": 3,
                "aggregator": "weighted_avg",
                "gate_hidden_ratio": 0.25,
            },
        })
        self.assertEqual(model.temporal_mode, "memory")
        self.assertEqual(model.temporal_memory_len, 3)
        self.assertIsNotNone(model.temporal_memory_fusion)

    def test_temporal_memory_buffer_update(self):
        model = build_model({
            "type": "YOLODualModalOBB",
            "num_classes": 1,
            "channels": [32, 64, 128, 256],
            "fusion": {"type": "SimpleConcatFusion"},
            "temporal": {
                "enabled": True,
                "mode": "memory",
                "memory_len": 2,
            },
        })
        feats_a = self._make_features()
        feats_b = self._make_features()
        feats_c = self._make_features()

        model.reset_temporal_memory()
        model.update_temporal_memory(feats_a)
        model.update_temporal_memory(feats_b)
        model.update_temporal_memory(feats_c)

        memory_bank = model.get_temporal_memory()
        self.assertEqual(len(memory_bank), 2)
        self.assertEqual(len(memory_bank[0]), 4)

    def test_two_frame_path_still_works(self):
        model = build_model({
            "type": "YOLODualModalOBB",
            "num_classes": 1,
            "channels": [32, 64, 128, 256],
            "fusion": {"type": "SimpleConcatFusion"},
            "temporal": {
                "enabled": True,
                "mode": "two_frame",
            },
        })
        self.assertEqual(model.temporal_mode, "two_frame")
        self.assertIsNotNone(model.temporal_fpn)

    def test_legacy_fusion_att_type_still_builds(self):
        # Keep one explicit compatibility test for the deprecated fusion entry.
        model = build_model({
            "type": "YOLODualModalOBB",
            "num_classes": 1,
            "channels": [32, 64, 128, 256],
            "fusion_att_type": "SimpleConcatFusion",
            "temporal": {
                "enabled": False,
                "mode": "off",
            },
        })
        self.assertEqual(model.fusion.__class__.__name__, "SimpleConcatFusion")


if __name__ == "__main__":
    unittest.main()
