import unittest

import torch

from src.tracking.appearance import maybe_extract_detection_appearance_features
from src.tracking.association import associate_tracks_to_detections
from src.tracking.track import Track
from src.tracking.tracker import MultiObjectTracker
from src.tracking.utils import normalize_tracking_cfg
from src.utils.config import load_config


class TrackingAssociationTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = normalize_tracking_cfg(
            {
                'enabled': True,
                'max_age': 2,
                'min_hits': 1,
                'init_score_threshold': 0.1,
                'match_iou_threshold': 0.1,
                'max_center_distance': 30.0,
                'use_class_constraint': True,
                'use_kalman': True,
                'appearance': {
                    'enabled': True,
                    'embedding_dim': 8,
                    'update_mode': 'queue_mean',
                    'history_size': 5,
                    'ema_momentum': 0.8,
                    'sampling_radius': 1,
                    'use_rgb_ir_branches': False,
                },
                'association': {
                    'use_appearance': True,
                    'w_motion': 0.25,
                    'w_iou': 0.60,
                    'w_app': 1.0,
                    'w_score': 0.15,
                    'class_mismatch_penalty': 1000000.0,
                    'appearance_gate': 0.6,
                },
            }
        )

    def _det(self, cx, cy, w=20.0, h=10.0, angle=0.0, score=0.9, class_id=0):
        return torch.tensor([cx, cy, w, h, angle, score, class_id], dtype=torch.float32)

    def _feature_payload(self):
        fused_feats = (
            torch.arange(1, 1 + 64 * 16 * 16, dtype=torch.float32).view(1, 64, 16, 16),
            torch.arange(1, 1 + 128 * 8 * 8, dtype=torch.float32).view(1, 128, 8, 8),
        )
        rgb_feats = (fused_feats[0] * 0.5, fused_feats[1] * 0.5)
        ir_feats = (fused_feats[0] * 1.5, fused_feats[1] * 1.5)
        return {
            'fused_feats': fused_feats,
            'rgb_feats': rgb_feats,
            'ir_feats': ir_feats,
            'input_hw': (128, 128),
        }

    def test_embedding_extraction_optional_path(self):
        detections = torch.stack([self._det(32, 32), self._det(96, 96)])
        disabled_cfg = normalize_tracking_cfg({'enabled': True, 'appearance': {'enabled': False}, 'association': {'use_appearance': False}})
        self.assertIsNone(maybe_extract_detection_appearance_features(self._feature_payload(), detections, disabled_cfg, base_size=128))

        payload = maybe_extract_detection_appearance_features(self._feature_payload(), detections, self.cfg, base_size=128)
        self.assertIsNotNone(payload)
        self.assertIn('fused', payload)
        self.assertEqual(payload['fused'].shape, (2, 8))
        self.assertTrue(torch.all(payload['masks']['fused']))

    def test_association_with_appearance_prefers_correct_track(self):
        appearance_cfg = self.cfg['appearance']
        track_a = Track(1, self._det(50, 50)[:5], 0.9, 0, appearance={'fused': torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}, appearance_cfg=appearance_cfg)
        track_b = Track(2, self._det(50, 50)[:5], 0.9, 0, appearance={'fused': torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}, appearance_cfg=appearance_cfg)
        detections = torch.stack([
            self._det(50, 50, score=0.88),
            self._det(50, 50, score=0.87),
        ])
        appearance_payload = {
            'fused': torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        }
        matches, _, _, _ = associate_tracks_to_detections([track_a, track_b], detections, self.cfg, appearance_features=appearance_payload)
        self.assertEqual(matches, [(0, 0), (1, 1)])

    def test_tracker_embedding_memory_updates(self):
        tracker = MultiObjectTracker(self.cfg)
        det = torch.stack([self._det(80, 80)])
        emb1 = {'fused': torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)}
        tracker.update(det, appearance_features=emb1)

        emb2 = {'fused': torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)}
        tracker.update(torch.stack([self._det(81, 80, score=0.88)]), appearance_features=emb2)

        track = tracker.get_active_tracks()[0]
        self.assertEqual(len(track.embedding_history), 2)
        expected = torch.tensor([0.7071, 0.7071, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(track.embedding, expected, atol=1e-3))

    def test_fallback_to_stage1_when_no_embedding(self):
        tracker = MultiObjectTracker(self.cfg)
        frame1 = tracker.update(torch.stack([self._det(20, 20)]), appearance_features=None)
        self.assertEqual(len(frame1), 1)
        frame2 = tracker.update(torch.stack([self._det(21, 20, score=0.88)]), appearance_features=None)
        self.assertEqual(len(frame2), 1)
        self.assertEqual(frame1[0]['track_id'], frame2[0]['track_id'])

    def test_tracking_assoc_config_loads(self):
        cfg = load_config('configs/exp_tracking_assoc.yaml')
        self.assertTrue(cfg.tracking.enabled)
        self.assertTrue(cfg.tracking.appearance.enabled)
        self.assertTrue(cfg.tracking.association.use_appearance)

    def test_class_constraint_and_appearance_work_together(self):
        appearance_cfg = self.cfg['appearance']
        track = Track(1, self._det(100, 100, class_id=0)[:5], 0.9, 0, appearance={'fused': torch.ones(8)}, appearance_cfg=appearance_cfg)
        detections = torch.stack([self._det(100, 100, class_id=1)])
        appearance_payload = {'fused': torch.ones((1, 8), dtype=torch.float32)}
        matches, unmatched_tracks, unmatched_dets, _ = associate_tracks_to_detections([track], detections, self.cfg, appearance_features=appearance_payload)
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(unmatched_dets, [0])


if __name__ == '__main__':
    unittest.main()
