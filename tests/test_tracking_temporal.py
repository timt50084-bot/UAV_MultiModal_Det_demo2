import unittest

import torch

from src.tracking.association import associate_tracks_to_detections
from src.tracking.memory import TrackMemoryBank
from src.tracking.track import Track
from src.tracking.tracker import MultiObjectTracker
from src.tracking.utils import normalize_tracking_cfg
from src.utils.config import load_config


class TrackingTemporalMemoryTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = normalize_tracking_cfg(
            {
                'enabled': True,
                'max_age': 3,
                'min_hits': 1,
                'init_score_threshold': 0.1,
                'match_iou_threshold': 0.1,
                'max_center_distance': 30.0,
                'use_class_constraint': True,
                'use_kalman': True,
                'appearance': {
                    'enabled': True,
                    'embedding_dim': 8,
                    'update_mode': 'ema',
                    'history_size': 5,
                    'ema_momentum': 0.8,
                    'sampling_radius': 1,
                    'use_rgb_ir_branches': False,
                },
                'memory': {
                    'enabled': True,
                    'size': 3,
                    'fusion': 'weighted_mean',
                    'decay': 0.8,
                    'use_temporal_consistency': True,
                    'lost_track_expansion': 1.25,
                },
                'smoothing': {
                    'enabled': True,
                    'bbox_ema': 0.7,
                    'angle_ema': 0.6,
                },
                'association': {
                    'use_appearance': True,
                    'use_temporal_memory': True,
                    'w_motion': 0.25,
                    'w_iou': 0.60,
                    'w_app': 0.60,
                    'w_temporal': 0.30,
                    'w_score': 0.15,
                    'class_mismatch_penalty': 1000000.0,
                    'appearance_gate': 0.6,
                    'lost_track_expansion': 1.25,
                },
            }
        )

    def _det(self, cx, cy, w=20.0, h=10.0, angle=0.0, score=0.9, class_id=0):
        return torch.tensor([cx, cy, w, h, angle, score, class_id], dtype=torch.float32)

    def test_track_memory_bank_updates(self):
        bank = TrackMemoryBank({'enabled': True, 'size': 3, 'fusion': 'weighted_mean', 'decay': 0.8})
        for index in range(4):
            bank.append(self._det(10 + index, 10)[:5], score=0.9, frame_index=index, embedding={'fused': torch.ones(8) * (index + 1)})
        self.assertEqual(len(bank), 3)
        self.assertEqual(bank.get_entries()[0]['frame_index'], 1)
        self.assertEqual(bank.get_entries()[-1]['frame_index'], 3)

    def test_aggregated_embedding_computation(self):
        bank = TrackMemoryBank({'enabled': True, 'size': 5, 'fusion': 'weighted_mean', 'decay': 0.5})
        first = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        second = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        bank.append(self._det(0, 0)[:5], score=0.9, frame_index=0, embedding={'fused': first})
        aggregated_single = bank.get_aggregated_embedding('fused')
        self.assertTrue(torch.allclose(aggregated_single, first, atol=1e-5))

        bank.append(self._det(1, 0)[:5], score=0.9, frame_index=1, embedding={'fused': second})
        aggregated = bank.get_aggregated_embedding('fused')
        expected = torch.tensor([0.4472, 0.8944, 0.0, 0.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(aggregated, expected, atol=1e-3))

    def test_temporal_memory_association_prefers_stable_track(self):
        appearance_cfg = self.cfg['appearance']
        memory_cfg = self.cfg['memory']
        smoothing_cfg = self.cfg['smoothing']

        track_a = Track(
            1,
            self._det(50, 50)[:5],
            0.9,
            0,
            appearance={'fused': torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
            appearance_cfg=appearance_cfg,
            memory_cfg=memory_cfg,
            smoothing_cfg=smoothing_cfg,
            frame_index=0,
        )
        track_a.update(self._det(52, 50, score=0.88), appearance={'fused': torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}, frame_index=1)
        track_a.mark_lost()

        track_b = Track(
            2,
            self._det(80, 50)[:5],
            0.9,
            0,
            appearance={'fused': torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
            appearance_cfg=appearance_cfg,
            memory_cfg=memory_cfg,
            smoothing_cfg=smoothing_cfg,
            frame_index=0,
        )
        track_b.update(self._det(81, 50, score=0.87), appearance={'fused': torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}, frame_index=1)
        track_b.mark_lost()

        detections = torch.stack([self._det(53, 50, score=0.86)])
        appearance_payload = {'fused': torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)}
        matches, unmatched_tracks, unmatched_dets, _ = associate_tracks_to_detections(
            [track_a, track_b],
            detections,
            self.cfg,
            appearance_features=appearance_payload,
        )
        self.assertEqual(matches, [(0, 0)])
        self.assertEqual(unmatched_tracks, [1])
        self.assertEqual(unmatched_dets, [])

    def test_fallback_to_stage2_when_memory_disabled(self):
        stage2_cfg = normalize_tracking_cfg(
            {
                'enabled': True,
                'appearance': {'enabled': True, 'embedding_dim': 8, 'history_size': 5},
                'memory': {'enabled': False},
                'association': {'use_appearance': True, 'use_temporal_memory': False},
            }
        )
        tracker = MultiObjectTracker(stage2_cfg)
        frame1 = tracker.update(
            torch.stack([self._det(20, 20)]),
            appearance_features={'fused': torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)},
        )
        frame2 = tracker.update(
            torch.stack([self._det(21, 20, score=0.88)]),
            appearance_features={'fused': torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)},
        )
        self.assertEqual(frame1[0]['track_id'], frame2[0]['track_id'])

    def test_tracking_temporal_config_loads(self):
        cfg = load_config('configs/exp_tracking_temporal.yaml')
        self.assertTrue(cfg.tracking.enabled)
        self.assertTrue(cfg.tracking.appearance.enabled)
        self.assertTrue(cfg.tracking.memory.enabled)
        self.assertTrue(cfg.tracking.association.use_temporal_memory)
        self.assertTrue(cfg.tracking.smoothing.enabled)

    def test_angle_smoothing_handles_periodicity(self):
        appearance_cfg = self.cfg['appearance']
        memory_cfg = self.cfg['memory']
        smoothing_cfg = self.cfg['smoothing']
        track = Track(
            1,
            self._det(0, 0, angle=3.05)[:5],
            0.9,
            0,
            appearance={'fused': torch.ones(8)},
            appearance_cfg=appearance_cfg,
            memory_cfg=memory_cfg,
            smoothing_cfg=smoothing_cfg,
            frame_index=0,
        )
        track.update(self._det(0, 0, angle=-3.05, score=0.88), appearance={'fused': torch.ones(8)}, frame_index=1)
        self.assertGreater(abs(track.bbox_obb[4].item()), 2.5)


if __name__ == '__main__':
    unittest.main()
