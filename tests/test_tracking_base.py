import unittest

import torch

from src.tracking.association import associate_tracks_to_detections
from src.tracking.track import Track
from src.tracking.tracker import MultiObjectTracker
from src.tracking.utils import build_tracker_from_cfg, detections_to_results, normalize_tracking_cfg
from src.utils.config import load_config


class TrackingBaseTestCase(unittest.TestCase):
    def setUp(self):
        self.tracking_cfg = normalize_tracking_cfg(
            {
                'enabled': True,
                'max_age': 2,
                'min_hits': 1,
                'init_score_threshold': 0.3,
                'match_iou_threshold': 0.1,
                'max_center_distance': 25.0,
                'use_class_constraint': True,
                'use_kalman': True,
            }
        )

    def _det(self, cx, cy, w, h, angle=0.0, score=0.9, class_id=0):
        return torch.tensor([cx, cy, w, h, angle, score, class_id], dtype=torch.float32)

    def test_tracker_initialization(self):
        cfg = load_config('configs/exp_tracking_base.yaml')
        self.assertTrue(cfg.tracking.enabled)
        tracker = MultiObjectTracker(cfg.tracking, class_names=cfg.dataset.class_names)
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.cfg['method'], 'tracking_by_detection')

    def test_track_creation_and_update(self):
        tracker = MultiObjectTracker(self.tracking_cfg)
        frame1 = tracker.update(torch.stack([self._det(100, 100, 20, 10)]))
        self.assertEqual(len(frame1), 1)
        track_id = frame1[0]['track_id']

        frame2 = tracker.update(torch.stack([self._det(102, 101, 20, 10, angle=0.05, score=0.88)]))
        self.assertEqual(len(frame2), 1)
        self.assertEqual(frame2[0]['track_id'], track_id)

    def test_track_lost_and_reactivation(self):
        tracker = MultiObjectTracker(self.tracking_cfg)
        frame1 = tracker.update(torch.stack([self._det(60, 60, 16, 8)]))
        track_id = frame1[0]['track_id']

        frame2 = tracker.update(torch.zeros((0, 7), dtype=torch.float32))
        self.assertEqual(frame2, [])
        self.assertEqual(len(tracker.get_active_tracks()), 0)
        self.assertEqual(tracker.tracks[0].state, Track.LOST)

        frame3 = tracker.update(torch.stack([self._det(61, 59, 16, 8, angle=0.03, score=0.92)]))
        self.assertEqual(len(frame3), 1)
        self.assertEqual(frame3[0]['track_id'], track_id)
        self.assertEqual(frame3[0]['state'], Track.TRACKED)

    def test_tracking_disabled_keeps_old_behavior(self):
        tracker = build_tracker_from_cfg({'enabled': False})
        self.assertIsNone(tracker)
        results = detections_to_results(torch.stack([self._det(10, 20, 8, 6)]))
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]['track_id'])
        self.assertEqual(results[0]['state'], 'detection')

    def test_obb_tracking_output_format(self):
        tracker = MultiObjectTracker(self.tracking_cfg)
        results = tracker.update(torch.stack([self._det(120, 80, 18, 12, angle=0.2, score=0.95, class_id=2)]))
        self.assertEqual(len(results), 1)
        self.assertTrue({'track_id', 'class_id', 'score', 'obb', 'state'}.issubset(results[0].keys()))
        self.assertEqual(len(results[0]['obb']), 5)

    def test_association_respects_class_constraint(self):
        tracker = MultiObjectTracker(self.tracking_cfg)
        tracker.update(torch.stack([self._det(200, 200, 20, 10, class_id=0)]))
        tracks = tracker.get_active_tracks()
        detections = torch.stack([self._det(201, 201, 20, 10, class_id=1)])
        matches, unmatched_tracks, unmatched_dets, _ = associate_tracks_to_detections(tracks, detections, tracker.cfg)
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(unmatched_dets, [0])


if __name__ == '__main__':
    unittest.main()
