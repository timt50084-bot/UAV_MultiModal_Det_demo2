import unittest

import torch

from src.tracking import MultiObjectTracker, TrackAwareRefiner, normalize_tracking_cfg
from src.utils.config import load_config


class TrackingJointLiteTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = normalize_tracking_cfg(
            {
                'enabled': True,
                'max_age': 5,
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
                    'use_rgb_ir_branches': True,
                },
                'memory': {
                    'enabled': True,
                    'size': 3,
                    'fusion': 'weighted_mean',
                    'decay': 0.8,
                    'use_temporal_consistency': True,
                    'lost_track_expansion': 1.25,
                },
                'modality': {
                    'enabled': True,
                    'use_scene_adaptation': False,
                    'reliability_source': 'auto',
                    'reliability_ema': 0.8,
                },
                'refinement': {
                    'enabled': True,
                    'rescue_low_score': True,
                    'rescue_score_threshold': 0.15,
                    'rescue_match_iou': 0.2,
                    'rescue_motion_gate': 0.5,
                    'enable_track_guided_prediction': True,
                    'max_prediction_only_steps': 2,
                    'predicted_track_score': 0.1,
                    'keep_small_tracked_candidates': True,
                    'keep_tracked_overlap_candidates': True,
                },
                'smoothing': {
                    'enabled': True,
                    'bbox_ema': 0.7,
                    'angle_ema': 0.6,
                },
                'association': {
                    'use_appearance': True,
                    'use_temporal_memory': True,
                    'use_modality_awareness': True,
                    'dynamic_weighting': True,
                    'w_motion': 0.25,
                    'w_iou': 0.60,
                    'w_app': 0.60,
                    'w_temporal': 0.30,
                    'w_score': 0.15,
                    'rgb_bias_gain': 0.5,
                    'ir_bias_gain': 0.5,
                    'low_conf_motion_boost': 0.3,
                    'class_mismatch_penalty': 1000000.0,
                    'appearance_gate': 0.6,
                    'lost_track_expansion': 1.25,
                },
            }
        )

    def _det(self, cx, cy, w=20.0, h=10.0, angle=0.0, score=0.9, class_id=0):
        return torch.tensor([cx, cy, w, h, angle, score, class_id], dtype=torch.float32)

    def _appearance(self, fused_value=1.0):
        return {
            'fused': torch.ones(8) * fused_value,
            'rgb': torch.ones(8) * fused_value,
            'ir': torch.ones(8) * fused_value,
        }

    def _reliability(self, rgb=0.6, ir=0.4, fused=0.6):
        return {
            'rgb_reliability': rgb,
            'ir_reliability': ir,
            'fused_reliability': fused,
        }

    def test_refinement_disabled_falls_back_stage5(self):
        cfg = normalize_tracking_cfg(self.cfg)
        cfg['refinement']['enabled'] = False
        refiner = TrackAwareRefiner(cfg)
        detections = torch.stack([self._det(20, 20, score=0.18)])
        refined, _, _, refinement_payload, summary = refiner.refine(
            detections,
            tracks=[],
            base_score_threshold=0.25,
            appearance_features=None,
            reliability_features=None,
        )
        self.assertEqual(refined.shape[0], 1)
        self.assertEqual(summary['rescued_detection_count'], 0)
        self.assertFalse(bool(refinement_payload['rescued_mask'].any().item()))

    def test_low_score_rescue_with_track_support(self):
        tracker = MultiObjectTracker(self.cfg)
        frame1 = tracker.update(
            torch.stack([self._det(20, 20, score=0.9)]),
            appearance_features={'fused': torch.stack([self._appearance()['fused']]), 'rgb': torch.stack([self._appearance()['rgb']]), 'ir': torch.stack([self._appearance()['ir']]), 'masks': {'fused': torch.tensor([True]), 'rgb': torch.tensor([True]), 'ir': torch.tensor([True])}},
            reliability_features={'rgb_reliability': torch.tensor([0.7]), 'ir_reliability': torch.tensor([0.3]), 'fused_reliability': torch.tensor([0.65]), 'masks': {'rgb_reliability': torch.tensor([True]), 'ir_reliability': torch.tensor([True]), 'fused_reliability': torch.tensor([True])}},
        )
        self.assertEqual(len(frame1), 1)

        refiner = TrackAwareRefiner(self.cfg)
        detections = torch.stack([self._det(21, 20, w=4.0, h=4.0, score=0.18)])
        appearance = {'fused': torch.stack([self._appearance()['fused']]), 'rgb': torch.stack([self._appearance()['rgb']]), 'ir': torch.stack([self._appearance()['ir']]), 'masks': {'fused': torch.tensor([True]), 'rgb': torch.tensor([True]), 'ir': torch.tensor([True])}}
        reliability = {'rgb_reliability': torch.tensor([0.7]), 'ir_reliability': torch.tensor([0.3]), 'fused_reliability': torch.tensor([0.65]), 'masks': {'rgb_reliability': torch.tensor([True]), 'ir_reliability': torch.tensor([True]), 'fused_reliability': torch.tensor([True])}}
        refined, _, _, refinement_payload, summary = refiner.refine(
            detections,
            tracks=tracker.tracks,
            base_score_threshold=0.25,
            appearance_features=appearance,
            reliability_features=reliability,
        )
        self.assertEqual(refined.shape[0], 1)
        self.assertEqual(summary['rescued_detection_count'], 1)
        self.assertTrue(bool(refinement_payload['rescued_mask'][0].item()))

    def test_no_track_no_rescue(self):
        refiner = TrackAwareRefiner(self.cfg)
        detections = torch.stack([self._det(40, 40, score=0.18)])
        refined, _, _, _, summary = refiner.refine(
            detections,
            tracks=[],
            base_score_threshold=0.25,
            appearance_features=None,
            reliability_features=None,
        )
        self.assertEqual(refined.shape[0], 0)
        self.assertEqual(summary['rescued_detection_count'], 0)

    def test_track_guided_prediction_limited_steps(self):
        tracker = MultiObjectTracker(self.cfg)
        refiner = TrackAwareRefiner(self.cfg)

        initial_det = torch.stack([self._det(20, 20, score=0.9)])
        appearance = {'fused': torch.stack([self._appearance()['fused']]), 'rgb': torch.stack([self._appearance()['rgb']]), 'ir': torch.stack([self._appearance()['ir']]), 'masks': {'fused': torch.tensor([True]), 'rgb': torch.tensor([True]), 'ir': torch.tensor([True])}}
        reliability = {'rgb_reliability': torch.tensor([0.7]), 'ir_reliability': torch.tensor([0.3]), 'fused_reliability': torch.tensor([0.65]), 'masks': {'rgb_reliability': torch.tensor([True]), 'ir_reliability': torch.tensor([True]), 'fused_reliability': torch.tensor([True])}}
        tracker.update(initial_det, frame_meta={'frame_index': 0}, appearance_features=appearance, reliability_features=reliability)

        predicted_states = []
        for frame_index in range(1, 5):
            detections = torch.zeros((0, 7), dtype=torch.float32)
            refined, refined_app, refined_rel, refinement_payload, summary = refiner.refine(
                detections,
                tracks=tracker.tracks,
                base_score_threshold=0.25,
                appearance_features=None,
                reliability_features=None,
                frame_meta={'frame_index': frame_index},
            )
            results = tracker.update(
                refined,
                frame_meta={'frame_index': frame_index},
                appearance_features=refined_app,
                reliability_features=refined_rel,
                refinement_payload=refinement_payload,
            )
            predicted_states.append((summary['track_guided_prediction_count'], len(results), results[0]['state'] if results else None))

        self.assertEqual(predicted_states[0][0], 1)
        self.assertEqual(predicted_states[0][2], 'predicted')
        self.assertEqual(predicted_states[1][0], 1)
        self.assertEqual(predicted_states[2][0], 0)
        self.assertEqual(predicted_states[3][0], 0)

    def test_jointlite_config_loads(self):
        cfg = load_config('configs/exp_tracking_jointlite.yaml')
        self.assertTrue(cfg.tracking.enabled)
        self.assertTrue(cfg.tracking.modality.enabled)
        self.assertTrue(cfg.tracking.refinement.enabled)
        self.assertTrue(cfg.tracking.refinement.enable_track_guided_prediction)

    def test_small_tracked_candidates_preserved_in_crowd(self):
        tracker = MultiObjectTracker(self.cfg)
        tracker.update(
            torch.stack([self._det(30, 30, w=4.0, h=4.0, score=0.9)]),
            appearance_features={'fused': torch.stack([self._appearance()['fused']]), 'rgb': torch.stack([self._appearance()['rgb']]), 'ir': torch.stack([self._appearance()['ir']]), 'masks': {'fused': torch.tensor([True]), 'rgb': torch.tensor([True]), 'ir': torch.tensor([True])}},
            reliability_features={'rgb_reliability': torch.tensor([0.7]), 'ir_reliability': torch.tensor([0.3]), 'fused_reliability': torch.tensor([0.65]), 'masks': {'rgb_reliability': torch.tensor([True]), 'ir_reliability': torch.tensor([True]), 'fused_reliability': torch.tensor([True])}},
        )
        refiner = TrackAwareRefiner(self.cfg)
        detections = torch.stack([self._det(30.5, 30.5, w=4.0, h=4.0, score=0.17)])
        appearance = {'fused': torch.stack([self._appearance()['fused']]), 'rgb': torch.stack([self._appearance()['rgb']]), 'ir': torch.stack([self._appearance()['ir']]), 'masks': {'fused': torch.tensor([True]), 'rgb': torch.tensor([True]), 'ir': torch.tensor([True])}}
        reliability = {'rgb_reliability': torch.tensor([0.7]), 'ir_reliability': torch.tensor([0.3]), 'fused_reliability': torch.tensor([0.65]), 'masks': {'rgb_reliability': torch.tensor([True]), 'ir_reliability': torch.tensor([True]), 'fused_reliability': torch.tensor([True])}}
        refined, _, _, refinement_payload, summary = refiner.refine(
            detections,
            tracks=tracker.tracks,
            base_score_threshold=0.25,
            appearance_features=appearance,
            reliability_features=reliability,
        )
        self.assertEqual(refined.shape[0], 1)
        self.assertTrue(bool(refinement_payload['rescued_small_mask'][0].item()))
        self.assertEqual(summary['rescued_small_object_count'], 1)


if __name__ == '__main__':
    unittest.main()
