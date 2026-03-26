import tempfile
import unittest

import torch

from src.tracking import MultiObjectTracker, TrackingEvaluator, associate_tracks_to_detections, normalize_tracking_cfg
from src.tracking.track import Track
from src.tracking.refinement import TrackAwareRefiner
from src.utils.config import load_config


class TrackingFinalTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = normalize_tracking_cfg(
            {
                'enabled': True,
                'max_age': 2,
                'min_hits': 1,
                'init_score_threshold': 0.1,
                'match_iou_threshold': 0.95,
                'max_center_distance': 10.0,
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
                    'max_prediction_only_steps': 1,
                    'predicted_track_score': 0.1,
                    'keep_small_tracked_candidates': True,
                    'keep_tracked_overlap_candidates': True,
                },
                'feature_assist': {
                    'enabled': True,
                    'source': 'temporal_fused',
                    'embedding_dim': 8,
                    'use_for_reactivation': True,
                    'use_for_overlap_resolution': True,
                    'sampling_radius': 1,
                },
                'reactivation': {
                    'enabled': True,
                    'max_reactivate_age': 4,
                    'use_memory_reactivation': True,
                    'use_feature_assist_reactivation': True,
                    'reactivation_gate': 0.4,
                },
                'overlap_disambiguation': {
                    'enabled': True,
                    'overlap_iou_threshold': 0.30,
                    'ambiguity_margin': 0.15,
                    'assist_margin': 0.05,
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
                    'use_feature_assist': True,
                    'dynamic_weighting': True,
                    'w_motion': 0.25,
                    'w_iou': 0.60,
                    'w_app': 0.60,
                    'w_temporal': 0.30,
                    'w_score': 0.15,
                    'w_assist': 0.35,
                    'rgb_bias_gain': 0.5,
                    'ir_bias_gain': 0.5,
                    'low_conf_motion_boost': 0.3,
                    'class_mismatch_penalty': 1000000.0,
                    'appearance_gate': 0.8,
                    'assist_gate': 0.8,
                    'lost_track_expansion': 1.25,
                },
            }
        )

    def _det(self, cx, cy, w=20.0, h=10.0, angle=0.0, score=0.9, class_id=0):
        return torch.tensor([cx, cy, w, h, angle, score, class_id], dtype=torch.float32)

    def _appearance_payload(self, vectors):
        fused = torch.stack(vectors).float()
        return {
            'fused': fused,
            'rgb': fused.clone(),
            'ir': fused.clone(),
            'masks': {
                'fused': torch.ones(fused.shape[0], dtype=torch.bool),
                'rgb': torch.ones(fused.shape[0], dtype=torch.bool),
                'ir': torch.ones(fused.shape[0], dtype=torch.bool),
            },
        }

    def _assist_payload(self, vectors):
        fused = torch.stack(vectors).float()
        return {
            'fused': fused,
            'rgb': fused.clone(),
            'ir': fused.clone(),
            'masks': {
                'fused': torch.ones(fused.shape[0], dtype=torch.bool),
                'rgb': torch.ones(fused.shape[0], dtype=torch.bool),
                'ir': torch.ones(fused.shape[0], dtype=torch.bool),
            },
            'source': 'temporal_fused',
        }

    def _reliability_payload(self, num=1, rgb=0.7, ir=0.3, fused=0.65):
        return {
            'rgb_reliability': torch.full((num,), float(rgb), dtype=torch.float32),
            'ir_reliability': torch.full((num,), float(ir), dtype=torch.float32),
            'fused_reliability': torch.full((num,), float(fused), dtype=torch.float32),
            'masks': {
                'rgb_reliability': torch.ones(num, dtype=torch.bool),
                'ir_reliability': torch.ones(num, dtype=torch.bool),
                'fused_reliability': torch.ones(num, dtype=torch.bool),
            },
        }

    def test_feature_assist_optional_path(self):
        cfg = normalize_tracking_cfg(self.cfg)
        cfg['feature_assist']['enabled'] = False
        cfg['association']['use_feature_assist'] = False
        cfg['reactivation']['enabled'] = False
        tracker = MultiObjectTracker(cfg)
        frame1 = tracker.update(
            torch.stack([self._det(20, 20)]),
            appearance_features=self._appearance_payload([torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]),
            reliability_features=self._reliability_payload(1),
        )
        frame2 = tracker.update(
            torch.stack([self._det(21, 20, score=0.88)]),
            appearance_features=self._appearance_payload([torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]),
            reliability_features=self._reliability_payload(1),
        )
        self.assertEqual(frame1[0]['track_id'], frame2[0]['track_id'])
        self.assertFalse(frame2[0]['has_feature_assist'])

    def test_memory_and_feature_assist_reactivation_priority(self):
        tracker = MultiObjectTracker(self.cfg)
        det0 = torch.stack([self._det(20, 20, score=0.92)])
        appearance0 = self._appearance_payload([torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
        assist0 = self._assist_payload([torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
        tracker.update(det0, frame_meta={'frame_index': 0}, appearance_features=appearance0, reliability_features=self._reliability_payload(1), feature_assist_features=assist0)
        track = tracker.tracks[0]
        track.mark_lost()

        score_feature, source_feature, detail_feature = tracker._score_reactivation_candidate(
            track,
            self._det(32, 20, score=0.86),
            0,
            appearance_features=self._appearance_payload([torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]),
            feature_assist_features=self._assist_payload([torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]),
            use_memory=True,
            use_feature_assist=True,
        )
        self.assertGreaterEqual(score_feature, self.cfg['reactivation']['reactivation_gate'])
        self.assertEqual(source_feature, 'feature_assist_reactivation')
        self.assertIsNotNone(detail_feature['feature_assist_similarity'])

        score_memory, source_memory, detail_memory = tracker._score_reactivation_candidate(
            track,
            self._det(32, 20, score=0.86),
            0,
            appearance_features=self._appearance_payload([torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]),
            feature_assist_features=self._assist_payload([torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]),
            use_memory=True,
            use_feature_assist=True,
        )
        self.assertGreaterEqual(score_memory, self.cfg['reactivation']['reactivation_gate'])
        self.assertEqual(source_memory, 'memory_reactivation')
        self.assertIsNotNone(detail_memory['memory_similarity'])

    def test_overlap_disambiguation_in_crowd(self):
        appearance_cfg = self.cfg['appearance']
        memory_cfg = self.cfg['memory']
        modality_cfg = self.cfg['modality']
        feature_assist_cfg = self.cfg['feature_assist']
        smoothing_cfg = self.cfg['smoothing']
        track_a = Track(1, self._det(50, 50)[:5], 0.9, 0, appearance={'fused': torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}, feature_assist={'fused': torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}, appearance_cfg=appearance_cfg, memory_cfg=memory_cfg, modality_cfg=modality_cfg, feature_assist_cfg=feature_assist_cfg, smoothing_cfg=smoothing_cfg, min_hits=1)
        track_b = Track(2, self._det(56, 50)[:5], 0.9, 0, appearance={'fused': torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}, feature_assist={'fused': torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}, appearance_cfg=appearance_cfg, memory_cfg=memory_cfg, modality_cfg=modality_cfg, feature_assist_cfg=feature_assist_cfg, smoothing_cfg=smoothing_cfg, min_hits=1)
        detections = torch.stack([self._det(52, 50, score=0.86), self._det(54, 50, score=0.87)])
        matches, _, _, info = associate_tracks_to_detections(
            [track_a, track_b],
            detections,
            self.cfg,
            appearance_features=self._appearance_payload([
                torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ]),
            reliability_features=self._reliability_payload(2),
            feature_assist_features=self._assist_payload([
                torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ]),
        )
        self.assertEqual(len(matches), 2)
        self.assertGreaterEqual(info['overlap_summary']['overlap_disambiguation_count'], 1)

    def test_track_state_machine_transitions(self):
        track = Track(
            1,
            self._det(20, 20)[:5],
            0.9,
            0,
            appearance={'fused': torch.ones(8)},
            feature_assist={'fused': torch.ones(8)},
            appearance_cfg=self.cfg['appearance'],
            memory_cfg=self.cfg['memory'],
            modality_cfg=self.cfg['modality'],
            feature_assist_cfg=self.cfg['feature_assist'],
            smoothing_cfg=self.cfg['smoothing'],
            min_hits=2,
            frame_index=0,
        )
        self.assertEqual(track.state, Track.TENTATIVE)
        track.update(self._det(20, 20, score=0.88), appearance={'fused': torch.ones(8)}, feature_assist={'fused': torch.ones(8)}, frame_index=1, association_summary={})
        self.assertEqual(track.state, Track.TRACKED)
        track.mark_lost()
        self.assertEqual(track.state, Track.LOST)
        track.update(self._det(20, 20, score=0.20), appearance={'fused': torch.ones(8)}, feature_assist={'fused': torch.ones(8)}, frame_index=2, association_summary={'predicted_candidate': True})
        self.assertEqual(track.state, Track.PREDICTED_ONLY)
        track.update(self._det(20, 20, score=0.85), appearance={'fused': torch.ones(8)}, feature_assist={'fused': torch.ones(8)}, frame_index=3, association_summary={})
        self.assertEqual(track.state, Track.TRACKED)
        track.mark_lost()
        track.update(self._det(20, 20, score=0.86), appearance={'fused': torch.ones(8)}, feature_assist={'fused': torch.ones(8)}, frame_index=4, association_summary={'reactivation_source': 'feature_assist_reactivation'})
        self.assertEqual(track.state, Track.REACTIVATING)
        track.mark_removed()
        self.assertEqual(track.state, Track.REMOVED)

    def test_tracking_final_config_loads(self):
        cfg = load_config('configs/exp_tracking_final.yaml')
        self.assertTrue(cfg.tracking.enabled)
        self.assertTrue(cfg.tracking.feature_assist.enabled)
        self.assertTrue(cfg.tracking.reactivation.enabled)
        self.assertTrue(cfg.tracking.overlap_disambiguation.enabled)
        self.assertTrue(cfg.tracking.association.use_feature_assist)

    def test_runtime_summary_generated_without_gt(self):
        pred_sequences = {
            'seq_final': {
                'sequence_id': 'seq_final',
                'frames': [
                    {
                        'frame_index': 0,
                        'image_id': '000000.jpg',
                        'results': [
                            {
                                'track_id': 1,
                                'class_id': 0,
                                'score': 0.85,
                                'obb': [10.0, 10.0, 8.0, 8.0, 0.0],
                                'state': 'reactivating',
                                'feature_assist_reactivation': True,
                                'overlap_disambiguated': True,
                                'overlap_disambiguation_helped': True,
                                'predicted_only_to_tracked': True,
                            }
                        ],
                        'advanced_summary': {
                            'feature_assist_reactivation_count': 1,
                            'memory_reactivation_count': 0,
                            'predicted_candidate_reactivation_count': 0,
                            'overlap_disambiguation_count': 1,
                            'overlap_disambiguation_helped_count': 1,
                            'reactivating_state_count': 1,
                            'predicted_only_to_tracked_count': 1,
                        },
                        'metadata': {'time_of_day': 'night'},
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = TrackingEvaluator({'enabled': True, 'output_dir': tmpdir, 'export_results': False})
            result = evaluator.evaluate_from_sequences(pred_sequences, gt_sequences=None)
            self.assertFalse(result['available'])
            summary = result['analysis']['summary']
            self.assertEqual(summary['feature_assist_reactivation_count'], 1)
            self.assertEqual(summary['overlap_disambiguation_count'], 1)
            self.assertEqual(summary['predicted_only_to_tracked_count'], 1)


if __name__ == '__main__':
    unittest.main()
