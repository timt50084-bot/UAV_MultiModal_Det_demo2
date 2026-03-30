import unittest

import torch

from src.metrics.task_metrics import normalize_eval_metrics_cfg

from src.tracking import (
    TrackingErrorAnalyzer,
    compute_dynamic_weight_profile,
    maybe_extract_detection_reliability_features,
    normalize_tracking_cfg,
)
from src.tracking.track import Track
from src.utils.config import load_config
from src.utils.postprocess_tuning import normalize_infer_cfg
from tools.infer import build_tracking_frame_meta, resolve_tracking_scene_context


class TrackingModalityTestCase(unittest.TestCase):
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
                    'use_scene_adaptation': True,
                    'reliability_source': 'auto',
                    'reliability_ema': 0.8,
                    'night_motion_boost': 0.2,
                    'fog_temporal_boost': 0.2,
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

    def test_reliability_fields_optional_path(self):
        detections = torch.stack([self._det(16, 16)])
        self.assertIsNone(maybe_extract_detection_reliability_features(None, detections, self.cfg))

        feature_payload = {
            'fused_feats': (torch.ones(1, 8, 16, 16),),
            'rgb_feats': (torch.ones(1, 8, 16, 16) * 2.0,),
            'ir_feats': (torch.ones(1, 8, 16, 16) * 0.5,),
            'input_hw': (64, 64),
        }
        reliability = maybe_extract_detection_reliability_features(feature_payload, detections, self.cfg)
        self.assertIsNotNone(reliability)
        self.assertEqual(reliability['rgb_reliability'].shape[0], 1)
        self.assertGreater(reliability['rgb_reliability'][0].item(), reliability['ir_reliability'][0].item())

        profile = compute_dynamic_weight_profile(
            self.cfg['association'],
            self.cfg['modality'],
            detection_reliability=None,
            track_reliability=None,
        )
        self.assertEqual(profile['association_mode'], 'stage3_fallback')

    def test_modality_dynamic_weighting_changes_cost(self):
        rgb_profile = compute_dynamic_weight_profile(
            self.cfg['association'],
            self.cfg['modality'],
            detection_reliability={'rgb_reliability': 0.9, 'ir_reliability': 0.1, 'fused_reliability': 0.85},
            track_reliability={'rgb_reliability': 0.8, 'ir_reliability': 0.2, 'fused_reliability': 0.80},
        )
        low_conf_profile = compute_dynamic_weight_profile(
            self.cfg['association'],
            self.cfg['modality'],
            detection_reliability={'rgb_reliability': 0.2, 'ir_reliability': 0.2, 'fused_reliability': 0.2},
            track_reliability={'rgb_reliability': 0.2, 'ir_reliability': 0.2, 'fused_reliability': 0.2},
        )
        self.assertEqual(rgb_profile['association_mode'], 'rgb_dominant')
        self.assertGreater(rgb_profile['w_app'], low_conf_profile['w_app'])
        self.assertGreater(low_conf_profile['w_motion'], rgb_profile['w_motion'])
        self.assertTrue(low_conf_profile['low_confidence_motion_fallback'])

    def test_track_reliability_memory_updates(self):
        track = Track(
            1,
            self._det(20, 20)[:5],
            0.9,
            0,
            appearance={'fused': torch.ones(8), 'rgb': torch.ones(8), 'ir': torch.zeros(8)},
            reliability={'rgb_reliability': 0.8, 'ir_reliability': 0.2, 'fused_reliability': 0.7},
            appearance_cfg=self.cfg['appearance'],
            memory_cfg=self.cfg['memory'],
            modality_cfg=self.cfg['modality'],
            smoothing_cfg=self.cfg['smoothing'],
            frame_index=0,
        )
        track.update(
            self._det(21, 20, score=0.88),
            appearance={'fused': torch.ones(8), 'rgb': torch.ones(8), 'ir': torch.zeros(8)},
            reliability={'rgb_reliability': 0.7, 'ir_reliability': 0.3, 'fused_reliability': 0.68},
            frame_index=1,
        )
        track.update(
            self._det(22, 20, score=0.87),
            appearance={'fused': torch.ones(8), 'rgb': torch.ones(8), 'ir': torch.zeros(8)},
            reliability={'rgb_reliability': 0.9, 'ir_reliability': 0.1, 'fused_reliability': 0.82},
            frame_index=2,
        )
        self.assertEqual(len(track.reliability_history), 3)
        summary = track.get_reliability_summary()
        self.assertIsNotNone(summary)
        self.assertGreater(summary['rgb_reliability'], summary['ir_reliability'])
        self.assertEqual(track.to_result()['aggregated_rgb_reliability'] is not None, True)

    def test_scene_adaptation_graceful_fallback(self):
        base_profile = compute_dynamic_weight_profile(
            self.cfg['association'],
            self.cfg['modality'],
            detection_reliability={'rgb_reliability': 0.6, 'ir_reliability': 0.4, 'fused_reliability': 0.6},
            track_reliability={'rgb_reliability': 0.6, 'ir_reliability': 0.4, 'fused_reliability': 0.6},
            frame_meta=None,
        )
        night_profile = compute_dynamic_weight_profile(
            self.cfg['association'],
            self.cfg['modality'],
            detection_reliability={'rgb_reliability': 0.6, 'ir_reliability': 0.4, 'fused_reliability': 0.6},
            track_reliability={'rgb_reliability': 0.6, 'ir_reliability': 0.4, 'fused_reliability': 0.6},
            frame_meta={'time_of_day': 'night', 'weather': 'fog'},
        )
        self.assertFalse(base_profile['scene_adapted'])
        self.assertTrue(night_profile['scene_adapted'])
        self.assertGreater(night_profile['w_temporal'], base_profile['w_temporal'])

    def test_visibility_only_scene_context_can_trigger_scene_adaptation(self):
        visibility_profile = compute_dynamic_weight_profile(
            self.cfg['association'],
            self.cfg['modality'],
            detection_reliability={'rgb_reliability': 0.6, 'ir_reliability': 0.4, 'fused_reliability': 0.6},
            track_reliability={'rgb_reliability': 0.6, 'ir_reliability': 0.4, 'fused_reliability': 0.6},
            frame_meta={'visibility': 'low'},
        )
        self.assertTrue(visibility_profile['scene_adapted'])

    def test_tracking_modality_config_loads(self):
        cfg = load_config('configs/exp_tracking_modality.yaml')
        self.assertTrue(cfg.tracking.enabled)
        self.assertTrue(cfg.tracking.appearance.enabled)
        self.assertTrue(cfg.tracking.memory.enabled)
        self.assertTrue(cfg.tracking.modality.enabled)
        self.assertTrue(cfg.tracking.association.use_modality_awareness)
        self.assertTrue(cfg.tracking.association.dynamic_weighting)

    def test_tracking_final_default_keeps_scene_adaptation_disabled(self):
        cfg = load_config('configs/main/tracking_final.yaml')
        self.assertFalse(cfg.tracking.modality.use_scene_adaptation)

    def test_formal_scene_adaptive_config_loads_without_enabling_other_eval_or_infer_features(self):
        cfg = load_config('configs/main/tracking_final_scene_adaptive.yaml')
        infer_cfg = normalize_infer_cfg(cfg.get('infer', {}), default_imgsz=cfg.dataset.imgsz, nms_cfg=cfg.val.nms)
        eval_cfg = normalize_eval_metrics_cfg(cfg.get('eval', {}))

        self.assertEqual(cfg.experiment.name, 'tracking_final_scene_adaptive')
        self.assertTrue(cfg.tracking.modality.use_scene_adaptation)
        self.assertEqual(cfg.tracking.modality.scene_context.time_of_day, 'night')
        self.assertEqual(cfg.tracking.modality.scene_context.weather, 'fog')
        self.assertEqual(cfg.tracking.modality.scene_context.visibility, 'low')
        self.assertFalse(infer_cfg['multi_scale']['enabled'])
        self.assertFalse(infer_cfg['tta']['enabled'])
        self.assertEqual(infer_cfg['classwise_conf_thresholds'], {})
        self.assertFalse(eval_cfg['cross_modal_robustness']['enabled'])
        self.assertFalse(eval_cfg['error_analysis']['enabled'])

    def test_tracking_frame_meta_includes_scene_context_when_provided(self):
        scene_context = resolve_tracking_scene_context(
            {
                'modality': {
                    'scene_context': {
                        'time_of_day': 'night',
                        'weather': 'fog',
                        'visibility': 'low',
                    }
                }
            }
        )
        frame_meta = build_tracking_frame_meta(3, 'D:/frames/seqA/000003.jpg', sequence_mode=True, scene_context=scene_context)

        self.assertEqual(frame_meta['frame_index'], 3)
        self.assertEqual(frame_meta['image_id'], '000003.jpg')
        self.assertEqual(frame_meta['sequence_id'], 'seqA')
        self.assertEqual(frame_meta['time_of_day'], 'night')
        self.assertEqual(frame_meta['weather'], 'fog')
        self.assertEqual(frame_meta['visibility'], 'low')

    def test_tracking_frame_meta_safe_fallback_without_scene_context(self):
        scene_context = resolve_tracking_scene_context({'modality': {'scene_context': {'time_of_day': '', 'weather': '', 'visibility': ''}}})
        frame_meta = build_tracking_frame_meta(0, 'D:/frames/seqA/000000.jpg', sequence_mode=True, scene_context=scene_context)

        self.assertEqual(scene_context, {})
        self.assertNotIn('time_of_day', frame_meta)
        self.assertNotIn('weather', frame_meta)
        self.assertNotIn('visibility', frame_meta)

    def test_modality_helped_reactivation_summary(self):
        fake_eval_result = {
            'available': True,
            'per_sequence': {
                'seq_modality': {
                    'metrics': {
                        'MOTA': 0.8,
                        'IDF1': 0.75,
                        'IDSwitches': 0,
                        'Fragmentations': 1,
                        'RecoveredTracks': 1,
                        'RecoveryOpportunities': 1,
                        'num_frames': 2,
                    },
                    'details': {
                        'frame_summaries': [
                            {
                                'frame_index': 0,
                                'image_id': '000000.jpg',
                                'metadata': {'time_of_day': 'night', 'weather': 'fog'},
                                'matches': [
                                    {
                                        'association_mode': 'rgb_dominant',
                                        'low_confidence_motion_fallback': False,
                                        'modality_helped_reactivation': True,
                                    }
                                ],
                                'id_switch_events': [],
                                'unmatched_gt_track_ids': [],
                                'unmatched_pred_track_ids': [],
                            }
                        ],
                        'gt_track_stats': {
                            '1': {
                                'class_id': 0,
                                'total_frames': 2,
                                'matched_frames': 2,
                                'matched_ratio': 1.0,
                                'id_switches': 0,
                                'fragmentations': 1,
                                'reactivated': True,
                                'areas': [16.0, 16.0],
                                'size_group': 'small',
                            }
                        },
                    },
                }
            },
        }
        analysis = TrackingErrorAnalyzer(
            {'grouped_analysis': {'enabled': True, 'keys': ['time_of_day', 'weather']}, 'small_object_area_threshold': 32},
            class_names=['car'],
        ).analyze(fake_eval_result)
        summary = analysis['summary']
        self.assertEqual(summary['rgb_dominant_association_count'], 1)
        self.assertEqual(summary['modality_helped_reactivation_count'], 1)
        self.assertIn('night', summary['grouped_analysis']['time_of_day'])


if __name__ == '__main__':
    unittest.main()
