import tempfile
import unittest
import warnings
from pathlib import Path

import cv2
import numpy as np

from src.tracking import TrackingEvaluator, normalize_tracking_eval_cfg, render_tracking_sequence
from src.utils.config import load_config


class TrackingEvalTestCase(unittest.TestCase):
    def _obj(self, track_id, cx, cy, w=10.0, h=6.0, angle=0.0, class_id=0, score=0.9, **extra):
        item = {
            'track_id': track_id,
            'class_id': class_id,
            'score': score,
            'obb': [cx, cy, w, h, angle],
        }
        item.update(extra)
        return item

    def _pred_sequences(self, with_metadata=True):
        metadata = {'time_of_day': 'night', 'weather': 'fog', 'size_group': 'small'} if with_metadata else {}
        return {
            'seq_eval': {
                'sequence_id': 'seq_eval',
                'frames': [
                    {
                        'frame_index': 0,
                        'image_id': '000000.jpg',
                        'results': [self._obj(10, 10, 10), self._obj(20, 40, 40, w=14.0, h=8.0, class_id=1)],
                        'metadata': dict(metadata),
                    },
                    {
                        'frame_index': 1,
                        'image_id': '000001.jpg',
                        'results': [self._obj(21, 41, 40, w=14.0, h=8.0, class_id=1)],
                        'metadata': dict(metadata),
                    },
                    {
                        'frame_index': 2,
                        'image_id': '000002.jpg',
                        'results': [self._obj(10, 12, 10), self._obj(21, 42, 40, w=14.0, h=8.0, class_id=1)],
                        'metadata': dict(metadata),
                    },
                ],
            }
        }

    def _gt_sequences(self, with_metadata=True):
        base_metadata = {'time_of_day': 'night', 'weather': 'fog', 'size_group': 'small'} if with_metadata else {}
        return {
            'seq_eval': {
                'sequence_id': 'seq_eval',
                'frames': [
                    {
                        'frame_index': 0,
                        'image_id': '000000.jpg',
                        'objects': [self._obj(1, 10, 10, w=8.0, h=8.0), self._obj(2, 40, 40, w=14.0, h=8.0, class_id=1)],
                        'metadata': dict(base_metadata),
                    },
                    {
                        'frame_index': 1,
                        'image_id': '000001.jpg',
                        'objects': [self._obj(1, 11, 10, w=8.0, h=8.0), self._obj(2, 41, 40, w=14.0, h=8.0, class_id=1)],
                        'metadata': dict(base_metadata),
                    },
                    {
                        'frame_index': 2,
                        'image_id': '000002.jpg',
                        'objects': [self._obj(1, 12, 10, w=8.0, h=8.0), self._obj(2, 42, 40, w=14.0, h=8.0, class_id=1)],
                        'metadata': dict(base_metadata),
                    },
                ],
            }
        }

    def test_tracking_eval_graceful_skip_without_gt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = normalize_tracking_eval_cfg({'enabled': True, 'output_dir': tmpdir, 'export_results': False})
            result = TrackingEvaluator(cfg).evaluate_from_sequences(self._pred_sequences(), gt_sequences=None)
            self.assertFalse(result['available'])
            self.assertEqual(result['reason'], 'missing_tracking_gt')

    def test_tracking_eval_reports_missing_predictions_when_only_gt_is_available(self):
        class DatasetWithTrackingGT:
            tracking_ground_truth = self._gt_sequences()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = normalize_tracking_eval_cfg({'enabled': True, 'output_dir': tmpdir, 'export_results': False})
            result = TrackingEvaluator(cfg).evaluate_from_dataset(DatasetWithTrackingGT())
            self.assertFalse(result['available'])
            self.assertEqual(result['reason'], 'missing_tracking_predictions')

    def test_tracking_error_analysis_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = normalize_tracking_eval_cfg({'enabled': True, 'output_dir': tmpdir, 'export_results': False})
            result = TrackingEvaluator(cfg, class_names=['car', 'truck']).evaluate_from_sequences(self._pred_sequences(), self._gt_sequences())
            self.assertTrue(result['available'])
            summary = result['analysis']['summary']
            self.assertEqual(result['metrics']['IDSwitches'], 1)
            self.assertEqual(summary['id_switch_count'], 1)
            self.assertGreaterEqual(summary['reactivated_tracks'], 1)
            self.assertIn('small_object_tracking', summary)

    def test_tracking_eval_exports_json_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = normalize_tracking_eval_cfg({'enabled': True, 'output_dir': tmpdir, 'export_results': True, 'export_mot_txt': True})
            result = TrackingEvaluator(cfg, class_names=['car', 'truck']).evaluate_from_sequences(self._pred_sequences(), self._gt_sequences())
            files = result['exported_files']
            self.assertTrue(Path(files['mot_metrics']).exists())
            self.assertTrue(Path(files['tracking_error_summary']).exists())
            self.assertTrue(Path(files['per_sequence_tracking_analysis']).exists())
            self.assertTrue(Path(files['per_track_analysis']).exists())
            self.assertTrue(Path(files['tracking_results']).exists())
            self.assertTrue(Path(files['tracking_results_mot_txt']).exists())

    def test_tracking_visualization_output_format(self):
        sequence = self._pred_sequences()['seq_eval']
        with tempfile.TemporaryDirectory() as tmpdir:
            image_root = Path(tmpdir) / 'images'
            output_dir = Path(tmpdir) / 'viz'
            image_root.mkdir(parents=True, exist_ok=True)
            for frame in sequence['frames']:
                canvas = np.zeros((64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(image_root / frame['image_id']), canvas)

            saved_paths = render_tracking_sequence(sequence, image_root=image_root, output_dir=output_dir, class_names=['car', 'truck'])
            self.assertEqual(len(saved_paths), 3)
            self.assertTrue(all(Path(path).exists() for path in saved_paths))

    def test_tracking_visualization_enabled_flag_is_respected(self):
        sequence = self._pred_sequences()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_root = Path(tmpdir) / 'images'
            image_root.mkdir(parents=True, exist_ok=True)
            for frame in sequence['seq_eval']['frames']:
                canvas = np.zeros((64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(image_root / frame['image_id']), canvas)

            cfg = normalize_tracking_eval_cfg({
                'enabled': True,
                'output_dir': tmpdir,
                'save_visualizations': True,
                'image_root': str(image_root),
                'visualization': {'enabled': False},
            })
            result = TrackingEvaluator(cfg).evaluate_from_sequences(sequence, gt_sequences=None)
            self.assertNotIn('visualizations', result.get('exported_files', {}))

    def test_tracking_eval_config_loads(self):
        cfg = load_config('configs/exp_tracking_eval.yaml')
        self.assertTrue(cfg.tracking_eval.enabled)
        self.assertTrue(cfg.tracking_eval.mot_metrics)
        self.assertTrue(cfg.tracking_eval.error_analysis)
        self.assertEqual(cfg.tracking_eval.long_track_min_length, 3)

    def test_tracking_eval_normalization_warns_and_ignores_mot_metrics_flag(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cfg = normalize_tracking_eval_cfg({'enabled': True, 'mot_metrics': False})
        self.assertTrue(cfg['mot_metrics'])
        self.assertTrue(any('tracking_eval.mot_metrics' in str(item.message) for item in caught))

    def test_grouped_tracking_analysis_graceful_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = normalize_tracking_eval_cfg({'enabled': True, 'output_dir': tmpdir, 'export_results': False})
            result = TrackingEvaluator(cfg).evaluate_from_sequences(self._pred_sequences(with_metadata=False), self._gt_sequences(with_metadata=False))
            grouped = result['analysis']['summary'].get('grouped_analysis', {})
            self.assertIsInstance(grouped, dict)
            self.assertTrue(all(value == {} for value in grouped.values()))


if __name__ == '__main__':
    unittest.main()
