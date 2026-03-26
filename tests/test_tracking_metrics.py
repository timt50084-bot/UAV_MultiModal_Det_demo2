import unittest

from src.tracking.metrics import evaluate_mot_sequence


class TrackingMetricsTestCase(unittest.TestCase):
    def _obj(self, track_id, cx, cy, w=10.0, h=6.0, angle=0.0, class_id=0, score=0.9):
        return {
            'track_id': track_id,
            'class_id': class_id,
            'score': score,
            'obb': [cx, cy, w, h, angle],
        }

    def test_basic_mot_metrics_computation(self):
        gt_sequence = {
            'sequence_id': 'seq_01',
            'frames': [
                {'frame_index': 0, 'objects': [self._obj(1, 10, 10)]},
                {'frame_index': 1, 'objects': [self._obj(1, 11, 10)]},
            ],
        }
        pred_sequence = {
            'sequence_id': 'seq_01',
            'frames': [
                {'frame_index': 0, 'results': [self._obj(100, 10, 10)]},
                {'frame_index': 1, 'results': [self._obj(101, 11, 10)]},
            ],
        }

        result = evaluate_mot_sequence(pred_sequence, gt_sequence, config={'matching': {'iou_threshold': 0.5, 'use_obb_iou': True, 'class_aware': True}})
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['metrics']['MOTA'], 0.5, places=4)
        self.assertAlmostEqual(result['metrics']['IDF1'], 0.5, places=4)
        self.assertEqual(result['metrics']['IDSwitches'], 1)


if __name__ == '__main__':
    unittest.main()
