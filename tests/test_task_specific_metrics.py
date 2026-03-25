import unittest

from src.metrics.task_specific_metrics import (
    append_task_metrics,
    compute_cross_modal_robustness,
    compute_small_object_metrics,
    compute_temporal_stability,
    infer_sequence_metadata,
    resolve_small_area_threshold,
)


class TaskSpecificMetricsTestCase(unittest.TestCase):
    def test_small_object_precision_recall_computation(self):
        area_threshold = resolve_small_area_threshold(32)
        preds = [
            {'image_id': 'img1', 'bbox': [10, 10, 12, 12, 0.0], 'score': 0.9, 'class': 0},
            {'image_id': 'img1', 'bbox': [40, 40, 10, 10, 0.0], 'score': 0.7, 'class': 0},
        ]
        gts = [
            {'image_id': 'img1', 'bbox': [10, 10, 12, 12, 0.0], 'class': 0},
        ]
        match_iou_fn = lambda pred_box, gt_box: 1.0 if pred_box[:4] == gt_box[:4] else 0.0

        metrics = compute_small_object_metrics(preds, gts, num_classes=1, area_threshold=area_threshold, match_iou_fn=match_iou_fn)
        self.assertAlmostEqual(metrics['Recall_S'], 1.0)
        self.assertAlmostEqual(metrics['Precision_S'], 0.5)

        empty_metrics = compute_small_object_metrics([], [], num_classes=1, area_threshold=area_threshold, match_iou_fn=match_iou_fn)
        self.assertEqual(empty_metrics['Recall_S'], 0.0)
        self.assertEqual(empty_metrics['Precision_S'], 0.0)

    def test_cross_modal_robustness_metric_fields_exist(self):
        metrics = compute_cross_modal_robustness(
            {'mAP_50': 0.62},
            {'mAP_50': 0.41},
            {'mAP_50': 0.35},
            base_metric='mAP_50',
        )
        self.assertIn('CrossModalRobustness_RGBDrop', metrics)
        self.assertIn('CrossModalRobustness_IRDrop', metrics)
        self.assertAlmostEqual(metrics['CrossModalRobustness_RGBDrop'], 0.21, places=6)
        self.assertAlmostEqual(metrics['CrossModalRobustness_IRDrop'], 0.27, places=6)

    def test_temporal_stability_graceful_skip_without_sequence_info(self):
        preds = [{'image_id': 'img1', 'bbox': [10, 10, 12, 12, 0.0], 'score': 0.9, 'class': 0}]
        stability = compute_temporal_stability(
            preds,
            image_metadata={'img1': None},
            match_iou_fn=lambda a, b: 1.0,
        )
        self.assertIsNone(stability)

    def test_result_dict_backward_compatible(self):
        base_metrics = {'mAP_50': 0.5, 'mAP_S': 0.2}
        merged = append_task_metrics(base_metrics, {})
        self.assertEqual(merged['mAP_50'], 0.5)
        self.assertEqual(merged['mAP_S'], 0.2)
        self.assertNotIn('Recall_S', merged)

    def test_sequence_metadata_parser_and_temporal_stability(self):
        image_a = 'video_01_0001.jpg'
        image_b = 'video_01_0002.jpg'
        metadata = {
            image_a: infer_sequence_metadata(image_a),
            image_b: infer_sequence_metadata(image_b),
        }
        preds = [
            {'image_id': image_a, 'bbox': [10, 10, 12, 12, 0.0], 'score': 0.9, 'class': 0},
            {'image_id': image_b, 'bbox': [10.3, 10.1, 12, 12, 0.05], 'score': 0.88, 'class': 0},
        ]
        stability = compute_temporal_stability(
            preds,
            metadata,
            conf_threshold=0.25,
            match_iou_threshold=0.3,
            max_center_shift_ratio=0.2,
            match_iou_fn=lambda a, b: 0.8,
        )
        self.assertIsNotNone(stability)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)


if __name__ == '__main__':
    unittest.main()
