import shutil
import unittest
from pathlib import Path

from src.metrics.error_analysis import ErrorAnalyzer


class ErrorAnalysisTestCase(unittest.TestCase):
    def test_basic_fp_fn_summary(self):
        analyzer = ErrorAnalyzer(cfg={'error_analysis': {'enabled': False}}, class_names=['car'])
        preds = [
            {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'score': 0.9, 'class': 0},
            {'image_id': 'img1', 'bbox': [60, 60, 18, 18, 0.0], 'score': 0.6, 'class': 0},
        ]
        gts = [
            {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'class': 0},
            {'image_id': 'img2', 'bbox': [30, 30, 20, 20, 0.0], 'class': 0},
        ]

        result = analyzer.analyze(preds, gts, image_metadata={})

        self.assertEqual(result['summary']['total_tp'], 1)
        self.assertEqual(result['summary']['total_fp'], 1)
        self.assertEqual(result['summary']['total_fn'], 1)

    def test_small_object_error_bucket(self):
        analyzer = ErrorAnalyzer(
            cfg={
                'error_analysis': {
                    'enabled': False,
                    'buckets': {
                        'small_object_area_threshold': 32,
                        'area_bins': [16, 32, 96],
                    },
                },
            },
            class_names=['car'],
        )
        preds = [
            {'image_id': 'img1', 'bbox': [80, 80, 60, 60, 0.0], 'score': 0.95, 'class': 0},
        ]
        gts = [
            {'image_id': 'img1', 'bbox': [10, 10, 8, 8, 0.0], 'class': 0},
            {'image_id': 'img1', 'bbox': [40, 40, 20, 20, 0.0], 'class': 0},
            {'image_id': 'img1', 'bbox': [80, 80, 60, 60, 0.0], 'class': 0},
        ]

        result = analyzer.analyze(preds, gts, image_metadata={})

        self.assertEqual(result['summary']['small_fn'], 2)
        self.assertEqual(result['summary']['area_bucket_summary']['fn']['tiny'], 1)
        self.assertEqual(result['summary']['area_bucket_summary']['fn']['small'], 1)
        self.assertEqual(result['summary']['area_bucket_summary']['tp']['medium'], 1)

    def test_confusion_summary_generation(self):
        analyzer = ErrorAnalyzer(cfg={'error_analysis': {'enabled': False}}, class_names=['car', 'truck', 'bus'])
        preds = [
            {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'score': 0.9, 'class': 1},
        ]
        gts = [
            {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'class': 0},
        ]

        result = analyzer.analyze(preds, gts, image_metadata={})

        self.assertEqual(result['summary']['class_confusion_summary']['car->truck'], 1)

    def test_missing_metadata_graceful_degradation(self):
        analyzer = ErrorAnalyzer(cfg={'error_analysis': {'enabled': False}}, class_names=['car'])
        preds = [
            {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'score': 0.9, 'class': 0},
        ]
        gts = [
            {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'class': 0},
        ]

        result = analyzer.analyze(preds, gts, image_metadata={})

        self.assertIn('angle_error_summary', result['summary'])
        self.assertEqual(len(result['per_image_records']), 1)
        self.assertIsNone(result['per_image_records'][0]['sequence_id'])
        self.assertIsNone(result['per_image_records'][0]['time_of_day'])

    def test_export_json_csv(self):
        tmp_dir = Path('tests') / 'tmp_error_analysis_exports'
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        try:
            analyzer = ErrorAnalyzer(
                cfg={
                    'error_analysis': {
                        'enabled': True,
                        'output_dir': str(tmp_dir),
                        'export_json': True,
                        'export_csv': True,
                        'include_per_image': True,
                    },
                },
                class_names=['car'],
            )
            preds = [
                {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'score': 0.9, 'class': 0},
            ]
            gts = [
                {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'class': 0},
            ]

            result = analyzer.analyze(preds, gts, image_metadata={})

            self.assertTrue((tmp_dir / 'summary.json').exists())
            self.assertTrue((tmp_dir / 'per_image_errors.csv').exists())
            self.assertIn('summary_json', result['exported_files'])
            self.assertIn('per_image_csv', result['exported_files'])
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def test_modality_contribution_optional_path(self):
        analyzer = ErrorAnalyzer(
            cfg={'error_analysis': {'enabled': False, 'modality_contribution': {'enabled': True}}},
            class_names=['car'],
        )
        baseline_preds = [
            {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'score': 0.9, 'class': 0},
        ]
        gts = [
            {'image_id': 'img1', 'bbox': [10, 10, 20, 20, 0.0], 'class': 0},
        ]

        no_drop_result = analyzer.analyze(baseline_preds, gts, image_metadata={})
        self.assertFalse(no_drop_result['summary']['modality_contribution_summary']['available'])

        rgb_drop_data = {'preds': [], 'gts': gts, 'image_metadata': {}}
        ir_drop_data = {'preds': baseline_preds, 'gts': gts, 'image_metadata': {}}
        drop_result = analyzer.analyze(
            baseline_preds,
            gts,
            image_metadata={},
            rgb_drop_data=rgb_drop_data,
            ir_drop_data=ir_drop_data,
        )

        self.assertTrue(drop_result['summary']['modality_contribution_summary']['available'])
        self.assertEqual(drop_result['summary']['modality_contribution_summary']['rgb_dominant_success_count'], 1)


if __name__ == '__main__':
    unittest.main()
