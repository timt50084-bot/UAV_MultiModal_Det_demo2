import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_detection_evaluators import (
    COMPARE_METRICS,
    build_compare_result,
    build_evaluator_specs,
    build_parity_eval_cfg,
    build_runtime_summary,
    compute_metric_drifts,
    evaluate_drift_gate,
    resolve_compare_device,
    write_compare_outputs,
)


class CompareDetectionEvaluatorsTestCase(unittest.TestCase):
    def test_build_evaluator_specs_returns_fixed_cpu_gpu_pairs(self):
        specs = build_evaluator_specs()

        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0]['name'], 'cpu_reference')
        self.assertEqual(specs[0]['evaluator'], 'cpu')
        self.assertEqual(specs[0]['obb_iou_backend'], 'cpu_polygon')
        self.assertEqual(specs[1]['name'], 'gpu_candidate')
        self.assertEqual(specs[1]['evaluator'], 'gpu')
        self.assertEqual(specs[1]['obb_iou_backend'], 'gpu_prob')

    def test_build_parity_eval_cfg_disables_auxiliary_sections(self):
        eval_cfg = build_parity_eval_cfg({
            'small_object': {'enabled': True, 'area_threshold': 24},
            'cross_modal_robustness': {'enabled': True},
            'temporal_stability': {'enabled': True},
            'group_eval': {'enabled': True, 'keys': ['weather']},
            'error_analysis': {'enabled': True, 'output_dir': 'outputs/error_analysis'},
        })

        self.assertTrue(eval_cfg['small_object']['enabled'])
        self.assertEqual(eval_cfg['small_object']['area_threshold'], 24)
        self.assertFalse(eval_cfg['cross_modal_robustness']['enabled'])
        self.assertFalse(eval_cfg['temporal_stability']['enabled'])
        self.assertFalse(eval_cfg['group_eval']['enabled'])
        self.assertFalse(eval_cfg['error_analysis']['enabled'])

    def test_compute_metric_drifts_and_gate_schema(self):
        cpu_metrics = {
            'mAP_50': 0.50,
            'mAP_50_95': 0.32,
            'Precision': 0.60,
            'Recall': 0.58,
            'mAP_S': 0.22,
        }
        gpu_metrics = {
            'mAP_50': 0.48,
            'mAP_50_95': 0.30,
            'Precision': 0.57,
            'Recall': 0.56,
            'mAP_S': 0.18,
        }

        drifts = compute_metric_drifts(cpu_metrics, gpu_metrics)
        gate = evaluate_drift_gate(drifts, strict=False)

        self.assertEqual(set(drifts.keys()), set(COMPARE_METRICS))
        self.assertAlmostEqual(drifts['mAP_50']['abs_diff'], 0.02, places=6)
        self.assertAlmostEqual(drifts['Precision']['gpu_minus_cpu'], -0.03, places=6)
        self.assertEqual(gate['status'], 'ok')
        self.assertEqual(gate['per_metric']['mAP_50']['status'], 'ok')

    def test_strict_gate_fails_large_drift(self):
        drifts = compute_metric_drifts({'mAP_50': 0.50}, {'mAP_50': 0.65}, metric_names=('mAP_50',))
        gate = evaluate_drift_gate(
            drifts,
            strict=True,
            tolerances={'mAP_50': {'warn_abs': 0.03, 'fail_abs': 0.10}},
        )

        self.assertEqual(gate['status'], 'fail')
        self.assertEqual(gate['per_metric']['mAP_50']['status'], 'fail')
        self.assertTrue(gate['failures'])

    def test_build_runtime_summary_includes_speedup_fields(self):
        summary = build_runtime_summary(cpu_runtime_s=20.0, gpu_runtime_s=8.0, num_images=40)

        self.assertAlmostEqual(summary['speedup_vs_cpu'], 2.5, places=6)
        self.assertAlmostEqual(summary['cpu_images_per_second'], 2.0, places=6)
        self.assertAlmostEqual(summary['gpu_images_per_second'], 5.0, places=6)
        self.assertAlmostEqual(summary['cpu_ms_per_image'], 500.0, places=6)
        self.assertAlmostEqual(summary['gpu_ms_per_image'], 200.0, places=6)

    def test_build_compare_result_contains_expected_schema(self):
        specs = build_evaluator_specs()
        result = build_compare_result(
            metadata={
                'config_entry': 'configs/main/full_project.yaml',
                'source_config': 'configs/main/full_project.yaml',
                'device': 'cuda:0',
                'num_images': 100,
                'compare_metrics': list(COMPARE_METRICS),
            },
            cpu_result={
                'spec': specs[0],
                'metrics': {
                    'mAP_50': 0.50,
                    'mAP_50_95': 0.32,
                    'Precision': 0.60,
                    'Recall': 0.58,
                    'mAP_S': 0.22,
                },
                'runtime_s': 20.0,
            },
            gpu_result={
                'spec': specs[1],
                'metrics': {
                    'mAP_50': 0.48,
                    'mAP_50_95': 0.30,
                    'Precision': 0.57,
                    'Recall': 0.56,
                    'mAP_S': 0.18,
                },
                'runtime_s': 8.0,
            },
            strict=False,
        )

        self.assertIn('metadata', result)
        self.assertIn('cpu_reference', result)
        self.assertIn('gpu_candidate', result)
        self.assertIn('drift', result)
        self.assertIn('runtime', result)
        self.assertIn('gate', result)
        self.assertIn('summary', result)
        self.assertAlmostEqual(result['drift']['mAP_50']['abs_diff'], 0.02, places=6)
        self.assertAlmostEqual(result['runtime']['speedup_vs_cpu'], 2.5, places=6)

    def test_write_compare_outputs_emits_json_and_markdown(self):
        specs = build_evaluator_specs()
        result = build_compare_result(
            metadata={'compare_metrics': list(COMPARE_METRICS), 'num_images': 10},
            cpu_result={'spec': specs[0], 'metrics': {'mAP_50': 0.5}, 'runtime_s': 5.0},
            gpu_result={'spec': specs[1], 'metrics': {'mAP_50': 0.48}, 'runtime_s': 2.5},
            strict=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_compare_outputs(result, tmpdir)
            json_path = Path(paths['json'])
            markdown_path = Path(paths['markdown'])

            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())

            payload = json.loads(json_path.read_text(encoding='utf-8'))
            markdown = markdown_path.read_text(encoding='utf-8')

            self.assertIn('output_files', payload)
            self.assertIn('Detection Evaluator Parity Report', markdown)
            self.assertIn('Metric Comparison', markdown)

    def test_resolve_compare_device_requires_cuda(self):
        with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
            resolve_compare_device(-1, cuda_available=True)
        with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
            resolve_compare_device(0, cuda_available=False)
        self.assertEqual(resolve_compare_device(1, cuda_available=True), 'cuda:1')


if __name__ == '__main__':
    unittest.main()
