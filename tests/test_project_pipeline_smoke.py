"""Project-level smoke tests for final repository completeness."""

import unittest
from pathlib import Path

from src.utils.config import load_config
from tools.run_experiment_suite import build_experiment_suite
from tools.summarize_results import collect_project_summary


MAIN_CONFIGS = [
    'configs/exp_baseline.yaml',
    'configs/exp_fusion_main.yaml',
    'configs/exp_assigner_main.yaml',
    'configs/exp_temporal_main.yaml',
    'configs/exp_full_project.yaml',
    'configs/exp_tracking_base.yaml',
    'configs/exp_tracking_assoc.yaml',
    'configs/exp_tracking_temporal.yaml',
    'configs/exp_tracking_modality.yaml',
    'configs/exp_tracking_jointlite.yaml',
    'configs/exp_tracking_final.yaml',
]

DOC_TEMPLATES = [
    'docs/EXPERIMENT_PLAN.md',
    'docs/ABLATION_TABLE_TEMPLATE.md',
    'docs/TECHNICAL_PLAN_TEMPLATE.md',
    'docs/PPT_OUTLINE.md',
    'docs/DEMO_SCRIPT.md',
    'docs/RESULTS_TRACKING_TEMPLATE.md',
]


class ProjectPipelineSmokeTestCase(unittest.TestCase):
    def test_core_configs_load(self):
        for config_path in MAIN_CONFIGS:
            with self.subTest(config_path=config_path):
                self.assertTrue(Path(config_path).exists(), msg=config_path)
                cfg = load_config(config_path)
                self.assertIsNotNone(cfg)

    def test_experiment_suite_plan_builds(self):
        suite = build_experiment_suite(mode='plan', subset='all')
        self.assertEqual(len(suite), 11)
        names = [item['name'] for item in suite]
        self.assertIn('full_project', names)
        self.assertIn('tracking_final', names)

    def test_summary_tool_collects_without_results(self):
        summary = collect_project_summary(experiments_root='outputs/experiments', subset='all')
        self.assertIn('detection', summary)
        self.assertIn('tracking', summary)
        self.assertEqual(len(summary['detection']), 5)
        self.assertEqual(len(summary['tracking']), 6)

    def test_docs_templates_exist(self):
        for doc_path in DOC_TEMPLATES:
            with self.subTest(doc_path=doc_path):
                self.assertTrue(Path(doc_path).exists(), msg=doc_path)


if __name__ == '__main__':
    unittest.main()