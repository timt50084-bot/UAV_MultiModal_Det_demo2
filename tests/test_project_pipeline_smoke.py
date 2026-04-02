"""Project-level smoke tests for final repository completeness."""

from io import StringIO
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

from tools import val as val_tool
from src.utils.config import load_config
from tools.run_experiment_suite import build_experiment_suite
from tools.summarize_results import collect_project_summary
from tools.val import parse_args as parse_val_args


MAIN_CONFIGS = [
    'configs/main/baseline.yaml',
    'configs/main/fusion_main.yaml',
    'configs/main/assigner_main.yaml',
    'configs/main/temporal_main.yaml',
    'configs/main/full_project.yaml',
    'configs/main/tracking_base.yaml',
    'configs/main/tracking_final.yaml',
    'configs/main/tracking_eval.yaml',
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
        self.assertEqual(len(suite), 7)
        names = [item['name'] for item in suite]
        self.assertIn('full_project', names)
        self.assertIn('tracking_final', names)

    def test_summary_tool_collects_without_results(self):
        summary = collect_project_summary(experiments_root='outputs/experiments', subset='all')
        self.assertIn('detection', summary)
        self.assertIn('tracking', summary)
        self.assertEqual(len(summary['detection']), 5)
        self.assertEqual(len(summary['tracking']), 2)

    def test_val_parser_accepts_omegaconf_style_overrides(self):
        args, overrides = parse_val_args([
            '--config', 'configs/main/tracking_eval.yaml',
            'tracking_eval.results_path=outputs/tracking_results.json',
            'tracking_eval.image_root=data/frames',
        ])
        self.assertEqual(args.config, 'configs/main/tracking_eval.yaml')
        self.assertEqual(
            overrides,
            [
                'tracking_eval.results_path=outputs/tracking_results.json',
                'tracking_eval.image_root=data/frames',
            ],
        )

    def test_val_entrypoint_warns_when_tracking_eval_has_no_results_path(self):
        cfg = OmegaConf.create({
            'dataset': {'class_names': ['car'], 'imgsz': 640},
            'model': {'num_classes': 1},
            'val': {'nms': {'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 50, 'max_wh': 4096.0}},
            'infer': {},
            'tracking': {'enabled': False},
            'tracking_eval': {'enabled': True, 'results_path': '', 'gt_path': '', 'image_root': ''},
            'eval': {'small_object': {'enabled': False}},
        })

        class DummyModel:
            def to(self, device):
                del device
                return self

            def load_state_dict(self, state_dict):
                del state_dict

        class FakeEvaluator:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def evaluate(self, model, epoch='Final'):
                del model, epoch
                return {'mAP_50': 0.5, 'mAP_50_95': 0.3, 'Precision': 0.4, 'Recall': 0.6}

        class FakeTrackingEvaluator:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def evaluate_from_dataset(self, dataset):
                del dataset
                return {
                    'available': False,
                    'reason': 'missing_tracking_predictions',
                    'metrics': None,
                    'analysis': {'summary': {'available': False, 'reason': 'missing_tracking_predictions'}},
                    'exported_files': {},
                }

        args = SimpleNamespace(config='configs/main/tracking_eval.yaml', weights='dummy.pt', device=-1)
        loader = SimpleNamespace(dataset=SimpleNamespace(class_names=['car'], tracking_ground_truth={'seq0': {'frames': []}}))
        stdout = StringIO()
        with patch('tools.val.parse_args', return_value=(args, [])), \
                patch('tools.val.load_config', return_value=cfg), \
                patch('tools.val.apply_experiment_runtime_overrides', return_value=(cfg, 'tracking_eval')), \
                patch('tools.val.build_dataloader', return_value=(loader, None)), \
                patch('tools.val.build_model', return_value=DummyModel()), \
                patch('tools.val.torch.load', return_value={}), \
                patch('tools.val.build_detection_evaluator', return_value=FakeEvaluator()), \
                patch('tools.val.TrackingEvaluator', FakeTrackingEvaluator), \
                patch('sys.stdout', stdout):
            val_tool.main()

        output = stdout.getvalue()
        self.assertIn('does not generate tracking predictions automatically', output)
        self.assertIn('Status: skipped (missing_tracking_predictions)', output)
        self.assertNotIn('mAP_S:', output)

    def test_docs_templates_exist(self):
        for doc_path in DOC_TEMPLATES:
            with self.subTest(doc_path=doc_path):
                self.assertTrue(Path(doc_path).exists(), msg=doc_path)


if __name__ == '__main__':
    unittest.main()
