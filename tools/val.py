import argparse
import random

import numpy as np
import torch

from src.data.dataloader import build_dataloader
from src.engine.evaluator import Evaluator
from src.metrics.task_metrics import (
    describe_cross_modal_robustness,
    describe_detection_error_analysis,
    normalize_eval_metrics_cfg,
)
from src.metrics.obb_metrics import OBBMetricsEvaluator
from src.model.builder import build_model
from src.tracking import TrackingEvaluator, normalize_tracking_cfg, normalize_tracking_eval_cfg
from src.utils.config import load_config
from src.utils.config_utils import apply_experiment_runtime_overrides
from src.utils.postprocess_tuning import describe_classwise_thresholds, describe_tta_settings, normalize_infer_cfg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Validate the dual-modal OBB detector.',
        epilog='OmegaConf-style overrides are also accepted after argparse flags, for example: '
               'tracking_eval.results_path=outputs/tracking_results.json. '
               'This entry validates detections by default and does not generate tracking predictions automatically.',
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    parser.add_argument('--weights', type=str, default='', help='Checkpoint to validate. Required for detection validation unless tracking_eval.results_path is used.')
    parser.add_argument('--device', type=int, default=0, help='GPU id. Use -1 for CPU.')
    return parser.parse_known_args(argv)


def print_tracking_eval_result(result):
    if not result:
        return
    print('\nTracking evaluation')
    print('-' * 48)
    if not result.get('available', False):
        print(f"Status: skipped ({result.get('reason', 'missing_tracking_gt')})")
        if result.get('exported_files'):
            print(f"TrackingEvalFiles: {result['exported_files']}")
        return

    metrics = result.get('metrics') or {}
    print(f"MOTA: {metrics.get('MOTA')}")
    print(f"IDF1: {metrics.get('IDF1')}")
    print(f"IDSwitches: {metrics.get('IDSwitches')}")
    print(f"MostlyTracked: {metrics.get('MostlyTracked')}")
    print(f"MostlyLost: {metrics.get('MostlyLost')}")
    print(f"Fragmentations: {metrics.get('Fragmentations')}")
    if result.get('analysis'):
        summary = result['analysis'].get('summary') or {}
        print(f"TrackingEvalAnalysis: {summary}")
        refinement_keys = ['rescued_detection_count', 'rescued_small_object_count', 'track_guided_prediction_count', 'predicted_only_track_count', 'refinement_helped_reactivation_count', 'refinement_suppressed_false_drop_count']
        refinement_summary = {key: summary.get(key) for key in refinement_keys if key in summary}
        if refinement_summary:
            print(f"TrackingEvalRefinementSummary: {refinement_summary}")
        advanced_keys = ['feature_assist_reactivation_count', 'memory_reactivation_count', 'overlap_disambiguation_count', 'overlap_disambiguation_helped_count', 'reactivating_state_count', 'predicted_only_to_tracked_count', 'long_track_continuity_score', 'small_object_track_survival_rate']
        advanced_summary = {key: summary.get(key) for key in advanced_keys if key in summary}
        if advanced_summary:
            print(f"TrackingEvalAdvancedSummary: {advanced_summary}")
    if result.get('exported_files'):
        print(f"TrackingEvalFiles: {result['exported_files']}")


def run_tracking_eval_only(cfg):
    tracking_eval_cfg = normalize_tracking_eval_cfg(cfg.get('tracking_eval', {}))
    class_names = list(getattr(cfg.dataset, 'class_names', [])) if 'dataset' in cfg else []
    tracking_evaluator = TrackingEvaluator(tracking_eval_cfg, class_names=class_names)
    result = tracking_evaluator.evaluate_from_files(
        tracking_eval_cfg.get('results_path'),
        gt_path=tracking_eval_cfg.get('gt_path') or None,
    )
    print_tracking_eval_result(result)
    return result


def main():
    args, cli_overrides = parse_args()
    cfg = load_config(args.config, cli_args=cli_overrides)
    cfg, run_name = apply_experiment_runtime_overrides(cfg, config_path=args.config)

    tracking_cfg = normalize_tracking_cfg(cfg.get('tracking', {}))
    tracking_eval_cfg = normalize_tracking_eval_cfg(cfg.get('tracking_eval', {}))

    if tracking_eval_cfg['enabled'] and tracking_eval_cfg.get('results_path') and not args.weights:
        print(f'Experiment name: {run_name}')
        run_tracking_eval_only(cfg)
        return

    if not args.weights:
        raise ValueError('--weights is required for detection validation unless tracking_eval.results_path is provided.')

    device = torch.device('cpu' if args.device < 0 or not torch.cuda.is_available() else f'cuda:{args.device}')
    print(f'\nValidation device: {device}')
    print(f'Experiment name: {run_name}')

    if tracking_cfg['enabled']:
        print('Tracking is enabled in this config. Detection validation remains the default path.')
    if tracking_eval_cfg['enabled'] and not tracking_eval_cfg.get('results_path'):
        print(
            'Tracking evaluation is configured, but no tracking_eval.results_path was provided. '
            'This validation entry does not generate tracking predictions automatically, '
            'so tracking evaluation may be skipped unless precomputed tracking results are supplied.'
        )

    set_seed(42)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    val_loader, _ = build_dataloader(cfg, is_training=False)
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    extra_metrics_cfg = normalize_eval_metrics_cfg(cfg.get('eval', {}) if hasattr(cfg, 'get') else {})
    infer_cfg = normalize_infer_cfg(
        cfg.get('infer', {}) if hasattr(cfg, 'get') else {},
        default_imgsz=cfg.dataset.imgsz,
        nms_cfg=cfg.val.nms,
    )
    print(f'TTA: {describe_tta_settings(infer_cfg)}')
    print(f'Classwise thresholds: {describe_classwise_thresholds(infer_cfg.get("classwise_conf_thresholds", {}))}')
    print(f'Cross-modal robustness eval: {describe_cross_modal_robustness(extra_metrics_cfg.get("cross_modal_robustness", {}))}')
    print(f'Detection error analysis: {describe_detection_error_analysis(extra_metrics_cfg.get("error_analysis", {}))}')
    metrics_evaluator = OBBMetricsEvaluator(num_classes=cfg.model.num_classes, extra_metrics_cfg=extra_metrics_cfg)
    evaluator = Evaluator(
        val_loader,
        metrics_evaluator,
        device,
        nms_kwargs=cfg.val.nms,
        extra_metrics_cfg=extra_metrics_cfg,
        infer_cfg=infer_cfg,
    )

    print('\nStarting validation...')
    metrics = evaluator.evaluate(model, epoch='Final')

    if tracking_eval_cfg['enabled']:
        tracking_evaluator = TrackingEvaluator(tracking_eval_cfg, class_names=cfg.dataset.class_names)
        if tracking_eval_cfg.get('results_path'):
            tracking_result = tracking_evaluator.evaluate_from_files(
                tracking_eval_cfg.get('results_path'),
                gt_path=tracking_eval_cfg.get('gt_path') or None,
            )
        else:
            tracking_result = tracking_evaluator.evaluate_from_dataset(getattr(val_loader, 'dataset', None))
        metrics['TrackingEval'] = tracking_result.get('metrics')
        metrics['TrackingEvalReason'] = tracking_result.get('reason')
        metrics['TrackingEvalAnalysis'] = tracking_result.get('analysis', {}).get('summary') if tracking_result.get('analysis') else None
        if tracking_result.get('exported_files'):
            metrics['TrackingEvalFiles'] = tracking_result['exported_files']

    print('\nFinal metrics')
    print('-' * 48)
    print(f"mAP_50: {metrics.get('mAP_50', 0.0) * 100:.2f}%")
    print(f"mAP_50_95: {metrics.get('mAP_50_95', 0.0) * 100:.2f}%")
    print(f"Precision: {metrics.get('Precision', 0.0) * 100:.2f}%")
    print(f"Recall: {metrics.get('Recall', 0.0) * 100:.2f}%")
    if 'mAP_S' in metrics:
        print(f"mAP_S: {metrics.get('mAP_S', 0.0) * 100:.2f}%")
    if 'Recall_S' in metrics:
        print(f"Recall_S: {metrics.get('Recall_S', 0.0) * 100:.2f}%")
    if 'Precision_S' in metrics:
        print(f"Precision_S: {metrics.get('Precision_S', 0.0) * 100:.2f}%")
    if 'RGBOnly_mAP50' in metrics:
        print(f"RGBOnly_mAP50: {metrics['RGBOnly_mAP50'] * 100:.2f}%")
    if 'IROnly_mAP50' in metrics:
        print(f"IROnly_mAP50: {metrics['IROnly_mAP50'] * 100:.2f}%")
    if 'RGBDrop_mAP50' in metrics:
        print(f"RGBDrop_mAP50: {metrics['RGBDrop_mAP50'] * 100:.2f}%")
    if 'IRDrop_mAP50' in metrics:
        print(f"IRDrop_mAP50: {metrics['IRDrop_mAP50'] * 100:.2f}%")
    if 'CrossModalRobustness_RGBDrop' in metrics:
        print(f"CrossModalRobustness_RGBDrop: {metrics['CrossModalRobustness_RGBDrop']:.4f}")
    if 'CrossModalRobustness_IRDrop' in metrics:
        print(f"CrossModalRobustness_IRDrop: {metrics['CrossModalRobustness_IRDrop']:.4f}")
    if 'TemporalStability' in metrics:
        print(f"TemporalStability: {metrics['TemporalStability']}")
    if 'GroupedMetrics' in metrics:
        print(f"GroupedMetrics: {metrics['GroupedMetrics']}")
    if 'ErrorAnalysis' in metrics:
        print(f"ErrorAnalysis: {metrics['ErrorAnalysis']}")
    if 'ErrorAnalysisFiles' in metrics:
        print(f"ErrorAnalysisFiles: {metrics['ErrorAnalysisFiles']}")
    if 'TrackingEval' in metrics or 'TrackingEvalFiles' in metrics:
        print_tracking_eval_result(
            {
                'available': metrics.get('TrackingEval') is not None,
                'reason': metrics.get('TrackingEvalReason', 'missing_tracking_gt') if metrics.get('TrackingEval') is None else None,
                'metrics': metrics.get('TrackingEval'),
                'analysis': {'summary': metrics.get('TrackingEvalAnalysis')} if metrics.get('TrackingEvalAnalysis') is not None else None,
                'exported_files': metrics.get('TrackingEvalFiles', {}),
            }
        )


if __name__ == '__main__':
    main()
