import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any, Dict, Iterable, List, Optional

from tools.experiment_registry import get_experiments, infer_run_root


DETECTION_METRIC_CANDIDATES = ('metrics.json', 'eval_metrics.json', 'val_metrics.json', 'detection_metrics.json')


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _to_cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{}]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_cell(row.get(key)) for key in fieldnames})
    return str(path)


def _write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return str(path)


def _collect_detection_record(spec, experiments_root='outputs/experiments'):
    run_root = infer_run_root(spec, experiments_root=experiments_root)
    metrics_path = _first_existing(run_root / name for name in DETECTION_METRIC_CANDIDATES)
    metrics_data = _read_json(metrics_path) if metrics_path else None
    error_summary_path = run_root / 'error_analysis' / 'summary.json'
    error_summary = _read_json(error_summary_path)

    record = {
        'experiment': spec.name,
        'category': spec.category,
        'config': spec.config,
        'run_root': str(run_root),
        'metrics_path': '' if metrics_path is None else str(metrics_path),
        'error_summary_path': str(error_summary_path) if error_summary_path.exists() else '',
        'has_detection_metrics': metrics_data is not None,
        'has_error_analysis': error_summary is not None,
        'mAP_50': None,
        'mAP_50_95': None,
        'Precision': None,
        'Recall': None,
        'mAP_S': None,
        'Recall_S': None,
        'Precision_S': None,
        'GroupedMetrics': None,
        'ErrorAnalysisSummary': error_summary,
        'total_fp': None,
        'total_fn': None,
        'small_fn': None,
    }
    if isinstance(metrics_data, dict):
        for key in ('mAP_50', 'mAP_50_95', 'Precision', 'Recall', 'mAP_S', 'Recall_S', 'Precision_S', 'GroupedMetrics'):
            record[key] = metrics_data.get(key)
    if isinstance(error_summary, dict):
        for key in ('total_fp', 'total_fn', 'small_fn'):
            record[key] = error_summary.get(key)
    return record


def _collect_tracking_record(spec, experiments_root='outputs/experiments'):
    run_root = infer_run_root(spec, experiments_root=experiments_root)
    tracking_output_dir = run_root / 'tracking_infer'
    tracking_eval_dir = run_root / 'tracking_eval'
    tracking_results_path = tracking_output_dir / 'tracking_results.json'
    mot_metrics_path = tracking_eval_dir / 'mot_metrics.json'
    tracking_error_summary_path = tracking_eval_dir / 'tracking_error_summary.json'

    mot_metrics = _read_json(mot_metrics_path)
    tracking_error_summary = _read_json(tracking_error_summary_path)
    runtime_results = _read_json(tracking_results_path)

    record = {
        'experiment': spec.name,
        'category': spec.category,
        'config': spec.config,
        'run_root': str(run_root),
        'tracking_results_path': str(tracking_results_path) if tracking_results_path.exists() else '',
        'mot_metrics_path': str(mot_metrics_path) if mot_metrics_path.exists() else '',
        'tracking_error_summary_path': str(tracking_error_summary_path) if tracking_error_summary_path.exists() else '',
        'has_tracking_results': runtime_results is not None,
        'has_tracking_metrics': mot_metrics is not None,
        'has_tracking_analysis': tracking_error_summary is not None,
        'MOTA': None,
        'IDF1': None,
        'IDSwitches': None,
        'Fragmentations': None,
        'MostlyTracked': None,
        'MostlyLost': None,
        'TrackRecall': None,
        'TrackPrecision': None,
        'long_track_continuity_score': None,
        'small_object_track_survival_rate': None,
        'feature_assist_reactivation_count': None,
        'memory_reactivation_count': None,
        'overlap_disambiguation_count': None,
        'overlap_disambiguation_helped_count': None,
        'reactivating_state_count': None,
        'predicted_only_to_tracked_count': None,
        'grouped_analysis': None,
        'runtime_only': None,
    }
    if isinstance(mot_metrics, dict):
        for key in ('MOTA', 'IDF1', 'IDSwitches', 'Fragmentations', 'MostlyTracked', 'MostlyLost', 'TrackRecall', 'TrackPrecision'):
            record[key] = mot_metrics.get(key)
    if isinstance(tracking_error_summary, dict):
        for key in (
            'long_track_continuity_score',
            'small_object_track_survival_rate',
            'feature_assist_reactivation_count',
            'memory_reactivation_count',
            'overlap_disambiguation_count',
            'overlap_disambiguation_helped_count',
            'reactivating_state_count',
            'predicted_only_to_tracked_count',
            'grouped_analysis',
            'runtime_only',
        ):
            record[key] = tracking_error_summary.get(key)
    return record


def collect_project_summary(experiments_root='outputs/experiments', subset='all'):
    detection_rows = []
    tracking_rows = []
    for spec in get_experiments(subset=subset):
        if spec.category == 'detection':
            detection_rows.append(_collect_detection_record(spec, experiments_root=experiments_root))
        else:
            tracking_rows.append(_collect_tracking_record(spec, experiments_root=experiments_root))
    return {
        'experiments_root': str(Path(experiments_root)),
        'detection': detection_rows,
        'tracking': tracking_rows,
    }


def export_project_summary(summary, output_dir='outputs/summary'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'detection_summary': _write_csv(output_dir / 'detection_summary.csv', summary.get('detection', [])),
        'tracking_summary': _write_csv(output_dir / 'tracking_summary.csv', summary.get('tracking', [])),
        'project_summary': _write_json(output_dir / 'project_summary.json', summary),
    }
    return files


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize detection and tracking results across the project experiment suite.')
    parser.add_argument('--experiments-root', type=str, default='outputs/experiments')
    parser.add_argument('--output-dir', type=str, default='outputs/summary')
    parser.add_argument('--subset', choices=['detection', 'tracking', 'all'], default='all')
    return parser.parse_args()


def main():
    args = parse_args()
    summary = collect_project_summary(experiments_root=args.experiments_root, subset=args.subset)
    files = export_project_summary(summary, output_dir=args.output_dir)
    print(json.dumps({'summary_files': files, 'num_detection_rows': len(summary['detection']), 'num_tracking_rows': len(summary['tracking'])}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
