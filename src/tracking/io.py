import csv
import json
from copy import deepcopy
from pathlib import Path

from .metrics import normalize_tracking_sequences


def _to_serializable(value):
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def load_tracking_json(path):
    with Path(path).open('r', encoding='utf-8') as file:
        return json.load(file)


def save_tracking_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(_to_serializable(data), file, ensure_ascii=False, indent=2)
    return str(path)


def export_tracking_mot_txt(sequence_results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sequences = normalize_tracking_sequences(sequence_results, kind='pred', default_sequence_id='pred')

    with output_path.open('w', encoding='utf-8') as file:
        for sequence_id, sequence in sequences.items():
            for frame in sequence.get('frames', []):
                frame_index = int(frame.get('frame_index', 0))
                for result in frame.get('results', []):
                    obb = result.get('obb', [0, 0, 0, 0, 0])
                    file.write(
                        f"{sequence_id},{frame_index},{result.get('track_id')},{result.get('class_id')},{float(result.get('score', 0.0)):.6f},"
                        f"{float(obb[0]):.6f},{float(obb[1]):.6f},{float(obb[2]):.6f},{float(obb[3]):.6f},{float(obb[4]):.6f},{result.get('state', 'tracked')}\n"
                    )
    return str(output_path)


def export_tracking_artifacts(output_dir, metrics_summary=None, analysis_result=None, sequence_results=None, export_json=True, export_csv=True, export_mot_txt=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}

    if export_json and metrics_summary is not None:
        exported['mot_metrics'] = save_tracking_json(metrics_summary, output_dir / 'mot_metrics.json')

    if analysis_result is not None:
        if export_json:
            exported['tracking_error_summary'] = save_tracking_json(analysis_result.get('summary', {}), output_dir / 'tracking_error_summary.json')
            exported['per_sequence_tracking_analysis'] = save_tracking_json(analysis_result.get('per_sequence_analysis', []), output_dir / 'per_sequence_tracking_analysis.json')
        if export_csv:
            csv_path = output_dir / 'per_track_analysis.csv'
            per_track_rows = analysis_result.get('per_track_analysis', [])
            fieldnames = [
                'sequence_id',
                'gt_track_id',
                'class_id',
                'class_name',
                'total_frames',
                'matched_frames',
                'matched_ratio',
                'id_switches',
                'fragmentations',
                'reactivated',
                'size_group',
                'avg_area',
                'is_small_track',
            ]
            with csv_path.open('w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_track_rows:
                    writer.writerow({key: row.get(key) for key in fieldnames})
            exported['per_track_analysis'] = str(csv_path)

    if sequence_results is not None and export_json:
        exported['tracking_results'] = save_tracking_json(deepcopy(sequence_results), output_dir / 'tracking_results.json')
    if sequence_results is not None and export_mot_txt:
        exported['tracking_results_mot_txt'] = export_tracking_mot_txt(sequence_results, output_dir / 'tracking_results_mot.txt')

    return exported
