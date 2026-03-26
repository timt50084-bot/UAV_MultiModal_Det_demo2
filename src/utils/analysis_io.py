import csv
import json
from pathlib import Path


def ensure_output_dir(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_json(path, data):
    path = Path(path)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return str(path)


def write_csv(path, records):
    path = Path(path)
    normalized_records = list(records or [])
    if not normalized_records:
        normalized_records = [{}]

    fieldnames = sorted({key for record in normalized_records for key in record.keys()})
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in normalized_records:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in record.items()
            })
    return str(path)


def export_error_analysis(summary, per_image_records, output_dir, export_json=True, export_csv=True, include_per_image=True):
    output_path = ensure_output_dir(output_dir)
    exported_files = {}

    if export_json:
        exported_files['summary_json'] = write_json(output_path / 'summary.json', summary)
        exported_files['confusion_json'] = write_json(
            output_path / 'confusion_summary.json',
            summary.get('class_confusion_summary', {}),
        )
        exported_files['modality_json'] = write_json(
            output_path / 'modality_contribution.json',
            summary.get('modality_contribution_summary', {}),
        )
        if include_per_image:
            exported_files['per_image_json'] = write_json(output_path / 'per_image_errors.json', per_image_records)

    if export_csv and include_per_image:
        exported_files['per_image_csv'] = write_csv(output_path / 'per_image_errors.csv', per_image_records)

    return exported_files
