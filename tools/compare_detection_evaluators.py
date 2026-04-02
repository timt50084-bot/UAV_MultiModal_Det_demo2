import argparse
import copy
import json
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


COMPARE_METRICS = ('mAP_50', 'mAP_50_95', 'Precision', 'Recall', 'mAP_S')

DEFAULT_DRIFT_TOLERANCES = {
    'mAP_50': {'warn_abs': 0.03, 'fail_abs': 0.08},
    'mAP_50_95': {'warn_abs': 0.03, 'fail_abs': 0.08},
    'Precision': {'warn_abs': 0.05, 'fail_abs': 0.12},
    'Recall': {'warn_abs': 0.05, 'fail_abs': 0.12},
    'mAP_S': {'warn_abs': 0.05, 'fail_abs': 0.15},
}

PARITY_DISABLED_SECTIONS = (
    'cross_modal_robustness',
    'temporal_stability',
    'group_eval',
    'error_analysis',
)


def _config_to_dict(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return {key: _config_to_dict(value) if isinstance(value, Mapping) else copy.deepcopy(value) for key, value in cfg.items()}
    if hasattr(cfg, 'items'):
        return {key: copy.deepcopy(value) for key, value in cfg.items()}
    return copy.deepcopy(cfg)


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_evaluator_specs():
    return [
        {
            'name': 'cpu_reference',
            'evaluator': 'cpu',
            'obb_iou_backend': 'cpu_polygon',
            'description': 'CPU evaluator + exact shapely polygon IoU reference path.',
        },
        {
            'name': 'gpu_mainline',
            'evaluator': 'gpu',
            'obb_iou_backend': 'gpu_prob',
            'description': 'GPU evaluator + ProbIoU surrogate similarity detection mainline path.',
        },
    ]


def build_parity_eval_cfg(eval_cfg):
    cfg = _config_to_dict(eval_cfg) if isinstance(eval_cfg, Mapping) or hasattr(eval_cfg, 'items') else {}
    for section in PARITY_DISABLED_SECTIONS:
        section_cfg = dict(cfg.get(section, {})) if isinstance(cfg.get(section), Mapping) else {}
        section_cfg['enabled'] = False
        cfg[section] = section_cfg
    return cfg


def filter_compare_metrics(metrics, metric_names=COMPARE_METRICS):
    metrics = dict(metrics or {})
    return {name: _safe_float(metrics.get(name)) for name in metric_names}


def compute_metric_drifts(cpu_metrics, gpu_metrics, metric_names=COMPARE_METRICS):
    drifts = {}
    for metric in metric_names:
        cpu_value = _safe_float((cpu_metrics or {}).get(metric))
        gpu_value = _safe_float((gpu_metrics or {}).get(metric))
        available = cpu_value is not None and gpu_value is not None
        abs_diff = abs(gpu_value - cpu_value) if available else None
        rel_diff = None
        if available and abs(cpu_value) > 1e-12:
            rel_diff = abs_diff / abs(cpu_value)
        drifts[metric] = {
            'cpu': cpu_value,
            'gpu': gpu_value,
            'available': bool(available),
            'gpu_minus_cpu': None if not available else gpu_value - cpu_value,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
        }
    return drifts


def evaluate_drift_gate(drifts, tolerances=None, strict=False):
    tolerances = tolerances or DEFAULT_DRIFT_TOLERANCES
    warnings = []
    failures = []
    missing = []
    per_metric = {}

    for metric, entry in (drifts or {}).items():
        tolerance = dict(tolerances.get(metric, {}))
        warn_abs = _safe_float(tolerance.get('warn_abs'))
        fail_abs = _safe_float(tolerance.get('fail_abs'))
        status = 'ok'
        reason = ''

        if not entry.get('available', False):
            status = 'missing'
            reason = f'{metric} is unavailable in one or both evaluator outputs.'
            missing.append(metric)
        else:
            abs_diff = float(entry.get('abs_diff', 0.0))
            if strict and fail_abs is not None and abs_diff > fail_abs:
                status = 'fail'
                reason = f'{metric} abs_diff={abs_diff:.4f} exceeded strict fail_abs={fail_abs:.4f}.'
                failures.append(reason)
            elif warn_abs is not None and abs_diff > warn_abs:
                status = 'warning'
                reason = f'{metric} abs_diff={abs_diff:.4f} exceeded advisory warn_abs={warn_abs:.4f}.'
                warnings.append(reason)

        per_metric[metric] = {
            **entry,
            'status': status,
            'warn_abs': warn_abs,
            'fail_abs': fail_abs,
            'reason': reason,
        }

    if failures:
        overall_status = 'fail'
    elif warnings or missing:
        overall_status = 'warning'
    else:
        overall_status = 'ok'

    return {
        'strict': bool(strict),
        'status': overall_status,
        'warnings': warnings,
        'failures': failures,
        'missing_metrics': missing,
        'per_metric': per_metric,
    }


def build_runtime_summary(cpu_runtime_s, gpu_runtime_s, num_images=None):
    cpu_runtime_s = _safe_float(cpu_runtime_s)
    gpu_runtime_s = _safe_float(gpu_runtime_s)
    summary = {
        'cpu_runtime_s': cpu_runtime_s,
        'gpu_runtime_s': gpu_runtime_s,
        'gpu_minus_cpu_s': None if cpu_runtime_s is None or gpu_runtime_s is None else gpu_runtime_s - cpu_runtime_s,
        'speedup_vs_cpu': None if cpu_runtime_s is None or gpu_runtime_s in (None, 0.0) else cpu_runtime_s / gpu_runtime_s,
        'num_images': int(num_images) if num_images is not None else None,
        'cpu_images_per_second': None,
        'gpu_images_per_second': None,
        'cpu_ms_per_image': None,
        'gpu_ms_per_image': None,
    }
    if num_images and cpu_runtime_s and cpu_runtime_s > 0:
        summary['cpu_images_per_second'] = float(num_images) / cpu_runtime_s
        summary['cpu_ms_per_image'] = (cpu_runtime_s / float(num_images)) * 1000.0
    if num_images and gpu_runtime_s and gpu_runtime_s > 0:
        summary['gpu_images_per_second'] = float(num_images) / gpu_runtime_s
        summary['gpu_ms_per_image'] = (gpu_runtime_s / float(num_images)) * 1000.0
    return summary


def build_compare_result(metadata, cpu_result, gpu_result, strict=False, tolerances=None):
    cpu_metrics = filter_compare_metrics((cpu_result or {}).get('metrics', {}))
    gpu_metrics = filter_compare_metrics((gpu_result or {}).get('metrics', {}))
    drifts = compute_metric_drifts(cpu_metrics, gpu_metrics)
    gate = evaluate_drift_gate(drifts, tolerances=tolerances, strict=strict)
    runtime = build_runtime_summary(
        cpu_runtime_s=(cpu_result or {}).get('runtime_s'),
        gpu_runtime_s=(gpu_result or {}).get('runtime_s'),
        num_images=(metadata or {}).get('num_images'),
    )

    headline = (
        'CPU/GPU evaluator parity comparison completed with advisory warnings.'
        if gate['status'] == 'warning'
        else 'CPU/GPU evaluator parity comparison passed the current gate.'
        if gate['status'] == 'ok'
        else 'CPU/GPU evaluator parity comparison failed the strict gate.'
    )

    gpu_mainline_payload = {
        'spec': dict((gpu_result or {}).get('spec', {})),
        'metrics': gpu_metrics,
        'runtime_s': _safe_float((gpu_result or {}).get('runtime_s')),
    }

    return {
        'metadata': dict(metadata or {}),
        'cpu_reference': {
            'spec': dict((cpu_result or {}).get('spec', {})),
            'metrics': cpu_metrics,
            'runtime_s': _safe_float((cpu_result or {}).get('runtime_s')),
        },
        'gpu_mainline': gpu_mainline_payload,
        # Backward-compatible alias for older parity artifacts or ad hoc readers
        # that still expect the Stage 4 migration-era key.
        'gpu_candidate': dict(gpu_mainline_payload),
        'drift': gate['per_metric'],
        'runtime': runtime,
        'gate': {
            'strict': gate['strict'],
            'status': gate['status'],
            'warnings': gate['warnings'],
            'failures': gate['failures'],
            'missing_metrics': gate['missing_metrics'],
        },
        'summary': {
            'headline': headline,
            'scope_note': (
                'Parity mode isolates the base detection evaluator path. '
                'cross_modal_robustness / temporal_stability / group_eval / error_analysis are disabled for comparison.'
            ),
        },
    }


def _format_optional_float(value, fmt='.4f'):
    numeric = _safe_float(value)
    if numeric is None:
        return 'n/a'
    return format(numeric, fmt)


def render_compare_markdown(result):
    metadata = result.get('metadata', {})
    runtime = result.get('runtime', {})
    gate = result.get('gate', {})
    cpu_metrics = result.get('cpu_reference', {}).get('metrics', {})
    gpu_metrics = result.get('gpu_mainline', result.get('gpu_candidate', {})).get('metrics', {})
    drifts = result.get('drift', {})

    lines = [
        '# Detection Evaluator Parity Report',
        '',
        '## Summary',
        f"- Status: `{gate.get('status', 'warning')}`",
        f"- Headline: {result.get('summary', {}).get('headline', '')}",
        f"- Scope: {result.get('summary', {}).get('scope_note', '')}",
        '',
        '## Run Metadata',
        f"- Config entry: `{metadata.get('config_entry', '')}`",
        f"- Source config: `{metadata.get('source_config', '')}`",
        f"- Weights: `{metadata.get('weights', '')}`",
        f"- Device: `{metadata.get('device', '')}`",
        f"- Num images: `{metadata.get('num_images')}`",
        f"- Timestamp (UTC): `{metadata.get('timestamp_utc', '')}`",
        '',
        '## Metric Comparison',
        '| Metric | CPU | GPU | Abs Diff | Rel Diff | Status |',
        '| --- | ---: | ---: | ---: | ---: | --- |',
    ]

    for metric in metadata.get('compare_metrics', COMPARE_METRICS):
        drift = drifts.get(metric, {})
        cpu_value = cpu_metrics.get(metric)
        gpu_value = gpu_metrics.get(metric)
        abs_diff = drift.get('abs_diff')
        rel_diff = drift.get('rel_diff')
        lines.append(
            f"| {metric} | "
            f"{_format_optional_float(cpu_value, '.4f')} | "
            f"{_format_optional_float(gpu_value, '.4f')} | "
            f"{_format_optional_float(abs_diff, '.4f')} | "
            f"{'n/a' if rel_diff is None else f'{rel_diff:.2%}'} | "
            f"{drift.get('status', 'missing')} |"
        )

    lines.extend([
        '',
        '## Runtime Comparison',
        '| Path | Runtime (s) | Images/s | ms/image |',
        '| --- | ---: | ---: | ---: |',
        f"| CPU reference | {_format_optional_float(runtime.get('cpu_runtime_s'), '.4f')} | "
        f"{_format_optional_float(runtime.get('cpu_images_per_second'), '.2f')} | "
        f"{_format_optional_float(runtime.get('cpu_ms_per_image'), '.2f')} |",
        f"| GPU mainline | {_format_optional_float(runtime.get('gpu_runtime_s'), '.4f')} | "
        f"{_format_optional_float(runtime.get('gpu_images_per_second'), '.2f')} | "
        f"{_format_optional_float(runtime.get('gpu_ms_per_image'), '.2f')} |",
        '',
        f"- GPU minus CPU runtime (s): `{runtime.get('gpu_minus_cpu_s')}`",
        f"- Speedup vs CPU: `{runtime.get('speedup_vs_cpu')}`",
        '',
        '## Gate Notes',
    ])

    if gate.get('warnings'):
        for warning in gate['warnings']:
            lines.append(f"- Warning: {warning}")
    if gate.get('failures'):
        for failure in gate['failures']:
            lines.append(f"- Failure: {failure}")
    if gate.get('missing_metrics'):
        lines.append(f"- Missing metrics: {', '.join(gate['missing_metrics'])}")
    if not gate.get('warnings') and not gate.get('failures') and not gate.get('missing_metrics'):
        lines.append('- No advisory drift warnings were triggered.')

    return '\n'.join(lines) + '\n'


def write_compare_outputs(result, output_dir, json_name='detection_evaluator_compare.json', markdown_name='detection_evaluator_compare.md'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / json_name
    md_path = output_dir / markdown_name
    output_files = {
        'json': str(json_path),
        'markdown': str(md_path),
    }
    result_to_write = dict(result or {})
    result_to_write['output_files'] = output_files

    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(result_to_write, handle, ensure_ascii=False, indent=2)
    with md_path.open('w', encoding='utf-8') as handle:
        handle.write(render_compare_markdown(result_to_write))

    return output_files


def resolve_compare_device(device_arg, cuda_available):
    if int(device_arg) < 0 or not bool(cuda_available):
        raise RuntimeError(
            'CPU-vs-GPU detection evaluator comparison requires CUDA. '
            'Use a CUDA-enabled environment and pass --device >= 0.'
        )
    return f'cuda:{int(device_arg)}'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare CPU and GPU detection evaluators on the same checkpoint and validation set.',
        epilog='OmegaConf-style overrides are also accepted after argparse flags. '
               'The compare tool isolates the base detection parity scope and does not compare tracking metrics.',
    )
    parser.add_argument('--config', type=str, default='configs/main/full_project.yaml')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--strict-gate', action='store_true', help='Fail the process if drift exceeds strict fail thresholds.')
    return parser.parse_known_args(argv)


def _set_seed(seed, torch_module, np_module, random_module):
    random_module.seed(seed)
    np_module.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


def _sync_device(torch_module, device):
    if str(getattr(device, 'type', device)) == 'cuda':
        torch_module.cuda.synchronize(device)


def _run_single_compare_pass(spec, model, val_loader, device, num_classes, nms_kwargs, eval_cfg, infer_cfg, build_detection_evaluator, torch_module):
    evaluator = build_detection_evaluator(
        dataloader=val_loader,
        device=device,
        num_classes=num_classes,
        nms_kwargs=nms_kwargs,
        eval_cfg=eval_cfg,
        infer_cfg=infer_cfg,
    )
    _sync_device(torch_module, device)
    start = time.perf_counter()
    metrics = evaluator.evaluate(model, epoch=spec['name'])
    _sync_device(torch_module, device)
    runtime_s = time.perf_counter() - start
    return {
        'spec': dict(spec),
        'metrics': metrics,
        'runtime_s': runtime_s,
    }


def compare_detection_evaluators(config_path, weights_path, device_index=0, output_dir='', cli_overrides=None, strict_gate=False):
    import random

    import numpy as np
    import torch

    from src.data.dataloader import build_dataloader
    from src.engine.evaluator_factory import build_detection_evaluator
    from src.model.builder import build_model
    from src.utils.config import load_config
    from src.utils.config_utils import apply_experiment_runtime_overrides, get_experiment_run_root
    from src.utils.postprocess_tuning import normalize_infer_cfg

    cfg, cfg_meta = load_config(config_path, cli_args=cli_overrides, return_meta=True)
    cfg, run_name = apply_experiment_runtime_overrides(cfg, config_path=config_path)

    device = torch.device(resolve_compare_device(device_index, torch.cuda.is_available()))
    _set_seed(42, torch, np, random)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    val_loader, _ = build_dataloader(cfg, is_training=False)
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    raw_eval_cfg = cfg.get('eval', {}) if hasattr(cfg, 'get') else {}
    base_eval_cfg = build_parity_eval_cfg(raw_eval_cfg)
    infer_cfg = normalize_infer_cfg(
        cfg.get('infer', {}) if hasattr(cfg, 'get') else {},
        default_imgsz=cfg.dataset.imgsz,
        nms_cfg=cfg.val.nms,
    )

    specs = build_evaluator_specs()
    compare_results = {}
    for spec in specs:
        eval_cfg = copy.deepcopy(base_eval_cfg)
        eval_cfg['evaluator'] = spec['evaluator']
        eval_cfg['obb_iou_backend'] = spec['obb_iou_backend']
        compare_results[spec['name']] = _run_single_compare_pass(
            spec=spec,
            model=model,
            val_loader=val_loader,
            device=device,
            num_classes=cfg.model.num_classes,
            nms_kwargs=cfg.val.nms,
            eval_cfg=eval_cfg,
            infer_cfg=infer_cfg,
            build_detection_evaluator=build_detection_evaluator,
            torch_module=torch,
        )

    dataset = getattr(val_loader, 'dataset', None)
    num_images = len(dataset) if dataset is not None and hasattr(dataset, '__len__') else None
    metadata = {
        'config_entry': config_path,
        'source_config': cfg_meta.get('source_config_path', config_path),
        'resolved_config': cfg_meta.get('resolved_config_path', config_path),
        'weights': str(Path(weights_path)),
        'device': str(device),
        'run_name': run_name,
        'num_images': num_images,
        'compare_metrics': list(COMPARE_METRICS),
        'comparison_scope': {
            'disabled_sections': list(PARITY_DISABLED_SECTIONS),
            'small_object_enabled': bool(base_eval_cfg.get('small_object', {}).get('enabled', True)),
        },
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    result = build_compare_result(
        metadata=metadata,
        cpu_result=compare_results['cpu_reference'],
        gpu_result=compare_results['gpu_mainline'],
        strict=bool(strict_gate),
        tolerances=DEFAULT_DRIFT_TOLERANCES,
    )

    run_root = get_experiment_run_root(cfg, run_name)
    final_output_dir = Path(output_dir) if output_dir else run_root / 'evaluator_compare'
    result['output_files'] = write_compare_outputs(result, final_output_dir)
    return result


def main(argv=None):
    args, cli_overrides = parse_args(argv)
    result = compare_detection_evaluators(
        config_path=args.config,
        weights_path=args.weights,
        device_index=args.device,
        output_dir=args.output_dir,
        cli_overrides=cli_overrides,
        strict_gate=args.strict_gate,
    )

    print(json.dumps({
        'status': result['gate']['status'],
        'json_report': result['output_files']['json'],
        'markdown_report': result['output_files']['markdown'],
        'cpu_runtime_s': result['runtime']['cpu_runtime_s'],
        'gpu_runtime_s': result['runtime']['gpu_runtime_s'],
        'speedup_vs_cpu': result['runtime']['speedup_vs_cpu'],
    }, ensure_ascii=False, indent=2))

    if args.strict_gate and result['gate']['status'] == 'fail':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
