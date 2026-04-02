from collections.abc import Mapping

from src.metrics.obb_iou_backend import normalize_obb_iou_backend_name

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - optional at import time in some test environments
    OmegaConf = None


def config_to_dict(cfg):
    if cfg is None:
        return {}

    if OmegaConf is not None and OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)

    if isinstance(cfg, Mapping):
        return {key: config_to_dict(value) if isinstance(value, Mapping) else value for key, value in cfg.items()}

    if hasattr(cfg, 'items'):
        return {key: value for key, value in cfg.items()}

    return {}


def normalize_eval_metrics_cfg(cfg):
    root_cfg = config_to_dict(cfg)
    nested_cfg = config_to_dict(root_cfg.get('extra_metrics', {}))

    def get_section(name, defaults):
        section = dict(defaults)
        if isinstance(nested_cfg.get(name), Mapping):
            section.update(dict(nested_cfg[name]))
        if isinstance(root_cfg.get(name), Mapping):
            section.update(dict(root_cfg[name]))
        return section

    small_object = get_section('small_object', {
        'enabled': True,
        'area_threshold': 32,
        'iou_threshold': 0.5,
    })
    cross_modal = get_section('cross_modal_robustness', {
        'enabled': False,
        'base_metric': 'mAP_50',
        'rgb_drop_mode': 'zero',
        'ir_drop_mode': 'zero',
    })
    temporal = get_section('temporal_stability', {
        'enabled': 'auto',
        'conf_threshold': 0.25,
        'match_iou_threshold': 0.3,
        'max_center_shift_ratio': 0.1,
    })
    group_eval = get_section('group_eval', {
        'enabled': False,
        'keys': [],
    })
    error_analysis = get_section('error_analysis', {
        'enabled': False,
        'output_dir': 'outputs/error_analysis',
        'export_json': True,
        'export_csv': True,
        'include_per_image': True,
    })
    error_buckets = {
        'small_object_area_threshold': small_object.get('area_threshold', 32),
        'area_bins': [16, 32, 96],
        'aspect_ratio_bins': [1.5, 3.0],
        'dense_scene_gt_threshold': 5,
    }
    if isinstance(error_analysis.get('buckets'), Mapping):
        error_buckets.update(dict(error_analysis['buckets']))
    error_analysis['buckets'] = error_buckets
    error_confusion = {'enabled': True}
    if isinstance(error_analysis.get('confusion'), Mapping):
        error_confusion.update(dict(error_analysis['confusion']))
    error_analysis['confusion'] = error_confusion
    error_modality = {'enabled': False}
    if isinstance(error_analysis.get('modality_contribution'), Mapping):
        error_modality.update(dict(error_analysis['modality_contribution']))
    error_analysis['modality_contribution'] = error_modality
    error_analysis['iou_threshold'] = float(error_analysis.get('iou_threshold', 0.5))

    enabled = root_cfg.get(
        'enabled',
        nested_cfg.get(
            'enabled',
            bool(
                small_object.get('enabled', False)
                or cross_modal.get('enabled', False)
                or group_eval.get('enabled', False)
                or error_analysis.get('enabled', False)
            ),
        ),
    )
    obb_iou_backend = normalize_obb_iou_backend_name(
        root_cfg.get(
            'obb_iou_backend',
            nested_cfg.get('obb_iou_backend', 'cpu_polygon'),
        )
    )

    return {
        'enabled': bool(enabled),
        'obb_iou_backend': obb_iou_backend,
        'small_object': small_object,
        'cross_modal_robustness': cross_modal,
        'temporal_stability': temporal,
        'group_eval': group_eval,
        'error_analysis': error_analysis,
    }


def describe_cross_modal_robustness(cfg=None):
    cfg = dict(cfg or {})
    if not cfg.get('enabled', False):
        return 'off (baseline validation only)'
    base_metric = str(cfg.get('base_metric', 'mAP_50'))
    rgb_drop_mode = str(cfg.get('rgb_drop_mode', 'zero'))
    ir_drop_mode = str(cfg.get('ir_drop_mode', 'zero'))
    return (
        f'enabled (base_metric={base_metric}; '
        f'rgb_drop={rgb_drop_mode}; ir_drop={ir_drop_mode})'
    )


def describe_detection_error_analysis(cfg=None):
    cfg = dict(cfg or {})
    if not cfg.get('enabled', False):
        return 'off (metrics only)'
    output_dir = str(cfg.get('output_dir', 'outputs/error_analysis'))
    export_json = bool(cfg.get('export_json', True))
    export_csv = bool(cfg.get('export_csv', True))
    include_per_image = bool(cfg.get('include_per_image', True))
    return (
        f'enabled (output_dir={output_dir}; '
        f'json={export_json}; csv={export_csv}; per_image={include_per_image})'
    )
