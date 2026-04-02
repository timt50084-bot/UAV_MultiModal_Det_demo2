import os

from omegaconf import OmegaConf


LEGACY_CONFIG_REDIRECTS = {
    'configs/exp_full_project.yaml': 'configs/main/full_project.yaml',
    'configs/exp_angle_aware_assigner.yaml': 'configs/archive/ablation/angle_aware_assigner.yaml',
    'configs/exp_assigner_main.yaml': 'configs/main/assigner_main.yaml',
    'configs/exp_baseline.yaml': 'configs/main/baseline.yaml',
    'configs/exp_competition_infer.yaml': 'configs/archive/infer/competition_infer.yaml',
    'configs/exp_error_analysis.yaml': 'configs/archive/eval/error_analysis.yaml',
    'configs/exp_eval_full.yaml': 'configs/archive/eval/full.yaml',
    'configs/exp_fusion_main.yaml': 'configs/main/fusion_main.yaml',
    'configs/exp_realistic_multimodal_aug.yaml': 'configs/archive/ablation/realistic_multimodal_aug.yaml',
    'configs/exp_reliability_fusion.yaml': 'configs/archive/ablation/reliability_fusion.yaml',
    'configs/exp_task_metrics.yaml': 'configs/archive/eval/task_metrics.yaml',
    'configs/exp_temporal_main.yaml': 'configs/main/temporal_main.yaml',
    'configs/exp_temporal_memory.yaml': 'configs/archive/ablation/temporal_memory.yaml',
    'configs/exp_tracking_assoc.yaml': 'configs/archive/tracking/assoc.yaml',
    'configs/exp_tracking_base.yaml': 'configs/main/tracking_base.yaml',
    'configs/exp_tracking_eval.yaml': 'configs/main/tracking_eval.yaml',
    'configs/exp_tracking_final.yaml': 'configs/main/tracking_final.yaml',
    'configs/exp_tracking_jointlite.yaml': 'configs/archive/tracking/jointlite.yaml',
    'configs/exp_tracking_modality.yaml': 'configs/archive/tracking/modality.yaml',
    'configs/exp_tracking_temporal.yaml': 'configs/archive/tracking/temporal.yaml',
    'configs/ablation/angle_aware_assigner.yaml': 'configs/archive/ablation/angle_aware_assigner.yaml',
    'configs/ablation/assigner_main.yaml': 'configs/archive/ablation/assigner_main.yaml',
    'configs/ablation/baseline.yaml': 'configs/archive/ablation/baseline.yaml',
    'configs/ablation/fusion_main.yaml': 'configs/archive/ablation/fusion_main.yaml',
    'configs/ablation/realistic_multimodal_aug.yaml': 'configs/archive/ablation/realistic_multimodal_aug.yaml',
    'configs/ablation/reliability_fusion.yaml': 'configs/archive/ablation/reliability_fusion.yaml',
    'configs/ablation/temporal_main.yaml': 'configs/archive/ablation/temporal_main.yaml',
    'configs/ablation/temporal_memory.yaml': 'configs/archive/ablation/temporal_memory.yaml',
    'configs/eval/error_analysis.yaml': 'configs/archive/eval/error_analysis.yaml',
    'configs/eval/full.yaml': 'configs/archive/eval/full.yaml',
    'configs/eval/task_metrics.yaml': 'configs/archive/eval/task_metrics.yaml',
    'configs/infer/competition_infer.yaml': 'configs/archive/infer/competition_infer.yaml',
    'configs/main/tracking_assoc.yaml': 'configs/archive/tracking/assoc.yaml',
    'configs/main/tracking_jointlite.yaml': 'configs/archive/tracking/jointlite.yaml',
    'configs/main/tracking_modality.yaml': 'configs/archive/tracking/modality.yaml',
    'configs/main/tracking_temporal.yaml': 'configs/archive/tracking/temporal.yaml',
    'configs/tracking/assoc.yaml': 'configs/archive/tracking/assoc.yaml',
    'configs/tracking/base.yaml': 'configs/main/tracking_base.yaml',
    'configs/tracking/eval.yaml': 'configs/main/tracking_eval.yaml',
    'configs/tracking/final.yaml': 'configs/main/tracking_final.yaml',
    'configs/tracking/jointlite.yaml': 'configs/archive/tracking/jointlite.yaml',
    'configs/tracking/modality.yaml': 'configs/archive/tracking/modality.yaml',
    'configs/tracking/temporal.yaml': 'configs/archive/tracking/temporal.yaml',
}


def _is_mapping(node):
    return isinstance(node, dict) or hasattr(node, 'keys')


def _path_parts(path):
    return [part for part in str(path).split('.') if part]


def _path_exists(node, path):
    current = node
    for part in _path_parts(path):
        if not _is_mapping(current) or part not in current:
            return False
        current = current[part]
    return True


def _path_get(node, path, default=None):
    current = node
    for part in _path_parts(path):
        if not _is_mapping(current) or part not in current:
            return default
        current = current[part]
    return current


def _path_set(node, path, value):
    current = node
    parts = _path_parts(path)
    for part in parts[:-1]:
        if not _is_mapping(current) or part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _normalized_path_key(path):
    return os.path.normpath(str(path)).replace('\\', '/')


def _display_path(path):
    try:
        return _normalized_path_key(os.path.relpath(path, os.getcwd()))
    except ValueError:
        return _normalized_path_key(path)


def _resolve_redirect(config_path):
    normalized_candidates = {_normalized_path_key(config_path)}
    abs_candidate = os.path.abspath(config_path)
    normalized_candidates.add(_display_path(abs_candidate))

    for candidate in normalized_candidates:
        redirect_target = LEGACY_CONFIG_REDIRECTS.get(candidate)
        if redirect_target:
            redirected_abs = os.path.abspath(redirect_target)
            warning = (
                f"Config path {candidate} has moved to {_normalized_path_key(redirect_target)}. "
                f"Using the new location for backward compatibility."
            )
            return redirected_abs, warning
    return os.path.abspath(config_path), None


def get_config_value(cfg, path, default=None):
    return _path_get(cfg, path, default)


def _resolve_layered_value(layers, canonical_path, aliases=None, default_value=None):
    aliases = aliases or []
    selected = None
    warnings = []

    for layer_name, layer_cfg in layers:
        if layer_cfg is None:
            continue

        layer_entries = []
        if _path_exists(layer_cfg, canonical_path):
            layer_entries.append((canonical_path, _path_get(layer_cfg, canonical_path), False))
        for alias_path in aliases:
            if _path_exists(layer_cfg, alias_path):
                layer_entries.append((alias_path, _path_get(layer_cfg, alias_path), True))

        if not layer_entries:
            continue

        canonical_entries = [entry for entry in layer_entries if not entry[2]]
        if canonical_entries:
            layer_selected = canonical_entries[0]
            for entry_path, entry_value, is_alias in layer_entries:
                if is_alias and entry_value != layer_selected[1]:
                    warnings.append(
                        f"[{layer_name}] Both {canonical_path} and {entry_path} are set with different values. "
                        f"Using {canonical_path}={layer_selected[1]!r}."
                    )
        else:
            layer_selected = layer_entries[0]
            warnings.append(
                f"[{layer_name}] {layer_selected[0]} is a legacy alias. Prefer {canonical_path}."
            )

        selected = {
            'value': layer_selected[1],
            'source': layer_name,
            'source_path': layer_selected[0],
            'used_legacy_alias': layer_selected[2],
        }

    if selected is None and default_value is not None:
        selected = {
            'value': default_value,
            'source': 'built-in default',
            'source_path': canonical_path,
            'used_legacy_alias': False,
        }

    return selected, warnings


def _load_config_tree(config_path, visited=None):
    visited = set() if visited is None else set(visited)
    resolved_path, redirect_warning = _resolve_redirect(config_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    abs_path = os.path.abspath(resolved_path)
    if abs_path in visited:
        raise ValueError(f"Cyclic config inheritance detected at: {_display_path(abs_path)}")

    raw_cfg = OmegaConf.load(abs_path)
    warnings = []
    if redirect_warning:
        warnings.append(redirect_warning)

    base_configs = raw_cfg.get('base_configs', []) if hasattr(raw_cfg, 'get') else []
    if isinstance(base_configs, str):
        base_configs = [base_configs]

    merged_cfg = OmegaConf.create({})
    next_visited = visited | {abs_path}
    for base_entry in base_configs:
        base_path = str(base_entry)
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(abs_path), base_path)
        inherited_cfg, inherited_meta = _load_config_tree(base_path, visited=next_visited)
        warnings.extend(inherited_meta.get('warnings', []))
        merged_cfg = OmegaConf.merge(merged_cfg, inherited_cfg)

    if 'base_configs' in raw_cfg:
        del raw_cfg['base_configs']

    config_meta = raw_cfg.get('config_meta', {}) if hasattr(raw_cfg, 'get') else {}
    deprecated_message = config_meta.get('deprecated_message', '') if hasattr(config_meta, 'get') else ''
    recommended_edit_path = config_meta.get('recommended_edit_path', '') if hasattr(config_meta, 'get') else ''
    if deprecated_message:
        warnings.append(str(deprecated_message))
    if 'config_meta' in raw_cfg:
        del raw_cfg['config_meta']

    merged_cfg = OmegaConf.merge(merged_cfg, raw_cfg)
    return merged_cfg, {
        'warnings': warnings,
        'resolved_path': abs_path,
        'display_path': _display_path(abs_path),
        'source_config_path': _normalized_path_key(str(recommended_edit_path)) if recommended_edit_path else _display_path(abs_path),
    }


def _append_misplaced_field_warnings(cfg, meta):
    misplaced_specs = [
        ('batch', 'dataloader.batch_size'),
        ('workers', 'dataloader.num_workers'),
        ('imgsz', 'dataset.imgsz'),
        ('img_size', 'dataset.imgsz'),
        ('epochs', 'train.epochs'),
        ('eval_interval', 'train.eval_interval'),
        ('lr', 'train.lr'),
        ('weight_decay', 'train.weight_decay'),
        ('accumulate', 'train.accumulate'),
        ('dataloader.batch', 'dataloader.batch_size'),
        ('dataloader.workers', 'dataloader.num_workers'),
        ('train.batch_size', 'dataloader.batch_size'),
        ('train.num_workers', 'dataloader.num_workers'),
        ('profile_train', 'performance.profile_train'),
        ('persistent_workers', 'performance.dataloader.persistent_workers'),
        ('prefetch_factor', 'performance.dataloader.prefetch_factor'),
    ]
    for misplaced_path, canonical_path in misplaced_specs:
        if _path_exists(cfg, misplaced_path):
            meta['warnings'].append(
                f"{misplaced_path} is not a canonical training entry field. Use {canonical_path} instead."
            )


def _normalize_config(cfg, layers):
    meta = {
        'warnings': [],
        'sources': {},
    }

    specs = [
        {
            'canonical': 'dataloader.batch_size',
            'aliases': ['batch_size'],
            'default': 1,
        },
        {
            'canonical': 'dataloader.num_workers',
            'aliases': ['num_workers'],
            'default': 0,
        },
        {
            'canonical': 'dataloader.use_log_sampler',
            'aliases': ['use_log_sampler'],
            'default': False,
        },
        {
            'canonical': 'dataloader.pin_memory',
            'aliases': [],
            'default': True,
        },
        {
            'canonical': 'dataset.imgsz',
            'aliases': [],
        },
        {
            'canonical': 'train.epochs',
            'aliases': [],
        },
        {
            'canonical': 'train.eval_interval',
            'aliases': [],
            'default': 1,
        },
        {
            'canonical': 'train.lr',
            'aliases': [],
        },
        {
            'canonical': 'train.weight_decay',
            'aliases': [],
        },
        {
            'canonical': 'train.accumulate',
            'aliases': [],
        },
        {
            'canonical': 'performance.profile_train',
            'aliases': [],
            'default': False,
        },
        {
            'canonical': 'performance.dataloader.persistent_workers',
            'aliases': [],
            'default': True,
        },
        {
            'canonical': 'performance.dataloader.prefetch_factor',
            'aliases': [],
            'default': 2,
        },
        {
            'canonical': 'model.temporal.enabled',
            'aliases': ['model.temporal_enabled'],
            'default': False,
        },
        {
            'canonical': 'model.temporal.stride',
            'aliases': ['model.temporal_stride'],
            'default': 1,
        },
    ]

    for spec in specs:
        resolved, warnings = _resolve_layered_value(
            layers=layers,
            canonical_path=spec['canonical'],
            aliases=spec.get('aliases', []),
            default_value=spec.get('default', None),
        )
        meta['warnings'].extend(warnings)
        if resolved is None:
            continue

        _path_set(cfg, spec['canonical'], resolved['value'])
        meta['sources'][spec['canonical']] = {
            'source': resolved['source'],
            'path': resolved['source_path'],
            'used_legacy_alias': resolved['used_legacy_alias'],
        }

    known_dataloader_keys = {'batch_size', 'num_workers', 'use_log_sampler', 'pin_memory'}
    dataloader_cfg = _path_get(cfg, 'dataloader', {})
    if _is_mapping(dataloader_cfg):
        for key in dataloader_cfg.keys():
            if str(key) not in known_dataloader_keys:
                meta['warnings'].append(
                    f"dataloader.{key} is not consumed by the current training code and will be ignored."
                )

    known_perf_dataloader_keys = {'persistent_workers', 'prefetch_factor'}
    perf_dataloader_cfg = _path_get(cfg, 'performance.dataloader', {})
    if _is_mapping(perf_dataloader_cfg):
        for key in perf_dataloader_cfg.keys():
            if str(key) not in known_perf_dataloader_keys:
                meta['warnings'].append(
                    f"performance.dataloader.{key} is not consumed by the current training code and will be ignored."
                )

    _append_misplaced_field_warnings(cfg, meta)
    return cfg, meta


def load_config(config_path: str, default_path: str = 'configs/default.yaml', cli_args=None, return_meta=False):
    requested_config_path = _normalized_path_key(config_path)

    if os.path.exists(default_path) and _normalized_path_key(config_path) != _normalized_path_key(default_path):
        base_cfg, base_meta = _load_config_tree(default_path)
        custom_cfg, custom_meta = _load_config_tree(config_path)
    else:
        base_cfg = OmegaConf.create({})
        base_meta = {'warnings': [], 'display_path': _normalized_path_key(default_path)}
        custom_cfg, custom_meta = _load_config_tree(config_path)

    cli_cfg = OmegaConf.from_cli() if cli_args is None else OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(base_cfg, custom_cfg, cli_cfg)
    cfg, meta = _normalize_config(
        cfg,
        layers=[
            (base_meta.get('display_path', _normalized_path_key(default_path)), base_cfg),
            (custom_meta.get('display_path', requested_config_path), custom_cfg),
            ('CLI', cli_cfg),
        ],
    )
    meta['warnings'] = base_meta.get('warnings', []) + custom_meta.get('warnings', []) + meta['warnings']
    meta['requested_config_path'] = requested_config_path
    meta['resolved_config_path'] = custom_meta.get('display_path', requested_config_path)
    meta['source_config_path'] = custom_meta.get('source_config_path', meta['resolved_config_path'])

    if return_meta:
        return cfg, meta
    return cfg
