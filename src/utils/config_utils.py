from pathlib import Path

from omegaconf import OmegaConf

from src.utils.config import get_config_value


def infer_experiment_name(cfg, config_path=None):
    experiment_cfg = cfg.get('experiment', {}) if hasattr(cfg, 'get') else {}
    if hasattr(experiment_cfg, 'get'):
        explicit_name = experiment_cfg.get('name', '')
        if explicit_name:
            return str(explicit_name)

    if config_path:
        return Path(config_path).stem.replace('exp_', '')
    return 'default'


def get_experiment_run_root(cfg, run_name):
    experiment_cfg = cfg.get('experiment', {}) if hasattr(cfg, 'get') else {}
    output_root = experiment_cfg.get('output_root', 'outputs/experiments') if hasattr(experiment_cfg, 'get') else 'outputs/experiments'
    return Path(output_root) / run_name


def apply_experiment_runtime_overrides(cfg, config_path=None):
    if not hasattr(cfg, 'get'):
        return cfg, infer_experiment_name(cfg, config_path)

    experiment_cfg = cfg.get('experiment', {})
    enable_unified_dirs = bool(experiment_cfg.get('enable_unified_dirs', False)) if hasattr(experiment_cfg, 'get') else False
    run_name = infer_experiment_name(cfg, config_path)

    if not enable_unified_dirs:
        return cfg, run_name

    run_root = get_experiment_run_root(cfg, run_name)

    if 'train' in cfg and cfg.train is not None:
        current_save_dir = cfg.train.get('save_dir', 'outputs/weights')
        if current_save_dir in {'', 'outputs/weights'}:
            cfg.train.save_dir = str(run_root / 'weights')

    if 'eval' in cfg and cfg.eval is not None:
        error_cfg = cfg.eval.get('error_analysis', None)
        if error_cfg is not None:
            current_output_dir = error_cfg.get('output_dir', 'outputs/error_analysis')
            if current_output_dir in {'', 'outputs/error_analysis'}:
                cfg.eval.error_analysis.output_dir = str(run_root / 'error_analysis')

    if 'tracking_eval' in cfg and cfg.tracking_eval is not None:
        current_output_dir = cfg.tracking_eval.get('output_dir', 'outputs/tracking_eval')
        if current_output_dir in {'', 'outputs/tracking_eval'}:
            cfg.tracking_eval.output_dir = str(run_root / 'tracking_eval')

    return cfg, run_name


def save_resolved_config(cfg, run_name, filename='resolved_config.yaml'):
    run_root = get_experiment_run_root(cfg, run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / filename
    with resolved_path.open('w', encoding='utf-8') as handle:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))
    return resolved_path


def format_effective_train_config_summary(cfg, config_path, resolved_config_path, source_config_path=None):
    lines = [
        '',
        'Effective training config:',
        f'  config entry: {config_path}',
    ]
    if source_config_path and source_config_path != config_path:
        lines.append(f'  source config: {source_config_path}')
    lines.extend([
        f'  resolved config: {resolved_config_path}',
        f"  Effective batch_size: {get_config_value(cfg, 'dataloader.batch_size')}",
        f"  Effective num_workers: {get_config_value(cfg, 'dataloader.num_workers')}",
        f"  Effective dataset.root_dir: {get_config_value(cfg, 'dataset.root_dir')}",
        f"  Effective dataset.imgsz: {get_config_value(cfg, 'dataset.imgsz')}",
        f"  Effective train.epochs: {get_config_value(cfg, 'train.epochs')}",
        f"  Effective train.patience: {get_config_value(cfg, 'train.patience')}",
        f"  Effective train.eval_interval: {get_config_value(cfg, 'train.eval_interval')}",
        f"  Effective train.lr: {get_config_value(cfg, 'train.lr')}",
        f"  Effective train.use_amp: {get_config_value(cfg, 'train.use_amp')}",
        f"  Effective use_log_sampler: {get_config_value(cfg, 'dataloader.use_log_sampler')}",
        f"  Effective performance.profile_train: {get_config_value(cfg, 'performance.profile_train')}",
        f"  Effective performance.dataloader.persistent_workers: {get_config_value(cfg, 'performance.dataloader.persistent_workers')}",
        f"  Effective performance.dataloader.prefetch_factor: {get_config_value(cfg, 'performance.dataloader.prefetch_factor')}",
    ])
    return '\n'.join(lines)
