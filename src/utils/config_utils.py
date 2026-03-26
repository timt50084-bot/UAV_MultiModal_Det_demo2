from pathlib import Path


def infer_experiment_name(cfg, config_path=None):
    experiment_cfg = cfg.get('experiment', {}) if hasattr(cfg, 'get') else {}
    if hasattr(experiment_cfg, 'get'):
        explicit_name = experiment_cfg.get('name', '')
        if explicit_name:
            return str(explicit_name)

    if config_path:
        return Path(config_path).stem.replace('exp_', '')
    return 'default'


def apply_experiment_runtime_overrides(cfg, config_path=None):
    if not hasattr(cfg, 'get'):
        return cfg, infer_experiment_name(cfg, config_path)

    experiment_cfg = cfg.get('experiment', {})
    enable_unified_dirs = bool(experiment_cfg.get('enable_unified_dirs', False)) if hasattr(experiment_cfg, 'get') else False
    run_name = infer_experiment_name(cfg, config_path)

    if not enable_unified_dirs:
        return cfg, run_name

    output_root = Path(experiment_cfg.get('output_root', 'outputs/experiments'))
    run_root = output_root / run_name

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
