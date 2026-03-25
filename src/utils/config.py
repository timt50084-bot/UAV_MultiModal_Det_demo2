# ⭐ 核心：基于 OmegaConf 的配置解析与合并
import os
from omegaconf import OmegaConf


def load_config(config_path: str, default_path: str = 'configs/default.yaml'):
    """
    加载并合并 YAML 配置文件。
    支持使用特定的 config 覆盖 default_config 中的参数。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    # 先加载默认配置作为打底
    if os.path.exists(default_path) and config_path != default_path:
        base_cfg = OmegaConf.load(default_path)
        custom_cfg = OmegaConf.load(config_path)
        # 深度合并，custom_cfg 会覆盖 base_cfg 中同名的键值
        cfg = OmegaConf.merge(base_cfg, custom_cfg)
    else:
        cfg = OmegaConf.load(config_path)

    # 允许在命令行通过点分格式临时修改参数，例如 python train.py train.lr=0.001
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg