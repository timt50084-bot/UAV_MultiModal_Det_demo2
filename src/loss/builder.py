from importlib import import_module

from src.registry.loss_registry import ASSIGNERS, LOSSES


def _ensure_loss_modules_registered():
    modules = [
        'src.loss.detection_loss',
        'src.loss.assigners.target_assigner',
    ]
    for module in modules:
        import_module(module)


def build_loss(cfg):
    _ensure_loss_modules_registered()
    return LOSSES.build(cfg)


def build_assigner(cfg):
    _ensure_loss_modules_registered()
    return ASSIGNERS.build(cfg)
