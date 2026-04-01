from importlib import import_module

from src.registry.model_registry import DETECTORS


def _ensure_model_modules_registered():
    modules = [
        'src.model.backbones.dual_backbone',
        'src.model.fusion.cross_attn_fusion',
        'src.model.fusion.rdm_fusion',
        'src.model.fusion.reliability_fusion',
        'src.model.fusion.simple_concat',
        'src.model.necks.enhanced_neck',
        'src.model.temporal.temporal_fpn',
        'src.model.heads.obb_decoupled_head',
        'src.model.detectors.yolo_dual',
    ]
    for module in modules:
        import_module(module)


def build_model(cfg):
    _ensure_model_modules_registered()
    return DETECTORS.build(cfg)
