from collections.abc import Mapping

import torch


def normalize_infer_cfg(infer_cfg=None, default_imgsz=1024, nms_cfg=None):
    infer_cfg = dict(infer_cfg) if isinstance(infer_cfg, Mapping) else {}
    nms_cfg = dict(nms_cfg) if isinstance(nms_cfg, Mapping) else {}

    mode = str(infer_cfg.get('mode', 'fast')).lower()
    if mode not in {'fast', 'robust', 'competition'}:
        mode = 'fast'

    conf_threshold = infer_cfg.get('conf_threshold')
    if conf_threshold is None:
        conf_threshold = nms_cfg.get('conf_thres', 0.25)
    iou_threshold = infer_cfg.get('iou_threshold')
    if iou_threshold is None:
        iou_threshold = nms_cfg.get('iou_thres', 0.45)

    multi_scale_cfg = dict(infer_cfg.get('multi_scale', {}))
    tta_cfg = dict(infer_cfg.get('tta', {}))
    merge_cfg = dict(infer_cfg.get('merge', {}))

    default_robust_sizes = sorted({int(default_imgsz), max(32, int(round(default_imgsz * 1.125 / 32.0) * 32))})
    default_competition_sizes = sorted({
        max(32, int(round(default_imgsz * 0.75 / 32.0) * 32)),
        int(default_imgsz),
        max(32, int(round(default_imgsz * 1.125 / 32.0) * 32)),
    })

    multi_scale_enabled = bool(multi_scale_cfg.get('enabled', mode in {'robust', 'competition'}))
    if multi_scale_enabled:
        sizes = multi_scale_cfg.get('sizes') or (default_competition_sizes if mode == 'competition' else default_robust_sizes)
    else:
        sizes = [int(default_imgsz)]
    sizes = [int(size) for size in sizes]

    tta_enabled = bool(tta_cfg.get('enabled', mode in {'robust', 'competition'}))
    horizontal_flip = bool(tta_cfg.get('horizontal_flip', mode in {'robust', 'competition'}))

    return {
        'mode': mode,
        'conf_threshold': float(conf_threshold),
        'iou_threshold': float(iou_threshold),
        'classwise_conf_thresholds': dict(infer_cfg.get('classwise_conf_thresholds', {})),
        'multi_scale': {
            'enabled': multi_scale_enabled,
            'sizes': sizes,
        },
        'tta': {
            'enabled': tta_enabled,
            'horizontal_flip': horizontal_flip,
        },
        'merge': {
            'method': str(merge_cfg.get('method', 'nms')).lower(),
            'iou_threshold': float(merge_cfg.get('iou_threshold', iou_threshold if iou_threshold is not None else 0.55)),
            'max_det': int(merge_cfg.get('max_det', nms_cfg.get('max_det', 300))),
        },
        'enabled': multi_scale_enabled or (tta_enabled and horizontal_flip) or bool(infer_cfg.get('classwise_conf_thresholds')),
    }


def describe_classwise_thresholds(classwise_conf_thresholds=None):
    classwise_conf_thresholds = classwise_conf_thresholds or {}
    normalized = {str(key): float(value) for key, value in classwise_conf_thresholds.items()}
    if not normalized:
        return 'off (global confidence threshold only)'
    items = ', '.join(f'{key}={value:.2f}' for key, value in normalized.items())
    return f'enabled ({items})'


def describe_tta_settings(infer_cfg=None):
    infer_cfg = infer_cfg or {}
    multi_scale_cfg = infer_cfg.get('multi_scale', {}) or {}
    tta_cfg = infer_cfg.get('tta', {}) or {}
    merge_cfg = infer_cfg.get('merge', {}) or {}

    multi_scale_enabled = bool(multi_scale_cfg.get('enabled', False))
    horizontal_flip = bool(tta_cfg.get('enabled', False) and tta_cfg.get('horizontal_flip', False))
    if not multi_scale_enabled and not horizontal_flip:
        return 'off (single-scale inference)'

    sizes = multi_scale_cfg.get('sizes') or []
    size_desc = ','.join(str(int(size)) for size in sizes) if multi_scale_enabled and sizes else 'base'
    merge_method = str(merge_cfg.get('method', 'nms')).lower()
    merge_iou = float(merge_cfg.get('iou_threshold', 0.55))
    return f'enabled (sizes={size_desc}; horizontal_flip={horizontal_flip}; merge={merge_method}@{merge_iou:.2f})'


def apply_classwise_thresholds(preds, class_names, global_conf_threshold, classwise_conf_thresholds=None):
    """Apply optional per-class confidence overrides after NMS/merge.

    Mainline configs are expected to use dataset class names as dictionary keys.
    If the mapping is empty, this reduces to the legacy global-threshold behavior.
    """
    if preds is None or len(preds) == 0:
        if torch.is_tensor(preds):
            return preds[:0]
        return preds

    thresholds = {}
    classwise_conf_thresholds = classwise_conf_thresholds or {}
    for key, value in classwise_conf_thresholds.items():
        thresholds[str(key)] = float(value)

    keep_mask = []
    for pred in preds:
        cls_id = int(pred[6].item())
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        threshold = thresholds.get(cls_name, global_conf_threshold)
        keep_mask.append(float(pred[5].item()) >= float(threshold))

    keep_mask = torch.tensor(keep_mask, dtype=torch.bool, device=preds.device)
    return preds[keep_mask]
