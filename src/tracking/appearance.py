from copy import deepcopy

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.utils.tta import wrap_obb_angle


APPEARANCE_DEFAULTS = {
    'enabled': False,
    'embedding_dim': 128,
    'update_mode': 'ema',
    'history_size': 5,
    'ema_momentum': 0.8,
    'sampling_radius': 1,
    'use_rgb_ir_branches': False,
}

FEATURE_ASSIST_DEFAULTS = {
    'enabled': False,
    'source': 'temporal_fused',
    'embedding_dim': 128,
    'use_for_reactivation': True,
    'use_for_overlap_resolution': True,
}


# Stage-2/7 note:
# We keep the detector-tracker connection lightweight. Both appearance features and
# stage-7 feature-assist descriptors are extracted from the detector's existing fused
# / temporal-enhanced feature maps without introducing a new ReID backbone.


def normalize_appearance_cfg(appearance_cfg=None):
    cfg = deepcopy(APPEARANCE_DEFAULTS)
    if appearance_cfg is None:
        return cfg

    if OmegaConf.is_config(appearance_cfg):
        appearance_cfg = OmegaConf.to_container(appearance_cfg, resolve=True)

    if isinstance(appearance_cfg, dict):
        for key, value in appearance_cfg.items():
            cfg[key] = value
    return cfg


def normalize_feature_assist_cfg(feature_assist_cfg=None):
    cfg = deepcopy(FEATURE_ASSIST_DEFAULTS)
    if feature_assist_cfg is None:
        return cfg

    if OmegaConf.is_config(feature_assist_cfg):
        feature_assist_cfg = OmegaConf.to_container(feature_assist_cfg, resolve=True)

    if isinstance(feature_assist_cfg, dict):
        for key, value in feature_assist_cfg.items():
            cfg[key] = value
    return cfg


def empty_appearance_payload(num_detections=0, embedding_dim=128, device='cpu', include_modalities=False):
    payload = {
        'fused': torch.zeros((num_detections, embedding_dim), dtype=torch.float32, device=device),
        'masks': {
            'fused': torch.zeros(num_detections, dtype=torch.bool, device=device),
        },
    }
    if include_modalities:
        payload['rgb'] = torch.zeros((num_detections, embedding_dim), dtype=torch.float32, device=device)
        payload['ir'] = torch.zeros((num_detections, embedding_dim), dtype=torch.float32, device=device)
        payload['masks']['rgb'] = torch.zeros(num_detections, dtype=torch.bool, device=device)
        payload['masks']['ir'] = torch.zeros(num_detections, dtype=torch.bool, device=device)
    else:
        payload['rgb'] = None
        payload['ir'] = None
    return payload


def empty_feature_assist_payload(num_detections=0, embedding_dim=128, device='cpu'):
    payload = empty_appearance_payload(
        num_detections=num_detections,
        embedding_dim=embedding_dim,
        device=device,
        include_modalities=True,
    )
    payload['source'] = None
    return payload


def normalize_appearance_payload(appearance_features, num_detections=None, device='cpu', appearance_cfg=None):
    cfg = normalize_appearance_cfg(appearance_cfg)
    if appearance_features is None:
        return None

    if torch.is_tensor(appearance_features):
        appearance_features = {'fused': appearance_features}
    elif OmegaConf.is_config(appearance_features):
        appearance_features = OmegaConf.to_container(appearance_features, resolve=True)

    if not isinstance(appearance_features, dict):
        raise TypeError('appearance_features must be a tensor or a dict with fused/rgb/ir embeddings.')

    if num_detections is None:
        first_tensor = next((value for value in appearance_features.values() if torch.is_tensor(value)), None)
        num_detections = 0 if first_tensor is None else int(first_tensor.shape[0])

    payload = empty_appearance_payload(
        num_detections=num_detections,
        embedding_dim=int(cfg['embedding_dim']),
        device=device,
        include_modalities=bool(cfg.get('use_rgb_ir_branches', False)),
    )

    for key in ('fused', 'rgb', 'ir'):
        value = appearance_features.get(key)
        if value is None:
            continue
        value = value.to(device=device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[0] != num_detections:
            raise ValueError(f'Appearance tensor for {key} must have shape [N, D] matching detections.')
        value = F.normalize(value, dim=1) if value.numel() > 0 else value
        payload[key] = value
        payload['masks'][key] = torch.ones(num_detections, dtype=torch.bool, device=device)
    return payload


def normalize_feature_assist_payload(feature_assist_features, num_detections=None, device='cpu', feature_assist_cfg=None):
    cfg = normalize_feature_assist_cfg(feature_assist_cfg)
    if feature_assist_features is None:
        return None

    payload = normalize_appearance_payload(
        feature_assist_features,
        num_detections=num_detections,
        device=device,
        appearance_cfg={
            'embedding_dim': int(cfg['embedding_dim']),
            'use_rgb_ir_branches': True,
        },
    )
    if payload is None:
        return None
    payload['source'] = feature_assist_features.get('source') if isinstance(feature_assist_features, dict) else cfg.get('source')
    return payload


def get_detection_appearance(appearance_payload, index):
    if appearance_payload is None:
        return None
    appearance = {}
    for key in ('fused', 'rgb', 'ir'):
        tensor = appearance_payload.get(key)
        mask = appearance_payload.get('masks', {}).get(key)
        if tensor is None or mask is None or index >= tensor.shape[0] or not bool(mask[index].item()):
            appearance[key] = None
        else:
            appearance[key] = tensor[index].detach().clone()
    if all(value is None for value in appearance.values()):
        return None
    return appearance


def get_detection_feature_assist(feature_assist_payload, index):
    return get_detection_appearance(feature_assist_payload, index)


def maybe_extract_detection_appearance_features(tracking_feature_payload, detections, tracking_cfg, transform_cfg=None, base_size=None):
    if tracking_cfg is None:
        return None
    appearance_cfg = normalize_appearance_cfg(tracking_cfg.get('appearance', {}))
    association_cfg = tracking_cfg.get('association', {}) if isinstance(tracking_cfg, dict) else {}
    if not appearance_cfg.get('enabled', False):
        return None
    if not association_cfg.get('use_appearance', False):
        return None
    return extract_detection_appearance_features(
        tracking_feature_payload,
        detections,
        appearance_cfg,
        transform_cfg=transform_cfg,
        base_size=base_size,
    )


def maybe_extract_detection_feature_assist_features(tracking_feature_payload, detections, tracking_cfg, transform_cfg=None, base_size=None):
    if tracking_cfg is None:
        return None
    feature_assist_cfg = normalize_feature_assist_cfg(tracking_cfg.get('feature_assist', {}))
    if not feature_assist_cfg.get('enabled', False):
        return None
    return extract_detection_feature_assist_features(
        tracking_feature_payload,
        detections,
        feature_assist_cfg,
        tracking_cfg=tracking_cfg,
        transform_cfg=transform_cfg,
        base_size=base_size,
    )


def extract_detection_appearance_features(tracking_feature_payload, detections, appearance_cfg, transform_cfg=None, base_size=None):
    cfg = normalize_appearance_cfg(appearance_cfg)
    detections = torch.as_tensor(detections, dtype=torch.float32)
    num_detections = int(detections.shape[0]) if detections.ndim > 1 else (1 if detections.numel() > 0 else 0)
    include_modalities = bool(cfg.get('use_rgb_ir_branches', False))

    if tracking_feature_payload is None or num_detections == 0:
        return empty_appearance_payload(
            num_detections=num_detections,
            embedding_dim=int(cfg['embedding_dim']),
            device=detections.device if torch.is_tensor(detections) else 'cpu',
            include_modalities=include_modalities,
        )

    fused_feats = tracking_feature_payload.get('fused_feats')
    rgb_feats = tracking_feature_payload.get('rgb_feats')
    ir_feats = tracking_feature_payload.get('ir_feats')
    input_hw = tracking_feature_payload.get('input_hw')

    if fused_feats is None or input_hw is None:
        return empty_appearance_payload(
            num_detections=num_detections,
            embedding_dim=int(cfg['embedding_dim']),
            device=detections.device if torch.is_tensor(detections) else 'cpu',
            include_modalities=include_modalities,
        )

    source_boxes = _transform_boxes_to_source(detections[:, :5], transform_cfg=transform_cfg, base_size=base_size)
    device = fused_feats[0].device
    source_boxes = source_boxes.to(device=device, dtype=torch.float32)

    payload = {
        'fused': _extract_level_pooled_embedding(fused_feats, source_boxes, input_hw, cfg),
        'rgb': None,
        'ir': None,
        'masks': {},
    }
    payload['masks']['fused'] = torch.ones(num_detections, dtype=torch.bool, device=device)

    if include_modalities:
        payload['rgb'] = _extract_level_pooled_embedding(rgb_feats, source_boxes, input_hw, cfg) if rgb_feats is not None else None
        payload['ir'] = _extract_level_pooled_embedding(ir_feats, source_boxes, input_hw, cfg) if ir_feats is not None else None
        payload['masks']['rgb'] = torch.ones(num_detections, dtype=torch.bool, device=device) if payload['rgb'] is not None else torch.zeros(num_detections, dtype=torch.bool, device=device)
        payload['masks']['ir'] = torch.ones(num_detections, dtype=torch.bool, device=device) if payload['ir'] is not None else torch.zeros(num_detections, dtype=torch.bool, device=device)
    return payload


def extract_detection_feature_assist_features(tracking_feature_payload, detections, feature_assist_cfg, tracking_cfg=None, transform_cfg=None, base_size=None):
    cfg = normalize_feature_assist_cfg(feature_assist_cfg)
    tracking_cfg = dict(tracking_cfg or {})
    appearance_cfg = normalize_appearance_cfg(tracking_cfg.get('appearance', {}))
    proxy_cfg = {
        'enabled': True,
        'embedding_dim': int(cfg.get('embedding_dim', appearance_cfg.get('embedding_dim', 128))),
        'sampling_radius': int(appearance_cfg.get('sampling_radius', 1)),
        'use_rgb_ir_branches': True,
    }
    payload = extract_detection_appearance_features(
        tracking_feature_payload,
        detections,
        proxy_cfg,
        transform_cfg=transform_cfg,
        base_size=base_size,
    )
    if payload is not None:
        payload['source'] = cfg.get('source', 'temporal_fused')
    return payload


def _transform_boxes_to_source(boxes, transform_cfg=None, base_size=None):
    boxes = boxes.detach().clone().to(dtype=torch.float32)
    if transform_cfg is None or base_size is None:
        return boxes

    source_size = float(transform_cfg.get('size', base_size))
    scale = source_size / float(base_size)
    boxes[:, :4] *= scale
    if transform_cfg.get('horizontal_flip', False):
        boxes[:, 0] = source_size - boxes[:, 0]
        boxes[:, 4] = wrap_obb_angle(-boxes[:, 4])
    return boxes


def _extract_level_pooled_embedding(feature_levels, boxes, input_hw, appearance_cfg):
    if feature_levels is None:
        return None

    sampled_levels = [
        _sample_feature_level(level, boxes, input_hw, radius=int(appearance_cfg.get('sampling_radius', 1)))
        for level in feature_levels
    ]
    combined = torch.cat(sampled_levels, dim=1) if len(sampled_levels) > 1 else sampled_levels[0]
    embedding = _reduce_embedding_dim(combined, int(appearance_cfg['embedding_dim']))
    return F.normalize(embedding, dim=1)


def _sample_feature_level(feature_map, boxes, input_hw, radius=1):
    _, channels, feat_h, feat_w = feature_map.shape
    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    scale_x = feat_w / max(float(input_w), 1.0)
    scale_y = feat_h / max(float(input_h), 1.0)
    centers_x = torch.clamp(torch.round(boxes[:, 0] * scale_x - 0.5).long(), 0, feat_w - 1)
    centers_y = torch.clamp(torch.round(boxes[:, 1] * scale_y - 0.5).long(), 0, feat_h - 1)

    pooled = []
    feature_map = feature_map[0]
    for center_x, center_y in zip(centers_x.tolist(), centers_y.tolist()):
        x0 = max(0, center_x - radius)
        x1 = min(feat_w, center_x + radius + 1)
        y0 = max(0, center_y - radius)
        y1 = min(feat_h, center_y + radius + 1)
        patch = feature_map[:, y0:y1, x0:x1]
        if patch.numel() == 0:
            pooled.append(torch.zeros(channels, dtype=feature_map.dtype, device=feature_map.device))
        else:
            pooled.append(patch.mean(dim=(1, 2)))
    return torch.stack(pooled, dim=0)


def _reduce_embedding_dim(vectors, embedding_dim):
    if vectors.shape[1] == embedding_dim:
        return vectors
    reduced = F.adaptive_avg_pool1d(vectors.unsqueeze(1), embedding_dim).squeeze(1)
    return reduced
