from copy import deepcopy

import torch
from omegaconf import OmegaConf

from src.utils.tta import wrap_obb_angle


MODALITY_DEFAULTS = {
    'enabled': False,
    'use_scene_adaptation': False,
    'reliability_source': 'auto',
    'reliability_ema': 0.8,
    'night_motion_boost': 0.2,
    'fog_temporal_boost': 0.2,
}

NIGHT_TOKENS = {'night', 'dark', 'low_light', 'evening'}
FOG_TOKENS = {'fog', 'foggy', 'mist', 'haze', 'smog', 'low_visibility'}
DAY_TOKENS = {'day', 'daytime', 'sunny', 'bright', 'strong_light'}


# Stage-5 note:
# Reliability estimation is intentionally lightweight. When detector fusion weights are
# not exposed directly, we fall back to proxy reliability from local RGB / IR / fused
# feature responses. Missing reliability always degrades back to stage-3 behavior.


def normalize_modality_cfg(modality_cfg=None):
    cfg = deepcopy(MODALITY_DEFAULTS)
    if modality_cfg is None:
        return cfg

    if OmegaConf.is_config(modality_cfg):
        modality_cfg = OmegaConf.to_container(modality_cfg, resolve=True)

    if isinstance(modality_cfg, dict):
        for key, value in modality_cfg.items():
            cfg[key] = value
    return cfg



def normalize_reliability_dict(reliability):
    if reliability is None:
        return None

    if OmegaConf.is_config(reliability):
        reliability = OmegaConf.to_container(reliability, resolve=True)

    if not isinstance(reliability, dict):
        return None

    normalized = {}
    for key in ('rgb_reliability', 'ir_reliability', 'fused_reliability'):
        value = reliability.get(key)
        if value is None:
            normalized[key] = None
        else:
            normalized[key] = float(max(0.0, min(1.0, value)))

    if all(value is None for value in normalized.values()):
        return None
    return normalized



def empty_reliability_payload(num_detections=0, device='cpu'):
    return {
        'rgb_reliability': torch.zeros(num_detections, dtype=torch.float32, device=device),
        'ir_reliability': torch.zeros(num_detections, dtype=torch.float32, device=device),
        'fused_reliability': torch.zeros(num_detections, dtype=torch.float32, device=device),
        'masks': {
            'rgb_reliability': torch.zeros(num_detections, dtype=torch.bool, device=device),
            'ir_reliability': torch.zeros(num_detections, dtype=torch.bool, device=device),
            'fused_reliability': torch.zeros(num_detections, dtype=torch.bool, device=device),
        },
    }



def normalize_reliability_payload(payload, num_detections=None, device='cpu'):
    if payload is None:
        return None

    if OmegaConf.is_config(payload):
        payload = OmegaConf.to_container(payload, resolve=True)

    if not isinstance(payload, dict):
        return None

    if num_detections is None:
        first_value = next((value for key, value in payload.items() if key.endswith('reliability') and value is not None), None)
        if first_value is None:
            return None
        first_value = torch.as_tensor(first_value, dtype=torch.float32)
        num_detections = int(first_value.reshape(-1).shape[0])

    normalized = empty_reliability_payload(num_detections=num_detections, device=device)
    valid = False
    for key in ('rgb_reliability', 'ir_reliability', 'fused_reliability'):
        value = payload.get(key)
        if value is None:
            continue
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
        if tensor.numel() == 1 and num_detections != 1:
            tensor = tensor.repeat(num_detections)
        if tensor.shape[0] != num_detections:
            raise ValueError(f'Reliability tensor for {key} must have shape [N] matching detections.')
        normalized[key] = tensor.clamp(0.0, 1.0)
        normalized['masks'][key] = torch.ones(num_detections, dtype=torch.bool, device=device)
        valid = True

    if not valid:
        return None
    return normalized



def get_detection_reliability(reliability_payload, index):
    if reliability_payload is None:
        return None

    reliability = {}
    for key in ('rgb_reliability', 'ir_reliability', 'fused_reliability'):
        tensor = reliability_payload.get(key)
        mask = reliability_payload.get('masks', {}).get(key)
        if tensor is None or mask is None or index >= tensor.shape[0] or not bool(mask[index].item()):
            reliability[key] = None
        else:
            reliability[key] = float(tensor[index].item())
    return normalize_reliability_dict(reliability)



def maybe_extract_detection_reliability_features(tracking_feature_payload, detections, tracking_cfg, transform_cfg=None, base_size=None):
    if tracking_cfg is None:
        return None
    modality_cfg = normalize_modality_cfg(tracking_cfg.get('modality', {}))
    association_cfg = tracking_cfg.get('association', {}) if isinstance(tracking_cfg, dict) else {}
    if not modality_cfg.get('enabled', False):
        return None
    if not association_cfg.get('use_modality_awareness', False):
        return None
    return extract_detection_reliability_features(
        tracking_feature_payload,
        detections,
        modality_cfg,
        transform_cfg=transform_cfg,
        base_size=base_size,
    )



def extract_detection_reliability_features(tracking_feature_payload, detections, modality_cfg, transform_cfg=None, base_size=None):
    modality_cfg = normalize_modality_cfg(modality_cfg)
    detections = torch.as_tensor(detections, dtype=torch.float32)
    if detections.ndim == 1:
        detections = detections.unsqueeze(0)
    if detections.numel() == 0:
        return None
    if tracking_feature_payload is None:
        return None

    fused_feats = tracking_feature_payload.get('fused_feats')
    rgb_feats = tracking_feature_payload.get('rgb_feats')
    ir_feats = tracking_feature_payload.get('ir_feats')
    input_hw = tracking_feature_payload.get('input_hw')

    if input_hw is None or (fused_feats is None and rgb_feats is None and ir_feats is None):
        return None

    first_level = next((levels[0] for levels in (fused_feats, rgb_feats, ir_feats) if levels is not None and len(levels) > 0), None)
    if first_level is None:
        return None

    source_boxes = _transform_boxes_to_source(detections[:, :5], transform_cfg=transform_cfg, base_size=base_size)
    device = first_level.device
    source_boxes = source_boxes.to(device=device, dtype=torch.float32)

    rgb_response = _extract_reliability_response(rgb_feats, source_boxes, input_hw) if rgb_feats is not None else None
    ir_response = _extract_reliability_response(ir_feats, source_boxes, input_hw) if ir_feats is not None else None
    fused_response = _extract_reliability_response(fused_feats, source_boxes, input_hw) if fused_feats is not None else None

    if rgb_response is None and ir_response is None and fused_response is None:
        return None

    num_detections = int(detections.shape[0])
    if rgb_response is not None and ir_response is not None:
        denom = (rgb_response + ir_response).clamp(min=1e-6)
        rgb_reliability = rgb_response / denom
        ir_reliability = ir_response / denom
    elif rgb_response is not None:
        rgb_reliability = torch.ones(num_detections, dtype=torch.float32, device=device)
        ir_reliability = torch.zeros(num_detections, dtype=torch.float32, device=device)
    elif ir_response is not None:
        rgb_reliability = torch.zeros(num_detections, dtype=torch.float32, device=device)
        ir_reliability = torch.ones(num_detections, dtype=torch.float32, device=device)
    else:
        rgb_reliability = torch.full((num_detections,), 0.5, dtype=torch.float32, device=device)
        ir_reliability = torch.full((num_detections,), 0.5, dtype=torch.float32, device=device)

    if fused_response is not None:
        fused_reliability = fused_response / (fused_response + 1.0)
    else:
        fused_reliability = 0.5 * (rgb_reliability + ir_reliability)

    return normalize_reliability_payload(
        {
            'rgb_reliability': rgb_reliability,
            'ir_reliability': ir_reliability,
            'fused_reliability': fused_reliability,
        },
        num_detections=num_detections,
        device=device,
    )



def merge_track_and_detection_reliability(track_reliability=None, detection_reliability=None):
    track_reliability = normalize_reliability_dict(track_reliability)
    detection_reliability = normalize_reliability_dict(detection_reliability)
    if track_reliability is None and detection_reliability is None:
        return None

    merged = {}
    for key in ('rgb_reliability', 'ir_reliability', 'fused_reliability'):
        values = []
        if track_reliability is not None and track_reliability.get(key) is not None:
            values.append(float(track_reliability[key]))
        if detection_reliability is not None and detection_reliability.get(key) is not None:
            values.append(float(detection_reliability[key]))
        merged[key] = float(sum(values) / len(values)) if values else None
    return normalize_reliability_dict(merged)



def resolve_modality_mix_weights(track_reliability=None, detection_reliability=None, available_modalities=('fused',)):
    available_modalities = [modality for modality in available_modalities if modality in {'fused', 'rgb', 'ir'}]
    if not available_modalities:
        return {}

    merged = merge_track_and_detection_reliability(track_reliability, detection_reliability)
    if merged is None:
        if 'fused' in available_modalities:
            return {'fused': 1.0}
        uniform = 1.0 / float(len(available_modalities))
        return {modality: uniform for modality in available_modalities}

    raw_weights = {}
    if 'rgb' in available_modalities:
        raw_weights['rgb'] = float(merged.get('rgb_reliability', 0.0) or 0.0)
    if 'ir' in available_modalities:
        raw_weights['ir'] = float(merged.get('ir_reliability', 0.0) or 0.0)
    if 'fused' in available_modalities:
        fused_default = 0.5 * ((merged.get('rgb_reliability', 0.5) or 0.5) + (merged.get('ir_reliability', 0.5) or 0.5))
        raw_weights['fused'] = float(merged.get('fused_reliability', fused_default) or fused_default)

    total = sum(raw_weights.values())
    if total <= 1e-6:
        if 'fused' in available_modalities:
            return {'fused': 1.0}
        uniform = 1.0 / float(len(available_modalities))
        return {modality: uniform for modality in available_modalities}
    return {key: float(value / total) for key, value in raw_weights.items()}



def compute_dynamic_weight_profile(association_cfg, modality_cfg, detection_reliability=None, track_reliability=None, frame_meta=None):
    association_cfg = dict(association_cfg or {})
    modality_cfg = normalize_modality_cfg(modality_cfg)

    profile = {
        'w_motion': float(association_cfg.get('w_motion', 0.25)),
        'w_iou': float(association_cfg.get('w_iou', 0.60)),
        'w_app': float(association_cfg.get('w_app', 0.50)),
        'w_temporal': float(association_cfg.get('w_temporal', 0.30)),
        'w_score': float(association_cfg.get('w_score', 0.15)),
        'association_mode': 'stage3_fallback',
        'low_confidence_motion_fallback': False,
        'scene_adapted': False,
        'merged_reliability': None,
    }

    if not modality_cfg.get('enabled', False):
        return profile
    if not association_cfg.get('use_modality_awareness', False):
        return profile
    if not association_cfg.get('dynamic_weighting', False):
        return profile

    merged = merge_track_and_detection_reliability(track_reliability, detection_reliability)
    if merged is None:
        return profile

    rgb = float(merged.get('rgb_reliability', 0.5) or 0.5)
    ir = float(merged.get('ir_reliability', 0.5) or 0.5)
    fused = float(merged.get('fused_reliability', 0.5) or 0.5)
    dominance = rgb - ir
    low_confidence = max(rgb, ir, fused) < 0.45

    if low_confidence:
        boost = float(association_cfg.get('low_conf_motion_boost', 0.3))
        profile['w_motion'] += boost
        profile['w_temporal'] += 0.75 * boost
        profile['w_app'] *= max(0.35, 1.0 - boost)
        profile['association_mode'] = 'low_confidence_motion_fallback'
        profile['low_confidence_motion_fallback'] = True
    else:
        if abs(dominance) <= 0.15:
            profile['association_mode'] = 'balanced'
            profile['w_app'] += 0.10 * fused
        elif dominance > 0.0:
            gain = float(association_cfg.get('rgb_bias_gain', 0.5))
            profile['association_mode'] = 'rgb_dominant'
            profile['w_app'] += gain * min(dominance, 1.0)
            profile['w_motion'] *= max(0.75, 1.0 - 0.20 * dominance)
        else:
            gain = float(association_cfg.get('ir_bias_gain', 0.5))
            profile['association_mode'] = 'ir_dominant'
            profile['w_app'] += gain * min(-dominance, 1.0)
            profile['w_motion'] *= max(0.75, 1.0 + 0.10 * dominance)

    if modality_cfg.get('use_scene_adaptation', False):
        scene_summary = _extract_scene_summary(frame_meta)
        if scene_summary['night']:
            boost = float(modality_cfg.get('night_motion_boost', 0.2))
            profile['w_motion'] += boost
            profile['w_temporal'] += 0.5 * boost
            profile['scene_adapted'] = True
        if scene_summary['fog']:
            boost = float(modality_cfg.get('fog_temporal_boost', 0.2))
            profile['w_temporal'] += boost
            profile['w_app'] *= max(0.5, 1.0 - 0.5 * boost)
            profile['scene_adapted'] = True
        if scene_summary['day'] and dominance > 0.0:
            profile['w_app'] += 0.10 * float(association_cfg.get('rgb_bias_gain', 0.5)) * min(dominance, 1.0)
            profile['scene_adapted'] = True

    base_total = float(association_cfg.get('w_motion', 0.25)) + float(association_cfg.get('w_iou', 0.60)) + float(association_cfg.get('w_app', 0.50)) + float(association_cfg.get('w_temporal', 0.30))
    dynamic_total = profile['w_motion'] + profile['w_iou'] + profile['w_app'] + profile['w_temporal']
    if dynamic_total > 1e-6:
        scale = base_total / dynamic_total
        profile['w_motion'] *= scale
        profile['w_iou'] *= scale
        profile['w_app'] *= scale
        profile['w_temporal'] *= scale

    profile['merged_reliability'] = merged
    return profile



def _extract_scene_summary(frame_meta=None):
    metadata = frame_meta if isinstance(frame_meta, dict) else {}
    time_of_day = str(metadata.get('time_of_day', '') or '').strip().lower()
    weather = str(metadata.get('weather', '') or metadata.get('visibility', '') or '').strip().lower()
    return {
        'night': any(token in time_of_day for token in NIGHT_TOKENS),
        'fog': any(token in weather for token in FOG_TOKENS),
        'day': any(token in time_of_day for token in DAY_TOKENS),
    }



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



def _extract_reliability_response(feature_levels, boxes, input_hw):
    if feature_levels is None:
        return None
    responses = [_sample_response_level(level, boxes, input_hw) for level in feature_levels]
    if not responses:
        return None
    stacked = torch.stack(responses, dim=0)
    return stacked.mean(dim=0)



def _sample_response_level(feature_map, boxes, input_hw, radius=1):
    _, _, feat_h, feat_w = feature_map.shape
    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    scale_x = feat_w / max(float(input_w), 1.0)
    scale_y = feat_h / max(float(input_h), 1.0)
    centers_x = torch.clamp(torch.round(boxes[:, 0] * scale_x - 0.5).long(), 0, feat_w - 1)
    centers_y = torch.clamp(torch.round(boxes[:, 1] * scale_y - 0.5).long(), 0, feat_h - 1)

    level = feature_map[0]
    responses = []
    for center_x, center_y in zip(centers_x.tolist(), centers_y.tolist()):
        x0 = max(0, center_x - radius)
        x1 = min(feat_w, center_x + radius + 1)
        y0 = max(0, center_y - radius)
        y1 = min(feat_h, center_y + radius + 1)
        patch = level[:, y0:y1, x0:x1]
        if patch.numel() == 0:
            responses.append(torch.tensor(0.0, dtype=level.dtype, device=level.device))
        else:
            responses.append(patch.abs().mean())
    return torch.stack(responses, dim=0)
