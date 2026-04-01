import math
from copy import deepcopy
import warnings

import torch
from omegaconf import OmegaConf

from .appearance import normalize_appearance_cfg, normalize_feature_assist_cfg
from .memory import normalize_memory_cfg
from .modality import normalize_modality_cfg
from .refinement import normalize_refinement_cfg


TRACKING_DEFAULTS = {
    'enabled': False,
    'method': 'tracking_by_detection',
    'max_age': 30,
    'min_hits': 3,
    'init_score_threshold': 0.3,
    'match_iou_threshold': 0.3,
    'max_center_distance': 50.0,
    'use_class_constraint': True,
    'class_mismatch_penalty': 1.0,
    'use_kalman': True,
    'angle_smoothing': 0.7,
    'keep_history': 20,
    'appearance': normalize_appearance_cfg(),
    'memory': normalize_memory_cfg(),
    'modality': normalize_modality_cfg(),
    'feature_assist': normalize_feature_assist_cfg(),
    'reactivation': {
        'enabled': False,
        'max_reactivate_age': 8,
        'use_memory_reactivation': True,
        'use_feature_assist_reactivation': True,
        'reactivation_gate': 0.4,
    },
    'overlap_disambiguation': {
        'enabled': False,
        'overlap_iou_threshold': 0.45,
        'ambiguity_margin': 0.10,
        'assist_margin': 0.05,
    },
    'refinement': normalize_refinement_cfg(),
    'smoothing': {
        'enabled': False,
        'bbox_ema': 0.7,
        'angle_ema': 0.6,
    },
    'association': {
        'use_appearance': False,
        'use_temporal_memory': False,
        'use_modality_awareness': False,
        'use_feature_assist': False,
        'dynamic_weighting': False,
        'w_motion': 0.25,
        'w_iou': 0.60,
        'w_app': 0.50,
        'w_temporal': 0.30,
        'w_score': 0.15,
        'w_assist': 0.35,
        'rgb_bias_gain': 0.5,
        'ir_bias_gain': 0.5,
        'low_conf_motion_boost': 0.3,
        'class_mismatch_penalty': 1000000.0,
        'appearance_gate': 0.5,
        'assist_gate': 0.6,
        'lost_track_expansion': 1.25,
    },
}


def normalize_tracking_cfg(tracking_cfg=None):
    cfg = deepcopy(TRACKING_DEFAULTS)
    if tracking_cfg is None:
        return cfg

    if OmegaConf.is_config(tracking_cfg):
        tracking_cfg = OmegaConf.to_container(tracking_cfg, resolve=True)

    if not isinstance(tracking_cfg, dict):
        return cfg
    requested_method = tracking_cfg.get('method', TRACKING_DEFAULTS['method'])

    appearance_input = tracking_cfg.get('appearance', {}) if isinstance(tracking_cfg.get('appearance', {}), dict) else {}
    memory_input = tracking_cfg.get('memory', {}) if isinstance(tracking_cfg.get('memory', {}), dict) else {}
    modality_input = tracking_cfg.get('modality', {}) if isinstance(tracking_cfg.get('modality', {}), dict) else {}
    feature_assist_input = tracking_cfg.get('feature_assist', {}) if isinstance(tracking_cfg.get('feature_assist', {}), dict) else {}
    reactivation_input = tracking_cfg.get('reactivation', {}) if isinstance(tracking_cfg.get('reactivation', {}), dict) else {}
    overlap_input = tracking_cfg.get('overlap_disambiguation', {}) if isinstance(tracking_cfg.get('overlap_disambiguation', {}), dict) else {}
    refinement_input = tracking_cfg.get('refinement', {}) if isinstance(tracking_cfg.get('refinement', {}), dict) else {}
    association_input = tracking_cfg.get('association', {}) if isinstance(tracking_cfg.get('association', {}), dict) else {}
    smoothing_input = tracking_cfg.get('smoothing', {}) if isinstance(tracking_cfg.get('smoothing', {}), dict) else {}

    nested_keys = {
        'appearance', 'memory', 'modality', 'feature_assist', 'reactivation',
        'overlap_disambiguation', 'refinement', 'association', 'smoothing'
    }
    for key, value in tracking_cfg.items():
        if key in nested_keys and isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value

    cfg['appearance'] = normalize_appearance_cfg(cfg.get('appearance', {}))
    cfg['memory'] = normalize_memory_cfg(cfg.get('memory', {}))
    cfg['modality'] = normalize_modality_cfg(cfg.get('modality', {}))
    cfg['feature_assist'] = normalize_feature_assist_cfg(cfg.get('feature_assist', {}))
    cfg['refinement'] = normalize_refinement_cfg(cfg.get('refinement', {}))

    if cfg['appearance'].get('enabled', False) and 'use_appearance' not in association_input:
        cfg['association']['use_appearance'] = True
    if cfg['memory'].get('enabled', False) and 'use_temporal_memory' not in association_input:
        cfg['association']['use_temporal_memory'] = True
    if cfg['modality'].get('enabled', False) and 'use_modality_awareness' not in association_input:
        cfg['association']['use_modality_awareness'] = True
    if cfg['feature_assist'].get('enabled', False) and 'use_feature_assist' not in association_input:
        cfg['association']['use_feature_assist'] = True
    if cfg['feature_assist'].get('enabled', False) and 'enabled' not in reactivation_input:
        cfg['reactivation']['enabled'] = True
    if cfg['feature_assist'].get('enabled', False) and 'enabled' not in overlap_input:
        cfg['overlap_disambiguation']['enabled'] = True
    if 'class_mismatch_penalty' in tracking_cfg and 'class_mismatch_penalty' not in association_input:
        cfg['association']['class_mismatch_penalty'] = tracking_cfg['class_mismatch_penalty']
    if 'angle_ema' not in smoothing_input and 'angle_smoothing' in tracking_cfg:
        cfg['smoothing']['angle_ema'] = tracking_cfg['angle_smoothing']
    if 'reliability_ema' in modality_input:
        cfg['modality']['reliability_ema'] = modality_input['reliability_ema']
    if 'method' in tracking_cfg and requested_method != TRACKING_DEFAULTS['method']:
        warnings.warn(
            'tracking.method is retained for compatibility, but the current runtime only supports '
            '`tracking_by_detection`. Custom method values are ignored.',
            stacklevel=2,
        )
        cfg['method'] = TRACKING_DEFAULTS['method']

    cfg['class_mismatch_penalty'] = cfg['association']['class_mismatch_penalty']
    return cfg



def build_tracker_from_cfg(tracking_cfg=None, class_names=None):
    cfg = normalize_tracking_cfg(tracking_cfg)
    if not cfg['enabled']:
        return None

    from .tracker import MultiObjectTracker

    return MultiObjectTracker(tracking_cfg=cfg, class_names=class_names)



def ensure_detection_tensor(detections, device='cpu'):
    if detections is None:
        return torch.zeros((0, 7), dtype=torch.float32, device=device)
    if isinstance(detections, list):
        if len(detections) == 0:
            return torch.zeros((0, 7), dtype=torch.float32, device=device)
        detections = torch.tensor(detections, dtype=torch.float32, device=device)
    elif not torch.is_tensor(detections):
        detections = torch.as_tensor(detections, dtype=torch.float32, device=device)
    else:
        detections = detections.to(device=device, dtype=torch.float32)

    if detections.ndim == 1:
        detections = detections.unsqueeze(0)
    if detections.numel() == 0:
        return torch.zeros((0, 7), dtype=torch.float32, device=device)
    if detections.shape[-1] < 7:
        raise ValueError('Tracking expects detections shaped as [N, 7] -> [cx, cy, w, h, angle, score, class_id].')
    return detections[:, :7].contiguous()



def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi



def smooth_angle(previous_angle, new_angle, momentum=0.7):
    if previous_angle is None:
        return wrap_angle(float(new_angle))
    delta = wrap_angle(float(new_angle) - float(previous_angle))
    return wrap_angle(float(previous_angle) + (1.0 - momentum) * delta)



def detections_to_results(detections):
    detections = ensure_detection_tensor(detections)
    results = []
    for det in detections:
        results.append(
            {
                'track_id': None,
                'class_id': int(det[6].item()),
                'score': float(det[5].item()),
                'obb': det[:5].tolist(),
                'state': 'detection',
            }
        )
    return results



def tracks_to_results(tracks):
    return [track.to_result() for track in tracks]
