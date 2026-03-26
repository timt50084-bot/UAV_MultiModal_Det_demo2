import torch
import torch.nn.functional as F

from src.model.bbox_utils import batch_prob_iou

from .appearance import normalize_appearance_payload, normalize_feature_assist_payload
from .modality import (
    compute_dynamic_weight_profile,
    get_detection_reliability,
    normalize_reliability_payload,
    resolve_modality_mix_weights,
)
from .refinement import get_detection_refinement_context, normalize_refinement_payload
from .utils import ensure_detection_tensor


# Stage-7 note:
# Association remains tracking-by-detection and keeps the same greedy matching skeleton.
# We only add lightweight feature-assist cost terms, runtime overlap disambiguation signals,
# and compatibility hooks for stronger lost-track reactivation in the tracker.


def pairwise_prob_iou(track_boxes, det_boxes):
    if track_boxes.numel() == 0 or det_boxes.numel() == 0:
        return torch.zeros((track_boxes.shape[0], det_boxes.shape[0]), dtype=torch.float32)
    merged = torch.cat([track_boxes, det_boxes], dim=0)
    iou_matrix = batch_prob_iou(merged)
    split = track_boxes.shape[0]
    return iou_matrix[:split, split:]



def build_association_cost(tracks, detections, tracking_cfg, appearance_features=None, reliability_features=None, feature_assist_features=None, refinement_payload=None, frame_meta=None):
    detections = ensure_detection_tensor(detections)
    num_tracks = len(tracks)
    num_dets = detections.shape[0]

    if num_tracks == 0 or num_dets == 0:
        empty_cost = torch.zeros((num_tracks, num_dets), dtype=torch.float32)
        empty_mask = torch.zeros((num_tracks, num_dets), dtype=torch.bool)
        return (
            empty_cost,
            empty_mask,
            {
                'prob_iou': empty_cost,
                'center_distance': empty_cost,
                'appearance_cost': empty_cost,
                'appearance_valid_mask': empty_mask,
                'feature_assist_cost': empty_cost,
                'feature_assist_valid_mask': empty_mask,
                'temporal_cost': empty_cost,
                'temporal_valid_mask': empty_mask,
                'dynamic_weights': {
                    'w_motion': empty_cost,
                    'w_iou': empty_cost,
                    'w_app': empty_cost,
                    'w_temporal': empty_cost,
                    'w_score': empty_cost,
                    'w_assist': empty_cost,
                },
                'association_mode': [['stage3_fallback' for _ in range(num_dets)] for _ in range(num_tracks)],
                'low_confidence_motion_fallback': empty_mask,
                'scene_adapted': empty_mask,
                'reliability_payload': None,
                'refinement_payload': None,
                'feature_assist_payload': None,
            },
        )

    association_cfg = tracking_cfg.get('association', {}) if isinstance(tracking_cfg, dict) else {}
    memory_cfg = tracking_cfg.get('memory', {}) if isinstance(tracking_cfg, dict) else {}
    modality_cfg = tracking_cfg.get('modality', {}) if isinstance(tracking_cfg, dict) else {}
    feature_assist_cfg = tracking_cfg.get('feature_assist', {}) if isinstance(tracking_cfg, dict) else {}
    use_appearance = bool(association_cfg.get('use_appearance', False))
    use_temporal_memory = bool(association_cfg.get('use_temporal_memory', False) and memory_cfg.get('enabled', False))
    use_modality_awareness = bool(association_cfg.get('use_modality_awareness', False) and modality_cfg.get('enabled', False))
    use_feature_assist = bool(association_cfg.get('use_feature_assist', False) and feature_assist_cfg.get('enabled', False))

    track_boxes = torch.stack([track.bbox_obb for track in tracks], dim=0)
    det_boxes = detections[:, :5]
    prob_iou = pairwise_prob_iou(track_boxes, det_boxes)
    center_distance = torch.cdist(track_boxes[:, :2], det_boxes[:, :2], p=2)
    distance_limits = build_distance_limits(tracks, tracking_cfg).to(device=det_boxes.device, dtype=det_boxes.dtype)
    normalized_distance = center_distance / distance_limits.unsqueeze(1).clamp(min=1e-6)
    det_scores = detections[:, 5].clamp(0.0, 1.0).unsqueeze(0).expand(num_tracks, -1)

    iou_cost = 1.0 - prob_iou
    motion_cost = normalized_distance.clamp(max=1.5)
    score_cost = 1.0 - det_scores
    valid_mask = center_distance <= distance_limits.unsqueeze(1)

    iou_thresholds = torch.full((num_tracks,), float(tracking_cfg['match_iou_threshold']), dtype=det_boxes.dtype, device=det_boxes.device)
    if use_temporal_memory:
        relaxed = iou_thresholds.clone()
        for index, track in enumerate(tracks):
            if track.state == track.LOST:
                relaxed[index] = relaxed[index] * 0.5
        iou_thresholds = relaxed
    valid_mask &= (prob_iou >= iou_thresholds.unsqueeze(1)) | (center_distance <= distance_limits.unsqueeze(1) * 0.5)

    if refinement_payload is not None:
        for det_index in range(num_dets):
            context = get_detection_refinement_context(refinement_payload, det_index)
            if context is None or not context.get('predicted_candidate', False):
                continue
            support_track_id = context.get('support_track_id')
            if support_track_id is None:
                continue
            allowed_tracks = torch.tensor([track.track_id == int(support_track_id) for track in tracks], dtype=torch.bool, device=det_boxes.device)
            valid_mask[:, det_index] &= allowed_tracks

    if tracking_cfg.get('use_class_constraint', True):
        track_classes = torch.tensor([track.class_id for track in tracks], dtype=torch.long, device=det_boxes.device)
        det_classes = detections[:, 6].to(device=det_boxes.device, dtype=torch.long)
        class_match = track_classes.unsqueeze(1) == det_classes.unsqueeze(0)
        valid_mask &= class_match
        class_penalty = torch.where(class_match, torch.zeros_like(iou_cost), torch.full_like(iou_cost, float(association_cfg.get('class_mismatch_penalty', 1e6))))
    else:
        class_mismatch = torch.zeros_like(iou_cost)
        class_penalty = class_mismatch * float(association_cfg.get('class_mismatch_penalty', 1.0))

    reliability_payload = normalize_reliability_payload(reliability_features, num_detections=num_dets, device=det_boxes.device)
    refinement_payload = normalize_refinement_payload(refinement_payload, num_detections=num_dets)
    feature_assist_payload = normalize_feature_assist_payload(feature_assist_features, num_detections=num_dets, device=det_boxes.device, feature_assist_cfg=feature_assist_cfg)

    appearance_cost, appearance_valid_mask, appearance_debug = build_appearance_cost(
        tracks,
        appearance_features,
        reliability_payload,
        num_dets=num_dets,
        association_cfg=association_cfg,
        use_temporal_memory=use_temporal_memory,
        use_modality_awareness=use_modality_awareness,
    )
    feature_assist_cost, feature_assist_valid_mask = build_feature_assist_cost(
        tracks,
        feature_assist_payload,
        num_dets=num_dets,
        use_temporal_memory=use_temporal_memory,
    )
    temporal_cost, temporal_valid_mask = build_temporal_cost(tracks, detections, tracking_cfg)

    dynamic_weights, modality_debug = build_dynamic_weight_matrices(
        tracks,
        num_dets=num_dets,
        tracking_cfg=tracking_cfg,
        reliability_payload=reliability_payload,
        frame_meta=frame_meta,
        device=det_boxes.device,
        dtype=det_boxes.dtype,
    )

    cost = (
        dynamic_weights['w_iou'] * iou_cost
        + dynamic_weights['w_motion'] * motion_cost
        + dynamic_weights['w_score'] * score_cost
        + class_penalty
    )

    if use_appearance and appearance_cost is not None:
        cost = cost + dynamic_weights['w_app'] * appearance_cost
        appearance_gate = association_cfg.get('appearance_gate', None)
        if appearance_gate is not None:
            valid_mask &= (~appearance_valid_mask) | (appearance_cost <= float(appearance_gate))

    if use_feature_assist and feature_assist_cost is not None:
        cost = cost + dynamic_weights['w_assist'] * feature_assist_cost
        assist_gate = association_cfg.get('assist_gate', None)
        if assist_gate is not None:
            valid_mask &= (~feature_assist_valid_mask) | (feature_assist_cost <= float(assist_gate))

    if use_temporal_memory and temporal_cost is not None:
        cost = cost + dynamic_weights['w_temporal'] * temporal_cost
        valid_mask &= (~temporal_valid_mask) | (temporal_cost <= 1.5)

    debug = {
        'prob_iou': prob_iou,
        'center_distance': center_distance,
        'appearance_cost': appearance_cost if appearance_cost is not None else torch.zeros_like(cost),
        'appearance_valid_mask': appearance_valid_mask if appearance_cost is not None else torch.zeros_like(valid_mask),
        'appearance_modalities': appearance_debug.get('appearance_modalities', {}),
        'feature_assist_cost': feature_assist_cost if feature_assist_cost is not None else torch.zeros_like(cost),
        'feature_assist_valid_mask': feature_assist_valid_mask if feature_assist_cost is not None else torch.zeros_like(valid_mask),
        'temporal_cost': temporal_cost if temporal_cost is not None else torch.zeros_like(cost),
        'temporal_valid_mask': temporal_valid_mask if temporal_cost is not None else torch.zeros_like(valid_mask),
        'dynamic_weights': dynamic_weights,
        'association_mode': modality_debug['association_mode'],
        'low_confidence_motion_fallback': modality_debug['low_confidence_motion_fallback'],
        'scene_adapted': modality_debug['scene_adapted'],
        'reliability_payload': reliability_payload,
        'refinement_payload': refinement_payload,
        'feature_assist_payload': feature_assist_payload,
    }
    return cost, valid_mask, debug



def build_distance_limits(tracks, tracking_cfg):
    base_limit = float(tracking_cfg['max_center_distance'])
    association_cfg = tracking_cfg.get('association', {})
    memory_cfg = tracking_cfg.get('memory', {})
    use_temporal_memory = bool(association_cfg.get('use_temporal_memory', False) and memory_cfg.get('enabled', False))
    expansion = float(association_cfg.get('lost_track_expansion', memory_cfg.get('lost_track_expansion', 1.25)))

    limits = []
    for track in tracks:
        limit = base_limit
        if use_temporal_memory and track.state == track.LOST:
            limit = base_limit * expansion
        limits.append(limit)
    return torch.tensor(limits, dtype=torch.float32)



def build_appearance_cost(tracks, appearance_features, reliability_payload, num_dets, association_cfg, use_temporal_memory=False, use_modality_awareness=False):
    if not association_cfg.get('use_appearance', False):
        return None, None, {}
    if len(tracks) == 0 or num_dets == 0:
        return None, None, {}

    appearance_payload = normalize_appearance_payload(appearance_features, num_detections=num_dets)
    if appearance_payload is None:
        return None, None, {}

    modality_costs = {}
    modality_masks = {}
    for modality in ('fused', 'rgb', 'ir'):
        cost_matrix, valid_mask = build_single_modality_cost(
            tracks,
            appearance_payload,
            modality=modality,
            num_dets=num_dets,
            use_temporal_memory=use_temporal_memory,
        )
        if cost_matrix is not None:
            modality_costs[modality] = cost_matrix
            modality_masks[modality] = valid_mask

    if not modality_costs:
        return None, None, {}

    if not use_modality_awareness or reliability_payload is None or len(modality_costs) == 1:
        fallback_modality = 'fused' if 'fused' in modality_costs else next(iter(modality_costs.keys()))
        return modality_costs[fallback_modality], modality_masks[fallback_modality], {'appearance_modalities': modality_costs}

    device = next(iter(modality_costs.values())).device
    dtype = next(iter(modality_costs.values())).dtype
    combined_cost = torch.zeros((len(tracks), num_dets), dtype=dtype, device=device)
    combined_valid = torch.zeros((len(tracks), num_dets), dtype=torch.bool, device=device)

    for track_index, track in enumerate(tracks):
        track_reliability = track.get_reliability_summary()
        for det_index in range(num_dets):
            available_modalities = [
                modality
                for modality, valid_mask in modality_masks.items()
                if bool(valid_mask[track_index, det_index].item())
            ]
            if not available_modalities:
                continue

            detection_reliability = get_detection_reliability(reliability_payload, det_index)
            mix_weights = resolve_modality_mix_weights(
                track_reliability=track_reliability,
                detection_reliability=detection_reliability,
                available_modalities=available_modalities,
            )
            combined_cost[track_index, det_index] = sum(
                float(mix_weights.get(modality, 0.0)) * modality_costs[modality][track_index, det_index]
                for modality in available_modalities
            )
            combined_valid[track_index, det_index] = True

    return combined_cost, combined_valid, {'appearance_modalities': modality_costs}



def build_single_modality_cost(tracks, payload, modality, num_dets, use_temporal_memory=False, use_feature_assist=False):
    det_embeddings = payload.get(modality)
    det_mask = payload.get('masks', {}).get(modality)
    if det_embeddings is None or det_mask is None or not bool(det_mask.any().item()):
        return None, None

    track_embeddings = []
    track_mask = []
    for track in tracks:
        if use_feature_assist:
            embedding = track.get_aggregated_feature_assist(modality) if use_temporal_memory else track.get_feature_assist(modality)
        else:
            embedding = track.get_aggregated_embedding(modality) if use_temporal_memory else track.get_embedding(modality)
        if embedding is None:
            track_mask.append(False)
            track_embeddings.append(torch.zeros(det_embeddings.shape[1], dtype=det_embeddings.dtype, device=det_embeddings.device))
        else:
            track_mask.append(True)
            track_embeddings.append(embedding.to(device=det_embeddings.device, dtype=det_embeddings.dtype))

    track_embeddings = torch.stack(track_embeddings, dim=0)
    track_mask = torch.tensor(track_mask, dtype=torch.bool, device=det_embeddings.device)
    cosine_similarity = F.cosine_similarity(track_embeddings.unsqueeze(1), det_embeddings.unsqueeze(0), dim=-1)
    appearance_cost = 0.5 * (1.0 - cosine_similarity.clamp(-1.0, 1.0))
    appearance_valid_mask = track_mask.unsqueeze(1) & det_mask.unsqueeze(0)
    return appearance_cost, appearance_valid_mask



def build_feature_assist_cost(tracks, feature_assist_features, num_dets, use_temporal_memory=False):
    if len(tracks) == 0 or num_dets == 0:
        return None, None
    feature_assist_payload = normalize_feature_assist_payload(feature_assist_features, num_detections=num_dets)
    if feature_assist_payload is None:
        return None, None
    return build_single_modality_cost(
        tracks,
        feature_assist_payload,
        modality='fused',
        num_dets=num_dets,
        use_temporal_memory=use_temporal_memory,
        use_feature_assist=True,
    )



def build_temporal_cost(tracks, detections, tracking_cfg):
    association_cfg = tracking_cfg.get('association', {})
    memory_cfg = tracking_cfg.get('memory', {})
    if not association_cfg.get('use_temporal_memory', False) or not memory_cfg.get('enabled', False):
        return None, None
    if len(tracks) == 0 or detections.shape[0] == 0:
        return None, None

    predicted_boxes = []
    valid_rows = []
    for track in tracks:
        summary = track.get_memory_summary()
        predicted_obb = summary.get('predicted_obb')
        predicted_boxes.append(predicted_obb if predicted_obb is not None else track.bbox_obb)
        valid_rows.append(summary.get('size', 0) >= 2 and memory_cfg.get('use_temporal_consistency', True))

    predicted_boxes = torch.stack(predicted_boxes, dim=0).to(device=detections.device, dtype=torch.float32)
    valid_rows = torch.tensor(valid_rows, dtype=torch.bool, device=detections.device)

    center_cost = torch.cdist(predicted_boxes[:, :2], detections[:, :2], p=2) / max(float(tracking_cfg['max_center_distance']), 1e-6)
    size_cost = (predicted_boxes[:, None, 2:4] - detections[None, :, 2:4]).abs()
    size_cost = size_cost / detections[None, :, 2:4].clamp(min=1.0)
    size_cost = size_cost.mean(dim=-1)
    angle_delta = torch.atan2(torch.sin(predicted_boxes[:, None, 4] - detections[None, :, 4]), torch.cos(predicted_boxes[:, None, 4] - detections[None, :, 4])).abs()
    angle_cost = angle_delta / torch.pi

    temporal_cost = 0.60 * center_cost.clamp(max=2.0) + 0.25 * size_cost.clamp(max=2.0) + 0.15 * angle_cost.clamp(max=1.0)
    temporal_valid_mask = valid_rows.unsqueeze(1).expand(-1, detections.shape[0])
    return temporal_cost, temporal_valid_mask



def build_dynamic_weight_matrices(tracks, num_dets, tracking_cfg, reliability_payload=None, frame_meta=None, device='cpu', dtype=torch.float32):
    association_cfg = tracking_cfg.get('association', {}) if isinstance(tracking_cfg, dict) else {}
    modality_cfg = tracking_cfg.get('modality', {}) if isinstance(tracking_cfg, dict) else {}
    num_tracks = len(tracks)

    weights = {
        'w_motion': torch.full((num_tracks, num_dets), float(association_cfg.get('w_motion', 0.25)), dtype=dtype, device=device),
        'w_iou': torch.full((num_tracks, num_dets), float(association_cfg.get('w_iou', 0.60)), dtype=dtype, device=device),
        'w_app': torch.full((num_tracks, num_dets), float(association_cfg.get('w_app', 0.50)), dtype=dtype, device=device),
        'w_temporal': torch.full((num_tracks, num_dets), float(association_cfg.get('w_temporal', 0.30)), dtype=dtype, device=device),
        'w_score': torch.full((num_tracks, num_dets), float(association_cfg.get('w_score', 0.15)), dtype=dtype, device=device),
        'w_assist': torch.full((num_tracks, num_dets), float(association_cfg.get('w_assist', 0.35)), dtype=dtype, device=device),
    }
    mode_matrix = [['stage3_fallback' for _ in range(num_dets)] for _ in range(num_tracks)]
    low_conf_mask = torch.zeros((num_tracks, num_dets), dtype=torch.bool, device=device)
    scene_adapted_mask = torch.zeros((num_tracks, num_dets), dtype=torch.bool, device=device)

    if not association_cfg.get('use_modality_awareness', False):
        return weights, {'association_mode': mode_matrix, 'low_confidence_motion_fallback': low_conf_mask, 'scene_adapted': scene_adapted_mask}
    if not association_cfg.get('dynamic_weighting', False):
        return weights, {'association_mode': mode_matrix, 'low_confidence_motion_fallback': low_conf_mask, 'scene_adapted': scene_adapted_mask}
    if not modality_cfg.get('enabled', False):
        return weights, {'association_mode': mode_matrix, 'low_confidence_motion_fallback': low_conf_mask, 'scene_adapted': scene_adapted_mask}
    if reliability_payload is None:
        return weights, {'association_mode': mode_matrix, 'low_confidence_motion_fallback': low_conf_mask, 'scene_adapted': scene_adapted_mask}

    for track_index, track in enumerate(tracks):
        track_reliability = track.get_reliability_summary()
        for det_index in range(num_dets):
            detection_reliability = get_detection_reliability(reliability_payload, det_index)
            profile = compute_dynamic_weight_profile(
                association_cfg,
                modality_cfg,
                detection_reliability=detection_reliability,
                track_reliability=track_reliability,
                frame_meta=frame_meta,
            )
            weights['w_motion'][track_index, det_index] = float(profile['w_motion'])
            weights['w_iou'][track_index, det_index] = float(profile['w_iou'])
            weights['w_app'][track_index, det_index] = float(profile['w_app'])
            weights['w_temporal'][track_index, det_index] = float(profile['w_temporal'])
            weights['w_score'][track_index, det_index] = float(profile['w_score'])
            mode_matrix[track_index][det_index] = profile['association_mode']
            low_conf_mask[track_index, det_index] = bool(profile['low_confidence_motion_fallback'])
            scene_adapted_mask[track_index, det_index] = bool(profile['scene_adapted'])

    return weights, {'association_mode': mode_matrix, 'low_confidence_motion_fallback': low_conf_mask, 'scene_adapted': scene_adapted_mask}



def greedy_assignment(cost_matrix, valid_mask):
    num_tracks, num_dets = cost_matrix.shape
    if num_tracks == 0 or num_dets == 0:
        return [], list(range(num_tracks)), list(range(num_dets))

    candidates = []
    for track_idx in range(num_tracks):
        for det_idx in range(num_dets):
            if valid_mask[track_idx, det_idx]:
                candidates.append((float(cost_matrix[track_idx, det_idx].item()), track_idx, det_idx))

    candidates.sort(key=lambda item: item[0])
    matches = []
    used_tracks = set()
    used_dets = set()

    for _, track_idx, det_idx in candidates:
        if track_idx in used_tracks or det_idx in used_dets:
            continue
        used_tracks.add(track_idx)
        used_dets.add(det_idx)
        matches.append((track_idx, det_idx))

    unmatched_tracks = [idx for idx in range(num_tracks) if idx not in used_tracks]
    unmatched_dets = [idx for idx in range(num_dets) if idx not in used_dets]
    return matches, unmatched_tracks, unmatched_dets



def build_overlap_disambiguation_summary(matches, prob_iou, feature_assist_cost, valid_mask, tracking_cfg):
    overlap_cfg = tracking_cfg.get('overlap_disambiguation', {}) if isinstance(tracking_cfg, dict) else {}
    if not overlap_cfg.get('enabled', False) or not matches:
        return {'overlap_disambiguation_count': 0, 'overlap_disambiguation_helped_count': 0}, {}

    overlap_threshold = float(overlap_cfg.get('overlap_iou_threshold', 0.45))
    ambiguity_margin = float(overlap_cfg.get('ambiguity_margin', 0.10))
    assist_margin = float(overlap_cfg.get('assist_margin', 0.05))
    summary = {'overlap_disambiguation_count': 0, 'overlap_disambiguation_helped_count': 0}
    pair_flags = {}

    for track_idx, det_idx in matches:
        current_iou = float(prob_iou[track_idx, det_idx].item()) if prob_iou.numel() > 0 else 0.0
        alt_track_mask = valid_mask[:, det_idx].clone()
        alt_track_mask[track_idx] = False
        alt_det_mask = valid_mask[track_idx].clone()
        alt_det_mask[det_idx] = False
        ambiguous = False
        if alt_track_mask.any():
            alt_iou = float(prob_iou[:, det_idx][alt_track_mask].max().item()) if prob_iou.numel() > 0 else 0.0
            ambiguous = ambiguous or (current_iou >= overlap_threshold and alt_iou >= current_iou - ambiguity_margin)
        if alt_det_mask.any():
            alt_iou = float(prob_iou[track_idx][alt_det_mask].max().item()) if prob_iou.numel() > 0 else 0.0
            ambiguous = ambiguous or (current_iou >= overlap_threshold and alt_iou >= current_iou - ambiguity_margin)
        if not ambiguous:
            continue

        summary['overlap_disambiguation_count'] += 1
        helped = False
        if feature_assist_cost is not None and feature_assist_cost.numel() > 0:
            current_assist = float(feature_assist_cost[track_idx, det_idx].item())
            alternatives = []
            if alt_track_mask.any():
                alternatives.extend(feature_assist_cost[:, det_idx][alt_track_mask].tolist())
            if alt_det_mask.any():
                alternatives.extend(feature_assist_cost[track_idx][alt_det_mask].tolist())
            if alternatives and current_assist + assist_margin <= min(float(value) for value in alternatives):
                helped = True
                summary['overlap_disambiguation_helped_count'] += 1
        pair_flags[(track_idx, det_idx)] = {
            'overlap_disambiguated': True,
            'overlap_disambiguation_helped': helped,
        }

    return summary, pair_flags



def associate_tracks_to_detections(tracks, detections, tracking_cfg, appearance_features=None, reliability_features=None, feature_assist_features=None, refinement_payload=None, frame_meta=None):
    detections = ensure_detection_tensor(detections)
    cost_matrix, valid_mask, debug = build_association_cost(
        tracks,
        detections,
        tracking_cfg,
        appearance_features=appearance_features,
        reliability_features=reliability_features,
        feature_assist_features=feature_assist_features,
        refinement_payload=refinement_payload,
        frame_meta=frame_meta,
    )
    matches, unmatched_tracks, unmatched_dets = greedy_assignment(cost_matrix, valid_mask)
    overlap_summary, overlap_pair_flags = build_overlap_disambiguation_summary(
        matches,
        debug['prob_iou'],
        debug.get('feature_assist_cost'),
        valid_mask,
        tracking_cfg,
    )
    association_info = {
        'cost_matrix': cost_matrix,
        'valid_mask': valid_mask,
        'prob_iou': debug['prob_iou'],
        'center_distance': debug['center_distance'],
        'appearance_cost': debug['appearance_cost'],
        'appearance_valid_mask': debug['appearance_valid_mask'],
        'appearance_modalities': debug.get('appearance_modalities', {}),
        'feature_assist_cost': debug.get('feature_assist_cost'),
        'feature_assist_valid_mask': debug.get('feature_assist_valid_mask'),
        'temporal_cost': debug['temporal_cost'],
        'temporal_valid_mask': debug['temporal_valid_mask'],
        'dynamic_weights': debug['dynamic_weights'],
        'association_mode': debug['association_mode'],
        'low_confidence_motion_fallback': debug['low_confidence_motion_fallback'],
        'scene_adapted': debug['scene_adapted'],
        'reliability_payload': debug.get('reliability_payload'),
        'refinement_payload': debug.get('refinement_payload'),
        'feature_assist_payload': debug.get('feature_assist_payload'),
        'overlap_summary': overlap_summary,
        'overlap_pair_flags': overlap_pair_flags,
    }
    return matches, unmatched_tracks, unmatched_dets, association_info
