from copy import deepcopy

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.metrics.task_specific_metrics import is_small_bbox
from src.model.bbox_utils import batch_prob_iou

from .appearance import (
    get_detection_appearance,
    get_detection_feature_assist,
    normalize_appearance_payload,
    normalize_feature_assist_cfg,
    normalize_feature_assist_payload,
)
from .modality import empty_reliability_payload, get_detection_reliability, normalize_reliability_payload


REFINEMENT_DEFAULTS = {
    'enabled': False,
    'rescue_low_score': True,
    'rescue_score_threshold': 0.15,
    'rescue_match_iou': 0.2,
    'rescue_motion_gate': 0.5,
    'enable_track_guided_prediction': True,
    'max_prediction_only_steps': 2,
    'predicted_track_score': 0.1,
    'keep_small_tracked_candidates': True,
    'keep_tracked_overlap_candidates': True,
}


# Stage-6/7 note:
# The refiner stays postprocess-only. Stage 7 only extends it with optional feature-assist
# payload propagation so detector-side local descriptors can travel together with rescued or
# predicted candidates into the tracker.


def normalize_refinement_cfg(refinement_cfg=None):
    cfg = deepcopy(REFINEMENT_DEFAULTS)
    if refinement_cfg is None:
        return cfg

    if OmegaConf.is_config(refinement_cfg):
        refinement_cfg = OmegaConf.to_container(refinement_cfg, resolve=True)

    if isinstance(refinement_cfg, dict):
        for key, value in refinement_cfg.items():
            cfg[key] = value
    return cfg



def empty_refinement_payload(num_detections=0):
    return {
        'refinement_source': ['raw'] * int(num_detections),
        'rescued_mask': torch.zeros(int(num_detections), dtype=torch.bool),
        'predicted_mask': torch.zeros(int(num_detections), dtype=torch.bool),
        'rescued_small_mask': torch.zeros(int(num_detections), dtype=torch.bool),
        'support_track_ids': [None] * int(num_detections),
    }



def normalize_refinement_payload(refinement_payload, num_detections=None):
    if refinement_payload is None:
        return None

    if OmegaConf.is_config(refinement_payload):
        refinement_payload = OmegaConf.to_container(refinement_payload, resolve=True)

    if not isinstance(refinement_payload, dict):
        return None

    if num_detections is None:
        if 'refinement_source' in refinement_payload:
            num_detections = len(refinement_payload['refinement_source'])
        elif 'rescued_mask' in refinement_payload:
            num_detections = int(torch.as_tensor(refinement_payload['rescued_mask']).numel())
        else:
            return None

    payload = empty_refinement_payload(num_detections)
    if 'refinement_source' in refinement_payload and refinement_payload['refinement_source'] is not None:
        sources = list(refinement_payload['refinement_source'])
        if len(sources) != num_detections:
            raise ValueError('refinement_source must have the same length as detections.')
        payload['refinement_source'] = sources
    if 'support_track_ids' in refinement_payload and refinement_payload['support_track_ids'] is not None:
        support_ids = list(refinement_payload['support_track_ids'])
        if len(support_ids) != num_detections:
            raise ValueError('support_track_ids must have the same length as detections.')
        payload['support_track_ids'] = support_ids

    for key in ('rescued_mask', 'predicted_mask', 'rescued_small_mask'):
        if key in refinement_payload and refinement_payload[key] is not None:
            tensor = torch.as_tensor(refinement_payload[key], dtype=torch.bool).reshape(-1)
            if tensor.numel() != num_detections:
                raise ValueError(f'{key} must have shape [N] matching detections.')
            payload[key] = tensor
    return payload



def get_detection_refinement_context(refinement_payload, index):
    payload = normalize_refinement_payload(refinement_payload)
    if payload is None or index >= len(payload['refinement_source']):
        return None
    return {
        'refinement_source': payload['refinement_source'][index],
        'rescued_detection': bool(payload['rescued_mask'][index].item()),
        'predicted_candidate': bool(payload['predicted_mask'][index].item()),
        'rescued_small_object': bool(payload['rescued_small_mask'][index].item()),
        'support_track_id': payload['support_track_ids'][index],
    }


class TrackAwareRefiner:
    def __init__(self, tracking_cfg=None):
        tracking_cfg = dict(tracking_cfg or {})
        self.tracking_cfg = tracking_cfg
        self.refinement_cfg = normalize_refinement_cfg(tracking_cfg.get('refinement', {}))
        self.appearance_cfg = dict(tracking_cfg.get('appearance', {}))
        self.modality_cfg = dict(tracking_cfg.get('modality', {}))
        self.feature_assist_cfg = normalize_feature_assist_cfg(tracking_cfg.get('feature_assist', {}))
        self.last_feature_assist_payload = None

    def refine(self, detections, tracks, base_score_threshold, appearance_features=None, reliability_features=None, feature_assist_features=None, frame_meta=None):
        detections = _ensure_detection_tensor(detections)
        appearance_payload = normalize_appearance_payload(
            appearance_features,
            num_detections=detections.shape[0],
            device=detections.device,
            appearance_cfg=self.appearance_cfg,
        )
        reliability_payload = normalize_reliability_payload(
            reliability_features,
            num_detections=detections.shape[0],
            device=detections.device,
        )
        feature_assist_payload = normalize_feature_assist_payload(
            feature_assist_features,
            num_detections=detections.shape[0],
            device=detections.device,
            feature_assist_cfg=self.feature_assist_cfg,
        )
        self.last_feature_assist_payload = feature_assist_payload

        if not self.refinement_cfg.get('enabled', False):
            return detections, appearance_payload, reliability_payload, empty_refinement_payload(detections.shape[0]), self.empty_summary()

        base_score_threshold = float(base_score_threshold)
        tracks = [track for track in list(tracks or []) if getattr(track, 'state', None) != getattr(track, 'REMOVED', 'removed')]
        num_dets = int(detections.shape[0])
        payload = empty_refinement_payload(num_dets)
        summary = self.empty_summary()

        if num_dets == 0 and not tracks:
            self.last_feature_assist_payload = feature_assist_payload
            return detections, appearance_payload, reliability_payload, payload, summary

        keep_mask = detections[:, 5] >= base_score_threshold if num_dets > 0 else torch.zeros(0, dtype=torch.bool)
        support_map = {}
        matched_track_ids = set()

        for det_index in range(num_dets):
            support = self._best_support_track(
                detections[det_index],
                tracks,
                appearance_payload=appearance_payload,
                reliability_payload=reliability_payload,
                feature_assist_payload=feature_assist_payload,
                det_index=det_index,
            )
            if support is not None:
                support_map[det_index] = support
                if bool(keep_mask[det_index].item()):
                    matched_track_ids.add(int(support['track'].track_id))
                    payload['support_track_ids'][det_index] = int(support['track'].track_id)

        if self.refinement_cfg.get('rescue_low_score', True) and num_dets > 0:
            low_score_threshold = float(self.refinement_cfg.get('rescue_score_threshold', 0.15))
            for det_index in range(num_dets):
                if bool(keep_mask[det_index].item()):
                    continue
                score = float(detections[det_index, 5].item())
                if score < low_score_threshold:
                    continue
                support = support_map.get(det_index)
                if support is None:
                    continue
                if not self._should_rescue_detection(detections[det_index], support):
                    continue
                keep_mask[det_index] = True
                matched_track_ids.add(int(support['track'].track_id))
                payload['refinement_source'][det_index] = 'rescued'
                payload['rescued_mask'][det_index] = True
                payload['support_track_ids'][det_index] = int(support['track'].track_id)
                summary['rescued_detection_count'] += 1
                if self._is_small_detection(detections[det_index]):
                    payload['rescued_small_mask'][det_index] = True
                    summary['rescued_small_object_count'] += 1

        kept_indices = keep_mask.nonzero(as_tuple=False).view(-1)
        kept_detections = detections[kept_indices] if kept_indices.numel() > 0 else detections.new_zeros((0, 7))
        kept_appearance = _slice_appearance_payload(appearance_payload, kept_indices)
        kept_reliability = _slice_reliability_payload(reliability_payload, kept_indices)
        kept_feature_assist = _slice_appearance_payload(feature_assist_payload, kept_indices)
        kept_refinement = _slice_refinement_payload(payload, kept_indices)

        predicted_rows = []
        predicted_appearance = []
        predicted_reliability = []
        predicted_feature_assist = []
        predicted_support_ids = []
        predicted_small_flags = []

        if self.refinement_cfg.get('enable_track_guided_prediction', True):
            max_steps = int(self.refinement_cfg.get('max_prediction_only_steps', 2))
            predicted_score = float(self.refinement_cfg.get('predicted_track_score', 0.1))
            min_hits = int(self.tracking_cfg.get('min_hits', 1))
            for track in tracks:
                if int(getattr(track, 'track_id', -1)) in matched_track_ids:
                    continue
                if int(getattr(track, 'hits', 0)) < min_hits:
                    continue
                if int(getattr(track, 'prediction_only_streak', 0)) >= max_steps:
                    continue
                predicted_box = track.estimate_temporal_obb()
                if predicted_box is None:
                    continue
                predicted_rows.append(
                    torch.tensor(
                        [
                            float(predicted_box[0].item()),
                            float(predicted_box[1].item()),
                            float(predicted_box[2].item()),
                            float(predicted_box[3].item()),
                            float(predicted_box[4].item()),
                            predicted_score,
                            float(track.class_id),
                        ],
                        dtype=torch.float32,
                        device=predicted_box.device,
                    )
                )
                predicted_appearance.append(
                    {
                        'fused': track.get_aggregated_embedding('fused'),
                        'rgb': track.get_aggregated_embedding('rgb'),
                        'ir': track.get_aggregated_embedding('ir'),
                    }
                )
                predicted_reliability.append(track.get_reliability_summary())
                predicted_feature_assist.append(
                    {
                        'fused': track.get_aggregated_feature_assist('fused'),
                        'rgb': track.get_aggregated_feature_assist('rgb'),
                        'ir': track.get_aggregated_feature_assist('ir'),
                    }
                )
                predicted_support_ids.append(int(track.track_id))
                predicted_small_flags.append(self._is_small_detection(predicted_box))
                summary['track_guided_prediction_count'] += 1
                summary['predicted_only_track_count'] += 1

        if predicted_rows:
            predicted_tensor = torch.stack(predicted_rows, dim=0)
            kept_detections = torch.cat([kept_detections, predicted_tensor.to(device=predicted_tensor.device)], dim=0) if kept_detections.numel() > 0 else predicted_tensor
            kept_appearance = _append_appearance_entries(kept_appearance, predicted_appearance, device=predicted_tensor.device, appearance_cfg=self.appearance_cfg)
            kept_reliability = _append_reliability_entries(kept_reliability, predicted_reliability, device=predicted_tensor.device)
            kept_feature_assist = _append_appearance_entries(kept_feature_assist, predicted_feature_assist, device=predicted_tensor.device, appearance_cfg={'embedding_dim': self.feature_assist_cfg.get('embedding_dim', self.appearance_cfg.get('embedding_dim', 128)), 'use_rgb_ir_branches': True})
            predicted_payload = {
                'refinement_source': ['predicted'] * len(predicted_rows),
                'rescued_mask': torch.zeros(len(predicted_rows), dtype=torch.bool),
                'predicted_mask': torch.ones(len(predicted_rows), dtype=torch.bool),
                'rescued_small_mask': torch.as_tensor(predicted_small_flags, dtype=torch.bool),
                'support_track_ids': predicted_support_ids,
            }
            kept_refinement = _append_refinement_payload(kept_refinement, predicted_payload)

        summary['refinement_suppressed_false_drop_count'] = int(summary['rescued_detection_count'] + summary['track_guided_prediction_count'])
        self.last_feature_assist_payload = kept_feature_assist
        return kept_detections, kept_appearance, kept_reliability, kept_refinement, summary

    def empty_summary(self):
        return {
            'rescued_detection_count': 0,
            'rescued_small_object_count': 0,
            'track_guided_prediction_count': 0,
            'predicted_only_track_count': 0,
            'refinement_helped_reactivation_count': 0,
            'refinement_suppressed_false_drop_count': 0,
        }

    def _should_rescue_detection(self, detection, support):
        rescue_iou = float(self.refinement_cfg.get('rescue_match_iou', 0.2))
        motion_gate = float(self.refinement_cfg.get('rescue_motion_gate', 0.5))
        if self._is_small_detection(detection) and self.refinement_cfg.get('keep_small_tracked_candidates', True):
            rescue_iou *= 0.8
            motion_gate *= 1.1
        if self.refinement_cfg.get('keep_tracked_overlap_candidates', True):
            rescue_iou *= 0.9
        return support['iou'] >= rescue_iou or support['normalized_motion'] <= motion_gate

    def _best_support_track(self, detection, tracks, appearance_payload, reliability_payload, feature_assist_payload, det_index):
        if not tracks:
            return None

        det_box = detection[:5].unsqueeze(0)
        track_boxes = []
        candidate_tracks = []
        for track in tracks:
            if self.tracking_cfg.get('use_class_constraint', True) and int(track.class_id) != int(detection[6].item()):
                continue
            candidate_tracks.append(track)
            predicted = track.estimate_temporal_obb()
            track_boxes.append(predicted if predicted is not None else track.bbox_obb)

        if not candidate_tracks:
            return None

        track_boxes = torch.stack(track_boxes, dim=0).to(device=detection.device, dtype=torch.float32)
        ious = batch_prob_iou(torch.cat([track_boxes, det_box.to(track_boxes.device)], dim=0))[:-1, -1]
        centers = torch.cdist(track_boxes[:, :2], det_box[:, :2]).squeeze(1)
        max_center_distance = max(float(self.tracking_cfg.get('max_center_distance', 50.0)), 1e-6)
        normalized_motion = centers / max_center_distance
        appearance = get_detection_appearance(appearance_payload, det_index)
        feature_assist = get_detection_feature_assist(feature_assist_payload, det_index)

        best_support = None
        best_score = None
        for index, track in enumerate(candidate_tracks):
            score = 0.50 * float(ious[index].item()) + 0.20 * max(0.0, 1.0 - float(normalized_motion[index].item()))
            if appearance is not None:
                track_embedding = track.get_aggregated_embedding('fused')
                det_embedding = appearance.get('fused') if isinstance(appearance, dict) else None
                if track_embedding is not None and det_embedding is not None:
                    similarity = float(F.cosine_similarity(track_embedding.view(1, -1), det_embedding.view(1, -1)).item())
                    score += 0.20 * max(similarity, 0.0)
            if feature_assist is not None:
                track_assist = track.get_aggregated_feature_assist('fused')
                det_assist = feature_assist.get('fused') if isinstance(feature_assist, dict) else None
                if track_assist is not None and det_assist is not None:
                    similarity = float(F.cosine_similarity(track_assist.view(1, -1), det_assist.view(1, -1)).item())
                    score += 0.10 * max(similarity, 0.0)
            reliability = get_detection_reliability(reliability_payload, det_index)
            if reliability is not None and reliability.get('fused_reliability') is not None:
                score += 0.05 * float(reliability.get('fused_reliability'))
            candidate = {
                'track': track,
                'iou': float(ious[index].item()),
                'normalized_motion': float(normalized_motion[index].item()),
                'score': score,
            }
            if best_score is None or score > best_score:
                best_score = score
                best_support = candidate
        return best_support

    def _is_small_detection(self, box):
        values = box.tolist() if torch.is_tensor(box) else list(box)
        return is_small_bbox([float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4])], 32)



def _ensure_detection_tensor(detections, device='cpu'):
    if detections is None:
        return torch.zeros((0, 7), dtype=torch.float32, device=device)
    tensor = detections if torch.is_tensor(detections) else torch.as_tensor(detections, dtype=torch.float32, device=device)
    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.numel() == 0:
        return torch.zeros((0, 7), dtype=torch.float32, device=device)
    return tensor[:, :7].contiguous()



def _slice_appearance_payload(payload, indices):
    if payload is None:
        return None
    indices = indices.to(dtype=torch.long, device=indices.device)
    sliced = {'masks': {}, 'source': payload.get('source') if isinstance(payload, dict) else None}
    for key in ('fused', 'rgb', 'ir'):
        tensor = payload.get(key)
        mask = payload.get('masks', {}).get(key)
        sliced[key] = tensor.index_select(0, indices) if tensor is not None and indices.numel() > 0 else (tensor.new_zeros((0, tensor.shape[1])) if tensor is not None else None)
        sliced['masks'][key] = mask.index_select(0, indices) if mask is not None and indices.numel() > 0 else (mask.new_zeros((0,), dtype=torch.bool) if mask is not None else torch.zeros((0,), dtype=torch.bool))
    return sliced



def _slice_reliability_payload(reliability_payload, indices):
    if reliability_payload is None:
        return None
    indices = indices.to(dtype=torch.long, device=indices.device)
    sliced = {'masks': {}}
    for key in ('rgb_reliability', 'ir_reliability', 'fused_reliability'):
        tensor = reliability_payload.get(key)
        mask = reliability_payload.get('masks', {}).get(key)
        sliced[key] = tensor.index_select(0, indices) if tensor is not None and indices.numel() > 0 else (tensor.new_zeros((0,)) if tensor is not None else None)
        sliced['masks'][key] = mask.index_select(0, indices) if mask is not None and indices.numel() > 0 else (mask.new_zeros((0,), dtype=torch.bool) if mask is not None else torch.zeros((0,), dtype=torch.bool))
    return sliced



def _slice_refinement_payload(refinement_payload, indices):
    payload = normalize_refinement_payload(refinement_payload)
    if payload is None:
        return None
    if indices.numel() == 0:
        return empty_refinement_payload(0)
    selected = [int(index.item()) for index in indices]
    return {
        'refinement_source': [payload['refinement_source'][index] for index in selected],
        'rescued_mask': payload['rescued_mask'].index_select(0, indices.to(dtype=torch.long)),
        'predicted_mask': payload['predicted_mask'].index_select(0, indices.to(dtype=torch.long)),
        'rescued_small_mask': payload['rescued_small_mask'].index_select(0, indices.to(dtype=torch.long)),
        'support_track_ids': [payload['support_track_ids'][index] for index in selected],
    }



def _append_appearance_entries(appearance_payload, entries, device='cpu', appearance_cfg=None):
    payload = normalize_appearance_payload(appearance_payload, appearance_cfg=appearance_cfg)
    if payload is None:
        cfg = dict(appearance_cfg or {})
        payload = normalize_appearance_payload({'fused': torch.zeros((0, int(cfg.get('embedding_dim', 128))), dtype=torch.float32)}, num_detections=0, device=device, appearance_cfg=cfg)
    payload.setdefault('source', appearance_payload.get('source') if isinstance(appearance_payload, dict) else None)
    num_existing = 0 if payload is None else payload['masks']['fused'].shape[0]
    for entry in entries:
        for key in ('fused', 'rgb', 'ir'):
            vector = entry.get(key) if isinstance(entry, dict) else None
            if vector is not None:
                vector = vector.detach().clone().to(device=device, dtype=torch.float32).view(1, -1)
                if payload.get(key) is None:
                    payload[key] = vector
                else:
                    payload[key] = torch.cat([payload[key].to(device=device), vector], dim=0)
                payload['masks'][key] = torch.cat([payload['masks'].get(key, torch.zeros(num_existing, dtype=torch.bool, device=device)), torch.ones(1, dtype=torch.bool, device=device)], dim=0)
            else:
                if payload.get(key) is not None:
                    zeros = torch.zeros((1, payload[key].shape[1]), dtype=payload[key].dtype, device=device)
                    payload[key] = torch.cat([payload[key].to(device=device), zeros], dim=0)
                    payload['masks'][key] = torch.cat([payload['masks'].get(key, torch.zeros(num_existing, dtype=torch.bool, device=device)), torch.zeros(1, dtype=torch.bool, device=device)], dim=0)
        num_existing += 1
    return payload



def _append_reliability_entries(reliability_payload, entries, device='cpu'):
    payload = normalize_reliability_payload(reliability_payload, device=device)
    if payload is None:
        payload = empty_reliability_payload(0, device=device)
    for entry in entries:
        for key in ('rgb_reliability', 'ir_reliability', 'fused_reliability'):
            value = None if entry is None else entry.get(key)
            if payload.get(key) is None:
                payload[key] = torch.zeros((0,), dtype=torch.float32, device=device)
            if payload.get('masks', {}).get(key) is None:
                payload.setdefault('masks', {})[key] = torch.zeros((0,), dtype=torch.bool, device=device)
            if value is None:
                payload[key] = torch.cat([payload[key].to(device=device), torch.zeros(1, dtype=torch.float32, device=device)], dim=0)
                payload['masks'][key] = torch.cat([payload['masks'][key].to(device=device), torch.zeros(1, dtype=torch.bool, device=device)], dim=0)
            else:
                payload[key] = torch.cat([payload[key].to(device=device), torch.tensor([float(value)], dtype=torch.float32, device=device)], dim=0)
                payload['masks'][key] = torch.cat([payload['masks'][key].to(device=device), torch.ones(1, dtype=torch.bool, device=device)], dim=0)
    return payload



def _append_refinement_payload(refinement_payload, appended_payload):
    payload = normalize_refinement_payload(refinement_payload)
    appended = normalize_refinement_payload(appended_payload)
    if payload is None:
        return appended
    if appended is None:
        return payload
    return {
        'refinement_source': list(payload['refinement_source']) + list(appended['refinement_source']),
        'rescued_mask': torch.cat([payload['rescued_mask'], appended['rescued_mask']], dim=0),
        'predicted_mask': torch.cat([payload['predicted_mask'], appended['predicted_mask']], dim=0),
        'rescued_small_mask': torch.cat([payload['rescued_small_mask'], appended['rescued_small_mask']], dim=0),
        'support_track_ids': list(payload['support_track_ids']) + list(appended['support_track_ids']),
    }
