import torch
import torch.nn.functional as F

from .appearance import (
    get_detection_appearance,
    get_detection_feature_assist,
    normalize_appearance_payload,
    normalize_feature_assist_payload,
)
from .association import associate_tracks_to_detections
from .kalman_filter import LightweightKalmanFilter
from .modality import get_detection_reliability, normalize_reliability_payload
from .refinement import get_detection_refinement_context, normalize_refinement_payload
from .track import Track
from .utils import ensure_detection_tensor, normalize_tracking_cfg, tracks_to_results


class MultiObjectTracker:
    def __init__(self, tracking_cfg=None, class_names=None):
        self.cfg = normalize_tracking_cfg(tracking_cfg)
        self.class_names = class_names or []
        self.kalman_filter = LightweightKalmanFilter() if self.cfg.get('use_kalman', True) else None
        self.appearance_cfg = self.cfg.get('appearance', {})
        self.memory_cfg = self.cfg.get('memory', {})
        self.modality_cfg = self.cfg.get('modality', {})
        self.feature_assist_cfg = self.cfg.get('feature_assist', {})
        self.reactivation_cfg = self.cfg.get('reactivation', {})
        self.overlap_cfg = self.cfg.get('overlap_disambiguation', {})
        self.smoothing_cfg = self.cfg.get('smoothing', {})
        self.refinement_cfg = self.cfg.get('refinement', {})
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_track_id = 1
        self.frame_index = 0
        self.sequence_id = None
        self.last_frame_summary = {}

    def update(self, detections, frame_meta=None, appearance_features=None, reliability_features=None, feature_assist_features=None, refinement_payload=None):
        detections = ensure_detection_tensor(detections)
        appearance_features = normalize_appearance_payload(
            appearance_features,
            num_detections=detections.shape[0],
            device=detections.device,
            appearance_cfg=self.appearance_cfg,
        )
        reliability_features = normalize_reliability_payload(
            reliability_features,
            num_detections=detections.shape[0],
            device=detections.device,
        )
        feature_assist_features = normalize_feature_assist_payload(
            feature_assist_features,
            num_detections=detections.shape[0],
            device=detections.device,
            feature_assist_cfg=self.feature_assist_cfg,
        )
        refinement_payload = normalize_refinement_payload(refinement_payload, num_detections=detections.shape[0])
        self._maybe_reset_on_sequence_change(frame_meta)
        current_frame_index = self._resolve_frame_index(frame_meta)
        self.frame_index = current_frame_index

        for track in self.tracks:
            if track.state != Track.REMOVED:
                track.predict()

        matches, unmatched_tracks, unmatched_dets, association_info = associate_tracks_to_detections(
            self.tracks,
            detections,
            self.cfg,
            appearance_features=appearance_features,
            reliability_features=reliability_features,
            feature_assist_features=feature_assist_features,
            refinement_payload=refinement_payload,
            frame_meta=frame_meta,
        )

        reactivation_summary = self._empty_advanced_summary()
        if self.reactivation_cfg.get('enabled', False):
            extra_matches, unmatched_tracks, unmatched_dets, reactivation_summary = self._reactivate_lost_tracks(
                detections,
                unmatched_tracks,
                unmatched_dets,
                appearance_features=appearance_features,
                reliability_features=reliability_features,
                feature_assist_features=feature_assist_features,
                refinement_payload=refinement_payload,
            )
            matches.extend(extra_matches)

        transition_counts = {}
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            previous_state = track.state
            was_lost = track.state == Track.LOST
            refinement_context = get_detection_refinement_context(refinement_payload, det_idx)
            track.update(
                detections[det_idx],
                appearance=get_detection_appearance(appearance_features, det_idx),
                reliability=get_detection_reliability(reliability_features, det_idx),
                feature_assist=get_detection_feature_assist(feature_assist_features, det_idx),
                frame_index=current_frame_index,
                association_summary=self._build_match_summary(
                    track_idx,
                    det_idx,
                    association_info,
                    reliability_features,
                    refinement_context=refinement_context,
                    was_lost=was_lost,
                ),
            )
            if track.last_transition:
                transition_counts[track.last_transition] = transition_counts.get(track.last_transition, 0) + 1

        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            if track.state != Track.REMOVED:
                track.mark_lost()
            if track.time_since_update > int(self.cfg['max_age']):
                track.mark_removed()

        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            refinement_context = get_detection_refinement_context(refinement_payload, det_idx)
            if refinement_context is not None and refinement_context.get('predicted_candidate', False):
                continue
            if float(detection[5].item()) < float(self.cfg['init_score_threshold']):
                continue
            self.tracks.append(
                Track(
                    track_id=self.next_track_id,
                    bbox_obb=detection[:5],
                    score=float(detection[5].item()),
                    class_id=int(detection[6].item()),
                    kalman_filter=self.kalman_filter,
                    keep_history=int(self.cfg.get('keep_history', 20)),
                    angle_momentum=float(self.cfg.get('angle_smoothing', 0.7)),
                    appearance=get_detection_appearance(appearance_features, det_idx),
                    reliability=get_detection_reliability(reliability_features, det_idx),
                    feature_assist=get_detection_feature_assist(feature_assist_features, det_idx),
                    appearance_cfg=self.appearance_cfg,
                    memory_cfg=self.memory_cfg,
                    modality_cfg=self.modality_cfg,
                    feature_assist_cfg=self.feature_assist_cfg,
                    smoothing_cfg=self.smoothing_cfg,
                    frame_index=current_frame_index,
                    association_summary={
                        'association_mode': 'new_track',
                        'low_confidence_motion_fallback': False,
                        'modality_helped_reactivation': False,
                        'scene_adapted': False,
                        'dynamic_weights': None,
                        'refinement_source': None if refinement_context is None else refinement_context.get('refinement_source'),
                        'rescued_detection': False if refinement_context is None else refinement_context.get('rescued_detection', False),
                        'rescued_small_object': False if refinement_context is None else refinement_context.get('rescued_small_object', False),
                        'predicted_candidate': False,
                        'refinement_helped_reactivation': False,
                        'support_track_id': None if refinement_context is None else refinement_context.get('support_track_id'),
                        'reactivation_source': None,
                        'memory_reactivation': False,
                        'feature_assist_reactivation': False,
                        'predicted_candidate_reactivation': False,
                        'overlap_disambiguated': False,
                        'overlap_disambiguation_helped': False,
                        'predicted_only_to_tracked': False,
                    },
                    min_hits=int(self.cfg.get('min_hits', 1)),
                )
            )
            self.next_track_id += 1

        self.tracks = [track for track in self.tracks if track.state != Track.REMOVED]
        frame_results = self.get_active_tracks(as_results=True, only_recent=True)
        self.last_frame_summary = self._build_frame_summary(frame_results, association_info, reactivation_summary, transition_counts)
        return frame_results

    def get_active_tracks(self, as_results=False, confirmed_only=False, only_recent=False):
        visible_states = {Track.TRACKED, Track.REACTIVATING, Track.PREDICTED_ONLY}
        if not confirmed_only:
            visible_states.add(Track.TENTATIVE)
        tracks = [track for track in self.tracks if track.state in visible_states]
        if confirmed_only:
            tracks = [track for track in tracks if track.hits >= int(self.cfg['min_hits'])]
        if only_recent:
            tracks = [track for track in tracks if track.time_since_update == 0]
        if as_results:
            return tracks_to_results(tracks)
        return tracks

    def _reactivate_lost_tracks(self, detections, unmatched_track_indices, unmatched_det_indices, appearance_features=None, reliability_features=None, feature_assist_features=None, refinement_payload=None):
        if not unmatched_track_indices or not unmatched_det_indices:
            return [], unmatched_track_indices, unmatched_det_indices, self._empty_advanced_summary()

        max_age = int(self.reactivation_cfg.get('max_reactivate_age', 8))
        gate = float(self.reactivation_cfg.get('reactivation_gate', 0.4))
        use_memory = bool(self.reactivation_cfg.get('use_memory_reactivation', True))
        use_feature_assist = bool(self.reactivation_cfg.get('use_feature_assist_reactivation', True) and self.feature_assist_cfg.get('enabled', False))
        candidates = []

        for track_idx in unmatched_track_indices:
            track = self.tracks[track_idx]
            if track.state != Track.LOST:
                continue
            if track.time_since_update > max_age:
                continue
            if track.hits < int(self.cfg.get('min_hits', 1)):
                continue
            for det_idx in unmatched_det_indices:
                detection = detections[det_idx]
                if self.cfg.get('use_class_constraint', True) and int(track.class_id) != int(detection[6].item()):
                    continue
                refinement_context = get_detection_refinement_context(refinement_payload, det_idx)
                support_track_id = None if refinement_context is None else refinement_context.get('support_track_id')
                if support_track_id is not None and int(support_track_id) != int(track.track_id):
                    continue
                score, source, detail = self._score_reactivation_candidate(
                    track,
                    detection,
                    det_idx,
                    appearance_features=appearance_features,
                    feature_assist_features=feature_assist_features,
                    refinement_context=refinement_context,
                    use_memory=use_memory,
                    use_feature_assist=use_feature_assist,
                )
                if score >= gate:
                    candidates.append((score, track_idx, det_idx, source, detail))

        candidates.sort(key=lambda item: item[0], reverse=True)
        used_tracks = set()
        used_dets = set()
        matches = []
        summary = self._empty_advanced_summary()

        for score, track_idx, det_idx, source, detail in candidates:
            if track_idx in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_idx)
            used_dets.add(det_idx)
            matches.append((track_idx, det_idx))
            summary[source + '_count'] = int(summary.get(source + '_count', 0)) + 1 if source in {'memory_reactivation', 'feature_assist_reactivation'} else summary.get(source + '_count', 0)
            if source == 'memory_reactivation':
                summary['memory_reactivation_count'] += 1
            elif source == 'feature_assist_reactivation':
                summary['feature_assist_reactivation_count'] += 1
            elif source == 'predicted_candidate_reactivation':
                summary['predicted_candidate_reactivation_count'] += 1
            self._register_reactivation_detail(track_idx, det_idx, source, detail)

        remaining_tracks = [index for index in unmatched_track_indices if index not in used_tracks]
        remaining_dets = [index for index in unmatched_det_indices if index not in used_dets]
        return matches, remaining_tracks, remaining_dets, summary

    def _register_reactivation_detail(self, track_idx, det_idx, source, detail):
        self.cfg.setdefault('_reactivation_details', {})[(track_idx, det_idx)] = {
            'reactivation_source': source,
            **detail,
        }

    def _score_reactivation_candidate(self, track, detection, det_idx, appearance_features=None, feature_assist_features=None, refinement_context=None, use_memory=True, use_feature_assist=True):
        predicted = track.estimate_temporal_obb()
        predicted = predicted if predicted is not None else track.bbox_obb
        det_box = detection[:5].view(1, 5)
        pred_box = predicted.view(1, 5)
        iou = float(torch.clamp(torch.exp(-((pred_box[:, :2] - det_box[:, :2]) ** 2).sum(dim=1) / 400.0), min=0.0, max=1.0).item())
        motion_distance = float(torch.norm(pred_box[0, :2] - det_box[0, :2], p=2).item())
        normalized_motion = max(0.0, 1.0 - motion_distance / max(float(self.cfg.get('max_center_distance', 50.0)), 1e-6))

        memory_similarity = None
        if use_memory:
            det_embedding = get_detection_appearance(appearance_features, det_idx)
            track_embedding = track.get_aggregated_embedding('fused')
            det_fused = det_embedding.get('fused') if det_embedding is not None else None
            if track_embedding is not None and det_fused is not None:
                memory_similarity = float(F.cosine_similarity(track_embedding.view(1, -1), det_fused.view(1, -1)).item())

        feature_assist_similarity = None
        if use_feature_assist:
            det_assist = get_detection_feature_assist(feature_assist_features, det_idx)
            track_assist = track.get_aggregated_feature_assist('fused')
            det_fused = det_assist.get('fused') if det_assist is not None else None
            if track_assist is not None and det_fused is not None:
                feature_assist_similarity = float(F.cosine_similarity(track_assist.view(1, -1), det_fused.view(1, -1)).item())

        score = 0.35 * iou + 0.25 * normalized_motion
        if memory_similarity is not None:
            score += 0.20 * max(memory_similarity, 0.0)
        if feature_assist_similarity is not None:
            score += 0.20 * max(feature_assist_similarity, 0.0)

        source = 'memory_reactivation'
        if refinement_context is not None and refinement_context.get('predicted_candidate', False):
            source = 'predicted_candidate_reactivation'
        elif feature_assist_similarity is not None and (memory_similarity is None or feature_assist_similarity >= memory_similarity + 0.03):
            source = 'feature_assist_reactivation'
        elif memory_similarity is not None:
            source = 'memory_reactivation'
        else:
            source = 'normal_match'

        detail = {
            'memory_similarity': memory_similarity,
            'feature_assist_similarity': feature_assist_similarity,
            'predicted_candidate': False if refinement_context is None else bool(refinement_context.get('predicted_candidate', False)),
        }
        return score, source, detail

    def _build_match_summary(self, track_idx, det_idx, association_info, reliability_features, refinement_context=None, was_lost=False):
        dynamic_weights = association_info.get('dynamic_weights', {})
        association_mode = None
        if association_info.get('association_mode'):
            association_mode = association_info['association_mode'][track_idx][det_idx]
        low_confidence_motion_fallback = bool(association_info.get('low_confidence_motion_fallback', [])[track_idx, det_idx].item()) if 'low_confidence_motion_fallback' in association_info else False
        scene_adapted = bool(association_info.get('scene_adapted', [])[track_idx, det_idx].item()) if 'scene_adapted' in association_info else False
        detection_reliability = get_detection_reliability(reliability_features, det_idx)
        refinement_context = dict(refinement_context or {})
        rescued_detection = bool(refinement_context.get('rescued_detection', False))
        predicted_candidate = bool(refinement_context.get('predicted_candidate', False))
        reactivation_detail = self.cfg.get('_reactivation_details', {}).pop((track_idx, det_idx), None)
        reactivation_source = None
        if was_lost:
            reactivation_source = 'normal_match'
        if reactivation_detail is not None:
            reactivation_source = reactivation_detail.get('reactivation_source', reactivation_source)

        overlap_pair_flags = association_info.get('overlap_pair_flags', {}) or {}
        overlap_flags = overlap_pair_flags.get((track_idx, det_idx), {})
        track = self.tracks[track_idx]
        predicted_only_to_tracked = bool(track.state == Track.PREDICTED_ONLY and not predicted_candidate)
        return {
            'association_mode': association_mode,
            'low_confidence_motion_fallback': low_confidence_motion_fallback,
            'scene_adapted': scene_adapted,
            'modality_helped_reactivation': bool(was_lost and association_mode in {'rgb_dominant', 'ir_dominant', 'balanced'} and not low_confidence_motion_fallback),
            'dynamic_weights': {
                key: float(value[track_idx, det_idx].item())
                for key, value in dynamic_weights.items()
            } if dynamic_weights else None,
            'detection_reliability': detection_reliability,
            'refinement_source': refinement_context.get('refinement_source'),
            'rescued_detection': rescued_detection,
            'rescued_small_object': bool(refinement_context.get('rescued_small_object', False)),
            'predicted_candidate': predicted_candidate,
            'support_track_id': refinement_context.get('support_track_id'),
            'refinement_helped_reactivation': bool(was_lost and (rescued_detection or predicted_candidate)),
            'reactivation_source': reactivation_source,
            'memory_reactivation': bool(reactivation_source == 'memory_reactivation'),
            'feature_assist_reactivation': bool(reactivation_source == 'feature_assist_reactivation'),
            'predicted_candidate_reactivation': bool(reactivation_source == 'predicted_candidate_reactivation'),
            'feature_assist_similarity': None if reactivation_detail is None else reactivation_detail.get('feature_assist_similarity'),
            'memory_similarity': None if reactivation_detail is None else reactivation_detail.get('memory_similarity'),
            'overlap_disambiguated': bool(overlap_flags.get('overlap_disambiguated', False)),
            'overlap_disambiguation_helped': bool(overlap_flags.get('overlap_disambiguation_helped', False)),
            'predicted_only_to_tracked': predicted_only_to_tracked,
        }

    def _build_frame_summary(self, frame_results, association_info, reactivation_summary, transition_counts):
        overlap_summary = association_info.get('overlap_summary', {}) or {}
        summary = {
            'feature_assist_reactivation_count': int(reactivation_summary.get('feature_assist_reactivation_count', 0)),
            'memory_reactivation_count': int(reactivation_summary.get('memory_reactivation_count', 0)),
            'predicted_candidate_reactivation_count': int(reactivation_summary.get('predicted_candidate_reactivation_count', 0)),
            'overlap_disambiguation_count': int(overlap_summary.get('overlap_disambiguation_count', 0)),
            'overlap_disambiguation_helped_count': int(overlap_summary.get('overlap_disambiguation_helped_count', 0)),
            'reactivating_state_count': sum(1 for item in frame_results if item.get('state') == 'reactivating'),
            'predicted_only_to_tracked_count': sum(1 for item in frame_results if bool(item.get('predicted_only_to_tracked', False))),
            'state_transitions': dict(transition_counts),
        }
        return summary

    def _empty_advanced_summary(self):
        return {
            'feature_assist_reactivation_count': 0,
            'memory_reactivation_count': 0,
            'predicted_candidate_reactivation_count': 0,
            'overlap_disambiguation_count': 0,
            'overlap_disambiguation_helped_count': 0,
            'reactivating_state_count': 0,
            'predicted_only_to_tracked_count': 0,
        }

    def _maybe_reset_on_sequence_change(self, frame_meta=None):
        if not frame_meta:
            return
        sequence_id = frame_meta.get('sequence_id') or frame_meta.get('video_id')
        if sequence_id is None:
            return
        if self.sequence_id is not None and sequence_id != self.sequence_id:
            self.reset()
        self.sequence_id = sequence_id

    def _resolve_frame_index(self, frame_meta=None):
        if frame_meta and frame_meta.get('frame_index') is not None:
            return int(frame_meta['frame_index'])
        return int(self.frame_index) + 1
