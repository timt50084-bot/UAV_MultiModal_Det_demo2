from collections import defaultdict, deque

import torch
import torch.nn.functional as F

from .appearance import normalize_feature_assist_cfg
from .memory import TrackMemoryBank
from .modality import normalize_reliability_dict
from .utils import smooth_angle


class Track:
    TENTATIVE = 'tentative'
    TRACKED = 'tracked'
    LOST = 'lost'
    REACTIVATING = 'reactivating'
    PREDICTED_ONLY = 'predicted_only'
    REMOVED = 'removed'

    def __init__(
        self,
        track_id,
        bbox_obb,
        score,
        class_id,
        kalman_filter=None,
        keep_history=20,
        angle_momentum=0.7,
        appearance=None,
        reliability=None,
        feature_assist=None,
        appearance_cfg=None,
        memory_cfg=None,
        modality_cfg=None,
        feature_assist_cfg=None,
        smoothing_cfg=None,
        frame_index=None,
        association_summary=None,
        min_hits=1,
    ):
        bbox_obb = bbox_obb.detach().clone().to(dtype=torch.float32)
        self.track_id = int(track_id)
        self.score = float(score)
        self.class_id = int(class_id)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.min_hits = max(int(min_hits), 1)
        self.state = self.TENTATIVE if self.min_hits > 1 else self.TRACKED
        self.kalman_filter = kalman_filter
        self.appearance_cfg = dict(appearance_cfg or {})
        self.memory_cfg = dict(memory_cfg or {})
        self.modality_cfg = dict(modality_cfg or {})
        self.feature_assist_cfg = normalize_feature_assist_cfg(feature_assist_cfg)
        self.smoothing_cfg = dict(smoothing_cfg or {})
        self.angle_momentum = float(self.smoothing_cfg.get('angle_ema', angle_momentum))
        self.angle = float(bbox_obb[4].item())
        self.history = deque(maxlen=int(keep_history))

        self.embedding = None
        self.embedding_history = deque(maxlen=int(self.appearance_cfg.get('history_size', 5)))
        self.modality_embeddings = {}

        self.feature_assist = None
        self.feature_assist_history = deque(maxlen=max(int(self.memory_cfg.get('size', 5)), int(self.appearance_cfg.get('history_size', 5))))
        self.feature_assist_modalities = {}

        self.reliability = None
        self.reliability_history = deque(maxlen=max(int(self.memory_cfg.get('size', 5)), int(self.appearance_cfg.get('history_size', 5))))
        self.memory_bank = TrackMemoryBank(self.memory_cfg)
        self.aggregated_embedding = None
        self.aggregated_motion_state = None
        self.aggregated_reliability = None
        self.aggregated_feature_assist = None
        self.association_summary = {}
        self.prediction_only_streak = 0
        self.refinement_event_counts = {'rescued': 0, 'predicted': 0}
        self.reactivation_counts = defaultdict(int)
        self.state_transition_counts = defaultdict(int)
        self.predicted_only_to_tracked_count = 0
        self.last_transition = None

        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.initiate(bbox_obb[:4])
        else:
            self.mean = bbox_obb[:4].clone()
            self.covariance = None

        self.output_xywh = bbox_obb[:4].clone()
        self._push_history()
        if appearance is not None:
            self.update_appearance(appearance)
        if feature_assist is not None:
            self.update_feature_assist(feature_assist)
        if reliability is not None:
            self.update_reliability(reliability)
        self.append_memory(obb=bbox_obb, score=score, appearance=appearance, reliability=reliability, frame_index=frame_index)
        self.set_association_summary(association_summary, previous_state=None)

    @property
    def bbox_obb(self):
        return torch.tensor(
            [self.output_xywh[0], self.output_xywh[1], self.output_xywh[2].clamp(min=1e-3), self.output_xywh[3].clamp(min=1e-3), self.angle],
            dtype=torch.float32,
        )

    @property
    def box(self):
        return self.bbox_obb

    @property
    def cls_id(self):
        return self.class_id

    def get_embedding(self, modality='fused'):
        if modality == 'fused':
            return self.embedding
        return self.modality_embeddings.get(modality)

    def get_aggregated_embedding(self, modality='fused'):
        if self.memory_cfg.get('enabled', False):
            aggregated = self.memory_bank.get_aggregated_embedding(modality=modality)
            if aggregated is not None:
                return aggregated
        return self.get_embedding(modality=modality)

    def get_feature_assist(self, modality='fused'):
        if modality == 'fused':
            return self.feature_assist
        return self.feature_assist_modalities.get(modality)

    def get_aggregated_feature_assist(self, modality='fused'):
        values = []
        if modality == 'fused':
            values = list(self.feature_assist_history)
        else:
            vector = self.feature_assist_modalities.get(modality)
            values = [vector] if vector is not None else []

        values = [value for value in values if value is not None]
        if not values:
            return self.get_feature_assist(modality=modality)
        if len(values) == 1:
            return F.normalize(values[0], dim=0)
        momentum = float(self.feature_assist_cfg.get('ema_momentum', self.appearance_cfg.get('ema_momentum', 0.8)))
        aggregated = values[0]
        for value in values[1:]:
            aggregated = F.normalize(momentum * aggregated + (1.0 - momentum) * value, dim=0)
        return aggregated

    def get_reliability_summary(self):
        if self.memory_cfg.get('enabled', False):
            aggregated = self.memory_bank.get_aggregated_reliability()
            if aggregated is not None:
                return aggregated
        return self.reliability

    def get_memory_summary(self):
        summary = self.memory_bank.get_memory_summary()
        summary['aggregated_embedding'] = self.get_aggregated_embedding('fused')
        summary['aggregated_reliability'] = self.get_reliability_summary()
        summary['aggregated_feature_assist'] = self.get_aggregated_feature_assist('fused')
        summary['current_obb'] = self.bbox_obb.detach().clone()
        return summary

    def estimate_temporal_obb(self):
        return self.memory_bank.predict_next_obb(fallback_obb=self.bbox_obb)

    def append_memory(self, obb, score, appearance=None, reliability=None, frame_index=None):
        self.memory_bank.append(
            obb=obb,
            score=score,
            frame_index=frame_index,
            embedding=appearance,
            motion_state=self.mean[:4].detach().clone() if torch.is_tensor(self.mean) else None,
            reliability=reliability,
        )
        self._refresh_temporal_views()

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.output_xywh = self.mean[:4].detach().clone()
        self._push_history()
        return self.bbox_obb

    def update(self, detection, appearance=None, reliability=None, feature_assist=None, frame_index=None, association_summary=None):
        detection = detection.detach().clone().to(dtype=torch.float32)
        previous_xywh = self.output_xywh.detach().clone()
        previous_state = self.state
        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, detection[:4])
            target_xywh = self.mean[:4].detach().clone()
        else:
            self.mean = detection[:4].clone()
            target_xywh = detection[:4].clone()

        if self.smoothing_cfg.get('enabled', False):
            bbox_ema = float(self.smoothing_cfg.get('bbox_ema', 0.7))
            self.output_xywh = bbox_ema * previous_xywh + (1.0 - bbox_ema) * target_xywh
        else:
            self.output_xywh = target_xywh

        self.angle = smooth_angle(self.angle, detection[4].item(), momentum=self.angle_momentum)
        self.score = float(detection[5].item())
        self.class_id = int(detection[6].item())
        self.hits += 1
        self.time_since_update = 0
        self._push_history()
        if appearance is not None:
            self.update_appearance(appearance)
        if feature_assist is not None:
            self.update_feature_assist(feature_assist)
        if reliability is not None:
            self.update_reliability(reliability)
        self.append_memory(obb=self.bbox_obb, score=self.score, appearance=appearance, reliability=reliability, frame_index=frame_index)
        self._apply_state_from_summary(previous_state, association_summary)
        self.set_association_summary(association_summary, previous_state=previous_state)

    def update_appearance(self, appearance):
        if appearance is None:
            return

        fused = appearance.get('fused') if isinstance(appearance, dict) else None
        if fused is not None:
            fused = F.normalize(fused.detach().clone().to(dtype=torch.float32), dim=0)
            self.embedding_history.append(fused)
            if self.embedding is None:
                self.embedding = fused
            elif self.appearance_cfg.get('update_mode', 'ema') == 'queue_mean':
                self.embedding = F.normalize(torch.stack(list(self.embedding_history), dim=0).mean(dim=0), dim=0)
            else:
                momentum = float(self.appearance_cfg.get('ema_momentum', 0.8))
                self.embedding = F.normalize(momentum * self.embedding + (1.0 - momentum) * fused, dim=0)

        for key in ('rgb', 'ir'):
            value = appearance.get(key) if isinstance(appearance, dict) else None
            if value is None:
                continue
            self.modality_embeddings[key] = F.normalize(value.detach().clone().to(dtype=torch.float32), dim=0)

        self._refresh_temporal_views()

    def update_feature_assist(self, feature_assist):
        if feature_assist is None:
            return
        fused = feature_assist.get('fused') if isinstance(feature_assist, dict) else None
        if fused is not None:
            fused = F.normalize(fused.detach().clone().to(dtype=torch.float32), dim=0)
            self.feature_assist_history.append(fused)
            if self.feature_assist is None:
                self.feature_assist = fused
            else:
                momentum = float(self.feature_assist_cfg.get('ema_momentum', self.appearance_cfg.get('ema_momentum', 0.8)))
                self.feature_assist = F.normalize(momentum * self.feature_assist + (1.0 - momentum) * fused, dim=0)
        for key in ('rgb', 'ir'):
            value = feature_assist.get(key) if isinstance(feature_assist, dict) else None
            if value is None:
                continue
            self.feature_assist_modalities[key] = F.normalize(value.detach().clone().to(dtype=torch.float32), dim=0)
        self._refresh_temporal_views()

    def update_reliability(self, reliability):
        reliability = normalize_reliability_dict(reliability)
        if reliability is None:
            return

        self.reliability_history.append(dict(reliability))
        if self.reliability is None:
            self.reliability = dict(reliability)
        else:
            momentum = float(self.modality_cfg.get('reliability_ema', 0.8))
            updated = {}
            for key in ('rgb_reliability', 'ir_reliability', 'fused_reliability'):
                previous = self.reliability.get(key)
                current = reliability.get(key)
                if current is None:
                    updated[key] = previous
                elif previous is None:
                    updated[key] = current
                else:
                    updated[key] = float(momentum * previous + (1.0 - momentum) * current)
            self.reliability = normalize_reliability_dict(updated)
        self._refresh_temporal_views()

    def _apply_state_from_summary(self, previous_state, summary=None):
        summary = dict(summary or {})
        reactivation_source = summary.get('reactivation_source')
        predicted_candidate = bool(summary.get('predicted_candidate', False))
        if predicted_candidate:
            self.state = self.PREDICTED_ONLY
        elif previous_state == self.LOST and reactivation_source and reactivation_source != 'normal_match':
            self.state = self.REACTIVATING
        elif self.hits < self.min_hits:
            self.state = self.TENTATIVE
        else:
            self.state = self.TRACKED

    def set_association_summary(self, summary=None, previous_state=None):
        self.association_summary = dict(summary or {})
        if self.association_summary.get('predicted_candidate', False):
            self.prediction_only_streak += 1
            self.refinement_event_counts['predicted'] += 1
        else:
            self.prediction_only_streak = 0
        if self.association_summary.get('rescued_detection', False):
            self.refinement_event_counts['rescued'] += 1

        reactivation_source = self.association_summary.get('reactivation_source')
        if reactivation_source:
            self.reactivation_counts[reactivation_source] += 1

        if previous_state is not None and previous_state != self.state:
            transition = f'{previous_state}->{self.state}'
            self.last_transition = transition
            self.state_transition_counts[transition] += 1
            if previous_state == self.PREDICTED_ONLY and self.state == self.TRACKED:
                self.predicted_only_to_tracked_count += 1
        else:
            self.last_transition = None

    def mark_lost(self):
        self.state = self.LOST

    def mark_removed(self):
        self.state = self.REMOVED

    def to_result(self):
        current_reliability = self.reliability or {}
        aggregated_reliability = self.get_reliability_summary() or {}
        refinement_source = self.association_summary.get('refinement_source')
        reactivation_source = self.association_summary.get('reactivation_source')
        output_state = self.state
        if self.state == self.PREDICTED_ONLY:
            output_state = 'predicted'
        elif refinement_source == 'rescued' and self.state in {self.TRACKED, self.REACTIVATING, self.TENTATIVE}:
            output_state = 'rescued'

        return {
            'track_id': self.track_id,
            'class_id': self.class_id,
            'score': self.score,
            'obb': self.bbox_obb.tolist(),
            'state': output_state,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'has_embedding': self.embedding is not None,
            'has_feature_assist': self.feature_assist is not None,
            'memory_size': len(self.memory_bank),
            'has_aggregated_embedding': self.aggregated_embedding is not None,
            'has_aggregated_feature_assist': self.aggregated_feature_assist is not None,
            'rgb_reliability': current_reliability.get('rgb_reliability'),
            'ir_reliability': current_reliability.get('ir_reliability'),
            'fused_reliability': current_reliability.get('fused_reliability'),
            'aggregated_rgb_reliability': aggregated_reliability.get('rgb_reliability'),
            'aggregated_ir_reliability': aggregated_reliability.get('ir_reliability'),
            'aggregated_fused_reliability': aggregated_reliability.get('fused_reliability'),
            'association_mode': self.association_summary.get('association_mode'),
            'low_confidence_motion_fallback': bool(self.association_summary.get('low_confidence_motion_fallback', False)),
            'modality_helped_reactivation': bool(self.association_summary.get('modality_helped_reactivation', False)),
            'scene_adapted': bool(self.association_summary.get('scene_adapted', False)),
            'dynamic_weights': self.association_summary.get('dynamic_weights'),
            'refinement_source': refinement_source,
            'rescued_detection': bool(self.association_summary.get('rescued_detection', False)),
            'rescued_small_object': bool(self.association_summary.get('rescued_small_object', False)),
            'predicted_candidate': bool(self.association_summary.get('predicted_candidate', False)),
            'prediction_only_streak': int(self.prediction_only_streak),
            'refinement_helped_reactivation': bool(self.association_summary.get('refinement_helped_reactivation', False)),
            'support_track_id': self.association_summary.get('support_track_id'),
            'reactivation_source': reactivation_source,
            'memory_reactivation': bool(reactivation_source == 'memory_reactivation'),
            'feature_assist_reactivation': bool(reactivation_source == 'feature_assist_reactivation'),
            'predicted_candidate_reactivation': bool(reactivation_source == 'predicted_candidate_reactivation'),
            'feature_assist_similarity': self.association_summary.get('feature_assist_similarity'),
            'memory_similarity': self.association_summary.get('memory_similarity'),
            'overlap_disambiguated': bool(self.association_summary.get('overlap_disambiguated', False)),
            'overlap_disambiguation_helped': bool(self.association_summary.get('overlap_disambiguation_helped', False)),
            'predicted_only_to_tracked': bool(self.association_summary.get('predicted_only_to_tracked', False)),
            'state_transition': self.last_transition,
        }

    def _refresh_temporal_views(self):
        self.aggregated_embedding = self.memory_bank.get_aggregated_embedding('fused') if self.memory_cfg.get('enabled', False) else self.embedding
        self.aggregated_motion_state = self.memory_bank.estimate_velocity() if self.memory_cfg.get('enabled', False) else None
        self.aggregated_reliability = self.memory_bank.get_aggregated_reliability() if self.memory_cfg.get('enabled', False) else self.reliability
        self.aggregated_feature_assist = self.get_aggregated_feature_assist('fused')

    def _push_history(self):
        self.history.append(self.bbox_obb.tolist())
