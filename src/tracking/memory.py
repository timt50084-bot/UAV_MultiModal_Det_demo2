from collections import deque
from copy import deepcopy

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .modality import normalize_reliability_dict


MEMORY_DEFAULTS = {
    'enabled': False,
    'size': 5,
    'fusion': 'weighted_mean',
    'decay': 0.8,
    'use_temporal_consistency': True,
    'lost_track_expansion': 1.25,
}


# Stage-3/5 note:
# The memory bank stays intentionally lightweight. It now keeps short-term appearance,
# motion, and optional modality-reliability summaries without introducing heavier
# detector-tracker joint training or external memory systems.


def normalize_memory_cfg(memory_cfg=None):
    cfg = deepcopy(MEMORY_DEFAULTS)
    if memory_cfg is None:
        return cfg

    if OmegaConf.is_config(memory_cfg):
        memory_cfg = OmegaConf.to_container(memory_cfg, resolve=True)

    if isinstance(memory_cfg, dict):
        for key, value in memory_cfg.items():
            cfg[key] = value
    return cfg


class TrackMemoryBank:
    def __init__(self, memory_cfg=None):
        self.cfg = normalize_memory_cfg(memory_cfg)
        self.entries = deque(maxlen=int(self.cfg['size']))

    def __len__(self):
        return len(self.entries)

    def append(self, obb, score, frame_index=None, embedding=None, motion_state=None, reliability=None):
        entry = {
            'obb': obb.detach().clone().to(dtype=torch.float32),
            'score': float(score),
            'frame_index': int(frame_index) if frame_index is not None else None,
            'embedding': self._normalize_embedding_dict(embedding),
            'motion_state': motion_state.detach().clone().to(dtype=torch.float32) if torch.is_tensor(motion_state) else motion_state,
            'reliability': normalize_reliability_dict(reliability),
        }
        self.entries.append(entry)

    def get_entries(self):
        return list(self.entries)

    def get_aggregated_embedding(self, modality='fused'):
        embeddings = [
            entry['embedding'].get(modality)
            for entry in self.entries
            if entry.get('embedding') is not None and entry['embedding'].get(modality) is not None
        ]
        if not embeddings:
            return None
        if len(embeddings) == 1:
            return F.normalize(embeddings[0], dim=0)

        if self.cfg.get('fusion', 'weighted_mean') == 'ema':
            aggregated = embeddings[0]
            momentum = float(self.cfg.get('decay', 0.8))
            for embedding in embeddings[1:]:
                aggregated = F.normalize(momentum * aggregated + (1.0 - momentum) * embedding, dim=0)
            return aggregated

        weights = self._temporal_weights(len(embeddings), embeddings[0].device, embeddings[0].dtype)
        stacked = torch.stack(embeddings, dim=0)
        aggregated = (stacked * weights.view(-1, 1)).sum(dim=0)
        return F.normalize(aggregated, dim=0)

    def get_aggregated_reliability(self):
        keys = ('rgb_reliability', 'ir_reliability', 'fused_reliability')
        aggregated = {}
        has_value = False
        for key in keys:
            values = []
            for entry in self.entries:
                reliability = entry.get('reliability') or {}
                value = reliability.get(key)
                if value is not None:
                    values.append(float(value))
            if not values:
                aggregated[key] = None
                continue
            if len(values) == 1:
                aggregated[key] = float(values[0])
                has_value = True
                continue
            weights = self._temporal_weights(len(values), torch.device('cpu'), torch.float32).cpu()
            aggregated[key] = float((torch.tensor(values, dtype=torch.float32) * weights).sum().item())
            has_value = True
        return aggregated if has_value else None

    def get_aggregated_obb(self):
        if len(self.entries) == 0:
            return None
        if len(self.entries) == 1:
            return self.entries[-1]['obb'].detach().clone()

        device = self.entries[-1]['obb'].device
        dtype = self.entries[-1]['obb'].dtype
        weights = self._temporal_weights(len(self.entries), device, dtype)
        boxes = torch.stack([entry['obb'] for entry in self.entries], dim=0)
        aggregated = torch.zeros(5, dtype=dtype, device=device)
        aggregated[:4] = (boxes[:, :4] * weights.view(-1, 1)).sum(dim=0)
        aggregated[4] = _weighted_angle_mean(boxes[:, 4], weights)
        return aggregated

    def estimate_velocity(self):
        if len(self.entries) < 2:
            return None

        deltas = []
        for previous, current in zip(list(self.entries)[:-1], list(self.entries)[1:]):
            delta = current['obb'][:4] - previous['obb'][:4]
            prev_frame = previous.get('frame_index')
            curr_frame = current.get('frame_index')
            if prev_frame is not None and curr_frame is not None and curr_frame > prev_frame:
                delta = delta / float(curr_frame - prev_frame)
            deltas.append(delta)

        stacked = torch.stack(deltas, dim=0)
        weights = self._temporal_weights(len(deltas), stacked.device, stacked.dtype)
        return (stacked * weights.view(-1, 1)).sum(dim=0)

    def predict_next_obb(self, fallback_obb=None):
        aggregated = self.get_aggregated_obb()
        if aggregated is None:
            return fallback_obb.detach().clone() if torch.is_tensor(fallback_obb) else None

        velocity = self.estimate_velocity()
        predicted = aggregated.detach().clone()
        if velocity is not None:
            predicted[:4] = predicted[:4] + velocity
        return predicted

    def get_memory_summary(self):
        latest_obb = self.entries[-1]['obb'].detach().clone() if self.entries else None
        aggregated_obb = self.get_aggregated_obb()
        predicted_obb = self.predict_next_obb(fallback_obb=aggregated_obb)
        velocity = self.estimate_velocity()
        return {
            'size': len(self.entries),
            'latest_obb': latest_obb,
            'aggregated_obb': aggregated_obb,
            'predicted_obb': predicted_obb,
            'velocity': velocity,
            'aggregated_embedding': self.get_aggregated_embedding('fused'),
            'aggregated_reliability': self.get_aggregated_reliability(),
        }

    def _temporal_weights(self, length, device, dtype):
        decay = float(self.cfg.get('decay', 0.8))
        exponents = torch.arange(length - 1, -1, -1, device=device, dtype=dtype)
        weights = torch.pow(torch.full((length,), decay, device=device, dtype=dtype), exponents)
        return weights / weights.sum().clamp(min=1e-6)

    def _normalize_embedding_dict(self, embedding):
        if embedding is None:
            return None
        normalized = {}
        for key, value in embedding.items():
            if value is None:
                normalized[key] = None
            else:
                normalized[key] = F.normalize(value.detach().clone().to(dtype=torch.float32), dim=0)
        return normalized



def _weighted_angle_mean(angles, weights):
    sin_sum = (torch.sin(angles) * weights).sum()
    cos_sum = (torch.cos(angles) * weights).sum()
    return torch.atan2(sin_sum, cos_sum)
