"""Legacy detector-side temporal-memory refinement kept for compatibility.

The maintained detection mainline uses two-frame temporal fusion. This module is
retained only so archived experiments and compatibility tests can still load.
It does not affect tracking-side memory used by tracking_final.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMemoryBlock(nn.Module):
    def __init__(self, channels, level_index, gate_hidden_ratio=0.25, aggregator='weighted_avg'):
        super().__init__()
        hidden = max(16, int(channels * gate_hidden_ratio))
        self.aggregator = aggregator

        self.memory_gate = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.score_proj = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        self.motion_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5 if level_index == 0 else 3, padding=2 if level_index == 0 else 1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))

    def _compute_memory_weights(self, scores):
        if self.aggregator == 'mean':
            return torch.ones_like(scores) / max(scores.shape[1], 1)
        return torch.softmax(scores, dim=1)

    def forward(self, current_feat, memory_level_feats):
        if not memory_level_feats:
            empty_map = current_feat.new_zeros((current_feat.shape[0], 1, current_feat.shape[2], current_feat.shape[3]))
            return current_feat, empty_map

        gated_memories = []
        diff_summaries = []
        score_list = []
        for memory_feat in memory_level_feats:
            temporal_diff = current_feat - memory_feat
            fusion_seed = torch.cat([current_feat, memory_feat, temporal_diff.abs()], dim=1)
            gate = self.memory_gate(fusion_seed)
            gated_memories.append(memory_feat * gate)
            diff_summaries.append(temporal_diff.abs())
            score_list.append(self.score_proj(fusion_seed).flatten(1))

        score_tensor = torch.stack(score_list, dim=1)
        memory_weights = self._compute_memory_weights(score_tensor)

        aggregated_memory = torch.zeros_like(current_feat)
        aggregated_diff = torch.zeros_like(current_feat)
        for idx, (gated_memory, diff_summary) in enumerate(zip(gated_memories, diff_summaries)):
            weight = memory_weights[:, idx].view(-1, 1, 1, 1)
            aggregated_memory = aggregated_memory + gated_memory * weight
            aggregated_diff = aggregated_diff + diff_summary * weight

        motion_seed = torch.cat(
            [aggregated_diff.mean(dim=1, keepdim=True), aggregated_diff.amax(dim=1, keepdim=True)],
            dim=1
        )
        motion_mask = self.motion_gate(motion_seed)
        fused = torch.cat(
            [current_feat, aggregated_memory * (0.5 + motion_mask), aggregated_diff * (0.5 + motion_mask)],
            dim=1
        )
        out = self.refine(fused) + torch.tanh(self.residual_scale) * current_feat
        return out, motion_mask


class TemporalMemoryFusion(nn.Module):
    """Legacy detector-side temporal-memory refinement over neck features."""

    def __init__(self, channels=None, memory_len=3, aggregator='weighted_avg', gate_hidden_ratio=0.25):
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        self.channels = channels
        self.memory_len = memory_len
        self.aggregator = aggregator
        self.blocks = nn.ModuleList([
            TemporalMemoryBlock(
                channel,
                level_index=i,
                gate_hidden_ratio=gate_hidden_ratio,
                aggregator=aggregator,
            )
            for i, channel in enumerate(channels)
        ])
        self.level_names = ['P2', 'P3', 'P4', 'P5']

    def _normalize_memory_feats(self, memory_feats, num_levels):
        if memory_feats is None:
            return []
        if isinstance(memory_feats, Sequence):
            memory_feats = list(memory_feats)
        else:
            return []
        if len(memory_feats) == 0:
            return []
        if all(torch.is_tensor(feat) for feat in memory_feats):
            if len(memory_feats) != num_levels:
                raise ValueError("Single-step memory_feats must match the number of feature levels.")
            return [tuple(memory_feats)]

        normalized_steps = []
        for step_feats in memory_feats:
            if not isinstance(step_feats, Sequence):
                raise TypeError("Each memory step must be a sequence of per-level tensors.")
            step_feats = tuple(step_feats)
            if len(step_feats) != num_levels:
                raise ValueError("Each memory step must match the number of feature levels.")
            normalized_steps.append(step_feats)
        return normalized_steps

    def forward(self, current_feats, memory_feats=None, return_attention_map=False, target_size=None):
        outputs = []
        temporal_maps = {}
        memory_steps = self._normalize_memory_feats(memory_feats, len(current_feats))

        for level_idx, (level_name, block, current_feat) in enumerate(zip(self.level_names, self.blocks, current_feats)):
            level_memory = [step[level_idx] for step in memory_steps]
            refined_feat, motion_mask = block(current_feat, level_memory)
            outputs.append(refined_feat)

            if return_attention_map:
                if target_size is not None:
                    motion_mask = F.interpolate(motion_mask, size=target_size, mode='bilinear', align_corners=False)
                temporal_maps[f'{level_name}_Temporal_Map'] = motion_mask

        if return_attention_map:
            return tuple(outputs), temporal_maps
        return tuple(outputs)
