import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalRefineBlock(nn.Module):
    def __init__(self, channels, level_index):
        super().__init__()
        hidden = max(16, channels // 4)
        self.level_index = level_index

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
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

    def forward(self, current_feat, prev_feat):
        temporal_diff = current_feat - prev_feat
        motion_seed = torch.cat(
            [temporal_diff.abs().mean(dim=1, keepdim=True), temporal_diff.abs().amax(dim=1, keepdim=True)],
            dim=1
        )
        motion_mask = self.motion_gate(motion_seed)
        gated_prev = prev_feat * self.gate(torch.cat([current_feat, prev_feat, temporal_diff.abs()], dim=1))
        fused = torch.cat([current_feat, gated_prev, temporal_diff * (0.5 + motion_mask)], dim=1)
        out = self.refine(fused) + torch.tanh(self.residual_scale) * current_feat
        return out, motion_mask


class TemporalFeaturePyramid(nn.Module):
    """Two-frame temporal refinement over neck features."""

    def __init__(self, channels=None):
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        self.blocks = nn.ModuleList([
            TemporalRefineBlock(channel, level_index=i) for i, channel in enumerate(channels)
        ])
        self.level_names = ['P2', 'P3', 'P4', 'P5']

    def forward(self, current_feats, prev_feats, return_attention_map=False, target_size=None):
        outputs = []
        temporal_maps = {}

        for level_name, block, current_feat, prev_feat in zip(self.level_names, self.blocks, current_feats, prev_feats):
            refined_feat, motion_mask = block(current_feat, prev_feat)
            outputs.append(refined_feat)

            if return_attention_map:
                if target_size is not None:
                    motion_mask = F.interpolate(motion_mask, size=target_size, mode='bilinear', align_corners=False)
                temporal_maps[f'{level_name}_Temporal_Map'] = motion_mask

        if return_attention_map:
            return tuple(outputs), temporal_maps
        return tuple(outputs)
