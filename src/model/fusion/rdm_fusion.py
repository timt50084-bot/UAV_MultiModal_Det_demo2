import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.fusion_registry import FUSIONS


class ReciprocalDifferenceBlock(nn.Module):
    """Shared-response + discrepancy-response fusion for RGB/IR features."""

    def __init__(self, channels, level_index):
        super().__init__()
        hidden = max(16, channels // 4)
        self.level_index = level_index

        self.diff_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=max(1, channels // 8), bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5 if level_index == 0 else 3, padding=2 if level_index == 0 else 1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.residual_alpha = nn.Parameter(torch.tensor(0.35 if level_index == 0 else 0.2, dtype=torch.float32))

    def forward(self, f_rgb, f_ir, return_attention_map=False):
        shared = 0.5 * (f_rgb + f_ir)
        diff = torch.abs(f_rgb - f_ir)
        diff_enhanced = self.diff_proj(diff)

        pooled = torch.cat([
            F.adaptive_avg_pool2d(f_rgb, 1),
            F.adaptive_avg_pool2d(f_ir, 1),
            F.adaptive_avg_pool2d(diff, 1),
        ], dim=1)
        weights = torch.softmax(self.gate_mlp(pooled), dim=1)
        w_rgb, w_ir, w_diff = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        spatial_seed = torch.cat([
            diff.mean(dim=1, keepdim=True),
            diff.amax(dim=1, keepdim=True),
        ], dim=1)
        discrepancy_mask = self.spatial_gate(spatial_seed)
        small_object_boost = 1.25 if self.level_index == 0 else 1.0

        fused = (
            w_rgb * f_rgb +
            w_ir * f_ir +
            w_diff * diff_enhanced * (0.5 + discrepancy_mask * small_object_boost)
        )
        out = self.refine(fused + torch.tanh(self.residual_alpha) * shared)

        if return_attention_map:
            return out, discrepancy_mask
        return out


@FUSIONS.register("RDMFusion")
class ReciprocalDifferenceFusion(nn.Module):
    """Reciprocal Difference Modulation Fusion."""

    def __init__(self, channel_list=None):
        super().__init__()
        channel_list = channel_list or [64, 128, 256, 512]
        self.level_names = ['P2', 'P3', 'P4', 'P5']
        self.blocks = nn.ModuleList([
            ReciprocalDifferenceBlock(channels, level_index=i)
            for i, channels in enumerate(channel_list)
        ])

    def forward(self, rgb_features, ir_features, return_attention_map=False, target_size=None):
        fused_features = []
        attention_maps = {}

        for level_name, block, f_rgb, f_ir in zip(self.level_names, self.blocks, rgb_features, ir_features):
            if return_attention_map:
                fused_out, attention_map = block(f_rgb, f_ir, return_attention_map=True)
                if target_size is not None:
                    attention_map = F.interpolate(attention_map, size=target_size, mode='bilinear', align_corners=False)
                attention_maps[f'{level_name}_Discrepancy_Map'] = attention_map
            else:
                fused_out = block(f_rgb, f_ir)
            fused_features.append(fused_out)

        if return_attention_map:
            return tuple(fused_features), attention_maps
        return tuple(fused_features)
