import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.fusion_registry import FUSIONS

from .base_fusion import BaseFusion


class ContextProjector(nn.Module):
    """Global context projection for reliability prediction."""

    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.project(self.pool(x))


class ReliabilityFusionBlock(nn.Module):
    """Reliability-aware RGB/IR fusion on one feature level."""

    def __init__(self, channels, hidden_ratio=0.25, level_index=0, p2_boost_init=0.1):
        super().__init__()
        hidden_channels = max(16, int(channels * hidden_ratio))
        self.level_index = level_index

        self.rgb_context = ContextProjector(channels, hidden_channels)
        self.ir_context = ContextProjector(channels, hidden_channels)
        self.diff_context = ContextProjector(channels, hidden_channels)

        self.diff_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

        self.reliability_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, bias=True),
        )

        self.p2_boost = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5 if level_index == 0 else 3, padding=2 if level_index == 0 else 1, bias=False),
            nn.Sigmoid(),
        )
        self.p2_boost_alpha = nn.Parameter(torch.tensor(float(p2_boost_init), dtype=torch.float32))

        self.fuse_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))

    def forward(self, f_rgb, f_ir, return_attention_map=False):
        diff_feature = self.diff_branch(torch.abs(f_rgb - f_ir))

        boost_map = None
        if self.level_index == 0:
            boost_seed = torch.cat(
                [diff_feature.mean(dim=1, keepdim=True), diff_feature.amax(dim=1, keepdim=True)],
                dim=1
            )
            boost_map = self.p2_boost(boost_seed)
            diff_feature = diff_feature * (1.0 + torch.sigmoid(self.p2_boost_alpha) * boost_map)

        rgb_ctx = self.rgb_context(f_rgb)
        ir_ctx = self.ir_context(f_ir)
        diff_ctx = self.diff_context(diff_feature)
        rel_logits = self.reliability_head(torch.cat([rgb_ctx, ir_ctx, diff_ctx], dim=1))

        rel = torch.sigmoid(rel_logits)
        rel_sum = rel.sum(dim=1, keepdim=True).clamp(min=1e-6)
        r_rgb = rel[:, 0:1] / rel_sum
        r_ir = rel[:, 1:2] / rel_sum
        r_diff = rel[:, 2:3] / rel_sum

        fused = r_rgb * f_rgb + r_ir * f_ir + r_diff * diff_feature
        shared = 0.5 * (f_rgb + f_ir)
        out = self.fuse_refine(fused) + torch.tanh(self.residual_scale) * shared

        if return_attention_map:
            reliability_map = r_diff
            if boost_map is not None:
                reliability_map = reliability_map * boost_map
            return out, reliability_map
        return out


@FUSIONS.register("ReliabilityAwareFusion")
class ReliabilityAwareFusion(BaseFusion):
    """Reliability-aware fusion with explicit RGB/IR/discrepancy weighting."""

    def __init__(self, channel_list=None, hidden_ratio=0.25, p2_boost_init=0.1):
        channel_list = channel_list or [64, 128, 256, 512]
        super().__init__(channel_list)
        self.level_names = ['P2', 'P3', 'P4', 'P5']
        self.blocks = nn.ModuleList([
            ReliabilityFusionBlock(
                channels=channels,
                hidden_ratio=hidden_ratio,
                level_index=level_index,
                p2_boost_init=p2_boost_init,
            )
            for level_index, channels in enumerate(channel_list)
        ])

    def forward(self, rgb_features, ir_features, return_attention_map=False, target_size=None):
        fused_features = []
        attention_maps = {}

        for level_name, block, f_rgb, f_ir in zip(self.level_names, self.blocks, rgb_features, ir_features):
            if return_attention_map:
                fused_out, attention_map = block(f_rgb, f_ir, return_attention_map=True)
                if target_size is not None:
                    attention_map = F.interpolate(attention_map, size=target_size, mode='bilinear', align_corners=False)
                attention_maps[f'{level_name}_Reliability_Map'] = attention_map
            else:
                fused_out = block(f_rgb, f_ir)
            fused_features.append(fused_out)

        if return_attention_map:
            return tuple(fused_features), attention_maps
        return tuple(fused_features)
