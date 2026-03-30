# 你的双向 Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from src.registry.fusion_registry import FUSIONS


# ==========================================
# 辅助注意力模块 (原 attention.py 内容)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class JointChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: Optional[int] = None):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduction_ratio = reduction_ratio or max(8, channels // 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, reduction_ratio, 1, bias=False),
            nn.BatchNorm2d(reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_ratio, channels * 2, 1, bias=False)
        )

    def forward(self, f_rgb: torch.Tensor, f_ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(self.gap(torch.cat([f_rgb, f_ir], dim=1)))
        B, C2, _, _ = x.shape
        x = x.view(B, 2, C2 // 2)
        weights = F.softmax(x, dim=1).unsqueeze(-1).unsqueeze(-1)
        return weights[:, 0, ...], weights[:, 1, ...]


# ==========================================
# 核心双向跨模态 Transformer
# ==========================================
class BiCrossTransformerBlock(nn.Module):
    """⭐ 引入 SRA思想，Query 保持全分辨率，修复小目标特征丢失问题"""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.pos_conv_q = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pos_conv_kv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # 深度可分离卷积降维，替代粗暴的 AvgPool
        self.sr_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(inplace=True),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, return_attn_map: bool = False):
        B, C, H, W = x.shape

        x_pos = x + self.pos_conv_q(x)
        ctx_pos = context + self.pos_conv_kv(context)

        q = self.norm1_q(x_pos.flatten(2).transpose(1, 2))  # [B, H*W, C]

        if H > 40 or W > 40:
            ctx_down = self.sr_conv(ctx_pos)
        else:
            ctx_down = ctx_pos

        k = v = self.norm1_kv(ctx_down.flatten(2).transpose(1, 2))

        attn_out, attn_weights = self.attn(q, k, v, need_weights=return_attn_map)

        x_flat = q + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        out = x_flat.transpose(1, 2).reshape(B, C, H, W)

        if return_attn_map:
            spatial_map = attn_weights.mean(dim=1).view(B, 1, H, W)
            spatial_map = F.avg_pool2d(spatial_map, kernel_size=3, stride=1, padding=1)
            return out, torch.sigmoid(spatial_map)
        return out


class MS_FFM_Block(nn.Module):
    def __init__(self, channels: int, level: str, is_p2_level: bool = False, init_alpha: float = 0.3):
        super().__init__()
        self.level = level
        self.is_p2_level = is_p2_level

        if self.level in ['P4', 'P5']:
            self.tr_rgb = BiCrossTransformerBlock(channels)
            self.tr_ir = BiCrossTransformerBlock(channels)
            self.ir_spatial = nn.Identity()
        elif self.is_p2_level:
            self.ir_spatial = SpatialAttention(kernel_size=5)
            self.tr_rgb = self.tr_ir = nn.Identity()
        else:
            self.ir_spatial = self.tr_rgb = self.tr_ir = nn.Identity()

        self.fusion_ca = JointChannelAttention(channels)

        self.res_weight_raw = nn.Parameter(torch.tensor([math.atanh(init_alpha)], dtype=torch.float32))
        self.smooth_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, f_rgb: torch.Tensor, f_ir: torch.Tensor, return_attention_map: bool = False):
        att_map_out = None

        if self.level in ['P4', 'P5']:
            if return_attention_map:
                f_rgb_new, map_rgb = self.tr_rgb(f_rgb, context=f_ir.detach(), return_attn_map=True)
                f_ir_new, map_ir = self.tr_ir(f_ir, context=f_rgb.detach(), return_attn_map=True)
                att_map_out = (map_rgb + map_ir) / 2.0
            else:
                f_rgb_new = self.tr_rgb(f_rgb, context=f_ir.detach())
                f_ir_new = self.tr_ir(f_ir, context=f_rgb.detach())
            f_rgb, f_ir = f_rgb_new, f_ir_new

        w_rgb, w_ir = self.fusion_ca(f_rgb, f_ir)
        f_fused = f_rgb * w_rgb + f_ir * w_ir

        if self.is_p2_level:
            att_map_out = self.ir_spatial(f_ir)
            f_fused = f_fused * (0.5 + att_map_out)

        out = self.act(self.bn(self.smooth_conv(f_fused)) + torch.tanh(self.res_weight_raw) * (f_rgb + f_ir))
        return (out, att_map_out) if return_attention_map and att_map_out is not None else out


@FUSIONS.register("DualStreamFusion")
class DualStreamFusion(nn.Module):
    def __init__(self, channel_list: list = [64, 128, 256, 512]):
        super().__init__()
        self.levels = ['P2', 'P3', 'P4', 'P5']
        self.ffm_blocks = nn.ModuleDict({
            lvl: MS_FFM_Block(ch, lvl, is_p2_level=(lvl == 'P2'), init_alpha=(0.5 if lvl == 'P2' else 0.3))
            for lvl, ch in zip(self.levels, channel_list)
        })

    def forward(self, rgb_features: list, ir_features: list, return_attention_map: bool = False,
                target_size: Optional[Tuple[int, int]] = None):
        fused_features = []
        attention_maps = {}

        for lvl, f_rgb, f_ir in zip(self.levels, rgb_features, ir_features):
            if return_attention_map and lvl in ['P2', 'P4', 'P5']:
                fused_out, att_mask = self.ffm_blocks[lvl](f_rgb, f_ir, return_attention_map=True)
                if target_size is not None:
                    att_mask = F.interpolate(att_mask, size=target_size, mode='bilinear', align_corners=False)
                attention_maps[f'{lvl}_Attention_Map'] = att_mask
            else:
                fused_out = self.ffm_blocks[lvl](f_rgb, f_ir)
            fused_features.append(fused_out)

        if return_attention_map:
            return tuple(fused_features), attention_maps
        return tuple(fused_features)
