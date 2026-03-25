# Baseline
import torch
import torch.nn as nn
from src.registry.fusion_registry import FUSIONS
from .base_fusion import BaseFusion


@FUSIONS.register("SimpleConcatFusion")
class SimpleConcatFusion(BaseFusion):
    """
    🔥 极简对照组 (Ablation Baseline 1)
    直接将 RGB 和 IR 沿通道维度拼接，然后用 1x1 卷积降维。
    没有任何注意力机制，计算极快，用于衬托你们创新方法的优越性。
    """

    def __init__(self, channel_list: list = [64, 128, 256, 512]):
        super().__init__(channel_list)

        # 为每个尺度创建一个 1x1 卷积降维层 (2C -> C)
        self.mixers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True)
            ) for c in channel_list
        ])

    def forward(self, rgb_features, ir_features, return_attention_map=False, target_size=None):
        fused_features = []

        for r_feat, i_feat, mixer in zip(rgb_features, ir_features, self.mixers):
            # 简单拼接后降维
            concat_feat = torch.cat([r_feat, i_feat], dim=1)
            fused_features.append(mixer(concat_feat))

        if return_attention_map:
            # 占位符：简单拼接没有注意力图，返回空的 dict 防报错
            return tuple(fused_features), {}

        return tuple(fused_features)