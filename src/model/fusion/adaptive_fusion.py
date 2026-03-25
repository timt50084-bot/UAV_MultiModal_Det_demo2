# ⭐ 动态门控/权重控制器
import torch
import torch.nn as nn
from src.registry.fusion_registry import FUSIONS
from .base_fusion import BaseFusion


@FUSIONS.register("AdaptiveWeightFusion")
class AdaptiveWeightFusion(BaseFusion):
    """
    🔥 动态门控对照组 (Ablation Baseline 2)
    为 RGB 和 IR 分别学习一个可学习的标量权重 (Alpha, Beta)。
    比 SimpleConcat 聪明一点，但远不如你的 Cross Attention。
    """

    def __init__(self, channel_list: list = [64, 128, 256, 512]):
        super().__init__(channel_list)

        # 初始化可学习的权重 (每个尺度有两个权重)
        # 使用 Parameter 包装，使其在反向传播中被更新
        self.rgb_weights = nn.Parameter(torch.ones(len(channel_list), dtype=torch.float32))
        self.ir_weights = nn.Parameter(torch.ones(len(channel_list), dtype=torch.float32))

        # 平滑层，用于融合后的特征过渡
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True)
            ) for c in channel_list
        ])

    def forward(self, rgb_features, ir_features, return_attention_map=False, target_size=None):
        fused_features = []

        # 归一化权重，确保 RGB_w + IR_w = 1
        rgb_w = torch.relu(self.rgb_weights)
        ir_w = torch.relu(self.ir_weights)
        total_w = rgb_w + ir_w + 1e-6

        rgb_w = rgb_w / total_w
        ir_w = ir_w / total_w

        for i, (r_feat, i_feat, smooth) in enumerate(zip(rgb_features, ir_features, self.smooth_convs)):
            # 标量加权融合: F = w1 * RGB + w2 * IR
            fused = r_feat * rgb_w[i] + i_feat * ir_w[i]
            fused_features.append(smooth(fused))

        if return_attention_map:
            return tuple(fused_features), {}

        return tuple(fused_features)