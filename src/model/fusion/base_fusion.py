# 定义 forward 接口规范
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Union
import torch

class BaseFusion(nn.Module, ABC):
    """
    【融合模块基类】
    定义了严格的输入输出规范。无论是 Baseline 还是你的创新 Transformer，
    只要继承该类，就能无缝插入 YOLO 引擎。
    """
    def __init__(self, channel_list: List[int]):
        super().__init__()
        self.channel_list = channel_list

    @abstractmethod
    def forward(self, rgb_features: List[torch.Tensor], ir_features: List[torch.Tensor],
                return_attention_map: bool = False, target_size: Optional[Tuple[int, int]] = None) \
                -> Union[Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]]:
        """
        Args:
            rgb_features: RGB 分支的多尺度特征 [P2, P3, P4, P5]
            ir_features: IR 分支的多尺度特征 [P2, P3, P4, P5]
            return_attention_map: 是否返回热力图 (用于可视化)
            target_size: 热力图插值到的目标尺寸
        Returns:
            fused_features: 融合后的特征元组 (P2, P3, P4, P5)
        """
        pass