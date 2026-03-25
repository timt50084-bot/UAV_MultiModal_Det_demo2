# 模态对齐损失
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry.loss_registry import LOSSES

@LOSSES.register("ContrastiveAlignmentLoss")
class ContrastiveAlignmentLoss(nn.Module):
    """
    🔥 模态对齐对比损失 (Contrastive Alignment Loss)
    将原本写在模型内部的方法剥离，负责在特征层级面对齐 RGB 和 IR 模态的语义分布。
    """
    def __init__(self, temperature=0.07, lambda_c=0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_c = lambda_c

    def forward(self, rgb_feats, ir_feats):
        """
        Args:
            rgb_feats: RGB分支特征列表 [P2, P3, P4, P5]
            ir_feats: IR分支特征列表 [P2, P3, P4, P5]
        """
        # 取最高层语义特征 (P5) 进行对比学习对齐
        r_p5 = rgb_feats[-1]
        i_p5 = ir_feats[-1]

        # 全局平均池化，提取特征向量
        r_vec = torch.mean(r_p5, dim=[2, 3])
        i_vec = torch.mean(i_p5, dim=[2, 3])

        # L2 归一化投影到单位超球面
        r_vec = F.normalize(r_vec, dim=1)
        i_vec = F.normalize(i_vec, dim=1)

        # 计算相似度矩阵与 InfoNCE Loss
        logits = torch.matmul(r_vec, i_vec.t()) / self.temperature
        labels = torch.arange(r_vec.size(0), device=r_vec.device)

        loss_r2i = F.cross_entropy(logits, labels)
        loss_i2r = F.cross_entropy(logits.t(), labels)

        # 双向对称损失合并
        loss = (loss_r2i + loss_i2r) * 0.5 * self.lambda_c
        return loss