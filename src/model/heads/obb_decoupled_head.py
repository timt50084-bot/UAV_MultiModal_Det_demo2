# OBB 解耦预测头
import torch
import torch.nn as nn
import math
from src.registry.model_registry import HEADS

@HEADS.register("OBBDecoupledHead")
class OBBDecoupledHead(nn.Module):
    def __init__(self, num_classes=5, channels=[64, 128, 256, 512], return_dict=True):
        super().__init__()
        self.nc = num_classes
        self.nl = len(channels)
        self.return_dict = return_dict

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.angle_preds = nn.ModuleList()

        for ch in channels:
            self.cls_preds.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True),
                nn.Conv2d(ch, self.nc, 1)
            ))
            self.reg_preds.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True),
                nn.Conv2d(ch, 4, 1)
            ))
            self.angle_preds.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 3, padding=1, bias=False), nn.BatchNorm2d(ch // 2), nn.SiLU(inplace=True),
                nn.Conv2d(ch // 2, 1, 1)
            ))
        self._initialize_biases()

    def _initialize_biases(self, prior_prob=0.01):
        for m in self.cls_preds:
            last_conv = m[-1]
            if hasattr(last_conv, 'bias') and last_conv.bias is not None:
                last_conv.bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            else:
                last_conv.bias = nn.Parameter(torch.ones(self.nc) * -math.log((1 - prior_prob) / prior_prob))

    def forward(self, x):
        outputs = []
        for i in range(self.nl):
            feat = x[i]
            cls_out = self.cls_preds[i](feat)

            reg_raw = self.reg_preds[i](feat)
            xy = torch.sigmoid(reg_raw[:, 0:2, ...])
            wh = torch.exp(torch.clamp(reg_raw[:, 2:4, ...], max=10.0))
            reg_out = torch.cat([xy, wh], dim=1)

            angle_raw = self.angle_preds[i](feat)
            angle_out = torch.tanh(angle_raw) * (math.pi / 2.0)

            if self.return_dict:
                outputs.append({'cls': cls_out, 'reg': reg_out, 'angle': angle_out})
            else:
                outputs.append(torch.cat([cls_out, reg_out, angle_out], dim=1))

        return outputs