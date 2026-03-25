import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry.model_registry import NECKS

class CNN_SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, max(8, channel // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, channel // reduction), channel, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

class SPDConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super().__init__()
        self.scale = scale
        hidden_channels = inc * (scale ** 2)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, groups=hidden_channels, bias=False)
        self.pw_conv = nn.Conv2d(hidden_channels, outc, 1, bias=False)
        self.bn = nn.BatchNorm2d(outc)
        self.act = nn.SiLU(inplace=True)
        self.se = CNN_SEBlock(outc)

    def forward(self, x):
        x_slices = [x[..., i::self.scale, j::self.scale] for i in range(self.scale) for j in range(self.scale)]
        x_spd = torch.cat(x_slices, dim=1)
        return self.se(self.act(self.bn(self.pw_conv(self.dw_conv(x_spd)))))

@NECKS.register("EnhancedNeck")
class EnhancedNeck(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        c2, c3, c4, c5 = channels

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.c5_to_c4 = nn.Conv2d(c5 + c4, c4, 1, bias=False)
        self.c4_to_c3 = nn.Conv2d(c4 + c3, c3, 1, bias=False)
        self.c3_to_c2 = nn.Conv2d(c3 + c2, c2, 1, bias=False)

        self.down2 = SPDConv(c2, c3)
        self.down3 = SPDConv(c3, c4)
        self.down4 = SPDConv(c4, c5)

        self.align3 = nn.Sequential(nn.Conv2d(c3, c3, 1, bias=False), nn.BatchNorm2d(c3))
        self.align4 = nn.Sequential(nn.Conv2d(c4, c4, 1, bias=False), nn.BatchNorm2d(c4))
        self.align5 = nn.Sequential(nn.Conv2d(c5, c5, 1, bias=False), nn.BatchNorm2d(c5))

        self.smooth_p2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.smooth_p3 = nn.Conv2d(c3, c3, 3, 1, 1, bias=False)
        self.smooth_p4 = nn.Conv2d(c4, c4, 3, 1, 1, bias=False)
        self.smooth_p5 = nn.Conv2d(c5, c5, 3, 1, 1, bias=False)

        self.alpha3 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.alpha4 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.alpha5 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, features):
        f2, f3, f4, f5 = features
        p5 = f5

        p4 = self.c5_to_c4(torch.cat([self.up(p5), f4], dim=1))
        p3 = self.c4_to_c3(torch.cat([self.up(p4), f3], dim=1))
        p2 = self.c3_to_c2(torch.cat([self.up(p3), f2], dim=1))

        n2 = self.smooth_p2(p2)

        w3 = F.relu(self.alpha3)
        n3 = self.smooth_p3((w3[0] * self.align3(p3) + w3[1] * self.down2(n2)) / (w3.sum() + self.eps))

        w4 = F.relu(self.alpha4)
        n4 = self.smooth_p4((w4[0] * self.align4(p4) + w4[1] * self.down3(n3)) / (w4.sum() + self.eps))

        w5 = F.relu(self.alpha5)
        n5 = self.smooth_p5((w5[0] * self.align5(p5) + w5[1] * self.down4(n4)) / (w5.sum() + self.eps))

        return (n2, n3, n4, n5)