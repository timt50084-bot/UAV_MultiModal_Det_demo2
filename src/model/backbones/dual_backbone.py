# 非对称双流骨干网
import torch
import torch.nn as nn
from src.registry.model_registry import BACKBONES

# ==========================================
# 基础算子定义 (高内聚于 Backbone 内部)
# ==========================================
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def build_norm(c, norm_type='GN'):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=c)
    else:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return nn.SyncBatchNorm(c)
        return nn.BatchNorm2d(c)

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# ==========================================
# 核心网络
# ==========================================
@BACKBONES.register("AsymmetricDualBackbone")
class AsymmetricDualBackbone(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], norm_type='GN'):
        super().__init__()
        c2, c3, c4, c5 = channels
        self.norm_type = norm_type

        # RGB Branch
        self.rgb_stem = Conv(3, c2 // 2, 3, 2)
        self.rgb_stage2 = nn.Sequential(Conv(c2 // 2, c2, 3, 2), C2f(c2, c2, n=3, shortcut=True))
        self.rgb_stage3 = nn.Sequential(Conv(c2, c3, 3, 2), C2f(c3, c3, n=6, shortcut=True))
        self.rgb_stage4 = nn.Sequential(Conv(c3, c4, 3, 2), C2f(c4, c4, n=6, shortcut=True))
        self.rgb_stage5 = nn.Sequential(Conv(c4, c5, 3, 2), C2f(c5, c5, n=3, shortcut=True), SPPF(c5, c5, k=5))

        # IR Branch
        self.ir_stem = Conv(3, c2 // 2, 3, 2)
        self.ir_stage2 = nn.Sequential(Conv(c2 // 2, c2, 3, 2), C2f(c2, c2, n=1, shortcut=True))
        self.ir_stage3 = nn.Sequential(Conv(c2, c3, 3, 2), C2f(c3, c3, n=2, shortcut=True))
        self.ir_stage4 = nn.Sequential(Conv(c3, c4, 3, 2), C2f(c4, c4, n=2, shortcut=True))
        self.ir_stage5 = nn.Sequential(Conv(c4, c5, 3, 2), C2f(c5, c5, n=1, shortcut=True), SPPF(c5, c5, k=5))

        self.align_shared = nn.ModuleList([build_norm(c, norm_type=self.norm_type) for c in channels])

    def forward(self, img_rgb, img_ir):
        r1 = self.rgb_stem(img_rgb)
        r2 = self.rgb_stage2(r1)
        r3 = self.rgb_stage3(r2)
        r4 = self.rgb_stage4(r3)
        r5 = self.rgb_stage5(r4)

        i1 = self.ir_stem(img_ir)
        i2 = self.ir_stage2(i1)
        i3 = self.ir_stage3(i2)
        i4 = self.ir_stage4(i3)
        i5 = self.ir_stage5(i4)

        rgb_feats, ir_feats = [], []
        for i, (r, ir) in enumerate(zip([r2, r3, r4, r5], [i2, i3, i4, i5])):
            x = torch.cat([r, ir], dim=0)
            x = self.align_shared[i](x)
            r_aligned, ir_aligned = x.chunk(2, dim=0)
            rgb_feats.append(r_aligned)
            ir_feats.append(ir_aligned)

        return tuple(rgb_feats), tuple(ir_feats)