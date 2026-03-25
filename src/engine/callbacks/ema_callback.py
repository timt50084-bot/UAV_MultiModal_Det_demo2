# 模型滑动平均
import math
import copy
import torch
import torch.nn as nn
from .base import Callback

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    return model.module if is_parallel(model) else model

class EMACallback(Callback):
    """模型指数移动平均回调 (接管在每个 Batch 后更新 EMA 权重的逻辑)"""
    def __init__(self, model, decay=0.9998, tau=2000):
        self.ema = copy.deepcopy(de_parallel(model)).eval()
        self.updates = 0
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters(): p.requires_grad_(False)

    def on_batch_end(self, trainer):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            msd = de_parallel(trainer.model).state_dict()
            esd = self.ema.state_dict()
            for k in list(esd.keys()):
                v, model_v = esd[k], msd[k].detach()
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * model_v
                else:
                    v.copy_(model_v)