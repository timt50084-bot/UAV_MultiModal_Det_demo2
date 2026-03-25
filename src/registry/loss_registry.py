# 实例化 LOSS 注册表
from .registry import Registry
LOSSES = Registry("Loss")
ASSIGNERS = Registry("Assigner")