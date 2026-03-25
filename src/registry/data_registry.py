# 实例化 DATASET, TRANSFORM 注册表
from .registry import Registry
DATASETS = Registry("Dataset")
TRANSFORMS = Registry("Transform")