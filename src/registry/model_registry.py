# 实例化 BACKBONE, NECK, HEAD, DETECTOR 注册表
from .registry import Registry
BACKBONES = Registry("Backbone")
NECKS = Registry("Neck")
HEADS = Registry("Head")
DETECTORS = Registry("Detector")