# 方便将 src 安装为包，避免 import 路径报错
from setuptools import setup, find_packages

setup(
    name='uav_multimodal_det',
    version='1.0.0',
    description='A highly decoupled, industrial-grade framework for UAV Dual-Modal OBB Object Detection.',
    author='Your Name',
    packages=find_packages(), # 会自动找到 src 目录
    python_requires='>=3.8',
)