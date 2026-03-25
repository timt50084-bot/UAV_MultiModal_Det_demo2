from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README_PATH = ROOT / 'README.md'


def read_readme():
    if README_PATH.exists():
        return README_PATH.read_text(encoding='utf-8-sig')
    return 'UAV multi-modal OBB detection project.'


def discover_packages():
    packages = ['src']
    packages.extend(find_packages(where='.', include=['src.*']))
    return packages


setup(
    name='uav-multimodal-obb-det',
    version='0.1.0',
    description='Configuration-driven UAV RGB-IR oriented object detection project.',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    packages=discover_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.13',
        'torchvision>=0.14',
        'numpy>=1.21',
        'opencv-python>=4.5',
        'omegaconf>=2.3',
        'PyYAML>=6.0',
        'tqdm>=4.64',
        'shapely>=1.8',
    ],
)
