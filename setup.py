from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README_PATH = ROOT / 'README.md'


def read_readme():
    if README_PATH.exists():
        for encoding in ('utf-8-sig', 'utf-8', 'gb18030'):
            try:
                return README_PATH.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
    return 'UAV multi-modal OBB detection project.'


def discover_packages():
    packages = ['src']
    packages.extend(find_packages(where='.', include=['src.*']))
    return packages


def load_requirements(path: str = 'requirements.txt'):
    req_path = ROOT / path
    requirements = []
    if not req_path.exists():
        return requirements

    for line in req_path.read_text(encoding='utf-8').splitlines():
        item = line.strip()
        if not item or item.startswith('#'):
            continue
        requirements.append(item)
    return requirements


setup(
    name='uav-multimodal-obb-det',
    version='0.1.0',
    description='Configuration-driven UAV RGB-IR oriented object detection project.',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    packages=discover_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=load_requirements(),
    extras_require={
        'export': [
            'onnx>=1.15,<2.0',
            'onnxsim>=0.4.33,<1.0',
            'onnxruntime>=1.17,<2.0',
        ],
    },
)
