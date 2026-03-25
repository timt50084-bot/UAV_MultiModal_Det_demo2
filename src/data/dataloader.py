from copy import deepcopy
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from tqdm import tqdm

from src.registry.data_registry import DATASETS

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None


def _ensure_data_modules_registered():
    import_module('src.data.datasets.drone_rgb_ir')


def _clone_dataset_cfg(dataset_cfg):
    if OmegaConf is not None and OmegaConf.is_config(dataset_cfg):
        return OmegaConf.to_container(dataset_cfg, resolve=True)
    return deepcopy(dataset_cfg)


def create_small_object_sampler(label_dir, dataset_len, small_threshold=0.05):
    label_dir = Path(label_dir)
    weights = []
    label_files = sorted(list(label_dir.glob('*.txt')))

    for label_path in tqdm(label_files, desc="Calculating Log-Scale weights"):
        small_obj_count = 0
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        if w < small_threshold and h < small_threshold:
                            small_obj_count += 1

        weights.append(1.0 + np.log1p(small_obj_count))

    if len(weights) < dataset_len:
        weights.extend([1.0] * (dataset_len - len(weights)))

    weights = torch.DoubleTensor(weights)
    return WeightedRandomSampler(weights, num_samples=dataset_len, replacement=True)


def collate_fn(batch):
    img_rgb, img_ir, labels, prev_rgb, prev_ir = zip(*batch)

    img_rgb = torch.stack(img_rgb, 0)
    img_ir = torch.stack(img_ir, 0)
    prev_rgb = torch.stack(prev_rgb, 0)
    prev_ir = torch.stack(prev_ir, 0)

    targets = []
    for i, label in enumerate(labels):
        if label.shape[0] > 0:
            batch_idx = torch.full((label.shape[0], 1), i, dtype=torch.float32)
            targets.append(torch.cat((batch_idx, label), dim=1))

    if len(targets) > 0:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros((0, 7), dtype=torch.float32)

    return img_rgb, img_ir, targets, prev_rgb, prev_ir


def build_dataloader(cfg, is_training=True):
    _ensure_data_modules_registered()

    dataset_cfg = _clone_dataset_cfg(cfg.dataset)
    dataset_cfg['split'] = 'train' if is_training else 'val'
    dataset_cfg['is_training'] = is_training
    dataset_cfg['use_temporal'] = bool(cfg.model.get('temporal_enabled', False))
    dataset_cfg['temporal_stride'] = int(cfg.model.get('temporal_stride', 1))
    dataset = DATASETS.build(dataset_cfg)

    sampler = None
    shuffle = is_training
    is_distributed = dist.is_available() and dist.is_initialized()

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    elif is_training and cfg.get('use_log_sampler', False):
        lbl_dir = Path(dataset_cfg['root_dir']) / dataset_cfg['split'] / 'labels' / 'merged'
        sampler = create_small_object_sampler(lbl_dir, len(dataset))
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_training
    )
    return dataloader, dataset
