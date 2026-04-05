from copy import deepcopy
from importlib import import_module
from pathlib import Path
import random
import time
import warnings

import cv2
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


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_DATALOADER_TIMEOUT_SECONDS = 120
TEMPORAL_DATALOADER_PREFETCH_FACTOR = 1
TEMPORAL_DATALOADER_WARN_WORKERS = 4


def _ensure_data_modules_registered():
    import_module('src.data.datasets.drone_rgb_ir')


def _clone_dataset_cfg(dataset_cfg):
    if OmegaConf is not None and OmegaConf.is_config(dataset_cfg):
        return OmegaConf.to_container(dataset_cfg, resolve=True)
    return deepcopy(dataset_cfg)


def _clone_performance_cfg(cfg):
    performance_cfg = cfg.get('performance', {}) if hasattr(cfg, 'get') else {}
    if OmegaConf is not None and OmegaConf.is_config(performance_cfg):
        return OmegaConf.to_container(performance_cfg, resolve=True)
    if isinstance(performance_cfg, dict):
        return deepcopy(performance_cfg)
    if hasattr(performance_cfg, 'items'):
        return dict(performance_cfg.items())
    return {}


def _clone_dataloader_cfg(cfg):
    dataloader_cfg = cfg.get('dataloader', {}) if hasattr(cfg, 'get') else {}
    if OmegaConf is not None and OmegaConf.is_config(dataloader_cfg):
        return OmegaConf.to_container(dataloader_cfg, resolve=True)
    if isinstance(dataloader_cfg, dict):
        return deepcopy(dataloader_cfg)
    if hasattr(dataloader_cfg, 'items'):
        return dict(dataloader_cfg.items())
    return {}


def _normalize_optional_cfg_dict(cfg_section):
    if isinstance(cfg_section, dict):
        return deepcopy(cfg_section)
    if hasattr(cfg_section, 'items'):
        return dict(cfg_section.items())
    return {}


def _has_explicit_cfg_value(cfg_dict, key):
    return key in cfg_dict and cfg_dict.get(key) not in (None, '')


def _coerce_non_negative_int(value, default=0):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _resolve_dataset_root_dir(root_dir):
    if root_dir in (None, ''):
        return root_dir

    root_path = Path(str(root_dir)).expanduser()
    if root_path.is_absolute():
        return str(root_path)

    repo_candidate = (REPO_ROOT / root_path).resolve()
    cwd_candidate = (Path.cwd() / root_path).resolve()
    if repo_candidate.exists() or not cwd_candidate.exists():
        return str(repo_candidate)
    return str(cwd_candidate)


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


def _normalize_temporal_debug_cfg(performance_cfg):
    debug_cfg = performance_cfg.get('temporal_debug', {}) if isinstance(performance_cfg, dict) else {}
    if not isinstance(debug_cfg, dict):
        debug_cfg = {}
    normalized = dict(debug_cfg)
    normalized['enabled'] = bool(normalized.get('enabled', False))
    normalized['sample_log_interval'] = max(0, int(normalized.get('sample_log_interval', 0)))
    normalized['slow_sample_ms'] = float(normalized.get('slow_sample_ms', 500.0))
    normalized['slow_stage_ms'] = float(normalized.get('slow_stage_ms', 150.0))
    normalized['slow_augment_ms'] = float(normalized.get('slow_augment_ms', 200.0))
    normalized['slow_collate_ms'] = float(normalized.get('slow_collate_ms', 200.0))
    normalized['trace_stage_starts'] = bool(normalized.get('trace_stage_starts', False))
    normalized['trace_main_process_stages'] = bool(normalized.get('trace_main_process_stages', True))
    normalized['trace_batch_wait'] = bool(normalized.get('trace_batch_wait', False))
    normalized['dataloader_wait_ms'] = float(normalized.get('dataloader_wait_ms', 1000.0))
    normalized['prev_fallback_warning_limit'] = max(1, int(normalized.get('prev_fallback_warning_limit', 5)))
    normalized['cmcp_max_elapsed_ms'] = float(normalized.get('cmcp_max_elapsed_ms', 120.0))
    normalized['cmcp_max_small_objects'] = max(0, int(normalized.get('cmcp_max_small_objects', 48)))
    normalized['mrre_max_elapsed_ms'] = float(normalized.get('mrre_max_elapsed_ms', 80.0))
    normalized['mrre_max_labels'] = max(0, int(normalized.get('mrre_max_labels', 256)))
    normalized['log_epoch_reset'] = bool(normalized.get('log_epoch_reset', True))
    return normalized


def _dataloader_worker_init_fn(worker_id):
    try:
        cv2.setNumThreads(0)
        if hasattr(cv2, 'ocl'):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    base_seed = torch.initial_seed() % (2 ** 32)
    random.seed(base_seed)
    np.random.seed(base_seed)


def _collate_batch(batch):
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


def build_collate_fn(debug_cfg=None):
    debug_cfg = debug_cfg or {}
    if not debug_cfg.get('enabled', False):
        return _collate_batch

    slow_collate_ms = float(debug_cfg.get('slow_collate_ms', 200.0))
    batch_counter = {'value': 0}

    def _debug_collate(batch):
        start_time = time.perf_counter()
        collated = _collate_batch(batch)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        batch_counter['value'] += 1
        if elapsed_ms >= slow_collate_ms:
            print(
                f"[TemporalData][Collate] batch={batch_counter['value']} "
                f"size={len(batch)} elapsed_ms={elapsed_ms:.1f}"
            )
        return collated

    return _debug_collate


def collate_fn(batch):
    return _collate_batch(batch)


def build_dataloader(cfg, is_training=True):
    _ensure_data_modules_registered()

    dataset_cfg = _clone_dataset_cfg(cfg.dataset)
    dataloader_cfg = _clone_dataloader_cfg(cfg)
    performance_cfg = _clone_performance_cfg(cfg)
    dataloader_perf_cfg = performance_cfg.get('dataloader', {}) if isinstance(performance_cfg, dict) else {}
    dataloader_perf_cfg = _normalize_optional_cfg_dict(dataloader_perf_cfg)
    temporal_debug_cfg = _normalize_temporal_debug_cfg(performance_cfg)
    num_workers = _coerce_non_negative_int(dataloader_cfg.get('num_workers', 0))
    if 'root_dir' in dataset_cfg:
        dataset_cfg['root_dir'] = _resolve_dataset_root_dir(dataset_cfg.get('root_dir'))
    dataset_cfg['split'] = 'train' if is_training else 'val'
    dataset_cfg['is_training'] = is_training
    model_cfg = cfg.get('model', {}) if hasattr(cfg, 'get') else {}
    temporal_cfg = model_cfg.get('temporal', {}) if hasattr(model_cfg, 'get') else {}
    dataset_cfg['use_temporal'] = bool(
        temporal_cfg.get('enabled', model_cfg.get('temporal_enabled', False))
    )
    dataset_cfg['temporal_stride'] = int(
        temporal_cfg.get('stride', model_cfg.get('temporal_stride', 1))
    )
    dataset_cfg['temporal_debug'] = dict(
        temporal_debug_cfg,
        num_workers=num_workers,
    )
    dataset = DATASETS.build(dataset_cfg)
    temporal_training = bool(is_training and dataset_cfg.get('use_temporal', False))

    sampler = None
    shuffle = is_training
    is_distributed = dist.is_available() and dist.is_initialized()

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    elif is_training and dataloader_cfg.get('use_log_sampler', False):
        lbl_dir = Path(dataset_cfg['root_dir']) / dataset_cfg['split'] / 'labels' / 'merged'
        sampler = create_small_object_sampler(lbl_dir, len(dataset))
        shuffle = False

    dataloader_kwargs = {}
    if num_workers > 0:
        if temporal_training and num_workers > TEMPORAL_DATALOADER_WARN_WORKERS:
            warnings.warn(
                f"Temporal training is using num_workers={num_workers}. "
                f"Values above {TEMPORAL_DATALOADER_WARN_WORKERS} can make worker stalls harder to recover from; "
                "prefer smaller worker counts when debugging throughput or stability issues.",
                RuntimeWarning,
                stacklevel=2,
            )
        persistent_workers = bool(dataloader_perf_cfg.get('persistent_workers', True))
        if temporal_training and persistent_workers and not bool(temporal_cfg.get('allow_persistent_workers', False)):
            persistent_workers = False
            print(
                '[DataLoader] Temporal training detected; forcing persistent_workers=False '
                'so epoch state and worker lifecycle stay deterministic.'
            )
        dataloader_kwargs['persistent_workers'] = persistent_workers
        prefetch_factor = dataloader_perf_cfg.get('prefetch_factor', None)
        if temporal_training and not _has_explicit_cfg_value(dataloader_perf_cfg, 'prefetch_factor'):
            prefetch_factor = TEMPORAL_DATALOADER_PREFETCH_FACTOR
        if prefetch_factor is not None:
            dataloader_kwargs['prefetch_factor'] = max(1, _coerce_non_negative_int(prefetch_factor, default=1))
        timeout_seconds = dataloader_perf_cfg.get('timeout_seconds', 0)
        if temporal_training and not _has_explicit_cfg_value(dataloader_perf_cfg, 'timeout_seconds'):
            timeout_seconds = TEMPORAL_DATALOADER_TIMEOUT_SECONDS
        timeout_seconds = _coerce_non_negative_int(timeout_seconds, default=0)
        if timeout_seconds > 0:
            dataloader_kwargs['timeout'] = timeout_seconds
        dataloader_kwargs['worker_init_fn'] = _dataloader_worker_init_fn

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_cfg.get('batch_size', 1),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=build_collate_fn(temporal_debug_cfg if temporal_training else None),
        pin_memory=bool(dataloader_cfg.get('pin_memory', True)),
        drop_last=is_training,
        **dataloader_kwargs,
    )
    return dataloader, dataset
