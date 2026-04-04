from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import get_worker_info

from src.data.transforms.augmentations import MultiModalAugmentationPipeline
from src.registry.data_registry import DATASETS

from .base_dataset import BaseDataset


def letterbox(img, new_shape=(1024, 1024), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh


@DATASETS.register("DroneDualDataset")
class DroneDualDataset(BaseDataset):
    def __init__(self, root_dir, split='train', img_size=1024, imgsz=None,
                 is_training=True, aug_cfg=None, class_names=None,
                 use_temporal=False, temporal_stride=1, **kwargs):
        super().__init__(root_dir, split, is_training)
        self.root_dir = Path(root_dir) / split
        self.img_size = imgsz if imgsz is not None else img_size
        self.class_names = class_names or []
        self.use_temporal = use_temporal
        self.temporal_stride = max(1, int(temporal_stride))
        self.rgb_files = sorted(list((self.root_dir / 'images' / 'img').glob('*.jpg')))
        temporal_debug = kwargs.get('temporal_debug', {})
        if not isinstance(temporal_debug, dict):
            temporal_debug = {}
        self.temporal_debug_enabled = bool(temporal_debug.get('enabled', False))
        self.temporal_sample_log_interval = max(0, int(temporal_debug.get('sample_log_interval', 0)))
        self.temporal_slow_sample_ms = float(temporal_debug.get('slow_sample_ms', 500.0))
        self._prev_fallback_warning_limit = max(1, int(temporal_debug.get('prev_fallback_warning_limit', 5)))
        self._prev_fallback_warning_count = 0

        aug_kwargs = aug_cfg if aug_cfg is not None else {}
        self.aug_pipeline = MultiModalAugmentationPipeline(**aug_kwargs) if is_training else None

    def __len__(self):
        return len(self.rgb_files)

    @staticmethod
    def _worker_id():
        worker_info = get_worker_info()
        return worker_info.id if worker_info is not None else 'main'

    def _format_sample_debug(self, idx, prev_idx, rgb_path, prev_rgb_path):
        return (
            f"worker={self._worker_id()} epoch={self.current_epoch + 1} idx={idx} prev_idx={prev_idx} "
            f"frame={rgb_path.name} prev_frame={prev_rgb_path.name}"
        )

    def _read_image_or_raise(self, image_path, idx, prev_idx, rgb_path, prev_rgb_path, role):
        image = cv2.imread(str(image_path))
        if image is not None:
            return image
        raise RuntimeError(
            f"[TemporalData] failed to read {role} image: {image_path} "
            f"({self._format_sample_debug(idx, prev_idx, rgb_path, prev_rgb_path)})"
        )

    def _read_prev_image_with_fallback(self, image_path, fallback_image, idx, prev_idx, rgb_path, prev_rgb_path, role):
        image = cv2.imread(str(image_path))
        if image is not None:
            return image
        if self._prev_fallback_warning_count < self._prev_fallback_warning_limit:
            print(
                f"[TemporalData] prev-frame fallback role={role} missing={image_path} "
                f"using_current_frame ({self._format_sample_debug(idx, prev_idx, rgb_path, prev_rgb_path)})"
            )
            self._prev_fallback_warning_count += 1
        return fallback_image.copy()

    def _load_labels(self, label_path, idx, prev_idx, rgb_path, prev_rgb_path):
        labels = []
        if not label_path.exists():
            return np.array(labels, dtype=np.float32)
        try:
            with open(label_path, 'r') as handle:
                labels = [[float(x) for x in line.strip().split()] for line in handle.readlines()]
        except Exception as exc:
            raise RuntimeError(
                f"[TemporalData] failed to parse labels: {label_path} "
                f"({self._format_sample_debug(idx, prev_idx, rgb_path, prev_rgb_path)})"
            ) from exc
        return np.array(labels, dtype=np.float32)

    def _maybe_log_sample(self, idx, prev_idx, rgb_path, prev_rgb_path, label_count, elapsed_ms):
        if not self.temporal_debug_enabled:
            return
        should_log = False
        if self.temporal_sample_log_interval > 0 and (idx % self.temporal_sample_log_interval == 0):
            should_log = True
        if elapsed_ms >= self.temporal_slow_sample_ms:
            should_log = True
        if should_log:
            print(
                f"[TemporalData] {self._format_sample_debug(idx, prev_idx, rgb_path, prev_rgb_path)} "
                f"labels={label_count} elapsed_ms={elapsed_ms:.1f}"
            )

    def __getitem__(self, idx):
        sample_start = time.perf_counter()
        rgb_path = self.rgb_files[idx]
        ir_path = self.root_dir / 'images' / 'imgr' / rgb_path.name
        label_path = self.root_dir / 'labels' / 'merged' / (rgb_path.stem + '.txt')
        prev_idx = max(0, idx - self.temporal_stride)
        prev_rgb_path = self.rgb_files[prev_idx]
        prev_ir_path = self.root_dir / 'images' / 'imgr' / prev_rgb_path.name

        img_rgb = self._read_image_or_raise(rgb_path, idx, prev_idx, rgb_path, prev_rgb_path, role='rgb')
        img_ir = self._read_image_or_raise(ir_path, idx, prev_idx, rgb_path, prev_rgb_path, role='ir')
        prev_rgb = self._read_prev_image_with_fallback(
            prev_rgb_path,
            fallback_image=img_rgb,
            idx=idx,
            prev_idx=prev_idx,
            rgb_path=rgb_path,
            prev_rgb_path=prev_rgb_path,
            role='prev_rgb',
        )
        prev_ir = self._read_prev_image_with_fallback(
            prev_ir_path,
            fallback_image=img_ir,
            idx=idx,
            prev_idx=prev_idx,
            rgb_path=rgb_path,
            prev_rgb_path=prev_rgb_path,
            role='prev_ir',
        )

        h_old, w_old = img_rgb.shape[:2]

        img_rgb, r, dw, dh = letterbox(img_rgb, (self.img_size, self.img_size))
        img_ir, _, _, _ = letterbox(img_ir, (self.img_size, self.img_size))
        prev_rgb, _, _, _ = letterbox(prev_rgb, (self.img_size, self.img_size))
        prev_ir, _, _, _ = letterbox(prev_ir, (self.img_size, self.img_size))

        labels = self._load_labels(label_path, idx, prev_idx, rgb_path, prev_rgb_path)

        if len(labels) > 0:
            labels[:, 1] = (labels[:, 1] * w_old * r + dw) / self.img_size
            labels[:, 2] = (labels[:, 2] * h_old * r + dh) / self.img_size
            labels[:, 3] = (labels[:, 3] * w_old * r) / self.img_size
            labels[:, 4] = (labels[:, 4] * h_old * r) / self.img_size

        if self.is_training and self.aug_pipeline is not None:
            if self.use_temporal:
                img_rgb, img_ir, prev_rgb, prev_ir, labels = self.aug_pipeline.apply_temporal_pair(
                    img_rgb, img_ir, prev_rgb, prev_ir, labels, epoch=self.current_epoch, max_epoch=self.max_epoch
                )
            else:
                img_rgb, img_ir, labels = self.aug_pipeline(
                    img_rgb, img_ir, labels, epoch=self.current_epoch, max_epoch=self.max_epoch
                )

        img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_ir = torch.from_numpy(img_ir).permute(2, 0, 1).float() / 255.0
        prev_rgb = torch.from_numpy(prev_rgb).permute(2, 0, 1).float() / 255.0
        prev_ir = torch.from_numpy(prev_ir).permute(2, 0, 1).float() / 255.0

        if len(labels) == 0:
            labels = torch.zeros((0, 6), dtype=torch.float32)
        else:
            labels = torch.from_numpy(labels).float()

        self._maybe_log_sample(
            idx=idx,
            prev_idx=prev_idx,
            rgb_path=rgb_path,
            prev_rgb_path=prev_rgb_path,
            label_count=int(labels.shape[0]),
            elapsed_ms=(time.perf_counter() - sample_start) * 1000.0,
        )
        return img_rgb, img_ir, labels, prev_rgb, prev_ir
