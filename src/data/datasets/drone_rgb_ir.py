from pathlib import Path

import cv2
import numpy as np
import torch

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

        aug_kwargs = aug_cfg if aug_cfg is not None else {}
        self.aug_pipeline = MultiModalAugmentationPipeline(**aug_kwargs) if is_training else None

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        ir_path = self.root_dir / 'images' / 'imgr' / rgb_path.name
        label_path = self.root_dir / 'labels' / 'merged' / (rgb_path.stem + '.txt')
        prev_idx = max(0, idx - self.temporal_stride)
        prev_rgb_path = self.rgb_files[prev_idx]
        prev_ir_path = self.root_dir / 'images' / 'imgr' / prev_rgb_path.name

        img_rgb = cv2.imread(str(rgb_path))
        img_ir = cv2.imread(str(ir_path))
        prev_rgb = cv2.imread(str(prev_rgb_path))
        prev_ir = cv2.imread(str(prev_ir_path))

        if img_rgb is None or img_ir is None or prev_rgb is None or prev_ir is None:
            raise ValueError(f"Failed to read image pair: {rgb_path}")

        h_old, w_old = img_rgb.shape[:2]

        img_rgb, r, dw, dh = letterbox(img_rgb, (self.img_size, self.img_size))
        img_ir, _, _, _ = letterbox(img_ir, (self.img_size, self.img_size))
        prev_rgb, _, _, _ = letterbox(prev_rgb, (self.img_size, self.img_size))
        prev_ir, _, _, _ = letterbox(prev_ir, (self.img_size, self.img_size))

        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = [[float(x) for x in line.strip().split()] for line in f.readlines()]
        labels = np.array(labels, dtype=np.float32)

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

        return img_rgb, img_ir, labels, prev_rgb, prev_ir
