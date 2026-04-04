import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.data.dataloader import build_dataloader


class TemporalDataloaderSmokeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.tmp_dir.name) / "dataset"
        self._create_split("train", count=6)
        self._create_split("val", count=4)
        self.cfg = OmegaConf.create({
            "dataloader": {
                "batch_size": 2,
                "num_workers": 1,
                "use_log_sampler": False,
                "pin_memory": False,
            },
            "dataset": {
                "type": "DroneDualDataset",
                "root_dir": str(self.dataset_root),
                "imgsz": 64,
                "class_names": ["car"],
                "aug_cfg": {
                    "enable_cmcp": False,
                    "enable_mrre": False,
                    "enable_weather": False,
                    "enable_modality_dropout": False,
                },
            },
            "model": {
                "temporal": {
                    "enabled": True,
                    "mode": "two_frame",
                    "stride": 1,
                },
            },
            "performance": {
                "dataloader": {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                    "timeout_seconds": 5,
                },
                "temporal_debug": {
                    "enabled": False,
                },
            },
        })

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_split(self, split, count):
        split_root = self.dataset_root / split
        (split_root / "images" / "img").mkdir(parents=True, exist_ok=True)
        (split_root / "images" / "imgr").mkdir(parents=True, exist_ok=True)
        (split_root / "labels" / "merged").mkdir(parents=True, exist_ok=True)

        for index in range(count):
            name = f"{index + 1:05d}.jpg"
            label_name = f"{index + 1:05d}.txt"
            rgb = np.full((64, 64, 3), fill_value=32 + index, dtype=np.uint8)
            ir = np.full((64, 64, 3), fill_value=96 + index, dtype=np.uint8)
            cv2.imwrite(str(split_root / "images" / "img" / name), rgb)
            cv2.imwrite(str(split_root / "images" / "imgr" / name), ir)
            with open(split_root / "labels" / "merged" / label_name, "w") as handle:
                handle.write("0 0.5 0.5 0.25 0.20 0.0\n")

    def test_temporal_loader_runs_multiple_epochs_without_persistent_workers(self):
        train_loader, dataset = build_dataloader(self.cfg, is_training=True)

        self.assertFalse(train_loader.persistent_workers)
        self.assertEqual(train_loader.timeout, 5)

        expected_batches_per_epoch = len(dataset) // self.cfg.dataloader.batch_size
        observed_batches = 0
        for epoch in range(3):
            dataset.set_epoch(epoch, 3)
            epoch_batches = 0
            for imgs_rgb, imgs_ir, targets, prev_rgb, prev_ir in train_loader:
                self.assertEqual(tuple(imgs_rgb.shape), (2, 3, 64, 64))
                self.assertEqual(tuple(imgs_ir.shape), (2, 3, 64, 64))
                self.assertEqual(tuple(prev_rgb.shape), (2, 3, 64, 64))
                self.assertEqual(tuple(prev_ir.shape), (2, 3, 64, 64))
                self.assertEqual(targets.shape[-1], 7)
                epoch_batches += 1
                observed_batches += 1
            self.assertEqual(epoch_batches, expected_batches_per_epoch)

        self.assertEqual(observed_batches, expected_batches_per_epoch * 3)


if __name__ == "__main__":
    unittest.main()
