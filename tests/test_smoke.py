import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.dataloader import build_dataloader
from src.engine.callbacks.checkpoint_callback import CheckpointCallback
from src.engine.callbacks.ema_callback import EMACallback
from src.engine.trainer import Trainer
from src.loss.builder import build_assigner, build_loss
from src.model.builder import build_model
from src.model.output_adapter import flatten_predictions


class SmokeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.tmp_dir.name) / "dataset"
        self.weights_dir = Path(self.tmp_dir.name) / "weights"

        self._create_split("train")
        self._create_split("val")

        self.cfg = OmegaConf.create({
            "batch_size": 1,
            "num_workers": 0,
            "use_log_sampler": False,
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
                "type": "YOLODualModalOBB",
                "num_classes": 1,
                "channels": [64, 128, 256, 512],
                "fusion_att_type": "SimpleConcatFusion",
                "norm_type": "GN",
                "use_contrastive": False,
                "temporal_enabled": True,
                "temporal_stride": 1,
            },
            "assigner": {
                "type": "DynamicTinyOBBAssigner",
                "num_classes": 1,
                "topk": 5,
                "alpha": 0.5,
                "beta": 6.0,
                "temperature": 2.0,
                "eps": 1e-9,
            },
            "loss": {
                "type": "UAVDualModalLoss",
                "num_classes": 1,
                "alpha": 0.25,
                "gamma": 2.0,
                "use_scale_weight": True,
            },
            "train": {
                "epochs": 1,
                "lr": 1e-3,
                "weight_decay": 5e-4,
                "lrf": 0.01,
                "accumulate": 1,
                "use_amp": False,
                "grad_clip": 10.0,
                "patience": 1,
                "save_dir": str(self.weights_dir),
            },
            "val": {
                "nms": {
                    "conf_thres": 0.001,
                    "iou_thres": 0.45,
                    "max_det": 50,
                    "max_wh": 4096.0,
                },
            },
        })

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_split(self, split):
        split_root = self.dataset_root / split
        (split_root / "images" / "img").mkdir(parents=True, exist_ok=True)
        (split_root / "images" / "imgr").mkdir(parents=True, exist_ok=True)
        (split_root / "labels" / "merged").mkdir(parents=True, exist_ok=True)

        rgb = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        ir = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)

        cv2.imwrite(str(split_root / "images" / "img" / "sample.jpg"), rgb)
        cv2.imwrite(str(split_root / "images" / "imgr" / "sample.jpg"), ir)

        with open(split_root / "labels" / "merged" / "sample.txt", "w") as f:
            f.write("0 0.5 0.5 0.25 0.20 0.0\n")

    def test_dataloader_and_forward_smoke(self):
        train_loader, _ = build_dataloader(self.cfg, is_training=True)
        val_loader, _ = build_dataloader(self.cfg, is_training=False)

        train_rgb, train_ir, train_targets, train_prev_rgb, train_prev_ir = next(iter(train_loader))
        val_rgb, val_ir, val_targets, val_prev_rgb, val_prev_ir = next(iter(val_loader))

        self.assertEqual(tuple(train_rgb.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(train_ir.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(train_prev_rgb.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(train_prev_ir.shape), (1, 3, 64, 64))
        self.assertEqual(train_targets.shape[-1], 7)
        self.assertEqual(tuple(val_rgb.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(val_ir.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(val_prev_rgb.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(val_prev_ir.shape), (1, 3, 64, 64))
        self.assertEqual(val_targets.shape[-1], 7)

        model = build_model(self.cfg.model)
        outputs, _, _ = model(train_rgb, train_ir, prev_rgb=train_prev_rgb, prev_ir=train_prev_ir)
        flat_preds, per_level_outputs = flatten_predictions(outputs)

        self.assertEqual(len(per_level_outputs), 4)
        self.assertEqual(flat_preds.ndim, 3)
        self.assertEqual(flat_preds.shape[0], 1)
        self.assertEqual(flat_preds.shape[-1], 6)
        self.assertTrue(torch.isfinite(flat_preds).all())

    def test_single_epoch_training_smoke(self):
        train_loader, _ = build_dataloader(self.cfg, is_training=True)

        model = build_model(self.cfg.model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        criterion = build_loss(self.cfg.loss)
        assigner = build_assigner(self.cfg.assigner)

        callbacks = [
            EMACallback(model),
            CheckpointCallback(save_dir=self.cfg.train.save_dir, patience=self.cfg.train.patience),
        ]

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            assigner=assigner,
            device=torch.device("cpu"),
            epochs=1,
            accumulate=1,
            use_amp=False,
            callbacks=callbacks,
        )
        trainer.train()

        self.assertTrue(hasattr(trainer, "ema_callback"))
        self.assertGreater(trainer.ema_callback.updates, 0)
        self.assertTrue((self.weights_dir / "latest.pt").exists())


if __name__ == "__main__":
    unittest.main()
