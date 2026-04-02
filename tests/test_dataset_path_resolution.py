import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.data.dataloader import build_dataloader


class DatasetPathResolutionTestCase(unittest.TestCase):
    def _create_split(self, root, split):
        split_root = root / split
        (split_root / 'images' / 'img').mkdir(parents=True, exist_ok=True)
        (split_root / 'images' / 'imgr').mkdir(parents=True, exist_ok=True)
        (split_root / 'labels' / 'merged').mkdir(parents=True, exist_ok=True)

        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        ir = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(split_root / 'images' / 'img' / 'sample.jpg'), rgb)
        cv2.imwrite(str(split_root / 'images' / 'imgr' / 'sample.jpg'), ir)

        with open(split_root / 'labels' / 'merged' / 'sample.txt', 'w', encoding='utf-8') as handle:
            handle.write('0 0.5 0.5 0.25 0.25 0.0\n')

    def test_repo_relative_dataset_root_stays_loadable_outside_repo_cwd(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as dataset_tmpdir, tempfile.TemporaryDirectory() as other_cwd:
            dataset_root = Path(dataset_tmpdir)
            self._create_split(dataset_root, 'train')
            self._create_split(dataset_root, 'val')

            relative_root = str(dataset_root.relative_to(repo_root)).replace('\\', '/')
            cfg = OmegaConf.create({
                'dataset': {
                    'type': 'DroneDualDataset',
                    'root_dir': relative_root,
                    'imgsz': 32,
                    'class_names': ['car'],
                    'aug_cfg': {
                        'enable_cmcp': False,
                        'enable_mrre': False,
                        'enable_weather': False,
                        'enable_modality_dropout': False,
                    },
                },
                'dataloader': {
                    'batch_size': 1,
                    'num_workers': 0,
                    'use_log_sampler': False,
                    'pin_memory': False,
                },
                'model': {
                    'temporal': {
                        'enabled': False,
                        'stride': 1,
                    },
                },
            })

            original_cwd = Path.cwd()
            try:
                os.chdir(other_cwd)
                dataloader, dataset = build_dataloader(cfg, is_training=False)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(dataloader), 1)


if __name__ == '__main__':
    unittest.main()
