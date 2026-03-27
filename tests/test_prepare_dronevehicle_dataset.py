import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.datasets.drone_rgb_ir import DroneDualDataset
from tools.prepare_dronevehicle_dataset import prepare_dataset


class PrepareDroneVehicleDatasetSmokeTestCase(unittest.TestCase):
    def _write_sample_xml(self, path: Path, cx=32.0, cy=32.0, w=20.0, h=12.0, angle=0.1):
        xml = f"""
<annotation>
  <object>
    <name>car</name>
    <robndbox>
      <cx>{cx}</cx>
      <cy>{cy}</cy>
      <w>{w}</w>
      <h>{h}</h>
      <angle>{angle}</angle>
    </robndbox>
  </object>
</annotation>
""".strip()
        path.write_text(xml, encoding='utf-8')

    def test_prepare_dataset_outputs_drone_dual_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            raw_root = tmp_root / 'raw'
            out_root = tmp_root / 'processed'

            for split in ('train', 'val'):
                (raw_root / split / f'{split}img').mkdir(parents=True, exist_ok=True)
                (raw_root / split / f'{split}imgr').mkdir(parents=True, exist_ok=True)
                (raw_root / split / f'{split}label').mkdir(parents=True, exist_ok=True)
                (raw_root / split / f'{split}labelr').mkdir(parents=True, exist_ok=True)

                rgb = np.full((64, 64, 3), 180, dtype=np.uint8)
                ir = np.full((64, 64, 3), 120, dtype=np.uint8)
                cv2.imwrite(str(raw_root / split / f'{split}img' / 'sample.jpg'), rgb)
                cv2.imwrite(str(raw_root / split / f'{split}imgr' / 'sample.jpg'), ir)
                self._write_sample_xml(raw_root / split / f'{split}label' / 'sample.xml')
                self._write_sample_xml(raw_root / split / f'{split}labelr' / 'sample.xml', angle=0.12)

            summary = prepare_dataset(
                raw_root=str(raw_root),
                output_root=str(out_root),
                splits=['train', 'val'],
                overwrite=False,
            )

            self.assertEqual(summary['total_processed_pairs'], 2)
            self.assertTrue((out_root / 'train' / 'images' / 'img' / 'sample.jpg').exists())
            self.assertTrue((out_root / 'train' / 'images' / 'imgr' / 'sample.jpg').exists())
            self.assertTrue((out_root / 'train' / 'labels' / 'merged' / 'sample.txt').exists())

            dataset = DroneDualDataset(
                root_dir=str(out_root),
                split='train',
                img_size=64,
                imgsz=64,
                is_training=False,
                aug_cfg=None,
                class_names=['car', 'truck', 'bus', 'van', 'freight_car'],
            )
            self.assertEqual(len(dataset), 1)
            img_rgb, img_ir, labels, prev_rgb, prev_ir = dataset[0]
            self.assertEqual(tuple(img_rgb.shape), (3, 64, 64))
            self.assertEqual(tuple(img_ir.shape), (3, 64, 64))
            self.assertEqual(tuple(prev_rgb.shape), (3, 64, 64))
            self.assertEqual(tuple(prev_ir.shape), (3, 64, 64))
            self.assertEqual(labels.shape[1], 6)


if __name__ == '__main__':
    unittest.main()