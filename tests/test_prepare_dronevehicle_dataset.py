import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.datasets.drone_rgb_ir import DroneDualDataset
from src.data.transforms.preprocess import get_dm_sop_crop_bbox, parse_xml_to_yolo_obb
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

    def _write_polygon_xml(self, path: Path, class_name='car'):
        xml = f"""
<annotation>
  <object>
    <name>{class_name}</name>
    <polygon>
      <x1>20</x1>
      <y1>20</y1>
      <x2>40</x2>
      <y2>20</y2>
      <x3>40</x3>
      <y3>50</y3>
      <x4>20</x4>
      <y4>50</y4>
    </polygon>
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

            saved_rgb = cv2.imread(str(out_root / 'train' / 'images' / 'img' / 'sample.jpg'))
            saved_ir = cv2.imread(str(out_root / 'train' / 'images' / 'imgr' / 'sample.jpg'))
            self.assertEqual(saved_rgb.shape[:2], saved_ir.shape[:2])

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

    def test_white_border_is_cropped(self):
        rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
        ir = np.full((100, 100, 3), 255, dtype=np.uint8)
        rgb[20:80, 20:80] = 120
        ir[25:75, 25:75] = 130

        x_min, y_min, x_max, y_max = get_dm_sop_crop_bbox(
            rgb,
            ir,
            Path('__missing_rgb__.xml'),
            Path('__missing_ir__.xml'),
        )

        self.assertGreater(x_min, 0)
        self.assertGreater(y_min, 0)
        self.assertLess(x_max, 99)
        self.assertLess(y_max, 99)
        self.assertLess((x_max - x_min + 1), 100)
        self.assertLess((y_max - y_min + 1), 100)

    def test_white_border_noise_is_ignored(self):
        rgb = np.full((120, 120, 3), 255, dtype=np.uint8)
        ir = np.full((120, 120, 3), 255, dtype=np.uint8)
        rgb[20:100, 18:102] = 80
        ir[22:98, 20:100] = 90

        rgb[0, 0] = 100
        rgb[1, 1] = 110
        ir[0, 119] = 100
        ir[2, 118] = 120

        x_min, y_min, x_max, y_max = get_dm_sop_crop_bbox(
            rgb,
            ir,
            Path('__missing_rgb__.xml'),
            Path('__missing_ir__.xml'),
        )

        self.assertGreaterEqual(x_min, 18)
        self.assertGreaterEqual(y_min, 20)
        self.assertLessEqual(x_max, 101)
        self.assertLessEqual(y_max, 99)

    def test_black_border_is_cropped(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        ir = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[15:85, 10:90] = 120
        ir[20:80, 12:88] = 140

        x_min, y_min, x_max, y_max = get_dm_sop_crop_bbox(
            rgb,
            ir,
            Path('__missing_rgb__.xml'),
            Path('__missing_ir__.xml'),
        )

        self.assertGreater(x_min, 0)
        self.assertGreater(y_min, 0)
        self.assertLess(x_max, 99)
        self.assertLess(y_max, 99)

    def test_xml_padding_protects_targets(self):
        rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
        ir = np.full((100, 100, 3), 255, dtype=np.uint8)
        rgb[35:65, 35:65] = 100
        ir[35:65, 35:65] = 110

        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / 'sample.xml'
            self._write_sample_xml(xml_path, cx=18.0, cy=50.0, w=12.0, h=12.0, angle=0.0)
            x_min, y_min, x_max, y_max = get_dm_sop_crop_bbox(rgb, ir, xml_path, xml_path)

        self.assertLessEqual(x_min, 5)
        self.assertGreaterEqual(x_max, 64)
        self.assertGreaterEqual(y_max, 64)

    def test_blank_image_falls_back_to_full_image(self):
        rgb = np.full((80, 90, 3), 255, dtype=np.uint8)
        ir = np.full((80, 90, 3), 255, dtype=np.uint8)

        x_min, y_min, x_max, y_max = get_dm_sop_crop_bbox(
            rgb,
            ir,
            Path('__missing_rgb__.xml'),
            Path('__missing_ir__.xml'),
        )

        self.assertEqual((x_min, y_min, x_max, y_max), (0, 0, 89, 79))

    def test_prepare_dataset_supports_training_trainingr_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            raw_root = tmp_root / 'raw'
            out_root = tmp_root / 'processed'

            (raw_root / 'train' / 'training').mkdir(parents=True, exist_ok=True)
            (raw_root / 'train' / 'trainingr').mkdir(parents=True, exist_ok=True)
            (raw_root / 'train' / 'trainlabel').mkdir(parents=True, exist_ok=True)
            (raw_root / 'train' / 'trainlabelr').mkdir(parents=True, exist_ok=True)
            (raw_root / 'val' / 'validation').mkdir(parents=True, exist_ok=True)
            (raw_root / 'val' / 'validationr').mkdir(parents=True, exist_ok=True)
            (raw_root / 'val' / 'vallabel').mkdir(parents=True, exist_ok=True)
            (raw_root / 'val' / 'vallabelr').mkdir(parents=True, exist_ok=True)

            train_rgb = np.full((64, 64, 3), 180, dtype=np.uint8)
            train_ir = np.full((64, 64, 3), 120, dtype=np.uint8)
            val_rgb = np.full((64, 64, 3), 170, dtype=np.uint8)
            val_ir = np.full((64, 64, 3), 110, dtype=np.uint8)

            cv2.imwrite(str(raw_root / 'train' / 'training' / 'sample.jpg'), train_rgb)
            cv2.imwrite(str(raw_root / 'train' / 'trainingr' / 'sample.jpg'), train_ir)
            cv2.imwrite(str(raw_root / 'val' / 'validation' / 'sample.jpg'), val_rgb)
            cv2.imwrite(str(raw_root / 'val' / 'validationr' / 'sample.jpg'), val_ir)

            self._write_sample_xml(raw_root / 'train' / 'trainlabel' / 'sample.xml')
            self._write_sample_xml(raw_root / 'train' / 'trainlabelr' / 'sample.xml', angle=0.12)
            self._write_sample_xml(raw_root / 'val' / 'vallabel' / 'sample.xml')
            self._write_sample_xml(raw_root / 'val' / 'vallabelr' / 'sample.xml', angle=0.12)

            summary = prepare_dataset(
                raw_root=str(raw_root),
                output_root=str(out_root),
                splits=['train', 'val'],
                overwrite=False,
            )

            self.assertEqual(summary['splits']['train']['rgb_dir'], str(raw_root / 'train' / 'training'))
            self.assertEqual(summary['splits']['train']['ir_dir'], str(raw_root / 'train' / 'trainingr'))
            self.assertFalse(summary['splits']['train']['rgb_ir_same_dir'])
            self.assertEqual(summary['splits']['val']['rgb_dir'], str(raw_root / 'val' / 'validation'))
            self.assertEqual(summary['splits']['val']['ir_dir'], str(raw_root / 'val' / 'validationr'))
            self.assertFalse(summary['splits']['val']['rgb_ir_same_dir'])
            self.assertEqual(summary['splits']['train']['empty_labels'], 0)
            self.assertEqual(summary['splits']['val']['empty_labels'], 0)

            train_label = (out_root / 'train' / 'labels' / 'merged' / 'sample.txt').read_text(encoding='utf-8').strip()
            val_label = (out_root / 'val' / 'labels' / 'merged' / 'sample.txt').read_text(encoding='utf-8').strip()
            self.assertTrue(train_label)
            self.assertTrue(val_label)

    def test_polygon_xml_is_converted_to_yolo_obb(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / 'polygon.xml'
            self._write_polygon_xml(xml_path)
            lines = parse_xml_to_yolo_obb(xml_path)

        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('0 '))

    def test_polygon_class_alias_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / 'polygon_alias.xml'
            self._write_polygon_xml(xml_path, class_name='feright car')
            lines = parse_xml_to_yolo_obb(xml_path)

        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('4 '))
