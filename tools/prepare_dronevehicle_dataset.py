"""Prepare raw DroneVehicle data into the DroneDualDataset training layout.

This script intentionally reuses the existing preprocessing helpers in
`src.data.transforms.preprocess` for:
- joint valid-region cropping
- XML to YOLO OBB conversion
- RGB / IR label fusion

Training-time augmentation is intentionally not part of this script.
"""

import argparse
import json
import logging
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms.preprocess import (  # noqa: E402
    cmlc_nms_fusion,
    get_dm_sop_crop_bbox,
    parse_xml_to_yolo_obb,
)


IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
MISSING_XML_PATH = Path('__missing__.xml')

RGB_DIR_CANDIDATES = {
    'train': ['training', 'trainimg', 'img', 'rgb', 'visible', 'images'],
    'val': ['validation', 'training', 'valimg', 'img', 'rgb', 'visible', 'images'],
    'test': ['testimg', 'img', 'rgb', 'visible', 'images'],
}
IR_DIR_CANDIDATES = {
    'train': ['trainingr', 'trainimgr', 'imgr', 'ir', 'infrared', 'images_ir'],
    'val': ['validationr', 'trainingr', 'valimgr', 'imgr', 'ir', 'infrared', 'images_ir'],
    'test': ['testimgr', 'imgr', 'ir', 'infrared', 'images_ir'],
}
RGB_LABEL_DIR_CANDIDATES = {
    'train': ['trainlabel', 'label', 'labels', 'labelrgb', 'labels_rgb'],
    'val': ['vallabel', 'validationlabel', 'label', 'labels', 'labelrgb', 'labels_rgb'],
    'test': ['testlabel', 'label', 'labels', 'labelrgb', 'labels_rgb'],
}
IR_LABEL_DIR_CANDIDATES = {
    'train': ['trainlabelr', 'labelr', 'label_ir', 'labels_ir', 'labelsr'],
    'val': ['vallabelr', 'validationlabelr', 'labelr', 'label_ir', 'labels_ir', 'labelsr'],
    'test': ['testlabelr', 'labelr', 'label_ir', 'labels_ir', 'labelsr'],
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Prepare raw DroneVehicle data into the DroneDualDataset directory structure.'
    )
    parser.add_argument(
        '--raw-root',
        required=False,
        type=str,
        default='D:/DataSet/DroneVehicle',
        help='Root directory of the raw DroneVehicle dataset.',
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='D:/DataSet/DroneVehicle_Processed',
        help='Output root for the processed dataset.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='Dataset splits to prepare. Example: --splits train val',
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing processed split directories.')
    parser.add_argument('--rgb-dir', type=str, default='', help='Optional explicit raw RGB image directory.')
    parser.add_argument('--ir-dir', type=str, default='', help='Optional explicit raw IR image directory.')
    parser.add_argument('--rgb-label-dir', type=str, default='', help='Optional explicit raw RGB XML directory.')
    parser.add_argument('--ir-label-dir', type=str, default='', help='Optional explicit raw IR XML directory.')
    parser.add_argument('--low-tol', type=int, default=10, help='Near-black threshold for valid-content masking.')
    parser.add_argument('--high-tol', type=int, default=245, help='Near-white threshold for valid-content masking.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser


def _configure_logging(level: str = 'INFO') -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='[%(levelname)s] %(message)s',
    )


def _candidate_roots(raw_root: Path, split: str) -> List[Path]:
    split_root = raw_root / split
    if split_root.is_dir():
        return [split_root, raw_root]
    return [raw_root]


def _find_existing_dir(raw_root: Path, split: str, override: str, candidates: Sequence[str]) -> Optional[Path]:
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = raw_root / override
        if path.is_dir():
            return path
        raise FileNotFoundError(f'Override directory does not exist: {path}')

    for base_root in _candidate_roots(raw_root, split):
        for candidate in candidates:
            path = base_root / candidate
            if path.is_dir():
                return path
    return None


def _index_files(directory: Optional[Path], suffixes: Iterable[str]) -> Dict[str, Path]:
    if directory is None or not directory.exists():
        return {}
    indexed: Dict[str, Path] = {}
    suffixes = tuple(s.lower() for s in suffixes)
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            indexed[path.stem.lower()] = path
    return indexed


def _safe_xml_path(path: Optional[Path], summary: Dict[str, int], counter_key: str) -> Path:
    if path is None or not path.exists():
        summary[counter_key] += 1
        return MISSING_XML_PATH
    try:
        ET.parse(path)
        return path
    except ET.ParseError:
        logging.warning('Bad XML skipped: %s', path)
        summary['bad_xml'] += 1
        return MISSING_XML_PATH


def _safe_parse_xml(
    xml_path: Path,
    x_offset: int,
    y_offset: int,
    crop_w: int,
    crop_h: int,
    summary: Dict[str, int],
) -> List[str]:
    if xml_path == MISSING_XML_PATH or not xml_path.exists():
        return []
    try:
        return parse_xml_to_yolo_obb(
            xml_path,
            x_offset=x_offset,
            y_offset=y_offset,
            crop_w=crop_w,
            crop_h=crop_h,
        )
    except ET.ParseError:
        logging.warning('Bad XML during parse skipped: %s', xml_path)
        summary['bad_xml'] += 1
        return []


def _prepare_output_dirs(output_root: Path, split: str, overwrite: bool = False) -> Dict[str, Path]:
    split_root = output_root / split
    if split_root.exists() and overwrite:
        shutil.rmtree(split_root)
    img_rgb_dir = split_root / 'images' / 'img'
    img_ir_dir = split_root / 'images' / 'imgr'
    label_dir = split_root / 'labels' / 'merged'
    img_rgb_dir.mkdir(parents=True, exist_ok=True)
    img_ir_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return {
        'split_root': split_root,
        'img_rgb_dir': img_rgb_dir,
        'img_ir_dir': img_ir_dir,
        'label_dir': label_dir,
    }


def _resolve_split_sources(
    raw_root: Path,
    split: str,
    rgb_dir: str = '',
    ir_dir: str = '',
    rgb_label_dir: str = '',
    ir_label_dir: str = '',
) -> Dict[str, Optional[Path]]:
    rgb_path = _find_existing_dir(raw_root, split, rgb_dir, RGB_DIR_CANDIDATES.get(split, ['img', 'rgb', 'images']))
    ir_path = _find_existing_dir(raw_root, split, ir_dir, IR_DIR_CANDIDATES.get(split, ['imgr', 'ir', 'infrared']))
    rgb_label_path = _find_existing_dir(
        raw_root,
        split,
        rgb_label_dir,
        RGB_LABEL_DIR_CANDIDATES.get(split, ['label', 'labels']),
    )
    ir_label_path = _find_existing_dir(
        raw_root,
        split,
        ir_label_dir,
        IR_LABEL_DIR_CANDIDATES.get(split, ['labelr', 'labels_ir']),
    )
    if rgb_path is None:
        raise FileNotFoundError(f'Could not resolve raw RGB directory for split `{split}` under {raw_root}')
    if ir_path is None:
        raise FileNotFoundError(f'Could not resolve raw IR directory for split `{split}` under {raw_root}')

    same_image_dir = rgb_path.resolve() == ir_path.resolve()
    if same_image_dir:
        logging.warning(
            'RGB and IR image directories resolved to the same path for split `%s`: %s',
            split,
            rgb_path,
        )

    return {
        'rgb_dir': rgb_path,
        'ir_dir': ir_path,
        'rgb_label_dir': rgb_label_path,
        'ir_label_dir': ir_label_path,
        'rgb_ir_same_dir': same_image_dir,
    }


def prepare_split(
    raw_root: Path,
    output_root: Path,
    split: str,
    overwrite: bool = False,
    rgb_dir: str = '',
    ir_dir: str = '',
    rgb_label_dir: str = '',
    ir_label_dir: str = '',
    low_tol: int = 10,
    high_tol: int = 245,
) -> Dict[str, object]:
    sources = _resolve_split_sources(
        raw_root=raw_root,
        split=split,
        rgb_dir=rgb_dir,
        ir_dir=ir_dir,
        rgb_label_dir=rgb_label_dir,
        ir_label_dir=ir_label_dir,
    )
    outputs = _prepare_output_dirs(output_root, split, overwrite=overwrite)

    rgb_index = _index_files(sources['rgb_dir'], IMAGE_SUFFIXES)
    ir_index = _index_files(sources['ir_dir'], IMAGE_SUFFIXES)
    rgb_xml_index = _index_files(sources['rgb_label_dir'], ('.xml',))
    ir_xml_index = _index_files(sources['ir_label_dir'], ('.xml',))

    summary: Dict[str, object] = {
        'split': split,
        'rgb_dir': '' if sources['rgb_dir'] is None else str(sources['rgb_dir']),
        'ir_dir': '' if sources['ir_dir'] is None else str(sources['ir_dir']),
        'rgb_label_dir': '' if sources['rgb_label_dir'] is None else str(sources['rgb_label_dir']),
        'ir_label_dir': '' if sources['ir_label_dir'] is None else str(sources['ir_label_dir']),
        'rgb_ir_same_dir': bool(sources.get('rgb_ir_same_dir', False)),
        'found_rgb_images': len(rgb_index),
        'found_ir_images': len(ir_index),
        'processed_pairs': 0,
        'missing_ir': 0,
        'missing_rgb_label': 0,
        'missing_ir_label': 0,
        'bad_xml': 0,
        'bad_image': 0,
        'empty_labels': 0,
        'warnings': 0,
        'crop_fallbacks': 0,
    }

    logging.info('Preparing split `%s` from %s', split, raw_root)
    logging.info('  RGB dir: %s', sources['rgb_dir'])
    logging.info('  IR dir: %s', sources['ir_dir'])
    logging.info('  RGB label dir: %s', sources['rgb_label_dir'])
    logging.info('  IR label dir: %s', sources['ir_label_dir'])
    logging.info('  RGB/IR same dir: %s', summary['rgb_ir_same_dir'])

    if summary['rgb_ir_same_dir']:
        summary['warnings'] += 1

    for stem, rgb_path in rgb_index.items():
        ir_path = ir_index.get(stem)
        if ir_path is None:
            logging.warning('Missing IR pair for RGB image: %s', rgb_path.name)
            summary['missing_ir'] += 1
            summary['warnings'] += 1
            continue

        img_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        img_ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
        if img_rgb is None or img_ir is None:
            logging.warning('Failed to read image pair: %s / %s', rgb_path, ir_path)
            summary['bad_image'] += 1
            summary['warnings'] += 1
            continue

        rgb_xml_path = _safe_xml_path(rgb_xml_index.get(stem), summary, 'missing_rgb_label')
        ir_xml_path = _safe_xml_path(ir_xml_index.get(stem), summary, 'missing_ir_label')
        if rgb_xml_path == MISSING_XML_PATH or ir_xml_path == MISSING_XML_PATH:
            summary['warnings'] += 1

        full_width = img_rgb.shape[1]
        full_height = img_rgb.shape[0]
        fallback = False
        try:
            x_min, y_min, x_max, y_max = get_dm_sop_crop_bbox(
                img_rgb,
                img_ir,
                rgb_xml_path,
                ir_xml_path,
                low_tol=low_tol,
                high_tol=high_tol,
            )
        except ET.ParseError:
            logging.warning('Crop skipped due to bad XML for sample `%s`; using full image.', stem)
            summary['bad_xml'] += 1
            x_min, y_min = 0, 0
            x_max, y_max = full_width - 1, full_height - 1
            fallback = True

        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(int(x_max), full_width - 1)
        y_max = min(int(y_max), full_height - 1)
        if x_max <= x_min or y_max <= y_min:
            x_min, y_min = 0, 0
            x_max, y_max = full_width - 1, full_height - 1
            fallback = True

        if x_min == 0 and y_min == 0 and x_max == full_width - 1 and y_max == full_height - 1:
            fallback = True
        if fallback:
            summary['crop_fallbacks'] += 1

        cropped_rgb = img_rgb[y_min:y_max + 1, x_min:x_max + 1]
        cropped_ir = img_ir[y_min:y_max + 1, x_min:x_max + 1]
        crop_h, crop_w = cropped_rgb.shape[:2]

        logging.info(
            'Sample `%s`: original=%sx%s crop=(%s,%s,%s,%s) cropped=%sx%s fallback=%s',
            stem,
            full_width,
            full_height,
            x_min,
            y_min,
            x_max,
            y_max,
            crop_w,
            crop_h,
            fallback,
        )

        rgb_lines = _safe_parse_xml(rgb_xml_path, x_min, y_min, crop_w, crop_h, summary)
        ir_lines = _safe_parse_xml(ir_xml_path, x_min, y_min, crop_w, crop_h, summary)
        merged_lines = cmlc_nms_fusion(rgb_lines, ir_lines)
        if not merged_lines:
            summary['empty_labels'] += 1

        out_name = f'{Path(rgb_path).stem}.jpg'
        out_rgb_path = outputs['img_rgb_dir'] / out_name
        out_ir_path = outputs['img_ir_dir'] / out_name
        out_label_path = outputs['label_dir'] / f'{Path(rgb_path).stem}.txt'

        cv2.imwrite(str(out_rgb_path), cropped_rgb)
        cv2.imwrite(str(out_ir_path), cropped_ir)
        out_label_path.write_text('\n'.join(merged_lines) + ('\n' if merged_lines else ''), encoding='utf-8')

        summary['processed_pairs'] += 1

    logging.info(
        'Finished split `%s`: processed=%s missing_ir=%s bad_xml=%s empty_labels=%s crop_fallbacks=%s',
        split,
        summary['processed_pairs'],
        summary['missing_ir'],
        summary['bad_xml'],
        summary['empty_labels'],
        summary['crop_fallbacks'],
    )
    return summary


def prepare_dataset(
    raw_root: str,
    output_root: str,
    splits: Sequence[str],
    overwrite: bool = False,
    rgb_dir: str = '',
    ir_dir: str = '',
    rgb_label_dir: str = '',
    ir_label_dir: str = '',
    low_tol: int = 10,
    high_tol: int = 245,
) -> Dict[str, object]:
    raw_root_path = Path(raw_root)
    output_root_path = Path(output_root)
    if not raw_root_path.exists():
        raise FileNotFoundError(f'Raw dataset root does not exist: {raw_root_path}')

    overall = {
        'raw_root': str(raw_root_path),
        'output_root': str(output_root_path),
        'splits': {},
        'total_processed_pairs': 0,
        'total_warnings': 0,
        'total_bad_xml': 0,
        'total_empty_labels': 0,
        'total_crop_fallbacks': 0,
    }

    for split in splits:
        split_summary = prepare_split(
            raw_root=raw_root_path,
            output_root=output_root_path,
            split=split,
            overwrite=overwrite,
            rgb_dir=rgb_dir,
            ir_dir=ir_dir,
            rgb_label_dir=rgb_label_dir,
            ir_label_dir=ir_label_dir,
            low_tol=low_tol,
            high_tol=high_tol,
        )
        overall['splits'][split] = split_summary
        overall['total_processed_pairs'] += int(split_summary['processed_pairs'])
        overall['total_warnings'] += int(split_summary['warnings'])
        overall['total_bad_xml'] += int(split_summary['bad_xml'])
        overall['total_empty_labels'] += int(split_summary['empty_labels'])
        overall['total_crop_fallbacks'] += int(split_summary['crop_fallbacks'])

    return overall


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    summary = prepare_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        splits=args.splits,
        overwrite=args.overwrite,
        rgb_dir=args.rgb_dir,
        ir_dir=args.ir_dir,
        rgb_label_dir=args.rgb_label_dir,
        ir_label_dir=args.ir_label_dir,
        low_tol=args.low_tol,
        high_tol=args.high_tol,
    )
    logging.info('Dataset preparation summary\n%s', json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
