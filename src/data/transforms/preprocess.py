"""Dataset preprocessing helpers reused by the formal dataset preparation CLI.

This module intentionally stays lightweight and file-oriented. Training-time
augmentation belongs in `augmentations.py` and should not be added here.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon


CLASS_MAP = {'car': 0, 'truck': 1, 'bus': 2, 'van': 3, 'freight_car': 4}
CLASS_NAME_ALIASES = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'van': 'van',
    'freight_car': 'freight_car',
    'freight car': 'freight_car',
    'feright car': 'freight_car',
    'feright_car': 'freight_car',
}


def obb_to_polygon(cx, cy, w, h, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2, h / 2
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return Polygon(np.dot(pts, rotation.T) + np.array([cx, cy], dtype=np.float32))


def _normalize_class_name(name):
    if not name:
        return None
    canonical = str(name).strip().lower().replace('-', ' ').replace('_', ' ')
    canonical = ' '.join(canonical.split())
    if canonical in CLASS_NAME_ALIASES:
        return CLASS_NAME_ALIASES[canonical]
    return CLASS_NAME_ALIASES.get(canonical.replace(' ', '_'))


def _parse_polygon_points(obj):
    polygon = obj.find('polygon')
    if polygon is None:
        return None

    points = []
    for index in range(1, 5):
        x_text = polygon.findtext(f'x{index}')
        y_text = polygon.findtext(f'y{index}')
        if x_text is None or y_text is None:
            return None
        points.append([float(x_text), float(y_text)])
    return np.asarray(points, dtype=np.float32)


def _extract_object_obb(obj):
    robndbox = obj.find('robndbox')
    if robndbox is not None:
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        theta = float(robndbox.find('angle').text)
    else:
        points = _parse_polygon_points(obj)
        if points is None or len(points) < 4:
            return None
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(points)
        theta = np.deg2rad(angle_deg)
        if h > w:
            w, h = h, w
            theta += np.pi / 2

    while theta >= np.pi / 2:
        theta -= np.pi
    while theta < -np.pi / 2:
        theta += np.pi
    return float(cx), float(cy), float(w), float(h), float(theta)


def _extract_object_bounds(obj):
    points = _parse_polygon_points(obj)
    if points is not None:
        return (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )

    obb = _extract_object_obb(obj)
    if obb is None:
        return None
    cx, cy, w, h, _ = obb
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _as_content_mask(image, low_tol=10, high_tol=245):
    """Return a mask that excludes both near-black and near-white background."""
    array = np.asarray(image)
    if array.ndim == 3:
        near_black = (array <= low_tol).all(axis=2)
        near_white = (array >= high_tol).all(axis=2)
    else:
        near_black = array <= low_tol
        near_white = array >= high_tol
    return ~(near_black | near_white)


def _find_stable_span(counts, min_pixels, window=3, min_hits=2):
    valid = counts >= min_pixels
    if not valid.any():
        return None

    kernel = np.ones(window, dtype=np.int32)
    smoothed = np.convolve(valid.astype(np.int32), kernel, mode='same')
    stable = smoothed >= min_hits
    indices = np.where(stable)[0]
    if indices.size == 0:
        indices = np.where(valid)[0]
    if indices.size == 0:
        return None
    return int(indices[0]), int(indices[-1])


def _mask_to_bounds(mask):
    """Convert a noisy content mask to a stable bbox for border trimming."""
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.sum() == 0:
        return None

    # Remove tiny isolated speckles so a few noisy pixels do not pin the crop
    # to the image border and leave residual white margins.
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    if cleaned.sum() == 0:
        cleaned = mask_uint8

    height, width = cleaned.shape
    min_row_pixels = max(5, int(round(width * 0.01)))
    min_col_pixels = max(5, int(round(height * 0.01)))

    row_counts = cleaned.sum(axis=1)
    col_counts = cleaned.sum(axis=0)
    row_span = _find_stable_span(row_counts, min_row_pixels)
    col_span = _find_stable_span(col_counts, min_col_pixels)

    if row_span is None or col_span is None:
        coords = np.argwhere(cleaned > 0)
        if coords.shape[0] == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return int(x_min), int(y_min), int(x_max), int(y_max)

    y_min, y_max = row_span
    x_min, x_max = col_span
    return int(x_min), int(y_min), int(x_max), int(y_max)


def _expand_bbox_by_xml(xml_path, current_bounds, image_width, image_height, padding):
    if xml_path is None:
        return current_bounds
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return current_bounds

    x_min, y_min, x_max, y_max = current_bounds
    tree = ET.parse(xml_path)
    for obj in tree.getroot().findall('object'):
        bounds = _extract_object_bounds(obj)
        if bounds is None:
            continue
        obj_x_min, obj_y_min, obj_x_max, obj_y_max = bounds
        x_min = min(x_min, max(0.0, obj_x_min - padding))
        y_min = min(y_min, max(0.0, obj_y_min - padding))
        x_max = max(x_max, min(image_width - 1.0, obj_x_max + padding))
        y_max = max(y_max, min(image_height - 1.0, obj_y_max + padding))

    return int(np.floor(x_min)), int(np.floor(y_min)), int(np.ceil(x_max)), int(np.ceil(y_max))


def get_dm_sop_crop_bbox(
    img_rgb,
    img_ir,
    xml_rgb,
    xml_ir,
    low_tol=10,
    high_tol=245,
    padding=20,
):
    """Compute a joint RGB/IR crop bbox that removes white and black borders."""
    height, width = img_rgb.shape[:2]

    mask_rgb = _as_content_mask(img_rgb, low_tol=low_tol, high_tol=high_tol)
    mask_ir = _as_content_mask(img_ir, low_tol=low_tol, high_tol=high_tol)
    mask_union = mask_rgb | mask_ir

    bounds = _mask_to_bounds(mask_union)
    if bounds is None:
        return 0, 0, width - 1, height - 1

    bounds = _expand_bbox_by_xml(xml_rgb, bounds, width, height, padding)
    bounds = _expand_bbox_by_xml(xml_ir, bounds, width, height, padding)

    x_min, y_min, x_max, y_max = bounds
    x_min = max(0, min(int(x_min), width - 1))
    y_min = max(0, min(int(y_min), height - 1))
    x_max = max(0, min(int(x_max), width - 1))
    y_max = max(0, min(int(y_max), height - 1))

    if x_max <= x_min or y_max <= y_min:
        return 0, 0, width - 1, height - 1
    return x_min, y_min, x_max, y_max


def fit_crop_bbox_to_target_size(
    x_min,
    y_min,
    x_max,
    y_max,
    image_width,
    image_height,
    target_width=640,
    target_height=512,
):
    """Fit a crop bbox to a fixed output size while staying inside the image."""
    crop_width = min(int(target_width), int(image_width))
    crop_height = min(int(target_height), int(image_height))

    center_x = (float(x_min) + float(x_max)) / 2.0
    center_y = (float(y_min) + float(y_max)) / 2.0

    left = int(round(center_x - crop_width / 2.0))
    top = int(round(center_y - crop_height / 2.0))

    max_left = max(0, int(image_width) - crop_width)
    max_top = max(0, int(image_height) - crop_height)
    left = min(max(left, 0), max_left)
    top = min(max(top, 0), max_top)

    right = left + crop_width - 1
    bottom = top + crop_height - 1
    return left, top, right, bottom


def cmlc_nms_fusion(yolo_lines_rgb, yolo_lines_ir, iou_thresh=0.4):
    all_boxes = [{'line': line, 'source': 'rgb'} for line in yolo_lines_rgb] + [
        {'line': line, 'source': 'ir'} for line in yolo_lines_ir
    ]
    if not all_boxes:
        return []

    parsed_boxes = []
    for item in all_boxes:
        parts = item['line'].split()
        cx, cy, w, h, theta = map(float, parts[1:])
        poly = obb_to_polygon(cx, cy, w, h, theta)
        parsed_boxes.append(
            {
                'cls': int(parts[0]),
                'poly': poly,
                'area': poly.area,
                'box_data': np.array([cx, cy, w, h, theta], dtype=np.float32),
                'source': item['source'],
            }
        )

    parsed_boxes.sort(key=lambda item: item['area'], reverse=True)
    keep_lines = []

    while parsed_boxes:
        current = parsed_boxes.pop(0)
        fused_box = current['box_data']
        weight = 1.0
        remaining = []

        for other in parsed_boxes:
            if current['cls'] != other['cls'] or not current['poly'].intersects(other['poly']):
                remaining.append(other)
                continue

            inter_area = current['poly'].intersection(other['poly']).area
            union_area = current['poly'].area + other['poly'].area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0

            if iou > iou_thresh:
                angle_diff = abs(fused_box[4] - other['box_data'][4])
                if angle_diff > np.pi / 2 - 0.1:
                    other['box_data'][4] = fused_box[4]
                fused_box = (fused_box * weight + other['box_data']) / (weight + 1.0)
                weight += 1.0
            else:
                remaining.append(other)

        keep_lines.append(
            f"{current['cls']} {fused_box[0]:.6f} {fused_box[1]:.6f} {fused_box[2]:.6f} {fused_box[3]:.6f} {fused_box[4]:.6f}"
        )
        parsed_boxes = remaining

    return keep_lines


def parse_xml_to_yolo_obb(xml_path, x_offset=0, y_offset=0, crop_w=None, crop_h=None):
    tree = ET.parse(xml_path)
    yolo_lines = []
    for obj in tree.getroot().findall('object'):
        cls_name = _normalize_class_name(obj.findtext('name'))
        if cls_name not in CLASS_MAP:
            continue

        obb = _extract_object_obb(obj)
        if obb is None:
            continue
        cx, cy, w, h, angle = obb
        cx -= x_offset
        cy -= y_offset

        if crop_w and crop_h:
            cx = cx / crop_w
            cy = cy / crop_h
            w = w / crop_w
            h = h / crop_h

        yolo_lines.append(f"{CLASS_MAP[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.6f}")
    return yolo_lines
