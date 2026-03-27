"""Dataset preprocessing helpers reused by the formal dataset preparation CLI.

This module intentionally stays lightweight and file-oriented. Training-time
augmentation belongs in `augmentations.py` and should not be added here.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


CLASS_MAP = {'car': 0, 'truck': 1, 'bus': 2, 'van': 3, 'freight_car': 4}


def obb_to_polygon(cx, cy, w, h, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2, h / 2
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return Polygon(np.dot(pts, rotation.T) + np.array([cx, cy], dtype=np.float32))


def _as_content_mask(image, low_tol=10, high_tol=245):
    """Return a mask that excludes both near-black and near-white background.

    Stage goal here is practical dataset cleanup rather than semantic image
    segmentation. We treat pixels as valid content only when they are neither
    near-black nor near-white, which fixes the previous white-border miss while
    keeping black-border trimming behavior.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        near_black = (array <= low_tol).all(axis=2)
        near_white = (array >= high_tol).all(axis=2)
    else:
        near_black = array <= low_tol
        near_white = array >= high_tol
    return ~(near_black | near_white)


def _expand_bbox_by_xml(xml_path, current_bounds, image_width, image_height, padding):
    if xml_path is None:
        return current_bounds
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return current_bounds

    x_min, y_min, x_max, y_max = current_bounds
    tree = ET.parse(xml_path)
    for obj in tree.getroot().findall('object'):
        robndbox = obj.find('robndbox')
        if robndbox is None:
            continue
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        x_min = min(x_min, max(0.0, cx - w / 2 - padding))
        y_min = min(y_min, max(0.0, cy - h / 2 - padding))
        x_max = max(x_max, min(image_width - 1.0, cx + w / 2 + padding))
        y_max = max(y_max, min(image_height - 1.0, cy + h / 2 + padding))

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
    """Compute a joint RGB/IR crop bbox that removes white and black borders.

    The crop is driven by the union of valid-content masks from RGB and IR,
    then safely expanded with XML annotations so targets are not trimmed away.
    Empty or abnormal images gracefully fall back to the full image extent.
    """
    height, width = img_rgb.shape[:2]

    mask_rgb = _as_content_mask(img_rgb, low_tol=low_tol, high_tol=high_tol)
    mask_ir = _as_content_mask(img_ir, low_tol=low_tol, high_tol=high_tol)
    mask_union = mask_rgb | mask_ir

    coords = np.argwhere(mask_union)
    if coords.shape[0] == 0:
        return 0, 0, width - 1, height - 1

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    bounds = (int(x_min), int(y_min), int(x_max), int(y_max))
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
        cls = obj.find('name').text.lower()
        if cls not in CLASS_MAP:
            continue

        robndbox = obj.find('robndbox')
        if robndbox is None:
            continue

        cx = float(robndbox.find('cx').text) - x_offset
        cy = float(robndbox.find('cy').text) - y_offset
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)

        if crop_w and crop_h:
            cx = cx / crop_w
            cy = cy / crop_h
            w = w / crop_w
            h = h / crop_h

        while angle >= np.pi / 2:
            angle -= np.pi
        while angle < -np.pi / 2:
            angle += np.pi

        yolo_lines.append(f"{CLASS_MAP[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.6f}")
    return yolo_lines
