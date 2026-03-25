# 归一化、Resize
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon

CLASS_MAP = {'car': 0, 'truck': 1, 'bus': 2, 'van': 3, 'freight_car': 4}

def obb_to_polygon(cx, cy, w, h, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2, h / 2
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return Polygon(np.dot(pts, R.T) + np.array([cx, cy]))

def get_dm_sop_crop_bbox(img_rgb, img_ir, xml_rgb, xml_ir, tol=10, padding=20):
    mask_rgb = (img_rgb > tol).any(2) if len(img_rgb.shape) == 3 else img_rgb > tol
    mask_ir = (img_ir > tol).any(2) if len(img_ir.shape) == 3 else img_ir > tol
    mask_union = mask_rgb | mask_ir

    coords = np.argwhere(mask_union)
    if coords.shape[0] == 0:
        return 0, 0, img_rgb.shape[1] - 1, img_rgb.shape[0] - 1

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    def expand_bbox_by_xml(xml_path, current_bounds):
        if not xml_path.exists(): return current_bounds
        x_min, y_min, x_max, y_max = current_bounds
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall('object'):
            robndbox = obj.find('robndbox')
            if robndbox is not None:
                cx, cy = float(robndbox.find('cx').text), float(robndbox.find('cy').text)
                w, h = float(robndbox.find('w').text), float(robndbox.find('h').text)
                x_min = min(x_min, max(0, cx - w / 2 - padding))
                y_min = min(y_min, max(0, cy - h / 2 - padding))
                x_max = max(x_max, min(img_rgb.shape[1], cx + w / 2 + padding))
                y_max = max(y_max, min(img_rgb.shape[0], cy + h / 2 + padding))
        return int(x_min), int(y_min), int(x_max), int(y_max)

    bounds = expand_bbox_by_xml(xml_rgb, (x_min, y_min, x_max, y_max))
    return expand_bbox_by_xml(xml_ir, bounds)

def cmlc_nms_fusion(yolo_lines_rgb, yolo_lines_ir, iou_thresh=0.4):
    all_boxes = [{'line': l, 'source': 'rgb'} for l in yolo_lines_rgb] + \
                [{'line': l, 'source': 'ir'} for l in yolo_lines_ir]
    if not all_boxes: return []

    parsed_boxes = []
    for item in all_boxes:
        parts = item['line'].split()
        cx, cy, w, h, theta = map(float, parts[1:])
        poly = obb_to_polygon(cx, cy, w, h, theta)
        parsed_boxes.append({
            'cls': int(parts[0]), 'poly': poly, 'area': poly.area,
            'box_data': np.array([cx, cy, w, h, theta]), 'source': item['source']
        })

    parsed_boxes.sort(key=lambda x: x['area'], reverse=True)
    keep_lines = []

    while parsed_boxes:
        current = parsed_boxes.pop(0)
        fused_box = current['box_data']
        weight = 1.0
        rem = []

        for other in parsed_boxes:
            if current['cls'] != other['cls'] or not current['poly'].intersects(other['poly']):
                rem.append(other)
                continue

            inter_area = current['poly'].intersection(other['poly']).area
            union_area = current['poly'].area + other['poly'].area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            if iou > iou_thresh:
                angle_diff = abs(fused_box[4] - other['box_data'][4])
                if angle_diff > np.pi / 2 - 0.1:
                    other['box_data'][4] = fused_box[4]
                # 🔥 已修复缩进 Bug: 这一步必须在 if 块内执行
                fused_box = (fused_box * weight + other['box_data']) / (weight + 1.0)
                weight += 1.0
            else:
                rem.append(other)

        keep_lines.append(f"{current['cls']} {fused_box[0]:.6f} {fused_box[1]:.6f} {fused_box[2]:.6f} {fused_box[3]:.6f} {fused_box[4]:.6f}")
        parsed_boxes = rem

    return keep_lines

def parse_xml_to_yolo_obb(xml_path, x_offset=0, y_offset=0, crop_w=None, crop_h=None):
    tree = ET.parse(xml_path)
    yolo_lines = []
    for obj in tree.getroot().findall('object'):
        cls = obj.find('name').text.lower()
        if cls not in CLASS_MAP: continue
        robndbox = obj.find('robndbox')
        if robndbox is None: continue

        cx = float(robndbox.find('cx').text) - x_offset
        cy = float(robndbox.find('cy').text) - y_offset
        w, h = float(robndbox.find('w').text), float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)

        if crop_w and crop_h:
            cx, cy, w, h = cx/crop_w, cy/crop_h, w/crop_w, h/crop_h

        while angle >= np.pi / 2: angle -= np.pi
        while angle < -np.pi / 2: angle += np.pi
        yolo_lines.append(f"{CLASS_MAP[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.6f}")
    return yolo_lines

# (process_dataset 保持不变，供你离线清洗数据使用)