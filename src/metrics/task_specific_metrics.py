import re
from pathlib import Path

import numpy as np


def resolve_small_area_threshold(area_threshold):
    area_threshold = float(area_threshold)
    return max(area_threshold, 0.0) ** 2


def bbox_area(obb):
    return max(float(obb[2]), 0.0) * max(float(obb[3]), 0.0)


def is_small_bbox(obb, area_threshold):
    return bbox_area(obb) < float(area_threshold)


def angle_distance(theta_a, theta_b):
    return abs(((theta_a - theta_b + np.pi / 2.0) % np.pi) - np.pi / 2.0)


def infer_sequence_metadata(image_id):
    path = Path(str(image_id))
    stem = path.stem
    match = re.match(r'(.+?)[-_]?(\d+)$', stem)
    if not match:
        return None

    seq_stem = match.group(1).rstrip('_-')
    frame_index = int(match.group(2))
    parent = path.parent.name
    sequence_id = f"{parent}/{seq_stem}" if parent else seq_stem
    return {
        'sequence_id': sequence_id,
        'frame_index': frame_index,
    }


def compute_small_object_metrics(preds, gts, num_classes, area_threshold, iou_threshold=0.5, match_iou_fn=None):
    if match_iou_fn is None:
        raise ValueError('match_iou_fn must be provided for small-object metrics.')

    small_gt_count = 0
    small_tp_count = 0
    small_fp_count = 0

    for class_id in range(num_classes):
        class_preds = sorted([p for p in preds if p['class'] == class_id], key=lambda x: x['score'], reverse=True)
        class_small_gts = [g for g in gts if g['class'] == class_id and is_small_bbox(g['bbox'], area_threshold)]
        small_gt_count += len(class_small_gts)

        gt_by_image = {}
        for gt in class_small_gts:
            image_entry = gt_by_image.setdefault(gt['image_id'], {'bboxes': [], 'matched': []})
            image_entry['bboxes'].append(gt['bbox'])
            image_entry['matched'].append(False)

        for pred in class_preds:
            pred_is_small = is_small_bbox(pred['bbox'], area_threshold)
            image_entry = gt_by_image.get(pred['image_id'])
            if image_entry is None:
                if pred_is_small:
                    small_fp_count += 1
                continue

            best_iou = 0.0
            best_idx = -1
            for gt_idx, gt_bbox in enumerate(image_entry['bboxes']):
                if image_entry['matched'][gt_idx]:
                    continue
                iou = match_iou_fn(pred['bbox'], gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx

            if best_iou >= iou_threshold and best_idx >= 0:
                image_entry['matched'][best_idx] = True
                small_tp_count += 1
            elif pred_is_small:
                small_fp_count += 1

    recall_s = small_tp_count / max(small_gt_count, 1)
    precision_s = small_tp_count / max(small_tp_count + small_fp_count, 1)
    return {
        'Recall_S': float(recall_s),
        'Precision_S': float(precision_s),
        'small_gt_count': int(small_gt_count),
        'small_tp_count': int(small_tp_count),
        'small_fp_count': int(small_fp_count),
    }


def compute_cross_modal_robustness(base_metrics, rgb_drop_metrics, ir_drop_metrics, base_metric='mAP_50'):
    baseline_value = float(base_metrics.get(base_metric, 0.0))
    rgb_drop_value = float(rgb_drop_metrics.get(base_metric, 0.0))
    ir_drop_value = float(ir_drop_metrics.get(base_metric, 0.0))

    return {
        'CrossModalRobustness_BaseMetric': base_metric,
        'RGBDrop_mAP50': rgb_drop_value,
        'IRDrop_mAP50': ir_drop_value,
        'CrossModalRobustness_RGBDrop': baseline_value - rgb_drop_value,
        'CrossModalRobustness_IRDrop': baseline_value - ir_drop_value,
    }


def compute_temporal_stability(
    preds,
    image_metadata,
    conf_threshold=0.25,
    match_iou_threshold=0.3,
    max_center_shift_ratio=0.1,
    match_iou_fn=None,
):
    if match_iou_fn is None:
        raise ValueError('match_iou_fn must be provided for temporal stability.')
    if not image_metadata:
        return None

    grouped = {}
    preds_by_image = {}
    for pred in preds:
        if pred['score'] < conf_threshold:
            continue
        preds_by_image.setdefault(pred['image_id'], []).append(pred)

    for image_id, metadata in image_metadata.items():
        if not metadata:
            continue
        sequence_id = metadata.get('sequence_id')
        frame_index = metadata.get('frame_index')
        if sequence_id is None or frame_index is None:
            continue
        grouped.setdefault(sequence_id, []).append((frame_index, preds_by_image.get(image_id, [])))

    pair_scores = []
    for sequence_frames in grouped.values():
        if len(sequence_frames) < 2:
            continue
        ordered_frames = sorted(sequence_frames, key=lambda x: x[0])
        for (prev_idx, prev_preds), (curr_idx, curr_preds) in zip(ordered_frames[:-1], ordered_frames[1:]):
            del prev_idx, curr_idx
            prev_used = [False] * len(prev_preds)
            matched_pairs = []

            for curr_pred in curr_preds:
                best_iou = 0.0
                best_match_idx = -1
                for pred_idx, prev_pred in enumerate(prev_preds):
                    if prev_used[pred_idx] or prev_pred['class'] != curr_pred['class']:
                        continue
                    iou = match_iou_fn(curr_pred['bbox'], prev_pred['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = pred_idx
                if best_iou >= match_iou_threshold and best_match_idx >= 0:
                    prev_used[best_match_idx] = True
                    matched_pairs.append((prev_preds[best_match_idx], curr_pred))

            denom = max(len(prev_preds), len(curr_preds), 1)
            matched_ratio = len(matched_pairs) / denom

            shift_scores = []
            angle_scores = []
            for prev_pred, curr_pred in matched_pairs:
                prev_bbox = prev_pred['bbox']
                curr_bbox = curr_pred['bbox']
                center_shift = np.linalg.norm(np.array(curr_bbox[:2]) - np.array(prev_bbox[:2]))
                norm = max(np.sqrt(max(bbox_area(prev_bbox), 1e-6)), 1e-6)
                shift_scores.append(max(0.0, 1.0 - center_shift / max(max_center_shift_ratio * norm, 1e-6)))

                delta_theta = angle_distance(curr_bbox[4], prev_bbox[4])
                angle_scores.append(max(0.0, 1.0 - delta_theta / (np.pi / 2.0)))

            shift_term = float(np.mean(shift_scores)) if shift_scores else 0.0
            angle_term = float(np.mean(angle_scores)) if angle_scores else 0.0
            pair_scores.append(0.6 * matched_ratio + 0.25 * shift_term + 0.15 * angle_term)

    if not pair_scores:
        return None
    return float(np.mean(pair_scores))


def append_task_metrics(base_metrics, extra_metrics):
    merged = dict(base_metrics)
    for key, value in (extra_metrics or {}).items():
        merged[key] = value
    return merged
