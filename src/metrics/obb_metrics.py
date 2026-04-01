import warnings

import numpy as np
from shapely.geometry import Polygon

from src.metrics.grouped_metrics import compute_grouped_metrics
from src.metrics.task_metrics import normalize_eval_metrics_cfg
from src.metrics.task_specific_metrics import (
    compute_small_object_metrics,
    compute_temporal_stability,
    resolve_small_area_threshold,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def obb2polygon(obb):
    cx, cy, w, h, theta = obb
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2.0, h / 2.0
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    pts = pts @ rotation.T + np.array([cx, cy])
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return None


def hbb_prescreening(obb1, obb2):
    """Horizontal bounding-box prescreening for faster OBB IoU."""
    cx1, cy1, w1, h1, t1 = obb1
    cx2, cy2, w2, h2, t2 = obb2
    w1h = w1 * abs(np.cos(t1)) + h1 * abs(np.sin(t1))
    h1h = w1 * abs(np.sin(t1)) + h1 * abs(np.cos(t1))
    w2h = w2 * abs(np.cos(t2)) + h2 * abs(np.sin(t2))
    h2h = w2 * abs(np.sin(t2)) + h2 * abs(np.cos(t2))
    if abs(cx1 - cx2) > (w1h + w2h) / 2:
        return False
    if abs(cy1 - cy2) > (h1h + h2h) / 2:
        return False
    return True


def polygon_iou(obb1, obb2, poly_cache):
    if not hbb_prescreening(obb1, obb2):
        return 0.0
    key1, key2 = tuple(obb1), tuple(obb2)
    if key1 not in poly_cache:
        poly_cache[key1] = obb2polygon(obb1)
    if key2 not in poly_cache:
        poly_cache[key2] = obb2polygon(obb2)
    poly1, poly2 = poly_cache[key1], poly_cache[key2]
    if poly1 is None or poly2 is None:
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


class OBBMetricsEvaluator:
    def __init__(self, num_classes=5, small_area_thresh=1024, extra_metrics_cfg=None):
        self.nc = num_classes
        self.default_small_area_thresh = small_area_thresh
        self.extra_metrics_cfg = normalize_eval_metrics_cfg(extra_metrics_cfg)
        self.small_area_thresh = resolve_small_area_threshold(
            self.extra_metrics_cfg['small_object'].get('area_threshold', np.sqrt(float(small_area_thresh)))
        )
        self.reset()

    def reset(self):
        self.preds = []
        self.gts = []
        self.image_metadata = {}

    def add_batch(self, image_ids, batch_preds, batch_gts, batch_metadata=None):
        batch_metadata = batch_metadata or [None] * len(image_ids)
        for i, img_id in enumerate(image_ids):
            self.image_metadata[img_id] = batch_metadata[i]
            if len(batch_preds[i]) > 0:
                for pred in batch_preds[i].cpu().numpy():
                    self.preds.append({
                        'image_id': img_id,
                        'bbox': pred[:5],
                        'score': float(pred[5]),
                        'class': int(pred[6]),
                    })
            if len(batch_gts[i]) > 0:
                for gt in batch_gts[i].cpu().numpy():
                    self.gts.append({
                        'image_id': img_id,
                        'class': int(gt[0]),
                        'bbox': gt[1:6],
                    })

    def _compute_detection_metrics(self, preds, gts, iou_thresh=0.5, eval_small_only=False):
        aps = []
        precisions = []
        recalls = []
        poly_cache = {}

        for class_id in range(self.nc):
            class_preds = sorted([p for p in preds if p['class'] == class_id], key=lambda x: x['score'], reverse=True)
            class_gts = [g for g in gts if g['class'] == class_id]
            if eval_small_only:
                class_gts = [g for g in class_gts if (g['bbox'][2] * g['bbox'][3]) < self.small_area_thresh]

            num_gts = len(class_gts)
            if num_gts == 0:
                continue

            gt_by_image = {}
            for gt in class_gts:
                image_entry = gt_by_image.setdefault(gt['image_id'], {'bboxes': [], 'matched': []})
                image_entry['bboxes'].append(gt['bbox'])
                image_entry['matched'].append(False)

            num_dets = len(class_preds)
            if num_dets == 0:
                aps.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            tp = np.zeros(num_dets)
            fp = np.zeros(num_dets)
            for pred_idx, pred in enumerate(class_preds):
                image_entry = gt_by_image.get(pred['image_id'])
                if image_entry is None:
                    fp[pred_idx] = 1
                    continue

                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_bbox in enumerate(image_entry['bboxes']):
                    if image_entry['matched'][gt_idx]:
                        continue
                    iou = polygon_iou(pred['bbox'], gt_bbox, poly_cache)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    tp[pred_idx] = 1
                    image_entry['matched'][best_gt_idx] = True
                else:
                    fp[pred_idx] = 1

            fp_cum = np.cumsum(fp)
            tp_cum = np.cumsum(tp)
            recall = tp_cum / (num_gts + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)
            aps.append(compute_ap(recall, precision))
            precisions.append(float(tp.sum() / max(tp.sum() + fp.sum(), 1.0)))
            recalls.append(float(tp.sum() / max(num_gts, 1)))

        return {
            'mAP': float(np.mean(aps)) if aps else 0.0,
            'Precision': float(np.mean(precisions)) if precisions else 0.0,
            'Recall': float(np.mean(recalls)) if recalls else 0.0,
            'poly_cache': poly_cache,
        }

    def evaluate(self, iou_thresh=0.5, eval_small_only=False):
        return self._compute_detection_metrics(self.preds, self.gts, iou_thresh=iou_thresh, eval_small_only=eval_small_only)['mAP']

    def _compute_map_range(self, preds, gts, thresholds=None):
        thresholds = thresholds if thresholds is not None else np.arange(0.5, 1.0, 0.05)
        map_values = [self._compute_detection_metrics(preds, gts, iou_thresh=thr, eval_small_only=False)['mAP'] for thr in thresholds]
        return float(np.mean(map_values)) if map_values else 0.0

    def _compute_metrics_dict(self, preds, gts, image_metadata, include_grouped_metrics=True):
        detection_metrics = self._compute_detection_metrics(preds, gts, iou_thresh=0.5, eval_small_only=False)
        small_cfg = self.extra_metrics_cfg['small_object']
        metrics = {
            'mAP_50': detection_metrics['mAP'],
            'mAP_50_95': self._compute_map_range(preds, gts),
            'Precision': detection_metrics['Precision'],
            'Recall': detection_metrics['Recall'],
        }
        if small_cfg.get('enabled', True):
            metrics['mAP_S'] = self._compute_detection_metrics(preds, gts, iou_thresh=0.5, eval_small_only=True)['mAP']

        poly_cache = detection_metrics['poly_cache']

        def match_iou_fn(pred_box, gt_box):
            return polygon_iou(pred_box, gt_box, poly_cache)

        if small_cfg.get('enabled', True):
            small_metrics = compute_small_object_metrics(
                preds,
                gts,
                num_classes=self.nc,
                area_threshold=resolve_small_area_threshold(small_cfg.get('area_threshold', 32)),
                iou_threshold=small_cfg.get('iou_threshold', 0.5),
                match_iou_fn=match_iou_fn,
            )
            metrics['Recall_S'] = small_metrics['Recall_S']
            metrics['Precision_S'] = small_metrics['Precision_S']

        temporal_cfg = self.extra_metrics_cfg['temporal_stability']
        temporal_enabled = temporal_cfg.get('enabled', 'auto')
        if temporal_enabled in (True, 'auto'):
            metrics['TemporalStability'] = compute_temporal_stability(
                preds,
                image_metadata,
                conf_threshold=temporal_cfg.get('conf_threshold', 0.25),
                match_iou_threshold=temporal_cfg.get('match_iou_threshold', 0.3),
                max_center_shift_ratio=temporal_cfg.get('max_center_shift_ratio', 0.1),
                match_iou_fn=match_iou_fn,
            )

        if include_grouped_metrics:
            metrics['GroupedMetrics'] = compute_grouped_metrics(
                preds,
                gts,
                image_metadata,
                self.extra_metrics_cfg['group_eval'],
                lambda sub_preds, sub_gts, sub_metadata: self._compute_metrics_dict(
                    sub_preds,
                    sub_gts,
                    sub_metadata,
                    include_grouped_metrics=False,
                ),
            )

        return metrics

    def get_full_metrics(self):
        return self._compute_metrics_dict(self.preds, self.gts, self.image_metadata, include_grouped_metrics=True)
