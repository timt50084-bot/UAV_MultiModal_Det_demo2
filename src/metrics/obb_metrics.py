# 计算 OBB 的 IoU 和 mAP
import numpy as np
from shapely.geometry import Polygon
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def obb2polygon(obb):
    cx, cy, w, h, theta = obb
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2.0, h / 2.0
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    pts = pts @ R.T + np.array([cx, cy])
    try:
        poly = Polygon(pts)
        if not poly.is_valid: poly = poly.buffer(0)
        return poly
    except:
        return None


def hbb_prescreening(obb1, obb2):
    """水平外接矩形初筛，加速评估"""
    cx1, cy1, w1, h1, t1 = obb1
    cx2, cy2, w2, h2, t2 = obb2
    w1h, h1h = w1 * abs(np.cos(t1)) + h1 * abs(np.sin(t1)), w1 * abs(np.sin(t1)) + h1 * abs(np.cos(t1))
    w2h, h2h = w2 * abs(np.cos(t2)) + h2 * abs(np.sin(t2)), w2 * abs(np.sin(t2)) + h2 * abs(np.cos(t2))
    if abs(cx1 - cx2) > (w1h + w2h) / 2: return False
    if abs(cy1 - cy2) > (h1h + h2h) / 2: return False
    return True


def polygon_iou(obb1, obb2, poly_cache):
    if not hbb_prescreening(obb1, obb2): return 0.0
    key1, key2 = tuple(obb1), tuple(obb2)
    if key1 not in poly_cache: poly_cache[key1] = obb2polygon(obb1)
    if key2 not in poly_cache: poly_cache[key2] = obb2polygon(obb2)
    poly1, poly2 = poly_cache[key1], poly_cache[key2]
    if poly1 is None or poly2 is None: return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / union if union > 0 else 0.0
    except:
        return 0.0


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


class OBBMetricsEvaluator:
    def __init__(self, num_classes=5, small_area_thresh=1024):
        self.nc = num_classes
        self.small_area_thresh = small_area_thresh
        self.reset()

    def reset(self):
        self.preds, self.gts = [], []

    def add_batch(self, image_ids, batch_preds, batch_gts):
        for i, img_id in enumerate(image_ids):
            if len(batch_preds[i]) > 0:
                for p in batch_preds[i].cpu().numpy():
                    self.preds.append({'image_id': img_id, 'bbox': p[:5], 'score': float(p[5]), 'class': int(p[6])})
            if len(batch_gts[i]) > 0:
                for g in batch_gts[i].cpu().numpy():
                    self.gts.append({'image_id': img_id, 'class': int(g[0]), 'bbox': g[1:6]})

    def evaluate(self, iou_thresh=0.5, eval_small_only=False):
        aps, poly_cache = [], {}
        for c in range(self.nc):
            c_preds = sorted([p for p in self.preds if p['class'] == c], key=lambda x: x['score'], reverse=True)
            c_gts = [g for g in self.gts if g['class'] == c]
            if eval_small_only:
                c_gts = [g for g in c_gts if (g['bbox'][2] * g['bbox'][3]) < self.small_area_thresh]

            num_gts = len(c_gts)
            if num_gts == 0: continue

            gt_by_img = {}
            for g in c_gts:
                img_id = g['image_id']
                if img_id not in gt_by_img: gt_by_img[img_id] = {'bboxes': [], 'matched': []}
                gt_by_img[img_id]['bboxes'].append(g['bbox'])
                gt_by_img[img_id]['matched'].append(False)

            nd = len(c_preds)
            if nd == 0:
                aps.append(0.0)
                continue

            tp, fp = np.zeros(nd), np.zeros(nd)
            for i, p in enumerate(c_preds):
                img_id = p['image_id']
                if img_id not in gt_by_img:
                    fp[i] = 1
                    continue

                gt_bboxes, matched = gt_by_img[img_id]['bboxes'], gt_by_img[img_id]['matched']
                best_iou, best_j = 0.0, -1
                for j, gt_box in enumerate(gt_bboxes):
                    if matched[j]: continue
                    iou = polygon_iou(p['bbox'], gt_box, poly_cache)
                    if iou > best_iou: best_iou, best_j = iou, j

                if best_iou >= iou_thresh:
                    tp[i] = 1
                    matched[best_j] = True
                else:
                    fp[i] = 1

            fpc, tpc = np.cumsum(fp), np.cumsum(tp)
            aps.append(compute_ap(tpc / (num_gts + 1e-16), tpc / (tpc + fpc + 1e-16)))
        return float(np.mean(aps)) if aps else 0.0

    def get_full_metrics(self):
        return {'mAP_50': self.evaluate(0.5, False), 'mAP_S': self.evaluate(0.5, True)}