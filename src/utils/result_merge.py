import torch

from src.model.bbox_utils import batch_prob_iou


def _pairwise_iou(reference_box, other_boxes):
    if other_boxes.shape[0] == 0:
        return other_boxes.new_zeros((0,))
    boxes = torch.cat([reference_box.unsqueeze(0), other_boxes], dim=0)
    return batch_prob_iou(boxes)[0, 1:]


def _weighted_merge_cluster(cluster):
    scores = cluster[:, 5]
    weights = scores / scores.sum().clamp(min=1e-6)
    merged = cluster[0].clone()
    merged[:4] = (cluster[:, :4] * weights.unsqueeze(1)).sum(dim=0)

    angles = cluster[:, 4]
    sin_term = (torch.sin(2.0 * angles) * weights).sum()
    cos_term = (torch.cos(2.0 * angles) * weights).sum()
    merged[4] = 0.5 * torch.atan2(sin_term, cos_term)
    merged[5] = scores.max()
    return merged


def merge_obb_predictions(prediction_sets, method='nms', iou_threshold=0.55, max_det=300):
    valid_sets = [pred for pred in (prediction_sets or []) if pred is not None and len(pred) > 0]
    if not valid_sets:
        if prediction_sets and torch.is_tensor(prediction_sets[0]):
            return prediction_sets[0][:0]
        return torch.zeros((0, 7), dtype=torch.float32)

    merged = torch.cat(valid_sets, dim=0)
    if merged.shape[0] == 0:
        return merged

    order = merged[:, 5].argsort(descending=True)
    merged = merged[order]
    kept = []

    while merged.shape[0] > 0 and len(kept) < int(max_det):
        current = merged[0]
        if merged.shape[0] == 1:
            kept.append(current)
            break

        rest = merged[1:]
        ious = _pairwise_iou(current[:5], rest[:, :5])
        same_class = rest[:, 6] == current[6]
        cluster_mask = same_class & (ious >= float(iou_threshold))

        if method == 'score_weighted':
            cluster = torch.cat([current.unsqueeze(0), rest[cluster_mask]], dim=0)
            kept.append(_weighted_merge_cluster(cluster))
        else:
            kept.append(current)

        suppress_mask = cluster_mask if method == 'score_weighted' else (same_class & (ious >= float(iou_threshold)))
        merged = rest[~suppress_mask]

    return torch.stack(kept, dim=0) if kept else valid_sets[0][:0]
