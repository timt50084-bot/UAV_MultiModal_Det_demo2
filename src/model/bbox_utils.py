import torch
import torchvision


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """【网格点生成器】根据特征图尺寸，动态生成每个网格的中心点"""
    anchor_points, stride_tensor = [], []
    assert len(feats) == len(strides)
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        device, dtype = feats[i].device, feats[i].dtype
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)
        anchors = grid * stride
        anchor_points.append(anchors)
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def normalize_anchor_points(anchor_points, input_hw):
    """Map pixel-space anchor centers to the normalized [0, 1] label space."""
    if input_hw is None:
        return anchor_points

    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    if input_h <= 0 or input_w <= 0:
        raise ValueError(f"input_hw must be positive, got: {input_hw}")

    normalized = anchor_points.clone()
    normalized[:, 0] /= float(input_w)
    normalized[:, 1] /= float(input_h)
    return normalized


def xywhr2xyxyxyxy(obb):
    cx, cy, w, h, theta = obb.unbind(dim=-1)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    vec1_x, vec1_y = (w / 2) * cos_t, (w / 2) * sin_t
    vec2_x, vec2_y = -(h / 2) * sin_t, (h / 2) * cos_t
    vec1 = torch.stack([vec1_x, vec1_y], dim=-1)
    vec2 = torch.stack([vec2_x, vec2_y], dim=-1)
    center = torch.stack([cx, cy], dim=-1)
    pt1, pt2 = center + vec1 + vec2, center - vec1 + vec2
    pt3, pt4 = center - vec1 - vec2, center + vec1 - vec2
    return torch.stack([pt1, pt2, pt3, pt4], dim=-2)


def obb2hbb(obb):
    cx, cy, w, h, theta = obb.unbind(dim=-1)
    cos_t, sin_t = torch.abs(torch.cos(theta)), torch.abs(torch.sin(theta))
    w_hbb = w * cos_t + h * sin_t
    h_hbb = w * sin_t + h * cos_t
    xmin, ymin = cx - w_hbb / 2, cy - h_hbb / 2
    xmax, ymax = cx + w_hbb / 2, cy + h_hbb / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def batch_prob_iou(boxes, eps=1e-7):
    """【神级优化】全矩阵化、纯 GPU 的 ProbIoU 近似计算"""
    K = boxes.shape[0]
    if K == 0: return torch.empty((0, 0), device=boxes.device)
    cx, cy, w, h, t = boxes.unbind(dim=-1)
    w, h = torch.clamp(w, min=1e-5), torch.clamp(h, min=1e-5)
    a, b = (w ** 2) / 12.0, (h ** 2) / 12.0
    cos_t, sin_t = torch.cos(t), torch.sin(t)

    s11 = a * cos_t ** 2 + b * sin_t ** 2
    s12 = (a - b) * cos_t * sin_t
    s22 = a * sin_t ** 2 + b * cos_t ** 2
    Sigma = torch.stack([s11, s12, s12, s22], dim=-1).view(-1, 2, 2)
    Sigma12 = (Sigma.unsqueeze(1) + Sigma.unsqueeze(0)) / 2.0

    det1 = torch.clamp(s11 * s22 - s12 ** 2, min=eps)
    s12_11, s12_12, s12_22 = Sigma12[..., 0, 0], Sigma12[..., 0, 1], Sigma12[..., 1, 1]
    det12 = torch.clamp(s12_11 * s12_22 - s12_12 ** 2, min=eps)

    inv_s12_11, inv_s12_22, inv_s12_12 = s12_22 / det12, s12_11 / det12, -s12_12 / det12
    dx, dy = cx.unsqueeze(1) - cx.unsqueeze(0), cy.unsqueeze(1) - cy.unsqueeze(0)

    B_part1 = 0.125 * (dx * (inv_s12_11 * dx + inv_s12_12 * dy) + dy * (inv_s12_12 * dx + inv_s12_22 * dy))
    B_part2 = 0.5 * (torch.log(det12 + eps) - 0.5 * torch.log(det1.unsqueeze(1) + eps) - 0.5 * torch.log(
        det1.unsqueeze(0) + eps))
    B_D = torch.clamp(B_part1 + B_part2, min=0.0, max=50.0)
    return torch.exp(-B_D)


def non_max_suppression_obb(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, max_wh=4096.0):
    """🔥 顶会霸榜级算子: Hybrid Matrix Soft-NMS"""
    bs = prediction.shape[0]
    output = [torch.zeros((0, 7), device=prediction.device) for _ in range(bs)]
    prediction = prediction.float()
    TOPK_PRE_NMS = 300

    for xi, x in enumerate(prediction):
        box, cls_scores = x[:, :5], x[:, 5:]
        i, j = (cls_scores > conf_thres).nonzero(as_tuple=True)
        x_filtered = torch.cat((box[i], cls_scores[i, j][:, None], j[:, None].float()), 1)

        if not x_filtered.shape[0]: continue
        x_filtered = x_filtered[x_filtered[:, 5].argsort(descending=True)[:5000]]
        obb_boxes, scores, classes = x_filtered[:, :5], x_filtered[:, 5], x_filtered[:, 6]

        # Stage 1: HBB
        hbb_boxes = obb2hbb(obb_boxes)
        c = classes * max_wh
        keep_hbb_idx = torchvision.ops.nms(hbb_boxes + c.unsqueeze(1), scores, iou_thres + 0.15)[:TOPK_PRE_NMS]

        obb_boxes_stage2 = obb_boxes[keep_hbb_idx]
        scores_stage2 = scores[keep_hbb_idx]
        classes_stage2 = classes[keep_hbb_idx]
        areas_stage2 = obb_boxes_stage2[:, 2] * obb_boxes_stage2[:, 3]

        if obb_boxes_stage2.shape[0] == 0: continue

        # Stage 2: Matrix Soft-NMS
        ious_matrix = batch_prob_iou(obb_boxes_stage2)
        class_match = classes_stage2.unsqueeze(0) == classes_stage2.unsqueeze(1)
        mask_pos = torch.triu(torch.ones_like(ious_matrix, dtype=torch.bool), diagonal=1)
        valid_ious = ious_matrix * class_match * mask_pos
        max_ious = valid_ious.max(dim=0)[0]

        scale = torch.clamp(areas_stage2 / 225.0, max=1.0)
        dynamic_sigma = 2.0 + 2.0 * (1.0 - scale)
        updated_scores = scores_stage2 * torch.exp(-(max_ious ** 2) / dynamic_sigma)

        keep_stage2_mask = updated_scores > conf_thres
        final_scores = updated_scores[keep_stage2_mask]
        final_keep_idx = keep_hbb_idx[keep_stage2_mask]

        if final_keep_idx.shape[0] > 0:
            order = final_scores.argsort(descending=True)
            final_keep_idx, final_scores = final_keep_idx[order][:max_det], final_scores[order][:max_det]
            final_boxes = x_filtered[final_keep_idx]
            final_boxes[:, 5] = final_scores
            output[xi] = final_boxes
    return output
