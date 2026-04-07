import torch
import torch.nn as nn

from src.registry.loss_registry import ASSIGNERS


@ASSIGNERS.register("DynamicTinyOBBAssigner")
class DynamicTinyOBBAssigner(nn.Module):
    def __init__(
        self,
        num_classes=5,
        topk=10,
        alpha=0.5,
        beta=6.0,
        temperature=2.0,
        eps=1e-9,
        lambda_theta=1.5,
        tiny_area_threshold=0.01,
        tiny_topk_boost=0,
        elongated_ratio_threshold=1e9,
        use_angle_aware_assign=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.eps = eps
        self.lambda_theta = lambda_theta
        self.tiny_area_threshold = tiny_area_threshold
        self.tiny_topk_boost = int(tiny_topk_boost)
        self.elongated_ratio_threshold = elongated_ratio_threshold
        self.use_angle_aware_assign = use_angle_aware_assign

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        batch_size, num_anchors = pred_scores.shape[:2]
        max_num_gts = gt_bboxes.shape[1]
        device = pred_scores.device

        if max_num_gts == 0:
            return (
                torch.full([batch_size, num_anchors], self.num_classes, dtype=torch.long, device=device),
                torch.zeros([batch_size, num_anchors, 5], device=device),
                torch.zeros([batch_size, num_anchors, self.num_classes], device=device),
                torch.zeros([batch_size, num_anchors], dtype=torch.bool, device=device),
            )

        align_metric, overlaps = self.get_box_metrics(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes)
        align_metric = torch.nan_to_num(align_metric.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        overlaps = torch.nan_to_num(overlaps.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)

        is_in_centers = self.select_candidates_in_gts(anchor_points, gt_bboxes)
        is_in_centers = self.apply_tiny_object_fallback(anchor_points, gt_bboxes, is_in_centers, mask_gt)
        align_metric *= is_in_centers.float()

        is_pos, target_labels, target_bboxes, target_scores = self.select_topk_candidates(
            align_metric, overlaps, pred_scores, gt_labels, gt_bboxes, mask_gt
        )

        return target_labels, target_bboxes, target_scores, is_pos

    def _get_gt_area(self, gt_bboxes):
        gt_wh = torch.clamp(gt_bboxes[..., 2:4], min=self.eps)
        return gt_wh[..., 0] * gt_wh[..., 1]

    def _get_aspect_ratio(self, gt_bboxes):
        gt_wh = torch.clamp(gt_bboxes[..., 2:4], min=self.eps)
        ratio_wh = gt_wh[..., 0] / gt_wh[..., 1]
        ratio_hw = gt_wh[..., 1] / gt_wh[..., 0]
        return torch.maximum(ratio_wh, ratio_hw)

    def _angle_distance(self, pred_theta, gt_theta):
        half_pi = pred_theta.new_tensor(torch.pi * 0.5)
        pi = pred_theta.new_tensor(torch.pi)
        wrapped = torch.remainder(pred_theta - gt_theta + half_pi, pi) - half_pi
        return wrapped.abs()

    def _compute_tiny_severity(self, gt_bboxes):
        gt_area = self._get_gt_area(gt_bboxes)
        if self.tiny_area_threshold <= 0:
            return torch.zeros_like(gt_area)
        severity = (self.tiny_area_threshold - gt_area) / (self.tiny_area_threshold + self.eps)
        return severity.clamp(min=0.0, max=1.0)

    def _compute_candidate_budget(self, gt_bboxes, num_anchors):
        batch_size, max_gts, _ = gt_bboxes.shape
        if max_gts == 0:
            return gt_bboxes.new_zeros((batch_size, 0), dtype=torch.long)

        tiny_extra = torch.ceil(self._compute_tiny_severity(gt_bboxes) * float(max(self.tiny_topk_boost, 0))).long()
        elongated_extra = (self._get_aspect_ratio(gt_bboxes) >= self.elongated_ratio_threshold).long()
        budget = self.topk + tiny_extra + elongated_extra
        return budget.clamp(min=1, max=max(int(num_anchors), 1))

    def get_box_metrics(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes):
        batch_size, num_anchors, _ = pred_scores.shape
        max_gts = gt_bboxes.shape[1]
        device = pred_scores.device

        indices = torch.zeros([2, batch_size, max_gts], dtype=torch.long, device=device)
        indices[0] = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, max_gts)
        indices[1] = gt_labels.squeeze(-1).long()

        cls_scores = pred_scores.detach().sigmoid().clamp(min=self.eps, max=1.0)
        bbox_scores = cls_scores.permute(0, 2, 1)[indices[0], indices[1], :]
        bbox_scores = bbox_scores.permute(0, 2, 1)

        pred_cxcy = pred_bboxes[:, :, None, :2]
        gt_cxcy = gt_bboxes[:, None, :, :2]
        distances = torch.norm(pred_cxcy - gt_cxcy, dim=-1)

        gt_wh = gt_bboxes[:, None, :, 2:4]
        scale = torch.sqrt(gt_wh[..., 0] * gt_wh[..., 1])
        overlaps = torch.exp(-((distances / (scale + 1e-6)) ** 2))

        align_metric = torch.exp(
            self.alpha * torch.log(bbox_scores + self.eps) +
            self.beta * torch.log(overlaps + self.eps)
        )

        if self.use_angle_aware_assign:
            pred_theta = pred_bboxes[:, :, None, 4]
            gt_theta = gt_bboxes[:, None, :, 4]
            delta_theta = self._angle_distance(pred_theta, gt_theta)
            angle_factor = torch.exp(-self.lambda_theta * delta_theta)
            align_metric = align_metric * angle_factor

        tiny_scale = 1.0 + self._compute_tiny_severity(gt_bboxes)[:, None, :] * (
            float(max(self.tiny_topk_boost, 0)) / max(float(self.topk), 1.0)
        )
        align_metric = align_metric * tiny_scale

        aspect_ratio = self._get_aspect_ratio(gt_bboxes)[:, None, :]
        elongated_strength = (
            (aspect_ratio - self.elongated_ratio_threshold) / (self.elongated_ratio_threshold + self.eps)
        ).clamp(min=0.0, max=1.0)
        elongated_factor = 1.0 + 0.25 * elongated_strength
        align_metric = align_metric * elongated_factor

        anchor_points_exp = anchor_points.view(1, num_anchors, 1, 2)
        center_dist = torch.norm(anchor_points_exp - gt_cxcy, dim=-1)
        center_weight = torch.exp(-((center_dist / (scale * 1.5 + 1e-6)) ** 2))

        align_metric = align_metric * center_weight
        return align_metric, overlaps

    def select_candidates_in_gts(self, anchor_points, gt_bboxes, radius=1.5):
        batch_size, max_gts, _ = gt_bboxes.shape
        num_anchors = anchor_points.shape[0]

        anchor_points = anchor_points.view(1, num_anchors, 1, 2)
        gt_cxcy = gt_bboxes[:, None, :, :2]
        gt_wh = gt_bboxes[:, None, :, 2:4]
        theta = gt_bboxes[:, None, :, 4]

        dx = anchor_points[..., 0] - gt_cxcy[..., 0]
        dy = anchor_points[..., 1] - gt_cxcy[..., 1]

        cos_t = torch.cos(-theta)
        sin_t = torch.sin(-theta)

        local_dx = dx * cos_t - dy * sin_t
        local_dy = dx * sin_t + dy * cos_t

        local_dist = torch.stack([torch.abs(local_dx), torch.abs(local_dy)], dim=-1)
        is_in_box = (local_dist < (gt_wh * radius)).all(dim=-1)
        return is_in_box

    def apply_tiny_object_fallback(self, anchor_points, gt_bboxes, is_in_centers, mask_gt):
        gt_area = self._get_gt_area(gt_bboxes)
        aspect_ratio = self._get_aspect_ratio(gt_bboxes)

        candidates_per_gt = is_in_centers.sum(dim=1)
        protected_missed_mask = (
            (mask_gt.squeeze(-1) > 0)
            & (candidates_per_gt == 0)
            & ((gt_area < self.tiny_area_threshold) | (aspect_ratio >= self.elongated_ratio_threshold))
        )

        if protected_missed_mask.any():
            candidate_budget = self._compute_candidate_budget(gt_bboxes, anchor_points.shape[0])
            fallback_k = (candidate_budget - self.topk + 3).clamp(min=3)
            gt_cxcy = gt_bboxes[..., :2]
            dist = torch.cdist(gt_cxcy, anchor_points.unsqueeze(0).repeat(gt_cxcy.size(0), 1, 1))

            max_fallback = min(anchor_points.shape[0], 3 + max(self.tiny_topk_boost, 0) + 1)
            closest_idx = dist.topk(max_fallback, largest=False).indices

            device = gt_bboxes.device
            seq = torch.arange(max_fallback, device=device).view(1, 1, max_fallback)
            keep_mask = seq < fallback_k.unsqueeze(-1)

            valid_fallback = protected_missed_mask.unsqueeze(-1) & keep_mask
            batch_idx, max_gts_idx, k_idx = torch.where(valid_fallback)

            actual_anchor_idx = closest_idx[batch_idx, max_gts_idx, k_idx]
            is_in_centers[batch_idx, actual_anchor_idx, max_gts_idx] = True

        return is_in_centers

    def select_topk_candidates(self, align_metric, overlaps, pred_scores, gt_labels, gt_bboxes, mask_gt):
        align_metric = align_metric.permute(0, 2, 1)
        overlaps = overlaps.permute(0, 2, 1)
        batch_size, max_gts, num_anchors = align_metric.shape
        device = align_metric.device

        align_metric *= mask_gt.view(batch_size, max_gts, 1)

        candidate_budget = self._compute_candidate_budget(gt_bboxes, num_anchors).to(device)
        max_topk = max(int(candidate_budget.max().item()) if candidate_budget.numel() > 0 else self.topk, 1)
        topk_metrics, topk_idxs = torch.topk(align_metric, max_topk, dim=-1, largest=True)
        topk_metrics = torch.nan_to_num(topk_metrics, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        topk_overlaps = torch.gather(overlaps, 2, topk_idxs)
        topk_overlaps = torch.nan_to_num(topk_overlaps, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
        dynamic_k = torch.round(topk_overlaps.sum(dim=-1))
        dynamic_k = torch.maximum(dynamic_k, torch.ones_like(dynamic_k))
        dynamic_k = torch.minimum(dynamic_k, candidate_budget.float()).int()

        seq = torch.arange(max_topk, device=device).view(1, 1, -1)
        k_mask = seq < dynamic_k.unsqueeze(-1)

        is_pos = torch.zeros_like(align_metric, dtype=torch.bool)
        is_pos.scatter_(-1, topk_idxs, (topk_metrics > self.eps) & k_mask)

        align_metric = align_metric.permute(0, 2, 1)
        is_pos = is_pos.permute(0, 2, 1)

        _, anchor_to_gt_idx = align_metric.max(dim=-1)
        is_pos_mask = is_pos.sum(dim=-1) > 0

        batch_idx = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, num_anchors)
        target_labels = gt_labels.squeeze(-1)[batch_idx, anchor_to_gt_idx]
        target_bboxes = gt_bboxes[batch_idx, anchor_to_gt_idx]

        align_metric_max = align_metric.max(dim=1, keepdim=True)[0]
        norm_align_metric = (align_metric / (align_metric_max + self.eps)) * overlaps.permute(0, 2, 1)
        norm_align_metric = torch.nan_to_num(norm_align_metric, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)

        target_scores_raw = norm_align_metric[
            batch_idx,
            torch.arange(num_anchors, device=device).unsqueeze(0).repeat(batch_size, 1),
            anchor_to_gt_idx
        ]
        target_scores_raw = torch.nan_to_num(target_scores_raw, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)

        target_scores = torch.zeros_like(pred_scores)
        valid_mask = (target_labels >= 0) & (target_labels < self.num_classes) & is_pos_mask
        soft_target_scores = target_scores_raw ** (1.0 / self.temperature)

        b_idx, a_idx = torch.where(valid_mask)
        l_idx = target_labels[valid_mask].long()
        target_scores[b_idx, a_idx, l_idx] = soft_target_scores[valid_mask]
        target_labels = target_labels.masked_fill(~is_pos_mask, self.num_classes)
        target_bboxes = target_bboxes * is_pos_mask.unsqueeze(-1).to(target_bboxes.dtype)

        return is_pos_mask, target_labels, target_bboxes, target_scores
