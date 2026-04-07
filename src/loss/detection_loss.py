import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.registry.loss_registry import LOSSES


def obb_to_covariance(boxes):
    cx, cy, w, h, theta = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]

    w = torch.clamp(w, min=1e-5)
    h = torch.clamp(h, min=1e-5)

    a = (w ** 2) / 12.0
    b = (h ** 2) / 12.0

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    sigma_11 = a * cos_t ** 2 + b * sin_t ** 2
    sigma_12 = (a - b) * cos_t * sin_t
    sigma_22 = a * sin_t ** 2 + b * cos_t ** 2

    sigma = torch.stack([sigma_11, sigma_12, sigma_12, sigma_22], dim=-1).view(-1, 2, 2)
    mu = torch.stack([cx, cy], dim=-1)
    return mu, sigma


class ProbIoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        mu1, sigma1 = obb_to_covariance(pred_boxes)
        mu2, sigma2 = obb_to_covariance(target_boxes)

        sigma12 = (sigma1 + sigma2) / 2.0

        det1 = torch.clamp(sigma1[:, 0, 0] * sigma1[:, 1, 1] - sigma1[:, 0, 1] ** 2, min=self.eps)
        det2 = torch.clamp(sigma2[:, 0, 0] * sigma2[:, 1, 1] - sigma2[:, 0, 1] ** 2, min=self.eps)
        det12 = torch.clamp(sigma12[:, 0, 0] * sigma12[:, 1, 1] - sigma12[:, 0, 1] ** 2, min=self.eps)

        inv_sigma12 = torch.zeros_like(sigma12)
        inv_sigma12[:, 0, 0] = sigma12[:, 1, 1] / det12
        inv_sigma12[:, 1, 1] = sigma12[:, 0, 0] / det12
        inv_sigma12[:, 0, 1] = -sigma12[:, 0, 1] / det12
        inv_sigma12[:, 1, 0] = -sigma12[:, 1, 0] / det12

        d_mu = (mu1 - mu2).unsqueeze(-1)
        b_part1 = 0.125 * torch.bmm(torch.bmm(d_mu.transpose(1, 2), inv_sigma12), d_mu).squeeze(-1).squeeze(-1)
        b_part2 = 0.5 * (torch.log(det12) - 0.5 * torch.log(det1) - 0.5 * torch.log(det2))

        bhattacharyya = torch.clamp(b_part1 + b_part2, max=50.0)
        prob_iou = torch.exp(-bhattacharyya).clamp(min=1e-6, max=1.0)
        return 1.0 - prob_iou, prob_iou


@LOSSES.register("UAVDualModalLoss")
class UAVDualModalLoss(nn.Module):
    def __init__(self, num_classes=5, alpha=0.25, gamma=2.0, use_scale_weight=True,
                 temporal_weight=0.1, temporal_low_motion_bias=0.75,
                 temporal_warmup_epochs=0.0, temporal_ramp_epochs=0.0,
                 temporal_max_loss=None, temporal_skip_loss_threshold=None,
                 angle_enabled=False, angle_weight=0.0, angle_type='wrapped_smooth_l1', angle_beta=0.1,
                 cls_loss_type='varifocal'):
        super().__init__()
        self.nc = num_classes
        self.use_scale_weight = use_scale_weight
        self.prob_iou_loss = ProbIoULoss()
        self.cls_loss_focal = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.temporal_weight = temporal_weight
        self.temporal_low_motion_bias = temporal_low_motion_bias
        self.temporal_warmup_epochs = max(0.0, float(temporal_warmup_epochs))
        self.temporal_ramp_epochs = max(0.0, float(temporal_ramp_epochs))
        self.temporal_max_loss = float(temporal_max_loss) if temporal_max_loss is not None else float('inf')
        self.temporal_skip_loss_threshold = (
            float(temporal_skip_loss_threshold)
            if temporal_skip_loss_threshold is not None else float('inf')
        )
        self.angle_enabled = bool(angle_enabled)
        self.angle_weight = float(angle_weight)
        self.angle_type = str(angle_type)
        self.angle_beta = float(angle_beta)
        self.cls_loss_type = str(cls_loss_type).lower()
        if self.cls_loss_type not in {'focal', 'varifocal'}:
            raise ValueError(f'Unsupported cls loss type: {self.cls_loss_type}')

    def _focal_loss(self, pred_logits, targets):
        bce_loss = self.cls_loss_focal(pred_logits, targets)
        probas = torch.sigmoid(pred_logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * ((1 - p_t) ** self.gamma)
        return focal_weight * bce_loss

    def _varifocal_loss(self, pred_logits, targets):
        bce_loss = self.cls_loss_focal(pred_logits, targets)
        pred_prob = torch.sigmoid(pred_logits)
        focal_weight = self.alpha * pred_prob.pow(self.gamma) * (1.0 - targets) + targets
        return focal_weight * bce_loss

    def _compute_cls_loss_raw(self, pred_logits, targets):
        if self.cls_loss_type == 'focal':
            return self._focal_loss(pred_logits, targets)
        if self.cls_loss_type == 'varifocal':
            return self._varifocal_loss(pred_logits, targets)
        raise ValueError(f'Unsupported cls loss type: {self.cls_loss_type}')

    @staticmethod
    def _reduce_normalizer(value):
        if value.ndim == 0:
            value = value.unsqueeze(0)
        value = value.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return torch.clamp(value.squeeze(0), min=1.0)

    @staticmethod
    def wrapped_angle_difference(pred_theta, target_theta):
        half_pi = pred_theta.new_tensor(torch.pi * 0.5)
        pi = pred_theta.new_tensor(torch.pi)
        return torch.remainder(pred_theta - target_theta + half_pi, pi) - half_pi

    def _compute_angle_loss_raw(self, pred_theta, target_theta):
        wrapped_delta = self.wrapped_angle_difference(pred_theta, target_theta)

        if self.angle_type == 'wrapped_smooth_l1':
            return F.smooth_l1_loss(
                wrapped_delta,
                torch.zeros_like(wrapped_delta),
                reduction='none',
                beta=self.angle_beta,
            )
        if self.angle_type == 'sin_cos':
            return 1.0 - torch.cos(2.0 * wrapped_delta)
        raise ValueError(f'Unsupported angle loss type: {self.angle_type}')

    def _forward_matched(self, matched_pred_cls, matched_pred_box, matched_tgt_cls, matched_tgt_box,
                         contrastive_loss=0.0, epoch=0, temporal_loss=0.0):
        matched_tgt_box = matched_tgt_box.to(device=matched_pred_box.device, dtype=matched_pred_box.dtype)
        if not torch.is_tensor(contrastive_loss):
            contrastive_loss = matched_pred_box.new_tensor(float(contrastive_loss))
        if not torch.is_tensor(temporal_loss):
            temporal_loss = matched_pred_box.new_tensor(float(temporal_loss))
        contrastive_loss = torch.nan_to_num(contrastive_loss, nan=0.0, posinf=0.0, neginf=0.0)
        temporal_loss = torch.nan_to_num(temporal_loss, nan=0.0, posinf=0.0, neginf=0.0)

        loss_reg_raw, prob_iou = self.prob_iou_loss(matched_pred_box, matched_tgt_box)

        tgt_cls_onehot = F.one_hot(matched_tgt_cls.long(), num_classes=self.nc).float()
        loss_cls_raw = self._focal_loss(matched_pred_cls, tgt_cls_onehot).sum(-1)

        with torch.no_grad():
            iou_weight = prob_iou.detach().clamp(min=0.2)
        loss_cls_raw = loss_cls_raw * iou_weight

        num_pos = self._reduce_normalizer(
            torch.tensor([matched_pred_box.size(0)], dtype=torch.float32, device=matched_pred_box.device)
        ).item()

        if self.use_scale_weight:
            tgt_area = torch.clamp(matched_tgt_box[:, 2] * matched_tgt_box[:, 3], min=1e-6)
            scale_weights = torch.clamp(1.0 / torch.sqrt(tgt_area), min=1.0, max=3.0).detach()
            loss_cls = (loss_cls_raw * scale_weights).sum() / num_pos
            loss_reg = (loss_reg_raw * scale_weights).sum() / num_pos
        else:
            scale_weights = None
            loss_cls = loss_cls_raw.sum() / num_pos
            loss_reg = loss_reg_raw.sum() / num_pos

        if self.angle_enabled and self.angle_weight > 0.0:
            loss_angle_raw = self._compute_angle_loss_raw(matched_pred_box[:, 4], matched_tgt_box[:, 4])
            if scale_weights is not None:
                loss_angle = (loss_angle_raw * scale_weights).sum() / num_pos
            else:
                loss_angle = loss_angle_raw.sum() / num_pos
        else:
            loss_angle = matched_pred_box.new_tensor(0.0)

        weight_cls = 2.0 if epoch < 5 else 1.0
        weight_reg = 1.0 if epoch < 5 else 2.0

        total_loss = (
            loss_cls * weight_cls
            + loss_reg * weight_reg
            + loss_angle * self.angle_weight
            + contrastive_loss
            + temporal_loss
        )
        return total_loss, loss_cls, loss_reg, loss_angle

    def _forward_dense(self, pred_scores, pred_bboxes, target_scores, target_bboxes, fg_mask,
                       contrastive_loss=0.0, epoch=0, temporal_loss=0.0):
        reference_tensor = pred_scores if pred_scores.numel() > 0 else pred_bboxes
        if not torch.is_tensor(contrastive_loss):
            contrastive_loss = reference_tensor.new_tensor(float(contrastive_loss))
        if not torch.is_tensor(temporal_loss):
            temporal_loss = reference_tensor.new_tensor(float(temporal_loss))
        contrastive_loss = torch.nan_to_num(contrastive_loss, nan=0.0, posinf=0.0, neginf=0.0)
        temporal_loss = torch.nan_to_num(temporal_loss, nan=0.0, posinf=0.0, neginf=0.0)

        target_scores = target_scores.to(device=pred_scores.device, dtype=pred_scores.dtype)
        target_bboxes = target_bboxes.to(device=pred_bboxes.device, dtype=pred_bboxes.dtype)
        fg_mask = fg_mask.to(device=pred_scores.device, dtype=torch.bool)

        cls_loss_raw = self._compute_cls_loss_raw(pred_scores, target_scores).sum(-1)
        num_pos = self._reduce_normalizer(fg_mask.sum(dtype=pred_scores.dtype))

        zero_term = pred_bboxes.sum() * 0.0
        loss_reg = zero_term
        loss_angle = zero_term
        cls_anchor_weights = torch.ones_like(cls_loss_raw)

        if bool(fg_mask.any().item()):
            matched_pred_box = pred_bboxes[fg_mask]
            matched_tgt_box = target_bboxes[fg_mask]
            loss_reg_raw, _ = self.prob_iou_loss(matched_pred_box, matched_tgt_box)

            scale_weights = None
            if self.use_scale_weight:
                tgt_area = torch.clamp(matched_tgt_box[:, 2] * matched_tgt_box[:, 3], min=1e-6)
                scale_weights = torch.clamp(1.0 / torch.sqrt(tgt_area), min=1.0, max=3.0).detach()
                cls_anchor_weights[fg_mask] = scale_weights.to(cls_anchor_weights.dtype)
                loss_reg = (loss_reg_raw * scale_weights).sum() / num_pos
            else:
                loss_reg = loss_reg_raw.sum() / num_pos

            if self.angle_enabled and self.angle_weight > 0.0:
                loss_angle_raw = self._compute_angle_loss_raw(matched_pred_box[:, 4], matched_tgt_box[:, 4])
                if scale_weights is not None:
                    loss_angle = (loss_angle_raw * scale_weights).sum() / num_pos
                else:
                    loss_angle = loss_angle_raw.sum() / num_pos

        loss_cls = (cls_loss_raw * cls_anchor_weights).sum() / num_pos

        weight_cls = 2.0 if epoch < 5 else 1.0
        weight_reg = 1.0 if epoch < 5 else 2.0

        total_loss = (
            loss_cls * weight_cls
            + loss_reg * weight_reg
            + loss_angle * self.angle_weight
            + contrastive_loss
            + temporal_loss
        )
        return total_loss, loss_cls, loss_reg, loss_angle

    def forward(self, pred_scores, pred_bboxes, target_scores_or_labels, target_bboxes,
                fg_mask=None, contrastive_loss=0.0, epoch=0, temporal_loss=0.0):
        dense_target_path = (
            fg_mask is not None
            and torch.is_tensor(target_scores_or_labels)
            and target_scores_or_labels.shape == pred_scores.shape
        )
        if dense_target_path:
            return self._forward_dense(
                pred_scores,
                pred_bboxes,
                target_scores_or_labels,
                target_bboxes,
                fg_mask,
                contrastive_loss=contrastive_loss,
                epoch=epoch,
                temporal_loss=temporal_loss,
            )
        return self._forward_matched(
            pred_scores,
            pred_bboxes,
            target_scores_or_labels,
            target_bboxes,
            contrastive_loss=contrastive_loss,
            epoch=epoch,
            temporal_loss=temporal_loss,
        )
