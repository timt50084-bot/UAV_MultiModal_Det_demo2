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
                 angle_enabled=False, angle_weight=0.0, angle_type='wrapped_smooth_l1', angle_beta=0.1):
        super().__init__()
        self.nc = num_classes
        self.use_scale_weight = use_scale_weight
        self.prob_iou_loss = ProbIoULoss()
        self.cls_loss_focal = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.temporal_weight = temporal_weight
        self.temporal_low_motion_bias = temporal_low_motion_bias
        self.angle_enabled = bool(angle_enabled)
        self.angle_weight = float(angle_weight)
        self.angle_type = str(angle_type)
        self.angle_beta = float(angle_beta)

    def _focal_loss(self, pred_logits, targets):
        bce_loss = self.cls_loss_focal(pred_logits, targets)
        probas = torch.sigmoid(pred_logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * ((1 - p_t) ** self.gamma)
        return focal_weight * bce_loss

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

    def forward(self, matched_pred_cls, matched_pred_box, matched_tgt_cls, matched_tgt_box,
                contrastive_loss=0.0, epoch=0, temporal_loss=0.0):
        if not torch.is_tensor(contrastive_loss):
            contrastive_loss = matched_pred_box.new_tensor(0.0)
        if not torch.is_tensor(temporal_loss):
            temporal_loss = matched_pred_box.new_tensor(0.0)

        loss_reg_raw, prob_iou = self.prob_iou_loss(matched_pred_box, matched_tgt_box)

        tgt_cls_onehot = F.one_hot(matched_tgt_cls.long(), num_classes=self.nc).float()
        loss_cls_raw = self._focal_loss(matched_pred_cls, tgt_cls_onehot).sum(-1)

        with torch.no_grad():
            iou_weight = prob_iou.detach().clamp(min=0.2)
        loss_cls_raw = loss_cls_raw * iou_weight

        num_pos = torch.tensor([matched_pred_box.size(0)], dtype=torch.float32, device=matched_pred_box.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_pos, op=dist.ReduceOp.SUM)
        num_pos = torch.clamp(num_pos, min=1.0).item()

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
