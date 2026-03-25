import re

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.model.bbox_utils import make_anchors
from src.model.output_adapter import flatten_predictions


class Trainer:
    def __init__(self, model, train_loader, optimizer, scheduler, criterion, assigner,
                 device, epochs, accumulate=1, grad_clip=10.0, use_amp=True, evaluator=None, callbacks=None):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.assigner = assigner
        self.device = device
        self.epochs = epochs
        self.accumulate = max(1, accumulate)
        self.grad_clip = grad_clip
        self.use_amp = use_amp

        self.evaluator = evaluator
        self.callbacks = callbacks or []
        self.scaler = GradScaler(enabled=self.use_amp)

        self.current_epoch = 0
        self.current_metrics = None
        self.stop_training = False

        self._bind_callbacks()

    def _bind_callbacks(self):
        for callback in self.callbacks:
            name = callback.__class__.__name__
            if name.startswith('EMA'):
                attr_name = 'ema_callback'
            else:
                attr_name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
            setattr(self, attr_name, callback)

    def trigger_callbacks(self, hook_name):
        for cb in self.callbacks:
            getattr(cb, hook_name, lambda x: None)(self)

    def format_targets(self, targets, batch_size):
        targets_np = targets.cpu().numpy()
        gt_dict = {b: [] for b in range(batch_size)}
        for target in targets_np:
            gt_dict[int(target[0])].append(target[1:])

        max_gts = max([len(v) for v in gt_dict.values()] + [0])

        gt_labels = torch.zeros((batch_size, max_gts, 1), device=self.device)
        gt_bboxes = torch.zeros((batch_size, max_gts, 5), device=self.device)
        mask_gt = torch.zeros((batch_size, max_gts, 1), device=self.device)

        if max_gts > 0:
            for batch_idx in range(batch_size):
                num_gts = len(gt_dict[batch_idx])
                if num_gts > 0:
                    gt_tensor = torch.tensor(gt_dict[batch_idx], device=self.device, dtype=torch.float32)
                    gt_labels[batch_idx, :num_gts, 0] = gt_tensor[:, 0]
                    gt_bboxes[batch_idx, :num_gts, :] = gt_tensor[:, 1:6]
                    mask_gt[batch_idx, :num_gts, 0] = 1.0

        return gt_labels, gt_bboxes, mask_gt

    def train(self):
        self.trigger_callbacks('on_train_begin')

        for epoch in range(self.epochs):
            if self.stop_training:
                break
            self.current_epoch = epoch

            self.model.train()
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch, self.epochs)

            self.trigger_callbacks('on_epoch_begin')

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            self.optimizer.zero_grad()
            total_loss_epoch = 0.0

            for i, (imgs_rgb, imgs_ir, targets, prev_rgb, prev_ir) in enumerate(pbar):
                self.trigger_callbacks('on_batch_begin')

                imgs_rgb = imgs_rgb.to(self.device, non_blocking=True)
                imgs_ir = imgs_ir.to(self.device, non_blocking=True)
                prev_rgb = prev_rgb.to(self.device, non_blocking=True)
                prev_ir = prev_ir.to(self.device, non_blocking=True)
                batch_size = imgs_rgb.shape[0]

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs, feat_rgb, feat_ir = self.model(imgs_rgb, imgs_ir, prev_rgb=prev_rgb, prev_ir=prev_ir)
                    flat_preds, feats_for_anchors = flatten_predictions(outputs)

                    strides = [4, 8, 16, 32]
                    anchor_points, _ = make_anchors(feats_for_anchors, strides)

                    gt_labels, gt_bboxes, mask_gt = self.format_targets(targets, batch_size)
                    pred_bboxes, pred_scores = flat_preds[..., :5], flat_preds[..., 5:]

                    target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
                        pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
                    )
                    pos_mask = fg_mask.bool()

                    if pos_mask.any():
                        matched_pred_cls = pred_scores[pos_mask]
                        matched_pred_box = pred_bboxes[pos_mask]
                        matched_tgt_cls = target_labels[pos_mask]
                        matched_tgt_box = target_bboxes[pos_mask]

                        contrastive_loss = 0.0
                        if getattr(self.model, 'use_contrastive', False) and epoch >= 10:
                            contrastive_loss = self.model.get_contrastive_alignment_loss(feat_rgb, feat_ir) * 0.1

                        temporal_loss = 0.0
                        if getattr(self.model, 'temporal_enabled', False):
                            temporal_loss = self.model.get_temporal_consistency_loss(
                                lambda_t=getattr(self.criterion, 'temporal_weight', 0.1),
                                low_motion_bias=getattr(self.criterion, 'temporal_low_motion_bias', 0.75)
                            )

                        loss_total, _, _ = self.criterion(
                            matched_pred_cls,
                            matched_pred_box,
                            matched_tgt_cls,
                            matched_tgt_box,
                            contrastive_loss,
                            epoch,
                            temporal_loss=temporal_loss,
                        )
                    else:
                        loss_total = flat_preds.sum() * 0.0

                    loss_total = loss_total / self.accumulate

                if not torch.isfinite(loss_total):
                    print("\nWarning: encountered non-finite loss, skipping batch.")
                    self.optimizer.zero_grad()
                    continue

                self.scaler.scale(loss_total).backward()

                if (i + 1) % self.accumulate == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.trigger_callbacks('on_batch_end')

                total_loss_epoch += loss_total.item() * self.accumulate
                pbar.set_postfix({"Loss": f"{total_loss_epoch / (i + 1):.4f}"})

            self.scheduler.step()

            if self.evaluator:
                eval_model = getattr(self, 'ema_callback').ema if hasattr(self, 'ema_callback') else self.model
                self.current_metrics = self.evaluator.evaluate(eval_model, epoch=epoch + 1)

            self.trigger_callbacks('on_epoch_end')

        self.trigger_callbacks('on_train_end')
