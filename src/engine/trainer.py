import math
import re
import time

import torch
from tqdm import tqdm

from src.model.bbox_utils import make_anchors, normalize_anchor_points
from src.model.output_adapter import flatten_predictions
from src.utils.torch_amp import autocast, make_grad_scaler


class TrainTimingProfile:
    STAGES = (
        'data_time',
        'target_time',
        'forward_time',
        'loss_time',
        'backward_time',
        'optim_time',
        'post_time',
        'iter_time',
    )

    def __init__(self, enabled=False, max_iters=50):
        self.enabled = enabled
        self.max_iters = max(1, int(max_iters))
        self.reset()

    def reset(self):
        self.count = 0
        self.totals = {name: 0.0 for name in self.STAGES}

    def should_profile(self, iteration_idx):
        return self.enabled and iteration_idx < self.max_iters

    def update(self, stage_times):
        if not self.enabled:
            return
        self.count += 1
        for name in self.STAGES:
            self.totals[name] += float(stage_times.get(name, 0.0))

    def has_samples(self):
        return self.count > 0

    def format_summary(self, epoch, eval_time=None):
        if not self.has_samples():
            return ''

        avg_iter_ms = (self.totals['iter_time'] / self.count) * 1000.0
        lines = [
            f"[Train Profile][Epoch {epoch}] averaged over first {self.count} iterations:",
        ]
        lines.append(
            "  "
            + " | ".join(
                f"{name.replace('_time', '')}={self.totals[name] / self.count * 1000.0:.1f}ms"
                f" ({(self.totals[name] / max(self.totals['iter_time'], 1e-12)) * 100.0:.1f}%)"
                for name in self.STAGES
                if name != 'iter_time'
            )
            + f" | iter={avg_iter_ms:.1f}ms"
        )
        if eval_time is not None:
            lines.append(f"  eval={eval_time:.2f}s")
        return "\n".join(lines)


class Trainer:
    def __init__(self, model, train_loader, optimizer, scheduler, criterion, assigner,
                 device, epochs, accumulate=1, grad_clip=10.0, use_amp=True, evaluator=None, callbacks=None,
                 performance_cfg=None, eval_interval=1):
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
        self.use_amp = bool(use_amp and self.device.type == 'cuda')
        self.eval_interval = max(1, int(eval_interval))
        self.performance_cfg = dict(performance_cfg or {})
        self.progress_log_interval = max(1, int(self.performance_cfg.get('log_interval', 10)))
        self.profile_train = bool(self.performance_cfg.get('profile_train', False))
        self.profile_iters = max(1, int(self.performance_cfg.get('profile_iters', 50)))
        self.profile_cuda = bool(self.performance_cfg.get('profile_cuda', self.device.type == 'cuda'))
        self.debug_loss_steps = max(0, int(self.performance_cfg.get('debug_loss_steps', 0)))
        self.debug_anomaly_steps = max(1, int(self.performance_cfg.get('debug_anomaly_steps', 5)))
        self.loss_alert_threshold = float(self.performance_cfg.get('loss_alert_threshold', 20.0))
        self.pred_alert_threshold = float(self.performance_cfg.get('pred_alert_threshold', 50.0))
        temporal_debug_cfg = self.performance_cfg.get('temporal_debug', {}) if isinstance(self.performance_cfg, dict) else {}
        if not isinstance(temporal_debug_cfg, dict):
            temporal_debug_cfg = {}
        self.temporal_debug_enabled = bool(temporal_debug_cfg.get('enabled', False))
        self.temporal_log_epoch_reset = bool(temporal_debug_cfg.get('log_epoch_reset', True))

        self.evaluator = evaluator
        self.callbacks = callbacks or []
        self.scaler = make_grad_scaler(self.device.type, enabled=self.use_amp)
        self._anchor_cache = {}
        self._debug_zero_pos_count = 0
        self._debug_loss_anomaly_count = 0
        self._named_warning_counts = {}

        self.current_epoch = 0
        self.current_metrics = None
        self.did_validate_this_epoch = False
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

    def _profile_stamp(self):
        if self.profile_cuda and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _get_anchor_points(self, feats_for_anchors, strides, input_hw):
        cache_key = (
            tuple(
                (tuple(feat.shape[-2:]), str(feat.device), str(feat.dtype))
                for feat in feats_for_anchors
            ),
            tuple(int(v) for v in input_hw),
        )
        anchor_points = self._anchor_cache.get(cache_key)
        if anchor_points is None:
            anchor_points, _ = make_anchors(feats_for_anchors, strides)
            anchor_points = normalize_anchor_points(anchor_points, input_hw)
            self._anchor_cache = {cache_key: anchor_points}
        return anchor_points

    def _should_log_zero_pos_debug(self, num_pos):
        if num_pos != 0:
            return False
        if self._debug_zero_pos_count >= self.debug_anomaly_steps:
            return False
        self._debug_zero_pos_count += 1
        return True

    def _should_log_loss_anomaly(self):
        if self._debug_loss_anomaly_count >= self.debug_anomaly_steps:
            return False
        self._debug_loss_anomaly_count += 1
        return True

    def _should_log_named_warning(self, name):
        count = self._named_warning_counts.get(name, 0)
        if count >= self.debug_anomaly_steps:
            return False
        self._named_warning_counts[name] = count + 1
        return True

    def _collect_grad_debug(self):
        total_tensors = 0
        grad_tensors = 0
        nonzero_grad_tensors = 0

        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            total_tensors += 1
            if parameter.grad is None:
                continue

            grad = parameter.grad.detach().float()
            grad_tensors += 1
            if bool(torch.any(grad != 0).item()):
                nonzero_grad_tensors += 1

        return {
            'total_tensors': total_tensors,
            'grad_tensors': grad_tensors,
            'nonzero_grad_tensors': nonzero_grad_tensors,
        }

    @staticmethod
    def _tensor_stats(tensor, apply_sigmoid=False):
        if tensor is None:
            return {'count': 0, 'finite': True, 'nonfinite_count': 0, 'min': None, 'max': None}

        detached = tensor.detach().float()
        if apply_sigmoid:
            detached = detached.sigmoid()
        if detached.numel() == 0:
            return {'count': 0, 'finite': True, 'nonfinite_count': 0, 'min': None, 'max': None}

        finite_mask = torch.isfinite(detached)
        nonfinite_count = int((~finite_mask).sum().item())
        if finite_mask.any():
            finite_values = detached[finite_mask]
            return {
                'count': detached.numel(),
                'finite': nonfinite_count == 0,
                'nonfinite_count': nonfinite_count,
                'min': float(finite_values.min().item()),
                'max': float(finite_values.max().item()),
            }

        return {
            'count': detached.numel(),
            'finite': False,
            'nonfinite_count': nonfinite_count,
            'min': None,
            'max': None,
        }

    @staticmethod
    def _format_tensor_stats(stats):
        if stats['count'] == 0:
            return 'empty'
        if stats['min'] is None or stats['max'] is None:
            return f"all_nonfinite(count={stats['nonfinite_count']})"
        suffix = '' if stats['finite'] else f" nonfinite={stats['nonfinite_count']}"
        return f"[{stats['min']:.4f}, {stats['max']:.4f}]{suffix}"

    def _compute_temporal_ramp(self, epoch, iteration_idx):
        if not getattr(self.model, 'temporal_enabled', False):
            return 0.0

        temporal_weight = float(getattr(self.criterion, 'temporal_weight', 0.0))
        if temporal_weight <= 0.0:
            return 0.0

        warmup_epochs = float(getattr(self.criterion, 'temporal_warmup_epochs', 0.0))
        ramp_epochs = float(getattr(self.criterion, 'temporal_ramp_epochs', 0.0))
        progress_epoch = float(epoch) + float(iteration_idx + 1) / max(len(self.train_loader), 1)
        if progress_epoch <= warmup_epochs:
            return 0.0
        if ramp_epochs <= 0.0:
            return 1.0
        return min(max((progress_epoch - warmup_epochs) / max(ramp_epochs, 1e-6), 0.0), 1.0)

    def _stabilize_auxiliary_loss(self, name, loss_tensor, zero_term, epoch, iteration_idx):
        if loss_tensor is None:
            return zero_term, None

        if not torch.is_tensor(loss_tensor):
            loss_tensor = zero_term + float(loss_tensor)

        if not bool(torch.isfinite(loss_tensor).all().item()):
            if self._should_log_named_warning(name):
                print(
                    f"[TrainDebug][Epoch {epoch + 1} Iter {iteration_idx + 1}] "
                    f"{name}_loss is non-finite; dropping auxiliary branch for this batch."
                )
            return zero_term, f'{name}_non_finite'

        max_loss = float(getattr(self.criterion, f'{name}_max_loss', float('inf')))
        skip_threshold = float(getattr(self.criterion, f'{name}_skip_loss_threshold', float('inf')))
        loss_abs = abs(float(loss_tensor.detach().item()))

        if math.isfinite(skip_threshold) and loss_abs > skip_threshold:
            if self._should_log_named_warning(name):
                print(
                    f"[TrainDebug][Epoch {epoch + 1} Iter {iteration_idx + 1}] "
                    f"{name}_loss={loss_abs:.6f} exceeded skip threshold {skip_threshold:.6f}; skipping."
                )
            return zero_term, f'{name}_skipped_large'

        if math.isfinite(max_loss) and loss_abs > max_loss:
            if self._should_log_named_warning(name):
                print(
                    f"[TrainDebug][Epoch {epoch + 1} Iter {iteration_idx + 1}] "
                    f"{name}_loss={loss_abs:.6f} exceeded cap {max_loss:.6f}; clamping."
                )
            loss_tensor = loss_tensor.clamp(min=-max_loss, max=max_loss)
            return loss_tensor, f'{name}_clamped'

        return loss_tensor, None

    def _detect_loss_anomaly(self, loss_components, loss_total_raw, pred_scores, matched_pred_box):
        if not bool(torch.isfinite(loss_total_raw).all().item()):
            return 'total_loss_non_finite'

        for name, value in loss_components.items():
            if not math.isfinite(value):
                return f'{name}_non_finite'
            if abs(value) > self.loss_alert_threshold:
                return f'{name}_too_large'

        pred_score_stats = self._tensor_stats(pred_scores)
        if not pred_score_stats['finite']:
            return 'pred_scores_non_finite'
        if pred_score_stats['count'] > 0 and max(abs(pred_score_stats['min']), abs(pred_score_stats['max'])) > self.pred_alert_threshold:
            return 'pred_scores_too_large'

        matched_box_stats = self._tensor_stats(matched_pred_box)
        if not matched_box_stats['finite']:
            return 'matched_pred_box_non_finite'
        if matched_box_stats['count'] > 0 and max(abs(matched_box_stats['min']), abs(matched_box_stats['max'])) > self.pred_alert_threshold:
            return 'matched_pred_box_too_large'

        return None

    def _log_loss_debug(
        self,
        epoch,
        iteration_idx,
        batch_gt_count,
        per_image_gt_counts,
        num_pos,
        flat_preds,
        pred_bboxes,
        pred_scores,
        target_scores,
        gt_bboxes,
        mask_gt,
        anchor_points,
        matched_pred_box,
        loss_components,
        loss_total_raw,
        loss_for_backward,
        temporal_ramp,
        grad_debug=None,
        reason=None,
    ):
        prefix = f"[TrainDebug][Epoch {epoch + 1} Iter {iteration_idx + 1}]"
        finite = bool(torch.isfinite(loss_for_backward).item())

        print(
            f"{prefix} gt_count={batch_gt_count} per_image_gt={per_image_gt_counts} "
            f"num_pos={num_pos}"
            + (f" reason={reason}" if reason else "")
        )
        print(
            f"{prefix} shapes flat_preds={tuple(flat_preds.shape)} "
            f"pred_bboxes={tuple(pred_bboxes.shape)} pred_scores={tuple(pred_scores.shape)} "
            f"gt_bboxes={tuple(gt_bboxes.shape)}"
        )
        print(
            f"{prefix} loss cls={loss_components['cls']:.6f} "
            f"box={loss_components['box']:.6f} "
            f"angle={loss_components['angle']:.6f} "
            f"contrastive={loss_components['contrastive']:.6f} "
            f"temporal={loss_components['temporal']:.6f} "
            f"temporal_ramp={temporal_ramp:.4f} "
            f"total={loss_total_raw.detach().item():.6f} "
            f"backward_loss={loss_for_backward.detach().item():.6f} "
            f"dfl=n/a"
        )
        print(
            f"{prefix} tensor dtype={loss_for_backward.dtype} device={loss_for_backward.device} "
            f"requires_grad={loss_for_backward.requires_grad} finite={finite}"
        )
        print(
            f"{prefix} pred_scores(logit)={self._format_tensor_stats(self._tensor_stats(pred_scores))} "
            f"pred_scores(prob)={self._format_tensor_stats(self._tensor_stats(pred_scores, apply_sigmoid=True))} "
            f"target_scores={self._format_tensor_stats(self._tensor_stats(target_scores))} "
            f"matched_pred_box={self._format_tensor_stats(self._tensor_stats(matched_pred_box))}"
        )

        if batch_gt_count > 0:
            valid_gt_boxes = gt_bboxes[mask_gt.squeeze(-1).bool()]
            gt_center_min = valid_gt_boxes[:, :2].amin(dim=0).detach().cpu().tolist()
            gt_center_max = valid_gt_boxes[:, :2].amax(dim=0).detach().cpu().tolist()
            anchor_min = anchor_points.amin(dim=0).detach().cpu().tolist()
            anchor_max = anchor_points.amax(dim=0).detach().cpu().tolist()
            print(
                f"{prefix} anchor_range=({anchor_min[0]:.4f}, {anchor_min[1]:.4f}) -> "
                f"({anchor_max[0]:.4f}, {anchor_max[1]:.4f}) "
                f"gt_center_range=({gt_center_min[0]:.4f}, {gt_center_min[1]:.4f}) -> "
                f"({gt_center_max[0]:.4f}, {gt_center_max[1]:.4f})"
            )

        if grad_debug is not None:
            print(
                f"{prefix} grad_tensors={grad_debug['grad_tensors']}/{grad_debug['total_tensors']} "
                f"nonzero_grad_tensors={grad_debug['nonzero_grad_tensors']}"
            )

    @staticmethod
    def _format_loss_for_display(loss_value):
        loss_value = float(loss_value)
        if abs(loss_value) < 5e-8:
            loss_value = 0.0
        return f"{loss_value:.4f}"

    def _should_run_evaluation(self, epoch):
        if not self.evaluator:
            return False
        epoch_number = epoch + 1
        return (epoch_number % self.eval_interval == 0) or (epoch_number == self.epochs)

    def _reset_model_temporal_state(self, clear_memory=False, epoch=None, iteration_idx=None, reason=None):
        if hasattr(self.model, 'reset_temporal_state'):
            self.model.reset_temporal_state(clear_memory=clear_memory)
        elif clear_memory and hasattr(self.model, 'reset_temporal_memory'):
            self.model.reset_temporal_memory()
        elif hasattr(self.model, 'clear_temporal_step_state'):
            self.model.clear_temporal_step_state()
        elif hasattr(self.model, 'last_temporal_state'):
            self.model.last_temporal_state = None

        if not (self.temporal_debug_enabled and self.temporal_log_epoch_reset):
            return
        if not getattr(self.model, 'temporal_enabled', False):
            return
        state_fn = getattr(self.model, 'get_temporal_debug_state', None)
        state = state_fn() if callable(state_fn) else {}
        prefix = '[TemporalState]'
        if epoch is not None:
            prefix += f"[Epoch {epoch + 1}]"
        if iteration_idx is not None:
            prefix += f"[Iter {iteration_idx + 1}]"
        print(
            f"{prefix} reset reason={reason or 'unspecified'} clear_memory={clear_memory} "
            f"mode={state.get('mode', 'unknown')} "
            f"memory={state.get('memory_size', 0)}/{state.get('memory_len', 0)} "
            f"has_step_state={state.get('has_step_state', False)}"
        )

    def format_targets(self, targets, batch_size):
        if targets.numel() == 0:
            gt_labels = torch.zeros((batch_size, 0, 1), device=self.device)
            gt_bboxes = torch.zeros((batch_size, 0, 5), device=self.device)
            mask_gt = torch.zeros((batch_size, 0, 1), device=self.device)
            return gt_labels, gt_bboxes, mask_gt

        targets = targets.to(self.device, non_blocking=True)
        batch_indices = targets[:, 0].long()
        gt_counts = torch.bincount(batch_indices, minlength=batch_size)
        max_gts = int(gt_counts.max().item())

        gt_labels = torch.zeros((batch_size, max_gts, 1), device=self.device)
        gt_bboxes = torch.zeros((batch_size, max_gts, 5), device=self.device)
        mask_gt = torch.zeros((batch_size, max_gts, 1), device=self.device)

        write_positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for target, batch_idx in zip(targets, batch_indices):
            position = write_positions[batch_idx]
            gt_labels[batch_idx, position, 0] = target[1]
            gt_bboxes[batch_idx, position, :] = target[2:7]
            mask_gt[batch_idx, position, 0] = 1.0
            write_positions[batch_idx] += 1

        return gt_labels, gt_bboxes, mask_gt

    def train(self):
        self.trigger_callbacks('on_train_begin')

        for epoch in range(self.epochs):
            if self.stop_training:
                break
            self.current_epoch = epoch

            self.model.train()
            self._reset_model_temporal_state(clear_memory=True, epoch=epoch, reason='epoch_begin')
            train_dataset = getattr(self.train_loader, 'dataset', None)
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch, self.epochs)

            self.trigger_callbacks('on_epoch_begin')

            num_batches = len(self.train_loader)
            train_iter = iter(self.train_loader)
            pbar = tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{self.epochs}")
            self.optimizer.zero_grad()
            total_loss_epoch = None
            total_angle_epoch = None
            timing_profile = TrainTimingProfile(enabled=self.profile_train, max_iters=self.profile_iters)
            last_iter_end = time.perf_counter()

            for i in range(num_batches):
                try:
                    imgs_rgb, imgs_ir, targets, prev_rgb, prev_ir = next(train_iter)
                except StopIteration:
                    break
                except Exception as exc:
                    raise RuntimeError(
                        f"Training dataloader failed at epoch {epoch + 1}, batch {i + 1}/{num_batches}."
                    ) from exc
                self.trigger_callbacks('on_batch_begin')
                if i == 0:
                    print(f'First train batch imgs_rgb.shape: {tuple(imgs_rgb.shape)}')
                profile_iter = timing_profile.should_profile(i)
                stage_times = {name: 0.0 for name in TrainTimingProfile.STAGES} if profile_iter else None
                iter_start = time.perf_counter()
                if profile_iter:
                    stage_times['data_time'] = iter_start - last_iter_end
                    stage_start = self._profile_stamp()

                imgs_rgb = imgs_rgb.to(self.device, non_blocking=True)
                imgs_ir = imgs_ir.to(self.device, non_blocking=True)
                prev_rgb = prev_rgb.to(self.device, non_blocking=True)
                prev_ir = prev_ir.to(self.device, non_blocking=True)
                batch_size = imgs_rgb.shape[0]
                if profile_iter:
                    stage_times['data_time'] += self._profile_stamp() - stage_start
                    stage_start = self._profile_stamp()
                gt_labels, gt_bboxes, mask_gt = self.format_targets(targets, batch_size)
                batch_gt_count = int(mask_gt.sum().item())
                per_image_gt_counts = [int(x) for x in mask_gt.squeeze(-1).sum(dim=1).tolist()]
                if profile_iter:
                    stage_times['target_time'] = self._profile_stamp() - stage_start

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    if profile_iter:
                        stage_start = self._profile_stamp()
                    outputs, feat_rgb, feat_ir = self.model(imgs_rgb, imgs_ir, prev_rgb=prev_rgb, prev_ir=prev_ir)
                    flat_preds, feats_for_anchors = flatten_predictions(outputs)

                    strides = [4, 8, 16, 32]
                    anchor_points = self._get_anchor_points(feats_for_anchors, strides, imgs_rgb.shape[-2:])
                    if profile_iter:
                        stage_times['forward_time'] = self._profile_stamp() - stage_start

                    pred_bboxes, pred_scores = flat_preds[..., :5], flat_preds[..., 5:]
                    if profile_iter:
                        stage_start = self._profile_stamp()

                    target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
                        pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
                    )
                    pos_mask = fg_mask.bool()
                    num_pos = int(pos_mask.sum().item())

                    zero_term = flat_preds[..., :1].sum() * 0.0
                    contrastive_loss = zero_term
                    if getattr(self.model, 'use_contrastive', False) and epoch >= 10:
                        contrastive_loss = self.model.get_contrastive_alignment_loss(feat_rgb, feat_ir) * 0.1

                    temporal_loss = zero_term
                    temporal_ramp = self._compute_temporal_ramp(epoch, i)
                    temporal_reason = None
                    if getattr(self.model, 'temporal_enabled', False) and temporal_ramp > 0.0:
                        temporal_candidate = self.model.get_temporal_consistency_loss(
                            lambda_t=getattr(self.criterion, 'temporal_weight', 0.1) * temporal_ramp,
                            low_motion_bias=getattr(self.criterion, 'temporal_low_motion_bias', 0.75)
                        )
                        temporal_loss, temporal_reason = self._stabilize_auxiliary_loss(
                            name='temporal',
                            loss_tensor=temporal_candidate,
                            zero_term=zero_term,
                            epoch=epoch,
                            iteration_idx=i,
                        )

                    matched_pred_box = None
                    if pos_mask.any():
                        matched_pred_cls = pred_scores[pos_mask]
                        matched_pred_box = pred_bboxes[pos_mask]
                        matched_tgt_cls = target_labels[pos_mask]
                        matched_tgt_box = target_bboxes[pos_mask]
                        loss_total_raw, loss_cls, loss_reg, loss_angle = self.criterion(
                            matched_pred_cls,
                            matched_pred_box,
                            matched_tgt_cls,
                            matched_tgt_box,
                            contrastive_loss,
                            epoch,
                            temporal_loss=temporal_loss,
                        )
                        zero_pos_reason = None
                    else:
                        loss_cls = zero_term
                        loss_reg = zero_term
                        loss_angle = zero_term
                        loss_total_raw = zero_term + contrastive_loss + temporal_loss
                        zero_pos_reason = (
                            'assigner returned zero positives for a non-empty GT batch'
                            if batch_gt_count > 0 else
                            'batch has no GT labels'
                        )

                    loss_total = loss_total_raw / self.accumulate
                    loss_components = {
                        'cls': float(loss_cls.detach().item()),
                        'box': float(loss_reg.detach().item()),
                        'angle': float(loss_angle.detach().item()),
                        'contrastive': float(contrastive_loss.detach().item()),
                        'temporal': float(temporal_loss.detach().item()),
                    }
                    anomaly_reason = temporal_reason or self._detect_loss_anomaly(
                        loss_components=loss_components,
                        loss_total_raw=loss_total_raw,
                        pred_scores=pred_scores,
                        matched_pred_box=matched_pred_box,
                    )
                    self._reset_model_temporal_state(clear_memory=False, epoch=epoch, iteration_idx=i, reason='post_forward')
                    if profile_iter:
                        stage_times['loss_time'] = self._profile_stamp() - stage_start

                should_debug_step = i < self.debug_loss_steps
                should_log_zero_pos = self._should_log_zero_pos_debug(num_pos)
                should_log_loss_anomaly = bool(anomaly_reason) and self._should_log_loss_anomaly()
                debug_reason = "; ".join(
                    item for item in (zero_pos_reason, anomaly_reason) if item
                ) or None

                if not torch.isfinite(loss_total):
                    if should_debug_step or should_log_zero_pos or should_log_loss_anomaly:
                        self._log_loss_debug(
                            epoch=epoch,
                            iteration_idx=i,
                            batch_gt_count=batch_gt_count,
                            per_image_gt_counts=per_image_gt_counts,
                            num_pos=num_pos,
                            flat_preds=flat_preds,
                            pred_bboxes=pred_bboxes,
                            pred_scores=pred_scores,
                            target_scores=target_scores,
                            gt_bboxes=gt_bboxes,
                            mask_gt=mask_gt,
                            anchor_points=anchor_points,
                            matched_pred_box=matched_pred_box,
                            loss_components=loss_components,
                            loss_total_raw=loss_total_raw,
                            loss_for_backward=loss_total,
                            temporal_ramp=temporal_ramp,
                            grad_debug=None,
                            reason=debug_reason or 'non_finite_total_loss',
                        )
                    print("\nWarning: encountered non-finite loss, skipping batch.")
                    self.optimizer.zero_grad()
                    pbar.update(1)
                    continue

                if profile_iter:
                    stage_start = self._profile_stamp()
                self.scaler.scale(loss_total).backward()
                if profile_iter:
                    stage_times['backward_time'] = self._profile_stamp() - stage_start

                if should_debug_step or should_log_zero_pos or should_log_loss_anomaly:
                    grad_debug = self._collect_grad_debug()
                    self._log_loss_debug(
                        epoch=epoch,
                        iteration_idx=i,
                        batch_gt_count=batch_gt_count,
                        per_image_gt_counts=per_image_gt_counts,
                        num_pos=num_pos,
                        flat_preds=flat_preds,
                        pred_bboxes=pred_bboxes,
                        pred_scores=pred_scores,
                        target_scores=target_scores,
                        gt_bboxes=gt_bboxes,
                        mask_gt=mask_gt,
                        anchor_points=anchor_points,
                        matched_pred_box=matched_pred_box,
                        loss_components=loss_components,
                        loss_total_raw=loss_total_raw,
                        loss_for_backward=loss_total,
                        temporal_ramp=temporal_ramp,
                        grad_debug=grad_debug,
                        reason=debug_reason,
                    )

                if profile_iter:
                    stage_start = self._profile_stamp()
                if (i + 1) % self.accumulate == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.trigger_callbacks('on_batch_end')
                if profile_iter:
                    stage_times['optim_time'] = self._profile_stamp() - stage_start

                batch_loss = loss_total.detach() * self.accumulate
                total_loss_epoch = batch_loss if total_loss_epoch is None else total_loss_epoch + batch_loss
                batch_angle = loss_angle.detach()
                total_angle_epoch = batch_angle if total_angle_epoch is None else total_angle_epoch + batch_angle
                should_log = ((i + 1) % self.progress_log_interval == 0) or ((i + 1) == len(self.train_loader))
                if should_log and total_loss_epoch is not None:
                    avg_loss = total_loss_epoch.item() / (i + 1)
                    postfix = {
                        "Loss": self._format_loss_for_display(avg_loss),
                        "Pos": num_pos,
                        "GT": batch_gt_count,
                    }
                    if getattr(self.criterion, 'angle_enabled', False) and getattr(self.criterion, 'angle_weight', 0.0) > 0.0:
                        avg_angle = 0.0 if total_angle_epoch is None else total_angle_epoch.item() / (i + 1)
                        postfix["Angle"] = self._format_loss_for_display(avg_angle)
                    pbar.set_postfix(postfix)
                pbar.update(1)

                if profile_iter:
                    iter_end = self._profile_stamp()
                    stage_times['iter_time'] = iter_end - iter_start
                    measured_time = sum(stage_times[name] for name in TrainTimingProfile.STAGES if name not in {'post_time', 'iter_time'})
                    stage_times['post_time'] = max(stage_times['iter_time'] - measured_time, 0.0)
                    timing_profile.update(stage_times)
                    last_iter_end = iter_end
                else:
                    last_iter_end = time.perf_counter()
            pbar.close()

            self.scheduler.step()

            eval_time = None
            self.current_metrics = None
            self.did_validate_this_epoch = False
            if self._should_run_evaluation(epoch):
                eval_model = getattr(self, 'ema_callback').ema if hasattr(self, 'ema_callback') else self.model
                eval_start = self._profile_stamp() if self.profile_train else time.perf_counter()
                self.current_metrics = self.evaluator.evaluate(eval_model, epoch=epoch + 1)
                eval_end = self._profile_stamp() if self.profile_train else time.perf_counter()
                eval_time = eval_end - eval_start
                self.did_validate_this_epoch = True

            if timing_profile.has_samples():
                print(f"\n{timing_profile.format_summary(epoch + 1, eval_time=eval_time)}")

            self.trigger_callbacks('on_epoch_end')
            self._reset_model_temporal_state(clear_memory=True, epoch=epoch, reason='epoch_end')

        self.trigger_callbacks('on_train_end')
