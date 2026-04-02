import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import autocast
from tqdm import tqdm

from src.metrics.error_analysis import ErrorAnalyzer
from src.metrics.task_metrics import normalize_eval_metrics_cfg
from src.metrics.task_specific_metrics import compute_cross_modal_robustness, infer_sequence_metadata
from src.model.bbox_utils import non_max_suppression_obb
from src.model.output_adapter import flatten_predictions
from src.utils.postprocess_tuning import apply_classwise_thresholds, normalize_infer_cfg
from src.utils.result_merge import merge_obb_predictions
from src.utils.tta import apply_tta_transforms, build_tta_transforms, invert_tta_predictions


class Evaluator:
    def __init__(self, dataloader, metrics_evaluator, device, nms_kwargs=None, extra_metrics_cfg=None, infer_cfg=None):
        self.dataloader = dataloader
        self.device = device
        self.metrics_evaluator = metrics_evaluator
        self.nms_kwargs = nms_kwargs or {'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 300}
        self.extra_metrics_cfg = normalize_eval_metrics_cfg(extra_metrics_cfg)
        dataset = getattr(dataloader, 'dataset', None)
        dataset_imgsz = getattr(dataset, 'img_size', getattr(dataset, 'imgsz', 1024)) if dataset is not None else 1024
        self.infer_cfg = normalize_infer_cfg(infer_cfg, default_imgsz=dataset_imgsz, nms_cfg=self.nms_kwargs)
        self.class_names = list(getattr(dataset, 'class_names', [])) if dataset is not None else []

        self.rank = 0
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()

    def _maybe_reset_model_temporal_state(self, model):
        if hasattr(model, 'reset_temporal_memory'):
            model.reset_temporal_memory()

    def _apply_drop_mode(self, imgs_rgb, imgs_ir, rgb_drop_mode=None, ir_drop_mode=None):
        dropped_rgb = imgs_rgb.clone()
        dropped_ir = imgs_ir.clone()
        if rgb_drop_mode == 'zero':
            dropped_rgb.zero_()
        if ir_drop_mode == 'zero':
            dropped_ir.zero_()
        return dropped_rgb, dropped_ir

    def _resolve_dataset_metadata(self, dataset, sample_idx):
        if dataset is None:
            return None

        if hasattr(dataset, 'get_eval_metadata'):
            metadata = dataset.get_eval_metadata(sample_idx)
            return dict(metadata) if isinstance(metadata, dict) else None

        metadata_source = getattr(dataset, 'metadata', None)
        if isinstance(metadata_source, list) and sample_idx < len(metadata_source):
            metadata = metadata_source[sample_idx]
            return dict(metadata) if isinstance(metadata, dict) else None

        if isinstance(metadata_source, dict):
            metadata = metadata_source.get(sample_idx)
            return dict(metadata) if isinstance(metadata, dict) else None

        return None

    def _build_image_ids_and_metadata(self, epoch, batch_idx, batch_size, sample_offset):
        dataset = getattr(self.dataloader, 'dataset', None)
        image_ids = []
        batch_metadata = []

        rgb_files = getattr(dataset, 'rgb_files', None)
        for sample_idx in range(batch_size):
            global_idx = sample_offset + sample_idx
            if rgb_files is not None and global_idx < len(rgb_files):
                image_path = Path(rgb_files[global_idx])
                image_id = str(image_path)
                metadata = infer_sequence_metadata(image_id) or {}
            else:
                image_id = f"{epoch}_{self.rank}_{batch_idx}_{sample_idx}"
                metadata = {}

            dataset_metadata = self._resolve_dataset_metadata(dataset, global_idx)
            if dataset_metadata:
                metadata.update(dataset_metadata)

            image_ids.append(image_id)
            batch_metadata.append(metadata or None)
        return image_ids, batch_metadata

    @staticmethod
    def _targets_to_batch_gts(targets, batch_size):
        if targets.numel() == 0:
            return [torch.zeros((0, 6), dtype=torch.float32) for _ in range(batch_size)]

        gt_counts = torch.bincount(targets[:, 0].long(), minlength=batch_size)
        max_gts = int(gt_counts.max().item())
        padded = torch.zeros((batch_size, max_gts, 6), dtype=torch.float32, device=targets.device)
        write_positions = torch.zeros(batch_size, dtype=torch.long, device=targets.device)

        for target in targets:
            batch_idx = int(target[0].item())
            position = write_positions[batch_idx]
            padded[batch_idx, position, :] = target[1:7]
            write_positions[batch_idx] += 1

        batch_gts = []
        for batch_idx in range(batch_size):
            count = int(gt_counts[batch_idx].item())
            batch_gts.append(padded[batch_idx, :count].cpu())
        return batch_gts

    def _prepare_metrics_batch(self, preds, targets, batch_size):
        batch_gts = self._targets_to_batch_gts(targets, batch_size)
        return [pred.cpu() for pred in preds], batch_gts

    def _snapshot_eval_artifacts(self):
        return {
            'preds': deepcopy(getattr(self.metrics_evaluator, 'preds', [])),
            'gts': deepcopy(getattr(self.metrics_evaluator, 'gts', [])),
            'image_metadata': deepcopy(getattr(self.metrics_evaluator, 'image_metadata', {})),
        }

    def _forward_flat_predictions(self, model, imgs_rgb, imgs_ir, prev_rgb, prev_ir):
        amp_enabled = self.device.type == 'cuda'
        with autocast(device_type=self.device.type, enabled=amp_enabled):
            model_out = model(imgs_rgb, imgs_ir, prev_rgb=prev_rgb, prev_ir=prev_ir)
            outputs = model_out[0] if isinstance(model_out, tuple) else model_out
            outputs, _ = flatten_predictions(outputs)

        outputs = outputs.detach()
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
        outputs[..., :5] = outputs[..., :5].clamp(-1e4, 1e4)

        topk_pre = 3000
        if outputs.shape[1] > topk_pre:
            scores = outputs[..., 5:].amax(dim=-1)
            _, idx = scores.topk(topk_pre, dim=-1)
            outputs = outputs.gather(1, idx.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]))

        outputs[..., 5:] = outputs[..., 5:].sigmoid()
        return outputs.float()

    def _run_single_nms(self, flat_preds):
        preds = non_max_suppression_obb(
            flat_preds,
            conf_thres=self.infer_cfg['conf_threshold'],
            iou_thres=self.infer_cfg['iou_threshold'],
            max_det=self.infer_cfg['merge']['max_det'],
            max_wh=self.nms_kwargs.get('max_wh', 4096.0),
        )
        return [
            apply_classwise_thresholds(
                pred,
                class_names=self.class_names,
                global_conf_threshold=self.infer_cfg['conf_threshold'],
                classwise_conf_thresholds=self.infer_cfg.get('classwise_conf_thresholds', {}),
            )
            for pred in preds
        ]

    def _predict_batch(self, model, imgs_rgb, imgs_ir, prev_rgb, prev_ir):
        base_size = imgs_rgb.shape[-1]
        use_toolbox = self.infer_cfg.get('enabled', False) or self.infer_cfg.get('mode', 'fast') != 'fast'

        if not use_toolbox:
            return self._run_single_nms(self._forward_flat_predictions(model, imgs_rgb, imgs_ir, prev_rgb, prev_ir))

        transforms = build_tta_transforms(self.infer_cfg, base_size)
        merged_per_image = [[] for _ in range(imgs_rgb.shape[0])]

        for transform_cfg in transforms:
            aug_rgb, aug_ir, aug_prev_rgb, aug_prev_ir = apply_tta_transforms(
                imgs_rgb,
                imgs_ir,
                prev_rgb,
                prev_ir,
                transform_cfg,
            )
            aug_preds = self._run_single_nms(self._forward_flat_predictions(model, aug_rgb, aug_ir, aug_prev_rgb, aug_prev_ir))
            for batch_idx, pred in enumerate(aug_preds):
                merged_per_image[batch_idx].append(
                    invert_tta_predictions(pred, transform_cfg=transform_cfg, base_size=base_size)
                )

        return [
            merge_obb_predictions(
                prediction_sets,
                method=self.infer_cfg['merge']['method'],
                iou_threshold=self.infer_cfg['merge']['iou_threshold'],
                max_det=self.infer_cfg['merge']['max_det'],
            )
            for prediction_sets in merged_per_image
        ]

    def _run_eval_pass(self, model, epoch=-1, rgb_drop_mode=None, ir_drop_mode=None, desc_prefix='Val'):
        model.eval()
        self.metrics_evaluator.reset()
        self._maybe_reset_model_temporal_state(model)

        pbar = tqdm(self.dataloader, desc=f"{desc_prefix} Epoch {epoch}", leave=False)
        amp_enabled = self.device.type == 'cuda'
        sample_offset = 0

        for batch_idx, (imgs_rgb, imgs_ir, targets, prev_rgb, prev_ir) in enumerate(pbar):
            imgs_rgb = imgs_rgb.to(self.device, non_blocking=True)
            imgs_ir = imgs_ir.to(self.device, non_blocking=True)
            prev_rgb = prev_rgb.to(self.device, non_blocking=True)
            prev_ir = prev_ir.to(self.device, non_blocking=True)

            eval_rgb, eval_ir = self._apply_drop_mode(imgs_rgb, imgs_ir, rgb_drop_mode=rgb_drop_mode, ir_drop_mode=ir_drop_mode)
            eval_prev_rgb, eval_prev_ir = self._apply_drop_mode(prev_rgb, prev_ir, rgb_drop_mode=rgb_drop_mode, ir_drop_mode=ir_drop_mode)

            profile_speed = amp_enabled and batch_idx < 10 and rgb_drop_mode is None and ir_drop_mode is None
            if profile_speed:
                torch.cuda.synchronize()
            t1 = time.time()
            preds = self._predict_batch(model, eval_rgb, eval_ir, eval_prev_rgb, eval_prev_ir)
            if profile_speed:
                torch.cuda.synchronize()
            t2 = time.time()
            t3 = t2

            image_ids, batch_metadata = self._build_image_ids_and_metadata(epoch, batch_idx, eval_rgb.shape[0], sample_offset)
            preds_for_metrics, batch_gts = self._prepare_metrics_batch(preds, targets, eval_rgb.shape[0])

            self.metrics_evaluator.add_batch(image_ids, preds_for_metrics, batch_gts, batch_metadata=batch_metadata)
            sample_offset += eval_rgb.shape[0]
            if rgb_drop_mode is None and ir_drop_mode is None:
                pbar.set_postfix({"infer(ms)": f"{(t2 - t1) * 1000:.1f}", "nms(ms)": f"{(t3 - t2) * 1000:.1f}"})

        return self.metrics_evaluator.get_full_metrics()

    @torch.no_grad()
    def evaluate(self, model, epoch=-1):
        baseline_metrics = self._run_eval_pass(model, epoch=epoch, desc_prefix='Val')
        baseline_artifacts = self._snapshot_eval_artifacts()
        metrics = dict(baseline_metrics)

        robustness_cfg = self.extra_metrics_cfg['cross_modal_robustness']
        rgb_artifacts = None
        ir_artifacts = None
        if robustness_cfg.get('enabled', False):
            rgb_metrics = self._run_eval_pass(
                model,
                epoch=epoch,
                rgb_drop_mode=robustness_cfg.get('rgb_drop_mode', 'zero'),
                desc_prefix='Val RGBDrop',
            )
            rgb_artifacts = self._snapshot_eval_artifacts()
            ir_metrics = self._run_eval_pass(
                model,
                epoch=epoch,
                ir_drop_mode=robustness_cfg.get('ir_drop_mode', 'zero'),
                desc_prefix='Val IRDrop',
                )
            ir_artifacts = self._snapshot_eval_artifacts()
            metrics.update(compute_cross_modal_robustness(
                baseline_metrics,
                rgb_metrics,
                ir_metrics,
                base_metric=robustness_cfg.get('base_metric', 'mAP_50'),
            ))

        error_cfg = self.extra_metrics_cfg['error_analysis']
        if error_cfg.get('enabled', False):
            dataset = getattr(self.dataloader, 'dataset', None)
            class_names = list(getattr(dataset, 'class_names', [])) if dataset is not None else []
            analysis = ErrorAnalyzer(
                cfg=self.extra_metrics_cfg,
                class_names=class_names,
            ).analyze(
                baseline_artifacts['preds'],
                baseline_artifacts['gts'],
                image_metadata=baseline_artifacts['image_metadata'],
                rgb_drop_data=rgb_artifacts,
                ir_drop_data=ir_artifacts,
                grouped_metrics=baseline_metrics.get('GroupedMetrics'),
            )
            metrics['ErrorAnalysis'] = analysis['summary']
            if analysis.get('exported_files'):
                metrics['ErrorAnalysisFiles'] = analysis['exported_files']

        if self.rank == 0:
            report_parts = [
                f"mAP@0.5: {metrics.get('mAP_50', 0):.4f}",
                f"mAP@0.5:0.95: {metrics.get('mAP_50_95', 0):.4f}",
                f"P: {metrics.get('Precision', 0):.4f}",
                f"R: {metrics.get('Recall', 0):.4f}",
            ]
            if 'mAP_S' in metrics:
                report_parts.append(f"AP_S: {metrics['mAP_S']:.4f}")
            print(f"\n[Val Report] {' | '.join(report_parts)}")
            if 'ErrorAnalysisFiles' in metrics:
                print(f"[Val ErrorAnalysis] exported to: {metrics['ErrorAnalysisFiles']}")

        model.train()
        return metrics


class GPUDetectionEvaluator(Evaluator):
    """Thin GPU evaluator wrapper that keeps batch predictions/targets on CUDA for metrics."""

    @staticmethod
    def _targets_to_batch_gts_device(targets, batch_size, device):
        if targets.numel() == 0:
            return [torch.zeros((0, 6), dtype=torch.float32, device=device) for _ in range(batch_size)]

        targets = targets.to(device, non_blocking=True)
        gt_counts = torch.bincount(targets[:, 0].long(), minlength=batch_size)
        max_gts = int(gt_counts.max().item())
        padded = torch.zeros((batch_size, max_gts, 6), dtype=torch.float32, device=device)
        write_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

        for target in targets:
            batch_idx = int(target[0].item())
            position = write_positions[batch_idx]
            padded[batch_idx, position, :] = target[1:7]
            write_positions[batch_idx] += 1

        batch_gts = []
        for batch_idx in range(batch_size):
            count = int(gt_counts[batch_idx].item())
            batch_gts.append(padded[batch_idx, :count])
        return batch_gts

    def _prepare_metrics_batch(self, preds, targets, batch_size):
        batch_gts = self._targets_to_batch_gts_device(targets, batch_size, self.device)
        return preds, batch_gts
