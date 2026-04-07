import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

from src.metrics.obb_iou_backend import OBB_IOU_BACKEND_GPU_PROB, build_obb_iou_backend
from src.metrics.task_metrics import normalize_eval_metrics_cfg
from src.metrics.task_specific_metrics import resolve_small_area_threshold


def compute_ap_torch(recall, precision):
    zero = torch.zeros(1, dtype=recall.dtype, device=recall.device)
    one = torch.ones(1, dtype=recall.dtype, device=recall.device)
    mrec = torch.cat((zero, recall, one))
    mpre = torch.cat((one, precision, zero))
    for idx in range(mpre.numel() - 1, 0, -1):
        mpre[idx - 1] = torch.maximum(mpre[idx - 1], mpre[idx])
    changed = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).squeeze(1)
    if changed.numel() == 0:
        return zero.squeeze(0)
    return torch.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1])


class GPUOBBMetricsEvaluator:
    """GPU-first detection metrics evaluator for the maintained detection path."""

    def __init__(self, num_classes=5, device=None, small_area_thresh=1024, extra_metrics_cfg=None):
        if torch is None:
            raise RuntimeError('GPUOBBMetricsEvaluator requires torch, but torch is not available.')

        self.nc = num_classes
        self.device = torch.device(device or 'cuda')
        if self.device.type != 'cuda':
            raise RuntimeError(
                f"GPUOBBMetricsEvaluator requires a CUDA device, but got '{self.device}'."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                'GPUOBBMetricsEvaluator requires CUDA, but torch.cuda.is_available() is False.'
            )

        self.default_small_area_thresh = small_area_thresh
        self.extra_metrics_cfg = normalize_eval_metrics_cfg(extra_metrics_cfg)
        self.obb_iou_backend_name = self.extra_metrics_cfg.get('obb_iou_backend', OBB_IOU_BACKEND_GPU_PROB)
        if self.obb_iou_backend_name != OBB_IOU_BACKEND_GPU_PROB:
            raise ValueError(
                "GPUOBBMetricsEvaluator requires eval.obb_iou_backend='gpu_prob' "
                f"but received '{self.obb_iou_backend_name}'."
            )
        self.keep_cpu_artifacts = bool(self.extra_metrics_cfg['error_analysis'].get('enabled', False))
        self.report_conf_threshold = float(self.extra_metrics_cfg.get('report_conf_threshold', 0.25))
        self.small_area_thresh = resolve_small_area_threshold(
            self.extra_metrics_cfg['small_object'].get('area_threshold', np.sqrt(float(small_area_thresh)))
        )
        self.reset()

    def reset(self):
        self.preds = []
        self.gts = []
        self.image_metadata = {}
        self._pred_rows = []
        self._gt_rows = []
        self._image_ids = []

    def _empty_pred_rows(self):
        return torch.zeros((0, 8), dtype=torch.float32, device=self.device)

    def _empty_gt_rows(self):
        return torch.zeros((0, 7), dtype=torch.float32, device=self.device)

    def _stack_pred_rows(self):
        if not self._pred_rows:
            return self._empty_pred_rows()
        return torch.cat(self._pred_rows, dim=0)

    def _stack_gt_rows(self):
        if not self._gt_rows:
            return self._empty_gt_rows()
        return torch.cat(self._gt_rows, dim=0)

    def _build_iou_backend(self):
        return build_obb_iou_backend(self.obb_iou_backend_name, device=self.device)

    def add_batch(self, image_ids, batch_preds, batch_gts, batch_metadata=None):
        batch_metadata = batch_metadata or [None] * len(image_ids)
        for batch_idx, img_id in enumerate(image_ids):
            image_index = len(self._image_ids)
            self._image_ids.append(img_id)
            self.image_metadata[img_id] = batch_metadata[batch_idx]

            pred_tensor = batch_preds[batch_idx]
            if not isinstance(pred_tensor, torch.Tensor):
                pred_tensor = torch.as_tensor(pred_tensor, dtype=torch.float32, device=self.device)
            pred_tensor = pred_tensor.to(self.device, dtype=torch.float32)
            if pred_tensor.numel() > 0:
                pred_tensor = pred_tensor.reshape(-1, 7)
                image_column = torch.full((pred_tensor.shape[0], 1), float(image_index), dtype=torch.float32, device=self.device)
                pred_rows = torch.cat(
                    [image_column, pred_tensor[:, 6:7], pred_tensor[:, 5:6], pred_tensor[:, :5]],
                    dim=1,
                )
                self._pred_rows.append(pred_rows)
                if self.keep_cpu_artifacts:
                    for pred in pred_tensor.detach().cpu().numpy():
                        self.preds.append({
                            'image_id': img_id,
                            'bbox': pred[:5],
                            'score': float(pred[5]),
                            'class': int(pred[6]),
                        })

            gt_tensor = batch_gts[batch_idx]
            if not isinstance(gt_tensor, torch.Tensor):
                gt_tensor = torch.as_tensor(gt_tensor, dtype=torch.float32, device=self.device)
            gt_tensor = gt_tensor.to(self.device, dtype=torch.float32)
            if gt_tensor.numel() > 0:
                gt_tensor = gt_tensor.reshape(-1, 6)
                image_column = torch.full((gt_tensor.shape[0], 1), float(image_index), dtype=torch.float32, device=self.device)
                gt_rows = torch.cat([image_column, gt_tensor[:, :1], gt_tensor[:, 1:6]], dim=1)
                self._gt_rows.append(gt_rows)
                if self.keep_cpu_artifacts:
                    for gt in gt_tensor.detach().cpu().numpy():
                        self.gts.append({
                            'image_id': img_id,
                            'class': int(gt[0]),
                            'bbox': gt[1:6],
                        })

    def _compute_detection_metrics(self, pred_rows, gt_rows, iou_thresh=0.5, eval_small_only=False,
                                   score_threshold=None, iou_backend=None):
        aps = []
        precisions = []
        recalls = []
        iou_backend = iou_backend or self._build_iou_backend()

        pred_rows = pred_rows if pred_rows is not None else self._stack_pred_rows()
        gt_rows = gt_rows if gt_rows is not None else self._stack_gt_rows()

        for class_id in range(self.nc):
            class_preds = pred_rows[pred_rows[:, 1].long() == class_id]
            class_gts = gt_rows[gt_rows[:, 1].long() == class_id]
            if score_threshold is not None:
                class_preds = class_preds[class_preds[:, 2] >= float(score_threshold)]
            if eval_small_only:
                class_gts = class_gts[(class_gts[:, 4] * class_gts[:, 5]) < float(self.small_area_thresh)]

            num_gts = int(class_gts.shape[0])
            if num_gts == 0:
                continue

            if class_preds.shape[0] == 0:
                zero = gt_rows.new_tensor(0.0)
                aps.append(zero)
                precisions.append(zero)
                recalls.append(zero)
                continue

            class_preds = class_preds[torch.argsort(class_preds[:, 2], descending=True)]
            pred_image_ids = class_preds[:, 0].long()
            tp = torch.zeros(class_preds.shape[0], dtype=torch.float32, device=self.device)
            fp = torch.zeros_like(tp)

            for image_index in torch.unique(pred_image_ids).tolist():
                pred_positions = torch.nonzero(pred_image_ids == image_index, as_tuple=False).squeeze(1)
                gt_positions = torch.nonzero(class_gts[:, 0].long() == image_index, as_tuple=False).squeeze(1)
                if gt_positions.numel() == 0:
                    fp[pred_positions] = 1.0
                    continue

                pred_boxes = class_preds[pred_positions][:, 3:8]
                gt_boxes = class_gts[gt_positions][:, 2:7]
                iou_matrix = iou_backend.pairwise_iou_tensor(pred_boxes, gt_boxes)
                matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=self.device)

                for local_pred_idx, global_pred_idx in enumerate(pred_positions.tolist()):
                    masked_row = iou_matrix[local_pred_idx].masked_fill(matched_gt, -1.0)
                    has_candidate = bool(torch.any(masked_row > -0.5).item())
                    if not has_candidate:
                        fp[global_pred_idx] = 1.0
                        continue

                    best_iou, best_gt_idx = masked_row.max(dim=0)
                    if bool((best_iou >= float(iou_thresh)).item()):
                        tp[global_pred_idx] = 1.0
                        matched_gt[best_gt_idx] = True
                    else:
                        fp[global_pred_idx] = 1.0

            fp_cum = torch.cumsum(fp, dim=0)
            tp_cum = torch.cumsum(tp, dim=0)
            recall = tp_cum / (float(num_gts) + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)
            aps.append(compute_ap_torch(recall, precision))
            precisions.append(tp.sum() / torch.clamp(tp.sum() + fp.sum(), min=1.0))
            recalls.append(tp.sum() / max(float(num_gts), 1.0))

        return {
            'mAP': float(torch.stack(aps).mean().item()) if aps else 0.0,
            'Precision': float(torch.stack(precisions).mean().item()) if precisions else 0.0,
            'Recall': float(torch.stack(recalls).mean().item()) if recalls else 0.0,
        }

    def _compute_map_range(self, pred_rows, gt_rows, thresholds=None, iou_backend=None):
        thresholds = thresholds if thresholds is not None else np.arange(0.5, 1.0, 0.05)
        iou_backend = iou_backend or self._build_iou_backend()
        map_values = [
            self._compute_detection_metrics(
                pred_rows,
                gt_rows,
                iou_thresh=thr,
                eval_small_only=False,
                iou_backend=iou_backend,
            )['mAP']
            for thr in thresholds
        ]
        return float(np.mean(map_values)) if map_values else 0.0

    def get_full_metrics(self):
        pred_rows = self._stack_pred_rows()
        gt_rows = self._stack_gt_rows()
        iou_backend = self._build_iou_backend()
        detection_metrics = self._compute_detection_metrics(
            pred_rows,
            gt_rows,
            iou_thresh=0.5,
            eval_small_only=False,
            iou_backend=iou_backend,
        )
        report_metrics = self._compute_detection_metrics(
            pred_rows,
            gt_rows,
            iou_thresh=0.5,
            eval_small_only=False,
            score_threshold=self.report_conf_threshold,
            iou_backend=iou_backend,
        )

        metrics = {
            'mAP_50': detection_metrics['mAP'],
            'mAP_50_95': self._compute_map_range(pred_rows, gt_rows, iou_backend=iou_backend),
            'Precision': report_metrics['Precision'],
            'Recall': report_metrics['Recall'],
            'ReportConfThreshold': float(self.report_conf_threshold),
        }

        if self.extra_metrics_cfg['small_object'].get('enabled', True):
            metrics['mAP_S'] = self._compute_detection_metrics(
                pred_rows,
                gt_rows,
                iou_thresh=0.5,
                eval_small_only=True,
                iou_backend=iou_backend,
            )['mAP']

        return metrics
