import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import autocast
from tqdm import tqdm

from src.metrics.task_specific_metrics import compute_cross_modal_robustness, infer_sequence_metadata
from src.model.bbox_utils import non_max_suppression_obb
from src.model.output_adapter import flatten_predictions


class Evaluator:
    def __init__(self, dataloader, metrics_evaluator, device, nms_kwargs=None, extra_metrics_cfg=None):
        self.dataloader = dataloader
        self.device = device
        self.metrics_evaluator = metrics_evaluator
        self.nms_kwargs = nms_kwargs or {'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 300}
        self.extra_metrics_cfg = self._build_extra_metrics_cfg(extra_metrics_cfg)

        self.rank = 0
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()

    def _build_extra_metrics_cfg(self, cfg):
        cfg = dict(cfg) if cfg is not None else {}
        robustness_cfg = dict(cfg.get('cross_modal_robustness', {}))
        return {
            'enabled': cfg.get('enabled', False),
            'cross_modal_robustness': {
                'enabled': robustness_cfg.get('enabled', False),
                'base_metric': robustness_cfg.get('base_metric', 'mAP_50'),
                'rgb_drop_mode': robustness_cfg.get('rgb_drop_mode', 'zero'),
                'ir_drop_mode': robustness_cfg.get('ir_drop_mode', 'zero'),
            },
        }

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
                metadata = infer_sequence_metadata(image_id)
            else:
                image_id = f"{epoch}_{self.rank}_{batch_idx}_{sample_idx}"
                metadata = None
            image_ids.append(image_id)
            batch_metadata.append(metadata)
        return image_ids, batch_metadata

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

            with autocast(device_type=self.device.type, enabled=amp_enabled):
                model_out = model(eval_rgb, eval_ir, prev_rgb=eval_prev_rgb, prev_ir=eval_prev_ir)
                outputs = model_out[0] if isinstance(model_out, tuple) else model_out
                outputs, _ = flatten_predictions(outputs)

            if profile_speed:
                torch.cuda.synchronize()
            t2 = time.time()

            outputs = outputs.detach()
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
            outputs[..., :5] = outputs[..., :5].clamp(-1e4, 1e4)

            topk_pre = 3000
            if outputs.shape[1] > topk_pre:
                scores = outputs[..., 5:].amax(dim=-1)
                _, idx = scores.topk(topk_pre, dim=-1)
                outputs = outputs.gather(1, idx.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]))

            outputs[..., 5:] = outputs[..., 5:].sigmoid()
            outputs = outputs.float()
            preds = non_max_suppression_obb(outputs, **self.nms_kwargs)

            if profile_speed:
                torch.cuda.synchronize()
            t3 = time.time()

            targets_np = targets.cpu().numpy()
            gt_dict = {b: [] for b in range(eval_rgb.shape[0])}
            for target in targets_np:
                gt_dict[int(target[0])].append(target[1:])
            batch_gts = [torch.tensor(gt_dict[i], dtype=torch.float32, device='cpu') for i in range(eval_rgb.shape[0])]

            image_ids, batch_metadata = self._build_image_ids_and_metadata(epoch, batch_idx, eval_rgb.shape[0], sample_offset)
            preds = [pred.cpu() for pred in preds]

            self.metrics_evaluator.add_batch(image_ids, preds, batch_gts, batch_metadata=batch_metadata)
            sample_offset += eval_rgb.shape[0]
            if rgb_drop_mode is None and ir_drop_mode is None:
                pbar.set_postfix({"infer(ms)": f"{(t2 - t1) * 1000:.1f}", "nms(ms)": f"{(t3 - t2) * 1000:.1f}"})

        return self.metrics_evaluator.get_full_metrics()

    @torch.no_grad()
    def evaluate(self, model, epoch=-1):
        baseline_metrics = self._run_eval_pass(model, epoch=epoch, desc_prefix='Val')
        metrics = dict(baseline_metrics)

        robustness_cfg = self.extra_metrics_cfg['cross_modal_robustness']
        if self.extra_metrics_cfg.get('enabled', False) and robustness_cfg.get('enabled', False):
            rgb_metrics = self._run_eval_pass(
                model,
                epoch=epoch,
                rgb_drop_mode=robustness_cfg.get('rgb_drop_mode', 'zero'),
                desc_prefix='Val RGBDrop',
            )
            ir_metrics = self._run_eval_pass(
                model,
                epoch=epoch,
                ir_drop_mode=robustness_cfg.get('ir_drop_mode', 'zero'),
                desc_prefix='Val IRDrop',
            )
            metrics.update(compute_cross_modal_robustness(
                baseline_metrics,
                rgb_metrics,
                ir_metrics,
                base_metric=robustness_cfg.get('base_metric', 'mAP_50'),
            ))

        if self.rank == 0:
            print(
                f"\n[Val Report] mAP@0.5: {metrics.get('mAP_50', 0):.4f} | "
                f"AP_S: {metrics.get('mAP_S', 0):.4f}"
            )

        model.train()
        return metrics
