import time

import torch
import torch.distributed as dist
from torch.amp import autocast
from tqdm import tqdm

from src.model.bbox_utils import non_max_suppression_obb
from src.model.output_adapter import flatten_predictions


class Evaluator:
    def __init__(self, dataloader, metrics_evaluator, device, nms_kwargs=None):
        self.dataloader = dataloader
        self.device = device
        self.metrics_evaluator = metrics_evaluator
        self.nms_kwargs = nms_kwargs or {'conf_thres': 0.001, 'iou_thres': 0.45, 'max_det': 300}

        self.rank = 0
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()

    @torch.no_grad()
    def evaluate(self, model, epoch=-1):
        model.eval()
        self.metrics_evaluator.reset()

        pbar = tqdm(self.dataloader, desc=f"Val Epoch {epoch}", leave=False)
        amp_enabled = self.device.type == 'cuda'

        for batch_idx, (imgs_rgb, imgs_ir, targets, prev_rgb, prev_ir) in enumerate(pbar):
            imgs_rgb = imgs_rgb.to(self.device, non_blocking=True)
            imgs_ir = imgs_ir.to(self.device, non_blocking=True)
            prev_rgb = prev_rgb.to(self.device, non_blocking=True)
            prev_ir = prev_ir.to(self.device, non_blocking=True)

            profile_speed = amp_enabled and batch_idx < 10
            if profile_speed:
                torch.cuda.synchronize()
            t1 = time.time()

            with autocast(device_type=self.device.type, enabled=amp_enabled):
                model_out = model(imgs_rgb, imgs_ir, prev_rgb=prev_rgb, prev_ir=prev_ir)
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
            gt_dict = {b: [] for b in range(imgs_rgb.shape[0])}
            for target in targets_np:
                gt_dict[int(target[0])].append(target[1:])
            batch_gts = [torch.tensor(gt_dict[i], dtype=torch.float32, device='cpu') for i in range(imgs_rgb.shape[0])]

            image_ids = [f"{epoch}_{self.rank}_{batch_idx}_{i}" for i in range(imgs_rgb.shape[0])]
            preds = [pred.cpu() for pred in preds]

            self.metrics_evaluator.add_batch(image_ids, preds, batch_gts)
            pbar.set_postfix({"infer(ms)": f"{(t2 - t1) * 1000:.1f}", "nms(ms)": f"{(t3 - t2) * 1000:.1f}"})

        metrics = self.metrics_evaluator.get_full_metrics()

        if self.rank == 0:
            print(
                f"\n[Val Report] mAP@0.5: {metrics.get('mAP_50', 0):.4f} | "
                f"AP_S: {metrics.get('mAP_S', 0):.4f}"
            )

        model.train()
        return metrics
