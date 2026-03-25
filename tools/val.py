import argparse
import random

import numpy as np
import torch

from src.data.dataloader import build_dataloader
from src.engine.evaluator import Evaluator
from src.metrics.obb_metrics import OBBMetricsEvaluator
from src.model.builder import build_model
from src.utils.config import load_config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Validate the dual-modal OBB detector.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    parser.add_argument('--weights', type=str, required=True, help='Checkpoint to validate.')
    parser.add_argument('--device', type=int, default=0, help='GPU id. Use -1 for CPU.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cpu' if args.device < 0 or not torch.cuda.is_available() else f'cuda:{args.device}')
    print(f'\nValidation device: {device}')

    set_seed(42)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    val_loader, _ = build_dataloader(cfg, is_training=False)
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    extra_metrics_cfg = cfg.get('eval', {}).get('extra_metrics', {}) if hasattr(cfg, 'get') else {}
    metrics_evaluator = OBBMetricsEvaluator(num_classes=cfg.model.num_classes, extra_metrics_cfg=extra_metrics_cfg)
    evaluator = Evaluator(val_loader, metrics_evaluator, device, nms_kwargs=cfg.val.nms, extra_metrics_cfg=extra_metrics_cfg)

    print('\nStarting validation...')
    metrics = evaluator.evaluate(model, epoch='Final')

    print('\nFinal metrics')
    print('-' * 48)
    print(f"mAP_50: {metrics.get('mAP_50', 0.0) * 100:.2f}%")
    print(f"mAP_S: {metrics.get('mAP_S', 0.0) * 100:.2f}%")
    if 'Recall_S' in metrics:
        print(f"Recall_S: {metrics.get('Recall_S', 0.0) * 100:.2f}%")
    if 'Precision_S' in metrics:
        print(f"Precision_S: {metrics.get('Precision_S', 0.0) * 100:.2f}%")
    if 'CrossModalRobustness_RGBDrop' in metrics:
        print(f"CrossModalRobustness_RGBDrop: {metrics['CrossModalRobustness_RGBDrop']:.4f}")
    if 'CrossModalRobustness_IRDrop' in metrics:
        print(f"CrossModalRobustness_IRDrop: {metrics['CrossModalRobustness_IRDrop']:.4f}")
    if 'TemporalStability' in metrics:
        print(f"TemporalStability: {metrics['TemporalStability']}")


if __name__ == '__main__':
    main()
