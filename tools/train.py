import argparse
import math
import os
import random

import numpy as np
import torch

from src.data.dataloader import build_dataloader
from src.engine.callbacks.checkpoint_callback import CheckpointCallback
from src.engine.callbacks.ema_callback import EMACallback
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.loss.builder import build_assigner, build_loss
from src.metrics.obb_metrics import OBBMetricsEvaluator
from src.model.builder import build_model
from src.utils.config import load_config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the dual-modal OBB detector.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    parser.add_argument('--device', type=int, default=0, help='GPU id. Use -1 for CPU.')
    parser.add_argument('--resume', type=str, default='', help='Optional checkpoint to resume from.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cpu' if args.device < 0 or not torch.cuda.is_available() else f'cuda:{args.device}')
    print(f'\nTraining device: {device}')
    set_seed(42)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_loader, _ = build_dataloader(cfg, is_training=True)
    val_loader, _ = build_dataloader(cfg, is_training=False)

    model = build_model(cfg.model).to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f'Resumed weights from: {args.resume}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    lr_lambda = lambda step: ((1 - math.cos(step * math.pi / cfg.train.epochs)) / 2) * (cfg.train.lrf - 1) + 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = build_loss(cfg.loss).to(device)
    assigner = build_assigner(cfg.assigner).to(device)

    extra_metrics_cfg = cfg.get('eval', {}).get('extra_metrics', {}) if hasattr(cfg, 'get') else {}
    metrics_evaluator = OBBMetricsEvaluator(num_classes=cfg.model.num_classes, extra_metrics_cfg=extra_metrics_cfg)
    evaluator = Evaluator(val_loader, metrics_evaluator, device, nms_kwargs=cfg.val.nms, extra_metrics_cfg=extra_metrics_cfg)

    callbacks = [
        EMACallback(model),
        CheckpointCallback(save_dir=cfg.train.save_dir, patience=cfg.train.patience),
    ]

    engine = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        assigner=assigner,
        device=device,
        epochs=cfg.train.epochs,
        accumulate=cfg.train.accumulate,
        use_amp=cfg.train.use_amp,
        evaluator=evaluator,
        callbacks=callbacks,
    )

    print('\nStarting training loop...')
    engine.train()


if __name__ == '__main__':
    main()
