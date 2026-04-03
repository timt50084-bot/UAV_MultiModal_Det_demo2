import argparse
import math
import os
import random

import numpy as np
import torch

from src.data.dataloader import build_dataloader
from src.engine.callbacks.checkpoint_callback import CheckpointCallback
from src.engine.callbacks.ema_callback import EMACallback
from src.engine.evaluator_factory import build_detection_evaluator
from src.engine.trainer import Trainer
from src.loss.builder import build_assigner, build_loss
from src.model.builder import build_model
from src.utils.config import load_config
from src.utils.config_utils import (
    apply_experiment_runtime_overrides,
    format_effective_train_config_summary,
    save_resolved_config,
)
from src.utils.detection_cuda import resolve_detection_device


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the dual-modal OBB detector.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id for the detection mainline.')
    parser.add_argument('--resume', type=str, default='', help='Optional checkpoint to resume from.')
    return parser.parse_known_args()


def main():
    args, cli_overrides = parse_args()
    cfg, cfg_meta = load_config(args.config, cli_args=cli_overrides, return_meta=True)
    cfg, run_name = apply_experiment_runtime_overrides(cfg, config_path=args.config)
    resolved_config_path = save_resolved_config(cfg, run_name)

    device = resolve_detection_device(args.device)
    print(f'\nTraining device: {device}')
    print(f'Experiment name: {run_name}')
    print(
        format_effective_train_config_summary(
            cfg,
            args.config,
            resolved_config_path,
            source_config_path=cfg_meta.get('source_config_path'),
        )
    )
    for warning in cfg_meta.get('warnings', []):
        print(f'[Config Warning] {warning}')
    aug_cfg = cfg.get('dataset', {}).get('aug_cfg', {}) if hasattr(cfg, 'get') else {}
    misalignment_cfg = aug_cfg.get('cross_modal_misalignment', {}) if hasattr(aug_cfg, 'get') else {}
    if misalignment_cfg and misalignment_cfg.get('enabled', False):
        print(
            'Cross-modal misalignment aug: '
            f"enabled (prob={float(misalignment_cfg.get('prob', 0.0)):.2f}, "
            f"apply_to={misalignment_cfg.get('apply_to', 'ir')}, "
            f"translate<={float(misalignment_cfg.get('max_translate_ratio', 0.0)):.3f}, "
            f"rotate<={float(misalignment_cfg.get('max_rotate_deg', 0.0)):.2f}deg, "
            f"scale_delta<={float(misalignment_cfg.get('max_scale_delta', 0.0)):.3f})"
        )
    else:
        print('Cross-modal misalignment aug: off')
    sensor_degradation_cfg = aug_cfg.get('sensor_degradation', {}) if hasattr(aug_cfg, 'get') else {}
    if sensor_degradation_cfg and sensor_degradation_cfg.get('enabled', False):
        rgb_cfg = sensor_degradation_cfg.get('rgb', {}) if hasattr(sensor_degradation_cfg, 'get') else {}
        ir_cfg = sensor_degradation_cfg.get('ir', {}) if hasattr(sensor_degradation_cfg, 'get') else {}
        print(
            'Sensor degradation aug: '
            f"enabled (prob={float(sensor_degradation_cfg.get('prob', 0.0)):.2f}, "
            f"rgb[max_gain={float(rgb_cfg.get('max_exposure_gain', 0.0)):.2f}, "
            f"flare_p={float(rgb_cfg.get('flare_prob', 0.0)):.2f}, "
            f"haze_p={float(rgb_cfg.get('haze_prob', 0.0)):.2f}], "
            f"ir[noise_std={float(ir_cfg.get('noise_std', 0.0)):.3f}, "
            f"max_drift={float(ir_cfg.get('max_drift', 0.0)):.3f}, "
            f"hotspot_p={float(ir_cfg.get('hotspot_prob', 0.0)):.2f}, "
            f"stripe_p={float(ir_cfg.get('stripe_prob', 0.0)):.2f}])"
        )
    else:
        print('Sensor degradation aug: off')
    loss_cfg = cfg.get('loss', {}) if hasattr(cfg, 'get') else {}
    angle_enabled = bool(loss_cfg.get('angle_enabled', False)) if hasattr(loss_cfg, 'get') else False
    angle_weight = float(loss_cfg.get('angle_weight', 0.0)) if hasattr(loss_cfg, 'get') else 0.0
    if angle_enabled and angle_weight > 0.0:
        print(
            'Angle loss: '
            f"enabled (type={loss_cfg.get('angle_type', 'wrapped_smooth_l1')}, "
            f"weight={angle_weight:.3f}, beta={float(loss_cfg.get('angle_beta', 0.1)):.3f})"
        )
    elif angle_enabled:
        print('Angle loss: configured but inactive (weight=0.0, legacy-equivalent)')
    else:
        print('Angle loss: off')
    set_seed(42)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_loader, _ = build_dataloader(cfg, is_training=True)
    val_loader, _ = build_dataloader(cfg, is_training=False)
    print(f'Train loader steps per epoch: {len(train_loader)}')

    model = build_model(cfg.model).to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f'Resumed weights from: {args.resume}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    lr_lambda = lambda step: ((1 - math.cos(step * math.pi / cfg.train.epochs)) / 2) * (cfg.train.lrf - 1) + 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = build_loss(cfg.loss).to(device)
    assigner = build_assigner(cfg.assigner).to(device)

    extra_metrics_cfg = cfg.get('eval', {}) if hasattr(cfg, 'get') else {}
    infer_cfg = cfg.get('infer', {}) if hasattr(cfg, 'get') else {}
    evaluator = build_detection_evaluator(
        dataloader=val_loader,
        device=device,
        num_classes=cfg.model.num_classes,
        nms_kwargs=cfg.val.nms,
        eval_cfg=extra_metrics_cfg,
        infer_cfg=infer_cfg,
    )
    print(
        f"[Eval Route] requested={evaluator.requested_backend}/{evaluator.requested_obb_iou_backend} "
        f"resolved={evaluator.resolved_backend}/{evaluator.resolved_obb_iou_backend} "
        f"role={evaluator.evaluator_role} reason={evaluator.resolution_reason} "
        f"evaluator={type(evaluator).__name__} metrics={type(evaluator.metrics_evaluator).__name__}"
    )
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
        grad_clip=cfg.train.grad_clip,
        use_amp=cfg.train.use_amp,
        evaluator=evaluator,
        callbacks=callbacks,
        performance_cfg=cfg.get('performance', {}) if hasattr(cfg, 'get') else {},
        eval_interval=cfg.train.get('eval_interval', 1),
    )

    print('\nStarting training loop...')
    engine.train()

if __name__ == '__main__':
    main()
