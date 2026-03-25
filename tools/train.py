# 解析配置 -> Build -> 启动 Trainer
import os
import argparse
import torch
import random
import numpy as np

# 导入我们重构后的强大架构组件
from src.utils.config import load_config
from src.data.dataloader import build_dataloader
from src.model.builder import build_model
from src.loss.builder import build_loss, build_assigner
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.metrics.obb_metrics import OBBMetricsEvaluator
from src.engine.callbacks.ema_callback import EMACallback
from src.engine.callbacks.checkpoint_callback import CheckpointCallback


def set_seed(seed=42):
    """🔥 锁死世界线：确保实验绝对可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="多模态极小目标 OBB 训练引擎点火器")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='全局配置文件路径')
    parser.add_argument('--device', type=int, default=0, help='使用的 GPU ID (设为 -1 强制 CPU)')
    parser.add_argument('--resume', type=str, default='', help='断点续训权重路径')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 解析合并全局配置 (OmegaConf 发威)
    cfg = load_config(args.config)

    # 2. 硬件防呆与种子锁定
    device = torch.device('cpu' if args.device < 0 or not torch.cuda.is_available() else f'cuda:{args.device}')
    print(f"\n🚀 正在点燃 V8 训练引擎... 当前挂载设备: {device}")
    set_seed(42)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # 激活极限吞吐

    # 3. 构建数据管道
    train_loader, _ = build_dataloader(cfg, is_training=True)
    val_loader, _ = build_dataloader(cfg, is_training=False)

    # 4. 构建模型架构
    model = build_model(cfg.model).to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"🔄 成功恢复断点权重: {args.resume}")

    # 5. 构建优化器与误差引擎
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    lf = lambda x: ((1 - math.cos(x * math.pi / cfg.train.epochs)) / 2) * (cfg.train.lrf - 1) + 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    criterion = build_loss(cfg.loss).to(device)
    assigner = build_assigner(cfg.assigner).to(device)

    # 6. 构建评估器与 Callbacks
    metrics_evaluator = OBBMetricsEvaluator(num_classes=cfg.model.num_classes)
    evaluator = Evaluator(val_loader, metrics_evaluator, device, nms_kwargs=cfg.val.nms)

    callbacks = [
        EMACallback(model),
        CheckpointCallback(save_dir=cfg.train.save_dir, patience=cfg.train.patience)
    ]

    # 7. 组装并点火！
    engine = Trainer(
        model=model, train_loader=train_loader, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, assigner=assigner, device=device, epochs=cfg.train.epochs,
        accumulate=cfg.train.accumulate, use_amp=cfg.train.use_amp,
        evaluator=evaluator, callbacks=callbacks
    )

    print("\n🔥 V8 引擎全状态 Check 完毕。Main Loop 已接管。")
    engine.train()


if __name__ == '__main__':
    import math  # 用于 lr_scheduler

    main()