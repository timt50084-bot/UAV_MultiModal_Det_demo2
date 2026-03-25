# 解析配置 -> Build -> 启动 Evaluator
import argparse
import torch
import random
import numpy as np

from src.utils.config import load_config
from src.data.dataloader import build_dataloader
from src.model.builder import build_model
from src.engine.evaluator import Evaluator
from src.metrics.obb_metrics import OBBMetricsEvaluator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="多模态极小目标 OBB 验证与刷榜系统")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='全局配置文件路径')
    parser.add_argument('--weights', type=str, required=True, help='要验证的权重文件')
    parser.add_argument('--device', type=int, default=0, help='使用的 GPU ID')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cpu' if args.device < 0 or not torch.cuda.is_available() else f'cuda:{args.device}')
    print(f"\n📊 启动赛场记分牌系统... 挂载节点: {device}")

    set_seed(42)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # 保障 mAP 绝对复现

    # 构建验证数据与模型
    val_loader, _ = build_dataloader(cfg, is_training=False)
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # 组装 Evaluator
    metrics_evaluator = OBBMetricsEvaluator(num_classes=cfg.model.num_classes)
    evaluator = Evaluator(val_loader, metrics_evaluator, device, nms_kwargs=cfg.val.nms)

    print("\n" + "=" * 50)
    print("🚀 [Validation 启动] 正在计算全量 OBB 多边形 IoU 与 AP 指标...")

    # 开始算分
    metrics = evaluator.evaluate(model, epoch="Final")

    print("\n" + "🏆 最终评测报告 (Final Leaderboard) 🏆".center(50))
    print("-" * 55)
    print(f"| {'指标 (Metrics)':<20} | {'分数 (Score)':<26} |")
    print("-" * 55)
    print(f"| {'mAP@0.5 (全量目标)':<21} | {metrics.get('mAP_50', 0) * 100:>23.2f} % |")
    print(f"| {'mAP_S@0.5 (极小目标)':<22} | {metrics.get('mAP_S', 0) * 100:>23.2f} % |")
    print("-" * 55)


if __name__ == '__main__':
    main()