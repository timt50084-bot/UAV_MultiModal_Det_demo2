# 负责保存 best.pt 和 latest.pt
import os
import torch
from .base import Callback


class CheckpointCallback(Callback):
    """模型保存与早停控制回调"""

    def __init__(self, save_dir='outputs/weights', patience=50, min_delta=1e-4, monitor='mAP_50'):
        self.save_dir = save_dir
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_fitness = 0.0
        self.best_epoch = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, trainer):
        # 提取模型，如果启用了 EMA 则优先保存 EMA
        model_to_save = trainer.ema_callback.ema if hasattr(trainer,
                                                            'ema_callback') and trainer.ema_callback else trainer.model

        # 每一轮结束都保存 latest.pt
        torch.save(model_to_save.state_dict(), os.path.join(self.save_dir, 'latest.pt'))

        # 检查是否突破最佳记录
        if trainer.current_metrics:
            fitness = trainer.current_metrics.get('mAP_50', 0) * 0.1 + trainer.current_metrics.get('mAP_S', 0) * 0.9

            if fitness > self.best_fitness + self.min_delta:
                self.best_fitness = fitness
                self.best_epoch = trainer.current_epoch
                torch.save(model_to_save.state_dict(), os.path.join(self.save_dir, 'best.pt'))
                print(f"🎉 突破历史最佳 ({fitness:.4f})！已保存 best.pt")
            elif trainer.current_epoch - self.best_epoch >= self.patience:
                print(f"\n⚠️ 触发早停机制！监控指标连续 {self.patience} 轮未提升。")
                trainer.stop_training = True