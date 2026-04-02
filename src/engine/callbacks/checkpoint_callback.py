import os

import torch

from .base import Callback


class CheckpointCallback(Callback):
    """Save latest/best checkpoints and trigger early stopping."""

    def __init__(self, save_dir='outputs/weights', patience=50, min_delta=1e-4, monitor='mAP_50'):
        self.save_dir = save_dir
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_fitness = 0.0
        self.best_epoch = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, trainer):
        model_to_save = (
            trainer.ema_callback.ema
            if hasattr(trainer, 'ema_callback') and trainer.ema_callback
            else trainer.model
        )
        validated_this_epoch = getattr(trainer, 'did_validate_this_epoch', True)

        torch.save(model_to_save.state_dict(), os.path.join(self.save_dir, 'latest.pt'))

        if validated_this_epoch and trainer.current_metrics:
            primary_metric = float(trainer.current_metrics.get(self.monitor, 0.0))
            if 'mAP_S' in trainer.current_metrics:
                fitness = primary_metric * 0.1 + float(trainer.current_metrics['mAP_S']) * 0.9
            else:
                fitness = primary_metric

            if fitness > self.best_fitness + self.min_delta:
                self.best_fitness = fitness
                self.best_epoch = trainer.current_epoch
                torch.save(model_to_save.state_dict(), os.path.join(self.save_dir, 'best.pt'))
                print(f"New best checkpoint saved: best.pt (fitness={fitness:.4f})")
            elif trainer.current_epoch - self.best_epoch >= self.patience:
                print(f"\nEarly stopping triggered: no improvement for {self.patience} epochs.")
                trainer.stop_training = True
