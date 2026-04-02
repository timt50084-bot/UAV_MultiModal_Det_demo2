import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.engine.callbacks.checkpoint_callback import CheckpointCallback


class CheckpointCallbackTestCase(unittest.TestCase):
    def _trainer(self, metrics):
        return SimpleNamespace(
            ema_callback=None,
            model=torch.nn.Linear(2, 2),
            current_metrics=metrics,
            current_epoch=1,
            did_validate_this_epoch=True,
            stop_training=False,
        )

    def test_checkpoint_falls_back_to_primary_metric_when_small_object_metric_is_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(save_dir=tmpdir, patience=5, monitor='mAP_50')
            trainer = self._trainer({'mAP_50': 0.42})
            with patch('src.engine.callbacks.checkpoint_callback.torch.save'), \
                    patch('builtins.print'):
                callback.on_epoch_end(trainer)
            self.assertAlmostEqual(callback.best_fitness, 0.42, places=6)

    def test_checkpoint_keeps_small_object_weighting_when_metric_is_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(save_dir=tmpdir, patience=5, monitor='mAP_50')
            trainer = self._trainer({'mAP_50': 0.40, 'mAP_S': 0.60})
            with patch('src.engine.callbacks.checkpoint_callback.torch.save'), \
                    patch('builtins.print'):
                callback.on_epoch_end(trainer)
            self.assertAlmostEqual(callback.best_fitness, 0.58, places=6)

    def test_checkpoint_ignores_stale_metrics_when_validation_was_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(save_dir=tmpdir, patience=3, monitor='mAP_50')
            callback.best_fitness = 0.50
            callback.best_epoch = 1
            trainer = self._trainer({'mAP_50': 0.40})
            trainer.current_epoch = 5
            trainer.did_validate_this_epoch = False
            with patch('src.engine.callbacks.checkpoint_callback.torch.save'), \
                    patch('builtins.print'):
                callback.on_epoch_end(trainer)
            self.assertAlmostEqual(callback.best_fitness, 0.50, places=6)
            self.assertFalse(trainer.stop_training)


if __name__ == '__main__':
    unittest.main()
