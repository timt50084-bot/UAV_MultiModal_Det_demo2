import unittest

import torch

from src.engine.trainer import Trainer


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, imgs_rgb, imgs_ir, prev_rgb=None, prev_ir=None):
        batch_size = imgs_rgb.shape[0]
        outputs = [
            self.weight.view(1, 1, 1, 1).expand(batch_size, 6, 1, 1)
            for _ in range(4)
        ]
        return outputs, outputs[0], outputs[0]


class _DummyAssigner(torch.nn.Module):
    def forward(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        target_labels = torch.zeros(pred_scores.shape[:2], dtype=torch.long, device=pred_scores.device)
        target_bboxes = torch.zeros_like(pred_bboxes)
        target_scores = torch.zeros_like(pred_scores)
        fg_mask = torch.ones(pred_scores.shape[:2], dtype=torch.bool, device=pred_scores.device)
        return target_labels, target_bboxes, target_scores, fg_mask


class _DummyCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.angle_enabled = False
        self.angle_weight = 0.0

    def forward(
        self,
        pred_scores,
        pred_bboxes,
        target_scores,
        target_bboxes,
        fg_mask=None,
        contrastive_loss=0.0,
        epoch=0,
        temporal_loss=None,
    ):
        temporal_loss = temporal_loss if temporal_loss is not None else pred_scores.sum() * 0.0
        loss_cls = (pred_scores - target_scores).square().mean()
        pos_mask = fg_mask.bool() if fg_mask is not None else torch.ones(pred_bboxes.shape[:2], dtype=torch.bool, device=pred_bboxes.device)
        if bool(pos_mask.any().item()):
            loss_reg = (pred_bboxes[pos_mask] - target_bboxes[pos_mask]).square().mean()
        else:
            loss_reg = pred_bboxes.sum() * 0.0
        loss_angle = pred_scores.sum() * 0.0
        loss_total = loss_cls + loss_reg + contrastive_loss + temporal_loss
        return loss_total, loss_cls, loss_reg, loss_angle


class _DummyTemporalModel(_DummyModel):
    def __init__(self, temporal_value=5.0):
        super().__init__()
        self.temporal_enabled = True
        self.temporal_mode = 'two_frame'
        self.temporal_value = float(temporal_value)

    def get_temporal_consistency_loss(self, lambda_t=0.1, low_motion_bias=0.75):
        del low_motion_bias
        return self.weight.new_tensor(self.temporal_value * float(lambda_t))


class _RecordingTemporalCriterion(_DummyCriterion):
    def __init__(self, warmup=0.0, ramp=0.0, max_loss=float('inf'), skip_loss=float('inf')):
        super().__init__()
        self.temporal_weight = 1.0
        self.temporal_low_motion_bias = 0.75
        self.temporal_warmup_epochs = float(warmup)
        self.temporal_ramp_epochs = float(ramp)
        self.temporal_max_loss = float(max_loss)
        self.temporal_skip_loss_threshold = float(skip_loss)
        self.seen_temporal_losses = []

    def forward(
        self,
        pred_scores,
        pred_bboxes,
        target_scores,
        target_bboxes,
        fg_mask=None,
        contrastive_loss=0.0,
        epoch=0,
        temporal_loss=None,
    ):
        total_loss, loss_cls, loss_reg, loss_angle = super().forward(
            pred_scores,
            pred_bboxes,
            target_scores,
            target_bboxes,
            fg_mask=fg_mask,
            contrastive_loss=contrastive_loss,
            epoch=epoch,
            temporal_loss=temporal_loss,
        )
        self.seen_temporal_losses.append(float(temporal_loss.detach().item()))
        return total_loss, loss_cls, loss_reg, loss_angle


class _DummyEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, model, epoch=-1):
        self.calls.append(epoch)
        return {'mAP_50': 0.1 * len(self.calls)}


class _FailingLoader:
    def __init__(self, exc):
        self.exc = exc
        self.num_workers = 8
        self.timeout = 120
        self.dataset = object()

    def __len__(self):
        return 1

    def __iter__(self):
        return self

    def __next__(self):
        raise self.exc


class TrainerEvalIntervalTestCase(unittest.TestCase):
    def test_trainer_validates_on_interval_and_final_epoch(self):
        batch = (
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((0, 7), dtype=torch.float32),
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
        )
        train_loader = [batch]
        model = _DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        evaluator = _DummyEvaluator()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=_DummyCriterion(),
            assigner=_DummyAssigner(),
            device=torch.device('cpu'),
            epochs=3,
            accumulate=1,
            grad_clip=1.0,
            use_amp=False,
            evaluator=evaluator,
            callbacks=[],
            eval_interval=2,
        )
        trainer.train()

        self.assertEqual(evaluator.calls, [2, 3])
        self.assertTrue(trainer.did_validate_this_epoch)
        self.assertAlmostEqual(float(trainer.current_metrics['mAP_50']), 0.2, places=6)

    def test_trainer_clamps_temporal_loss_before_backward(self):
        batch = (
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((0, 7), dtype=torch.float32),
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
        )
        train_loader = [batch]
        model = _DummyTemporalModel(temporal_value=5.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        criterion = _RecordingTemporalCriterion(max_loss=0.2, skip_loss=10.0)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            assigner=_DummyAssigner(),
            device=torch.device('cpu'),
            epochs=1,
            accumulate=1,
            grad_clip=1.0,
            use_amp=False,
            evaluator=None,
            callbacks=[],
        )
        trainer.train()

        self.assertEqual(len(criterion.seen_temporal_losses), 1)
        self.assertAlmostEqual(criterion.seen_temporal_losses[0], 0.2, places=6)

    def test_trainer_temporal_warmup_keeps_aux_loss_off(self):
        batch = (
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((0, 7), dtype=torch.float32),
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
        )
        train_loader = [batch]
        model = _DummyTemporalModel(temporal_value=5.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        criterion = _RecordingTemporalCriterion(warmup=2.0)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            assigner=_DummyAssigner(),
            device=torch.device('cpu'),
            epochs=1,
            accumulate=1,
            grad_clip=1.0,
            use_amp=False,
            evaluator=None,
            callbacks=[],
        )
        trainer.train()

        self.assertEqual(len(criterion.seen_temporal_losses), 1)
        self.assertAlmostEqual(criterion.seen_temporal_losses[0], 0.0, places=6)

    def test_trainer_raises_contextual_error_on_dataloader_failure(self):
        model = _DummyTemporalModel(temporal_value=1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        train_loader = _FailingLoader(RuntimeError("DataLoader worker (pid(s) 123) exited unexpectedly"))

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=_DummyCriterion(),
            assigner=_DummyAssigner(),
            device=torch.device('cpu'),
            epochs=1,
            accumulate=1,
            grad_clip=1.0,
            use_amp=False,
            evaluator=None,
            callbacks=[],
        )

        with self.assertRaises(RuntimeError) as raised:
            trainer.train()

        message = str(raised.exception)
        self.assertIn('epoch=1', message)
        self.assertIn('batch=1/1', message)
        self.assertIn('num_workers=8', message)
        self.assertIn('temporal_enabled=True', message)
        self.assertIn('temporal_mode=two_frame', message)
        self.assertIn('num_workers=0', message)
        self.assertIn('timeout_seconds=120', message)
        self.assertIn('DataLoader worker (pid(s) 123) exited unexpectedly', message)


if __name__ == '__main__':
    unittest.main()
