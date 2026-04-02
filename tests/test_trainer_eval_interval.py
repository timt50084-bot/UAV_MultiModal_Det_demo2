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
        target_labels = torch.zeros_like(pred_scores)
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
        matched_pred_cls,
        matched_pred_box,
        matched_tgt_cls,
        matched_tgt_box,
        contrastive_loss,
        epoch,
        temporal_loss=None,
    ):
        temporal_loss = temporal_loss if temporal_loss is not None else matched_pred_cls.sum() * 0.0
        loss_cls = matched_pred_cls.square().mean()
        loss_reg = matched_pred_box.square().mean()
        loss_angle = matched_pred_cls.sum() * 0.0
        loss_total = loss_cls + loss_reg + contrastive_loss + temporal_loss
        return loss_total, loss_cls, loss_reg, loss_angle


class _DummyEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, model, epoch=-1):
        self.calls.append(epoch)
        return {'mAP_50': 0.1 * len(self.calls)}


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


if __name__ == '__main__':
    unittest.main()
