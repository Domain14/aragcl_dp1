"""Regression checks for scoring and checkpoint selection with active dropout.

Run from the project directory: python -m unittest discover -s tests
"""
import contextlib
import io
import unittest

import torch
from torch import nn

from eval.robustness import early_detection_curve, robustness_sweep
from train.trainer import (
    TrainConfig,
    contrastive_pretrain,
    evaluate,
    evaluation_mode,
    supervised_train,
)


class RecordingClassifier(nn.Module):
    """Train on class 0 while class 1 validation prefers an early checkpoint."""

    def __init__(self):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(4.0))
        self.dropout = nn.Dropout(0.5)
        self.calls = []
        self.validation_margins = []

    def forward(self, x):
        self.calls.append((self.training, self.dropout.training,
                           torch.is_grad_enabled()))
        if not self.training:
            self.validation_margins.append(self.margin.detach().clone())
        score = self.dropout(x) * self.margin
        return torch.stack((-score, score), dim=1)


class TrainingModeTests(unittest.TestCase):
    def test_evaluate_repeats_predictions_with_dropout_enabled_before_call(self):
        torch.manual_seed(10)
        model = nn.Sequential(nn.Dropout(0.75), nn.Linear(32, 2))
        x, y = torch.randn(16, 32), torch.arange(16) % 2
        predictions = []

        def forward(batch):
            features, labels = batch
            logits = model(features)
            self.assertFalse(logits.requires_grad)
            self.assertFalse(model.training)
            predictions.append(logits.clone())
            return logits, labels

        first = evaluate(model, forward, [(x, y)])
        second = evaluate(model, forward, [(x, y)])
        self.assertTrue(torch.equal(predictions[0], predictions[1]))
        self.assertEqual(first, second)
        self.assertTrue(all(module.training for module in model.modules()))

    def test_context_restores_mixed_modes_and_gradients(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
        model[1].eval()
        before = [module.training for module in model.modules()]
        with torch.enable_grad():
            with evaluation_mode(model):
                self.assertFalse(any(module.training for module in model.modules()))
                self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(before, [module.training for module in model.modules()])

        model.eval()
        with evaluation_mode(model):
            pass
        self.assertFalse(any(module.training for module in model.modules()))

    def test_evaluate_restores_modes_when_forward_raises(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
        model[1].eval()
        before = [module.training for module in model.modules()]

        def fail(_):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            raise RuntimeError("bad validation batch")

        with torch.enable_grad():
            with self.assertRaisesRegex(RuntimeError, "bad validation batch"):
                evaluate(model, fail, [None])
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(before, [module.training for module in model.modules()])

    def test_supervised_training_restores_best_scored_checkpoint(self):
        torch.manual_seed(9)
        model = RecordingClassifier().eval()
        train = [(torch.ones(64), torch.zeros(64, dtype=torch.long))]
        validation = [(torch.ones(64), torch.ones(64, dtype=torch.long))]
        cfg = TrainConfig(epochs=4, patience=4, lr=2.0, weight_decay=0.0)

        def forward(batch):
            return model(batch[0]), batch[1]

        with contextlib.redirect_stdout(io.StringIO()):
            history = supervised_train("regression", forward, model,
                                       train, validation, cfg)

        self.assertEqual(len(history), 4)
        self.assertEqual(model.calls, [(True, True, True), (False, False, False)] * 4)
        self.assertGreater(max(item.f1 for item in history), history[-1].f1)
        best_index = max(range(len(history)), key=lambda i: history[i].f1)
        self.assertTrue(torch.equal(model.margin.detach(),
                                    model.validation_margins[best_index]))
        _, restored_metrics = evaluate(model, forward, validation)
        self.assertEqual(restored_metrics.f1, max(item.f1 for item in history))
        self.assertTrue(model.training)

    def test_pretraining_reenters_training_from_eval_mode(self):
        class Pretrainer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.dropout = nn.Dropout(0.5)
                self.calls = []

            def pretrain_step(self, graph, batch=None):
                self.calls.append((self.training, self.dropout.training,
                                   torch.is_grad_enabled()))
                return (self.dropout(graph) * self.weight - 1).square().mean()

        model = Pretrainer().eval()
        cfg = TrainConfig(pretrain_epochs=2)
        with contextlib.redirect_stdout(io.StringIO()):
            _, losses = contrastive_pretrain("regression", model,
                                             [(torch.ones(64), None)], cfg)
        self.assertEqual(len(losses), 2)
        self.assertEqual(model.calls, [(True, True, True)] * 2)

    def test_robustness_and_early_detection_disable_model_dropout(self):
        model = nn.Sequential(nn.Dropout(0.8), nn.Linear(2, 2))
        batch = (torch.ones(8, 2), torch.ones(8, dtype=torch.long))

        def forward(current_model, data, *perturbations):
            self.assertFalse(any(module.training for module in current_model.modules()))
            self.assertFalse(torch.is_grad_enabled())
            return current_model(data[0]), data[1]

        with contextlib.redirect_stdout(io.StringIO()):
            sweep = robustness_sweep(model, forward, [batch], [0.0], [0.0])
            curve = early_detection_curve(model, forward, [batch], [1.0])
        self.assertEqual(sweep[0].metrics, curve[1.0])
        self.assertTrue(all(module.training for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
