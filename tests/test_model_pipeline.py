"""Small integration check using fabricated JSON cascades and real PyG models.

The fixture is deliberately artificial. Its metrics have no research meaning;
it checks loader -> augmentation -> training -> restored-checkpoint scoring.
It reads none of the user's training, validation, or test data.
"""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_real_data
import train.trainer as trainer


def write_fixture(directory):
    for i in range(10):
        comments = []
        for j in range(6):
            # Two observed fixture rows share text; no synthetic node operation.
            content = f"repeated fixture comment {i}" if j in (0, 1) else f"unique fixture {i} reply {j}"
            comments.append({
                "comment id": j,
                "parent": 0 if j == 2 else -1,
                "children": [2] if j == 0 else [],
                "content": content,
                "time": f"26-1-1 00:{j + 1:02d}",
            })
        obj = {
            "source": {
                "tweet id": f"fixture-{i}", "label": i % 2,
                "content": f"source fixture class {i % 2} example {i}",
                "time": "26-1-1 00:00",
            },
            "comment": comments,
            "centrality": {"Degree": [1e-8] * 7},
        }
        (Path(directory) / f"fixture-{i}.json").write_text(json.dumps(obj))


class ModelPipelineTests(unittest.TestCase):
    def test_full_runner_reports_selected_checkpoint_metrics(self):
        random.seed(12)
        torch.manual_seed(12)
        prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, prior_threads)

        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory)
            data = run_real_data.load_and_prepare(directory, batch_size=4,
                                                  verbose=False, seed=12)

        self.assertGreater(data["val_graph"].duplication_edge_index.numel(), 0)
        best_by_model = {}
        original_supervised_train = trainer.supervised_train

        def record_selected(model_name, forward_fn, model, *args, **kwargs):
            history = original_supervised_train(model_name, forward_fn, model,
                                                *args, **kwargs)
            best_by_model[id(model)] = max(history, key=lambda item: item.f1)
            return history

        # Direct baselines call the runner import; finetune calls trainer's copy.
        with patch.object(trainer, "supervised_train", record_selected), \
             patch.object(run_real_data, "supervised_train", record_selected), \
             contextlib.redirect_stdout(io.StringIO()):
            metrics, models = run_real_data.train_all_models(
                data, data["feature_dim"],
                trainer.TrainConfig(epochs=2, patience=2, pretrain_epochs=1),
            )

        self.assertTrue({"Text-only", "Vanilla GCN", "R-GCN", "BiGCN", "ARAGCL-DP"}.issubset(models))
        for name, model in models.items():
            with self.subTest(model=name):
                self.assertEqual(metrics[name], best_by_model[id(model)])

        model = models["ARAGCL-DP"]
        graph, batch, labels = data["val_batches"][0]
        with trainer.evaluation_mode(model):
            first = model.classify(graph, batch=batch).clone()
        # Advance RNG between evaluations; evaluation must not depend on dropout.
        torch.rand(100)
        with trainer.evaluation_mode(model):
            second = model.classify(graph, batch=batch).clone()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all().item())


if __name__ == "__main__":
    unittest.main()
