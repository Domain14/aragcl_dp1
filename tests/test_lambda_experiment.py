"""Controlled duplicate-message weights on a shared graph and augmentation trace.

Tiny fabricated cascades verify the experiment protocol, not research scores.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_matched_comparisons as runner
from test_matched_comparisons import write_fixture


EXPECTED_LAMBDAS = {
    "lambda_0": 0.0,
    "lambda_0p1": 0.1,
    "lambda_0p25": 0.25,
    "lambda_0p5": 0.5,
}


class LambdaExperimentTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)

    def test_fixed_lambda_protocol_and_legacy_command_defaults(self):
        import run_lambda_comparison as entrypoint

        self.assertEqual(list(runner.LAMBDA_VARIANTS.items()),
                         list(EXPECTED_LAMBDAS.items()))
        arguments = entrypoint.parse_args([])
        self.assertEqual(arguments.models, list(EXPECTED_LAMBDAS))
        self.assertEqual(arguments.seeds, [0, 1, 2, 3, 4])
        self.assertEqual(Path(arguments.dataset_dir), Path("/storage/weibo1"))
        self.assertEqual(Path(arguments.output_dir),
                         Path("results/lambda-comparison-2026-09-29"))
        self.assertEqual(arguments.split_seed, 0)
        self.assertEqual(arguments.pretrain_epochs, 15)
        self.assertEqual(arguments.finetune_epochs, 25)
        self.assertEqual(arguments.batch_size, 8)
        self.assertEqual(arguments.patience, 6)
        legacy = runner.parse_args(["--dataset-dir", "data", "--output-dir", "results"])
        self.assertEqual(legacy.models, ["full", "no_dup_edges", "static"])

    def test_endpoints_reproduce_controls_and_lambda_views_match(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            write_fixture(dataset)
            arguments = dict(dataset_dir=dataset, seeds=[7], split_seed=0,
                             pretrain_epochs=1, finetune_epochs=2, patience=2,
                             batch_size=4, num_threads=1)
            controls, _ = runner.run_comparisons(
                output_dir=root / "controls", models=["no_dup_edges", "copy_to_reference"],
                **arguments)
            first, _ = runner.run_comparisons(
                output_dir=root / "first", models=list(EXPECTED_LAMBDAS), **arguments)
            second, _ = runner.run_comparisons(
                output_dir=root / "second", models=list(reversed(EXPECTED_LAMBDAS)),
                **arguments)
            all_rows = controls + first + second
            self.assertEqual(len({row["initial_state_sha256"] for row in all_rows}), 1)
            first_by_model = {row["model"]: row for row in first}
            control_by_model = {row["model"]: row for row in controls}
            for lambda_name, control_name in (("lambda_0", "no_dup_edges"),
                                               ("lambda_0p5", "copy_to_reference")):
                actual = first_by_model[lambda_name]
                expected = control_by_model[control_name]
                self.assertEqual(actual["trained_state_sha256"], expected["trained_state_sha256"])
                self.assertEqual(actual["metrics"], expected["metrics"])
                expected_predictions = json.loads(
                    (root / "controls" / "seed_7" / control_name / "predictions.json").read_text())
                actual_predictions = json.loads(
                    (root / "first" / "seed_7" / lambda_name / "predictions.json").read_text())
                self.assertEqual(actual_predictions, expected_predictions)

            traces, usage_counts = [], []
            configs_except_lambda = []
            for row in first + second:
                variant = row["model"]
                expected = first_by_model[variant]
                self.assertEqual(row["trained_state_sha256"], expected["trained_state_sha256"])
                self.assertEqual(row["metrics"], expected["metrics"])
                self.assertEqual(row["augmentation_trace_sha256"],
                                 expected["augmentation_trace_sha256"])
                self.assertEqual(row["graph_construction"],
                                 {"representation": "duplication", "duplicate_policy": "none"})
                self.assertTrue(row["model_config"]["use_duplication"])
                self.assertEqual(row["model_config"]["duplicate_message_direction"],
                                 "copy_to_reference")
                config = dict(row["model_config"])
                self.assertEqual(config.pop("lam"), EXPECTED_LAMBDAS[variant])
                configs_except_lambda.append(config)
            self.assertTrue(all(config == configs_except_lambda[0]
                                for config in configs_except_lambda))

            for run in ("first", "second"):
                for variant in EXPECTED_LAMBDAS:
                    directory = root / run / "seed_7" / variant
                    row = first_by_model[variant]
                    checkpoint = torch.load(directory / "best_checkpoint.pt", weights_only=True)
                    self.assertEqual(checkpoint["augmentation_trace_sha256"],
                                     row["augmentation_trace_sha256"])
                    self.assertEqual(checkpoint["model_config"], row["model_config"])
                    self.assertEqual(checkpoint["graph_construction"], row["graph_construction"])
                    self.assertEqual(runner.state_fingerprint(checkpoint["model_state_dict"]),
                                     row["trained_state_sha256"])
                    traces.append(json.loads((directory / "augmentation_trace.json").read_text()))
                    usage = json.loads((directory / "augmentation_usage.json").read_text())
                    self.assertEqual(usage["counts"]["view_count"], 4)
                    self.assertEqual(usage["counts"]["dropped_roots"], 0)
                    self.assertGreater(usage["counts"]["duplication_edges_before"], 0)
                    usage_counts.append(usage["counts"])
                manifest = json.loads((root / run / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "completed")
                self.assertEqual(manifest["augmentation_traces_match_within_seed"], {"7": True})
                self.assertNotIn("duplicate_filter_audit", manifest["dataset"])
                control_manifest = json.loads((root / "controls" / "manifest.json").read_text())
                for field in ("train_indices", "validation_indices", "tfidf", "input_fingerprints"):
                    self.assertEqual(manifest["dataset"][field], control_manifest["dataset"][field])
            self.assertEqual(len({row["augmentation_trace_sha256"] for row in first}), 1)
            self.assertTrue(all(trace == traces[0] for trace in traces))
            self.assertTrue(all(counts == usage_counts[0] for counts in usage_counts))

    def test_trace_mismatch_rejects_the_comparison_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            write_fixture(dataset)
            original_trace = runner.AugmentationTraceRecorder.trace_dict
            calls = []

            def mismatched_trace(recorder):
                trace = dict(original_trace(recorder))
                calls.append(trace["sequence_sha256"])
                if len(calls) == 2:
                    # Alter reported evidence while preserving all masks and
                    # random draws, to exercise the runner's matching guard.
                    trace["sequence_sha256"] = "deliberate-test-mismatch"
                return trace

            with patch.object(runner.AugmentationTraceRecorder, "trace_dict", mismatched_trace):
                with self.assertRaisesRegex(RuntimeError, "(?i)augmentation"):
                    runner.run_comparisons(
                        dataset_dir=dataset, output_dir=root / "failed",
                        models=["lambda_0", "lambda_0p1"], seeds=[7],
                        pretrain_epochs=1, finetune_epochs=1, patience=1,
                        batch_size=4, num_threads=1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], calls[1])
            manifest = json.loads((root / "failed" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["augmentation_traces_match_within_seed"], {"7": False})
            self.assertIn("RuntimeError", manifest["error"])
            # Preserve the mismatched report so failed comparisons are auditable.
            reported = json.loads((root / "failed" / "seed_7" / "lambda_0p1"
                                   / "augmentation_trace.json").read_text())
            self.assertEqual(reported["sequence_sha256"], "deliberate-test-mismatch")


if __name__ == "__main__":
    unittest.main()
