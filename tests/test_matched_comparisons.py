"""Regression checks for matching, leakage prevention, and saved research evidence.

Training fixtures are fabricated; their scores have no research interpretation.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from sklearn.model_selection import train_test_split

import run_matched_comparisons as runner


def write_fixture(directory):
    for index in range(10):
        # Unique Unicode code point makes train-only vocabulary membership testable.
        marker = chr(0x4E00 + index)
        content = f"fixture source {index % 2} {marker}"
        comments = [{"comment id": comment, "parent": -1,
                     "content": f"repeated {marker}" if comment < 2 else f"other {marker}",
                     "time": f"26-1-1 00:0{comment + 1}"} for comment in range(3)]
        document = {"source": {"tweet id": f"cascade-{index}", "label": index % 2,
                                "content": content, "time": "26-1-1 00:00"},
                    "comment": comments}
        (Path(directory) / f"cascade-{index}.json").write_text(json.dumps(document), encoding="utf-8")


class MatchedComparisonTests(unittest.TestCase):
    def test_stratified_sample_is_reproducible_and_reports_root_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_fixture(temporary)
            # A known repeated root must be detected across a split, even with
            # distinct source IDs. This diagnoses potential content leakage.
            for path in Path(temporary).glob("*.json"):
                document = json.loads(path.read_text())
                document["source"]["content"] = " SAME ROOT "
                path.write_text(json.dumps(document))
            _, first = runner.prepare_data(temporary, max_cascades=6, sample_seed=3)
            _, second = runner.prepare_data(temporary, max_cascades=6, sample_seed=3)
            self.assertEqual(first["sampling"]["selected_source_indices"],
                             second["sampling"]["selected_source_indices"])
            self.assertEqual(first["train_indices"], second["train_indices"])
            self.assertEqual(len(first["files"]), 6)
            self.assertEqual(sum(item["label"] for item in first["files"]), 3)
            self.assertEqual(first["normalized_root_overlap"]["cross_partition_group_count"], 1)
            self.assertEqual(first["normalized_root_overlap"]["cross_partition_file_count"], 6)
            self.assertEqual(len(first["sampling"]["source_inventory"]), 10)

    def test_shared_fixed_split_and_one_training_only_tfidf_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_fixture(temporary)
            original_fit = runner.TfidfVectorizer.fit
            fit_texts = []

            def record_fit(vectorizer, texts, *args, **kwargs):
                fit_texts.append(list(texts))
                return original_fit(vectorizer, texts, *args, **kwargs)

            with patch.object(runner.TfidfVectorizer, "fit", record_fit):
                data, manifest = runner.prepare_data(temporary, batch_size=4, split_seed=0)
            self.assertEqual(len(fit_texts), 1)
            expected_train, expected_val = train_test_split(list(range(10)), test_size=.2,
                                                            stratify=[i % 2 for i in range(10)], random_state=0)
            self.assertEqual(manifest["train_indices"], expected_train)
            self.assertEqual(manifest["validation_indices"], expected_val)
            vocabulary = manifest["tfidf"]["vocabulary"]
            for index in expected_val:
                self.assertNotIn(chr(0x4E00 + index), vocabulary)
            for full, static in zip(data["train_batches"], data["static_train_batches"]):
                torch.testing.assert_close(full[0].x, static[0].x, rtol=0, atol=0)
                torch.testing.assert_close(full[1], static[1], rtol=0, atol=0)
                torch.testing.assert_close(full[2], static[2], rtol=0, atol=0)
                self.assertGreater(full[0].duplication_edge_index.numel(), 0)
                self.assertEqual(static[0].duplication_edge_index.numel(), 0)
                self.assertEqual(static[0].duplication_freq.sum().item(), 0)

    def test_model_order_independence_and_saved_best_checkpoints(self):
        old_threads = torch.get_num_threads()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, old_threads)
        self.addCleanup(torch.use_deterministic_algorithms, old_deterministic)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            write_fixture(dataset)
            arguments = dict(dataset_dir=dataset, seeds=[7], split_seed=0,
                             pretrain_epochs=1, finetune_epochs=2, patience=2,
                             batch_size=4, num_threads=1)
            first, _ = runner.run_comparisons(output_dir=root / "first", **arguments)
            second, _ = runner.run_comparisons(output_dir=root / "second",
                                                models=list(reversed(runner.DEFAULT_MODELS)), **arguments)
            self.assertEqual(len({row["initial_state_sha256"] for row in first + second}), 1)
            by_model = {row["model"]: row for row in second}
            for row in first:
                other = by_model[row["model"]]
                self.assertEqual(row["metrics"], other["metrics"])
                self.assertEqual(row["trained_state_sha256"], other["trained_state_sha256"])
                directory = root / "first" / "seed_7" / row["model"]
                history = json.loads((directory / "history.json").read_text())
                selected = history["finetune"][history["selected_epoch_zero_based"]]
                self.assertEqual(row["metrics"]["f1"], selected["f1"])
                self.assertTrue(all(item["train_loss_logged"] is not None for item in history["finetune"]))
                predictions = json.loads((directory / "predictions.json").read_text())
                recalculated = runner.compute_metrics([p["label"] for p in predictions],
                                                       [p["prediction"] for p in predictions])
                self.assertEqual(row["metrics"], runner.asdict(recalculated))
                checkpoint = torch.load(directory / "best_checkpoint.pt", weights_only=True)
                self.assertEqual(runner.state_fingerprint(checkpoint["model_state_dict"]), row["trained_state_sha256"])
            manifest = json.loads((root / "first" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["dataset"]["tfidf"]["fit_count"], 1)
            self.assertEqual(len(manifest["runs"]), 3)
            self.assertTrue((root / "first" / "source_snapshot.zip").is_file())
            with self.assertRaises(FileExistsError):
                runner.run_comparisons(output_dir=root / "first", **arguments)

    def test_paired_differences_use_same_seed_and_sample_sd(self):
        def row(model, seed, f1):
            return {"model": model, "seed": seed,
                    "metrics": {name: f1 for name in runner.METRIC_NAMES}}
        rows = [row("full", 0, .8), row("full", 1, .6),
                row("no_dup_edges", 1, .5), row("no_dup_edges", 0, .7),
                row("static", 0, .6)]
        summary = runner.summary_statistics(rows)
        self.assertAlmostEqual(summary["models"]["full"]["f1"]["sample_sd"], .2 / 2**.5)
        self.assertIsNone(summary["models"]["static"]["f1"]["sample_sd"])
        pair = summary["paired_f1_differences"][0]
        self.assertEqual(pair["seeds"], [0, 1])
        self.assertAlmostEqual(pair["mean"], .1)
        self.assertAlmostEqual(pair["sample_sd"], 0)

    def test_direction_variant_preserves_inputs_and_records_reloadable_config(self):
        old_threads = torch.get_num_threads()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, old_threads)
        self.addCleanup(torch.use_deterministic_algorithms, old_deterministic)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            write_fixture(dataset)
            variants = ["full", "copy_to_reference", "no_dup_edges"]
            arguments = dict(dataset_dir=dataset, seeds=[7], pretrain_epochs=1,
                             finetune_epochs=2, patience=2, batch_size=4)
            first, summary = runner.run_comparisons(
                output_dir=root / "first", models=variants, **arguments)
            second, _ = runner.run_comparisons(
                output_dir=root / "second", models=variants[::-1], **arguments)
            self.assertEqual(len({row["initial_state_sha256"] for row in first + second}), 1)
            by_model = {row["model"]: row for row in second}
            for row in first:
                self.assertEqual(row["trained_state_sha256"],
                                 by_model[row["model"]]["trained_state_sha256"])
            configs = {row["model"]: dict(row["model_config"]) for row in first}
            self.assertEqual(configs["full"].pop("duplicate_message_direction"), "reference_to_copy")
            self.assertEqual(configs["copy_to_reference"].pop("duplicate_message_direction"), "copy_to_reference")
            self.assertEqual(configs["full"], configs["copy_to_reference"])
            checkpoint = torch.load(root / "first/seed_7/copy_to_reference/best_checkpoint.pt",
                                    weights_only=True)
            from models.aragcl_dp import ARAGCL_DP, ARAGCL_DP_Config
            restored = ARAGCL_DP(ARAGCL_DP_Config(**checkpoint["model_config"]))
            restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.assertEqual(restored.encoder.duplicate_message_direction, "copy_to_reference")
            self.assertEqual(runner.state_fingerprint(restored.state_dict()),
                             by_model["copy_to_reference"]["trained_state_sha256"])
            self.assertTrue(any("copy_to_reference" in item["direction"]
                                for item in summary["paired_f1_differences"]))
            # The legacy command still selects the original three variants.
            defaults = runner.parse_args(["--dataset-dir", str(dataset),
                                           "--output-dir", str(root / "unused")])
            self.assertEqual(defaults.models, list(runner.DEFAULT_MODELS))


if __name__ == "__main__":
    unittest.main()
