"""Protocol and replay checks for the content-matching experiment.

All cascades are fabricated; the training runs test reproducibility and
saved evidence, and their scores are not research results.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_matched_comparisons as runner
from models.aragcl_dp import ARAGCL_DP, ARAGCL_DP_Config
from test_matched_comparisons import write_fixture


def matching_fixture(directory):
    """Each cascade has a substantive exact pair, a marker pair and containment."""
    write_fixture(directory)
    for path in Path(directory).glob("*.json"):
        document = json.loads(path.read_text())
        marker = document["source"]["content"][-1]
        root = "当地居民发布了一条关于桥梁关闭的消息，请大家核实具体情况。" + marker
        repeated = "有目击者报告该消息缺乏可靠依据，需要查看原始记录。" + marker
        document["source"]["content"] = root
        document["comment"][0]["content"] = repeated
        document["comment"][1]["content"] = repeated
        document["comment"][2]["content"] = "请大家留意：" + root + "我正在寻找可靠来源。"
        document["comment"].extend([
            {"comment id": 3, "parent": -1, "content": " 转发微博 ", "time": "26-1-1 00:04"},
            {"comment id": 4, "parent": 3, "content": "转发微博", "time": "26-1-1 00:05"},
        ])
        path.write_text(json.dumps(document), encoding="utf-8")


def freeze_fixture(directory, destination):
    _, provenance = runner.prepare_data(directory, batch_size=4, split_seed=0)
    Path(destination).write_text(json.dumps({"dataset": provenance}), encoding="utf-8")
    return provenance


class MatchingExperimentTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.use_deterministic_algorithms,
                        torch.are_deterministic_algorithms_enabled())

    def test_wrapper_uses_bundled_frozen_cohort_and_fixed_model_controls(self):
        import run_matching_comparison as entrypoint

        args = entrypoint.parse_args([])
        self.assertEqual(args.models, ["matching_exact", "matching_filtered_exact",
                                       "matching_filtered_containment"])
        self.assertEqual(args.seeds, [0, 1, 2, 3, 4])
        self.assertEqual(args.dataset_dir, Path("/storage/weibo1"))
        self.assertEqual(args.output_dir.parent, Path("results"))
        self.assertTrue(args.output_dir.name.startswith("content-matching-"))
        self.assertEqual(args.cohort_manifest,
                         Path(entrypoint.__file__).resolve().parent /
                         "experiment_protocols/matching_cohort.json")
        bundled = json.loads(args.cohort_manifest.read_text())["dataset"]
        self.assertEqual(len(bundled["train_files_in_order"]), 400)
        self.assertEqual(len(bundled["validation_files_in_order"]), 100)
        self.assertTrue(set(bundled["train_files_in_order"]).isdisjoint(
            bundled["validation_files_in_order"]))
        self.assertEqual((args.split_seed, args.pretrain_epochs, args.finetune_epochs,
                          args.patience, args.batch_size), (0, 15, 25, 6, 8))
        with patch.object(entrypoint, "run_comparisons", return_value="dispatched") as run:
            self.assertEqual(entrypoint.main(["--seeds", "3", "--output-dir", "custom"]),
                             "dispatched")
        self.assertEqual(run.call_args.kwargs["seeds"], [3])
        self.assertEqual(run.call_args.kwargs["models"], args.models)
        self.assertEqual(run.call_args.kwargs["cohort_manifest"], args.cohort_manifest)

    def test_frozen_order_and_shared_features_survive_matching_and_extra_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            matching_fixture(dataset)
            frozen = root / "original-manifest.json"
            original = freeze_fixture(dataset, frozen)
            # A malformed additional JSON must not even be parsed when the
            # saved cohort is authoritative; directory size cannot change it.
            (dataset / "unrelated.json").write_text("not a cascade")
            with patch.object(runner, "train_test_split", side_effect=AssertionError("resplit")):
                data, manifest = runner.prepare_data(dataset, batch_size=4,
                                                     include_matching=True,
                                                     cohort_manifest=frozen)
            for key in ("train_indices", "validation_indices", "train_files_in_order",
                        "validation_files_in_order", "train_batch_file_order", "tfidf"):
                self.assertEqual(manifest[key], original[key])
            self.assertEqual(manifest["sampling"]["selected_file_count"], 10)
            self.assertTrue(manifest["frozen_cohort"]["source_hashes_verified"])
            self.assertTrue(manifest["frozen_cohort"]["prior_original_input_fingerprints_match"])
            for key, fingerprint in original["input_fingerprints"].items():
                self.assertEqual(manifest["input_fingerprints"][key], fingerprint)
            for partition in ("train", "val"):
                for strategy, edge_count in (("filtered_exact", 1), ("filtered_containment", 2)):
                    key = "matching_" + strategy + "_" + partition + "_batches"
                    for (before, batch, labels), (after, matched_batch, matched_labels) in zip(
                            data[partition + "_batches"], data[key]):
                        for field in ("x", "reply_edge_index", "timestamps",
                                      "propagation_depth", "root_mask"):
                            torch.testing.assert_close(getattr(before, field), getattr(after, field),
                                                       rtol=0, atol=0)
                        torch.testing.assert_close(batch, matched_batch, rtol=0, atol=0)
                        torch.testing.assert_close(labels, matched_labels, rtol=0, atol=0)
                        self.assertEqual(before.duplication_edge_index.size(1), 2 * len(labels))
                        self.assertEqual(after.duplication_edge_index.size(1), edge_count * len(labels))
            audit = manifest["content_matching_audit"]
            for partition, size in (("train", 8), ("validation", 2)):
                self.assertEqual(audit["summary"][partition]["exact"]["links"], 2 * size)
                self.assertEqual(audit["summary"][partition]["filtered_exact"]["links"], size)
                self.assertEqual(audit["summary"][partition]["filtered_containment"]["exact_links"], size)
                self.assertEqual(audit["summary"][partition]["filtered_containment"]["containment_links"], size)
            evidence = audit["training_link_evidence"]
            self.assertEqual({row["file"] for row in evidence}, set(original["train_files_in_order"]))
            self.assertTrue(all(row["file"] not in original["validation_files_in_order"]
                                for row in evidence))
            self.assertTrue(any(row["kind"] == "containment" for row in evidence))

    def test_frozen_cohort_rejects_changed_bytes_partitions_and_resampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            matching_fixture(dataset)
            frozen = root / "original-manifest.json"
            original = freeze_fixture(dataset, frozen)
            with self.assertRaisesRegex(ValueError, "resampled"):
                runner.prepare_data(dataset, max_cascades=6, cohort_manifest=frozen)
            with self.assertRaisesRegex(ValueError, "split_seed"):
                runner.prepare_data(dataset, split_seed=1, cohort_manifest=frozen)
            altered = dataset / original["files"][0]["file"]
            old_bytes = altered.read_bytes()
            altered.write_bytes(old_bytes + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner.prepare_data(dataset, batch_size=4, cohort_manifest=frozen)
            altered.write_bytes(old_bytes)
            for corruption in ("overlap", "missing", "unknown", "empty"):
                bad = deepcopy(original)
                if corruption == "overlap":
                    bad["validation_files_in_order"][0] = bad["train_files_in_order"][0]
                elif corruption == "missing":
                    bad["train_files_in_order"].pop()
                elif corruption == "unknown":
                    bad["train_files_in_order"][0] = "not-in-cohort.json"
                else:
                    bad["validation_files_in_order"] = []
                frozen.write_text(json.dumps({"dataset": bad}))
                with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, "partitions"):
                    runner.prepare_data(dataset, batch_size=4, cohort_manifest=frozen)
            for corruption in ("idf", "inputs"):
                bad = deepcopy(original)
                if corruption == "idf":
                    bad["tfidf"]["idf"][0] += 1
                    message = "TF-IDF"
                else:
                    bad["input_fingerprints"]["train"][0] = "invalid"
                    message = "graph/features"
                frozen.write_text(json.dumps({"dataset": bad}))
                with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, message):
                    runner.prepare_data(dataset, batch_size=4, cohort_manifest=frozen)

    def test_fixed_controls_order_independence_saved_evidence_and_checkpoint_replay(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            matching_fixture(dataset)
            frozen = root / "original-manifest.json"
            original = freeze_fixture(dataset, frozen)
            kwargs = dict(dataset_dir=dataset, cohort_manifest=frozen, seeds=[7],
                          pretrain_epochs=1, finetune_epochs=2, patience=2,
                          batch_size=4, num_threads=1)
            variants = list(runner.MATCHING_VARIANTS)
            first, _ = runner.run_comparisons(output_dir=root / "first", models=variants, **kwargs)
            reverse, _ = runner.run_comparisons(output_dir=root / "reverse", models=variants[::-1], **kwargs)
            control, _ = runner.run_comparisons(output_dir=root / "control", models=["lambda_0p25"], **kwargs)
            self.assertEqual(len({row["initial_state_sha256"] for row in first + reverse + control}), 1)
            self.assertTrue(all(row["model_config"] == first[0]["model_config"] for row in first))
            self.assertEqual(first[0]["model_config"]["lam"], .25)
            self.assertEqual(first[0]["model_config"]["duplicate_message_direction"], "copy_to_reference")
            reverse_by_name = {row["model"]: row for row in reverse}
            replay_data, _ = runner.prepare_data(dataset, batch_size=4, include_matching=True,
                                                 cohort_manifest=frozen)
            for row in first:
                variant = row["model"]
                other = reverse_by_name[variant]
                self.assertEqual(row["trained_state_sha256"], other["trained_state_sha256"])
                self.assertEqual(row["metrics"], other["metrics"])
                self.assertEqual(row["rng_stage_seeds"], other["rng_stage_seeds"])
                strategy = runner.MATCHING_VARIANTS[variant]
                self.assertEqual(row["graph_construction"]["matching_strategy"], strategy)
                directory = root / "first" / "seed_7" / variant
                predictions = json.loads((directory / "predictions.json").read_text())
                self.assertEqual([p["file"] for p in predictions], original["validation_files_in_order"])
                recomputed = runner.compute_metrics([p["label"] for p in predictions],
                                                     [p["prediction"] for p in predictions])
                self.assertEqual(row["metrics"], runner.asdict(recomputed))
                history = json.loads((directory / "history.json").read_text())
                selected = history["finetune"][history["selected_epoch_zero_based"]]
                self.assertEqual(selected["f1"], row["metrics"]["f1"])
                self.assertEqual(selected["f1"], max(item["f1"] for item in history["finetune"]))
                checkpoint = torch.load(directory / "best_checkpoint.pt", weights_only=True)
                self.assertEqual(checkpoint["graph_construction"], row["graph_construction"])
                restored = ARAGCL_DP(ARAGCL_DP_Config(**checkpoint["model_config"]))
                restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
                self.assertEqual(runner.state_fingerprint(restored.state_dict()), row["trained_state_sha256"])
                prefix = "" if strategy == "exact" else "matching_" + strategy + "_"
                graph, batch, labels = replay_data[prefix + "val_batches"][0]
                with runner.evaluation_mode(restored):
                    probabilities = restored.classify(graph, batch=batch).softmax(dim=1)
                torch.testing.assert_close(probabilities, torch.tensor([
                    [p["probability_0"], p["probability_1"]] for p in predictions]), rtol=0, atol=0)
                self.assertEqual(probabilities.argmax(dim=1).tolist(), [p["prediction"] for p in predictions])
                usage = json.loads((directory / "augmentation_usage.json").read_text())
                self.assertEqual(usage["counts"]["view_count"], 4)
                self.assertEqual(usage["counts"]["duplication_edges_before"],
                                 16 if strategy == "filtered_exact" else 32)
                other_usage = json.loads((root / "reverse" / "seed_7" / variant /
                                          "augmentation_usage.json").read_text())
                self.assertEqual(usage, other_usage)
            # Recording the matching audit and using its routing must leave
            # the exact baseline bit-identical to the old lambda=.25 command.
            exact = next(row for row in first if row["model"] == "matching_exact")
            self.assertEqual(exact["trained_state_sha256"], control[0]["trained_state_sha256"])
            self.assertEqual(exact["metrics"], control[0]["metrics"])
            actual_predictions = json.loads((root / "first/seed_7/matching_exact/predictions.json").read_text())
            control_predictions = json.loads((root / "control/seed_7/lambda_0p25/predictions.json").read_text())
            self.assertEqual(actual_predictions, control_predictions)
            manifest = json.loads((root / "first/manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["content_matching_protocol"]["lambda"], .25)
            self.assertNotIn("augmentation_traces_match_within_seed", manifest)
            audit = json.loads((root / "first/content_matching_audit.json").read_text())
            self.assertEqual(audit["summary"], manifest["dataset"]["content_matching_audit"]["summary"])
            self.assertNotIn("training_link_evidence", manifest["dataset"]["content_matching_audit"])
            self.assertEqual({r["file"] for r in audit["training_link_evidence"]},
                             set(original["train_files_in_order"]))
            self.assertEqual(json.loads((root / "first/matching_protocol.json").read_text()),
                             manifest["content_matching_protocol"])
            with self.assertRaisesRegex(ValueError, "only the three matching variants"):
                runner.run_comparisons(output_dir=root / "mixed", models=["matching_exact", "lambda_0p25"],
                                       **kwargs)


if __name__ == "__main__":
    unittest.main()
