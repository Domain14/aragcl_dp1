"""The marker experiment changes links/metadata, not its cohort or features.

Synthetic training scores here have no research interpretation.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

import run_matched_comparisons as runner
from test_matched_comparisons import write_fixture


def marker_fixture(directory):
    write_fixture(directory)
    for path in Path(directory).glob("*.json"):
        doc = json.loads(path.read_text())
        doc["comment"].extend([
            {"comment id": 3, "parent": -1, "content": " 转发微博 ", "time": "26-1-1 00:04"},
            {"comment id": 4, "parent": 3, "content": "转发微博", "time": "26-1-1 00:05"},
        ])
        path.write_text(json.dumps(doc))


class MarkerExperimentTests(unittest.TestCase):
    def test_filtered_batches_share_split_features_and_preserve_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker_fixture(temporary)
            _, legacy = runner.prepare_data(temporary, batch_size=4)
            data, audit = runner.prepare_data(temporary, batch_size=4, include_filtered=True)
            for key in ("files", "train_indices", "validation_indices", "tfidf"):
                self.assertEqual(audit[key], legacy[key])
            for key, fingerprint in legacy["input_fingerprints"].items():
                self.assertEqual(audit["input_fingerprints"][key], fingerprint)
            for partition in ("train", "val"):
                for (before, batch, y), (after, fb, fy) in zip(
                        data[partition + "_batches"], data["filtered_" + partition + "_batches"]):
                    for name in ("x", "reply_edge_index", "timestamps", "root_mask", "propagation_depth"):
                        torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
                    torch.testing.assert_close(batch, fb, rtol=0, atol=0)
                    torch.testing.assert_close(y, fy, rtol=0, atol=0)
                    self.assertEqual(before.duplication_edge_index.size(1), 2 * len(y))
                    self.assertEqual(after.duplication_edge_index.size(1), len(y))
                    self.assertEqual(after.duplication_freq.sum().item(), len(y))
            summary = audit["duplicate_filter_audit"]["summary"]
            self.assertEqual(summary["train"]["duplicate_edges_removed"], 8)
            self.assertEqual(summary["validation"]["duplicate_edges_removed"], 2)
            self.assertEqual(summary["train"]["nodes"], 48)
            self.assertEqual(summary["train"]["nodes_with_changed_duplication_frequency"], 8)

    def test_filter_routing_recording_and_control_reproducibility(self):
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            marker_fixture(dataset)
            controls = ["copy_to_reference", "no_dup_edges", "static"]
            variants = controls + list(runner.FILTERED_MODELS)
            kwargs = dict(dataset_dir=dataset, seeds=[3], pretrain_epochs=1,
                          finetune_epochs=2, patience=2, batch_size=4, num_threads=1)
            original, _ = runner.run_comparisons(output_dir=root / "original", models=controls, **kwargs)
            first, _ = runner.run_comparisons(output_dir=root / "first", models=variants, **kwargs)
            second, _ = runner.run_comparisons(output_dir=root / "second", models=variants[::-1], **kwargs)
            self.assertEqual(len({r["initial_state_sha256"] for r in first + second + original}), 1)
            first_by_model = {r["model"]: r for r in first}
            for row in original + second:
                expected = first_by_model[row["model"]]
                self.assertEqual(row["trained_state_sha256"], expected["trained_state_sha256"])
                self.assertEqual(row["metrics"], expected["metrics"])
            for row in first:
                variant = row["model"]
                directory = root / "first" / "seed_3" / variant
                expected_policy = ("exclude_weibo_marker" if variant in runner.FILTERED_MODELS else "none")
                self.assertEqual(row["graph_construction"]["duplicate_policy"], expected_policy)
                checkpoint = torch.load(directory / "best_checkpoint.pt", weights_only=True)
                self.assertEqual(checkpoint["graph_construction"], row["graph_construction"])
                usage = json.loads((directory / "augmentation_usage.json").read_text())
                self.assertEqual(usage["counts"]["view_count"], 4)
                self.assertEqual(usage["counts"]["original_nodes"], 96)
                expected_edges = 0 if variant == "static" else 16 if variant in runner.FILTERED_MODELS else 32
                self.assertEqual(usage["counts"]["duplication_edges_before"], expected_edges)
                self.assertEqual(usage["counts"]["dropped_roots"], 0)
                self.assertEqual(usage["counts"]["attribute_masks_sampled_roots"], 0)
                other_usage = json.loads((root / "second" / "seed_3" / variant / "augmentation_usage.json").read_text())
                self.assertEqual(usage, other_usage)
            filtered = first_by_model["filtered_copy_to_reference"]
            no_dup = first_by_model["filtered_no_dup_edges"]
            self.assertEqual(filtered["model_config"]["duplicate_message_direction"], "copy_to_reference")
            self.assertEqual(filtered["model_config"]["lam"], .5)
            self.assertFalse(no_dup["model_config"]["use_duplication"])
            self.assertTrue((root / "first" / "duplicate_filter_audit.json").is_file())
            manifest = json.loads((root / "first" / "manifest.json").read_text())
            self.assertTrue(manifest["augmentation_usage_recorded"])
            self.assertEqual(manifest["status"], "completed")


if __name__ == "__main__":
    unittest.main()
