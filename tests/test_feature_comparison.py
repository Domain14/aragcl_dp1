"""Feature-capacity controls on fabricated data, never research measurements."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_feature_comparison as entrypoint
import run_matched_comparisons as runner
from test_matched_comparisons import write_fixture


def feature_fixture(directory):
    write_fixture(directory)
    # More than 2048 distinct character unigrams/bigrams, shared across the
    # cohort. Per-cascade markers remain outside this synthetic alphabet.
    alphabet = "".join(chr(0x6000 + index) for index in range(1100))
    for path in Path(directory).glob("*.json"):
        document = json.loads(path.read_text())
        document["source"]["content"] += " " + alphabet
        path.write_text(json.dumps(document), encoding="utf-8")


def freeze(directory, destination):
    data, manifest = runner.prepare_data(directory, batch_size=4)
    destination.write_text(json.dumps({"dataset": manifest}))
    return data, manifest


class FeatureComparisonTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.use_deterministic_algorithms,
                        torch.are_deterministic_algorithms_enabled())

    def test_wrapper_defaults_and_paired_summary_match_by_seed(self):
        args = entrypoint.parse_args([])
        self.assertEqual(entrypoint.FEATURE_CAPACITIES, (512, 2048))
        self.assertEqual(args.dataset_dir, Path("/storage/weibo1"))
        self.assertTrue(args.cohort_manifest.is_file())
        self.assertEqual(args.seeds, [0, 1, 2, 3, 4])
        self.assertEqual((args.pretrain_epochs, args.finetune_epochs, args.patience,
                          args.batch_size), (15, 25, 6, 8))
        self.assertEqual(args.output_dir.parent, Path("results"))
        self.assertTrue(args.output_dir.name.startswith("feature-capacity-"))
        def row(width, seed, value):
            return {"tfidf_max_features": width, "seed": seed,
                    "metrics": {metric: value for metric in ("f1", "accuracy", "precision", "recall")}}
        summary = entrypoint.summarize([
            row(2048, 2, .6), row(512, 1, .8), row(2048, 0, .8),
            row(512, 2, .6), row(2048, 1, .7), row(512, 0, .7)])
        paired = summary["paired_f1"]
        self.assertEqual(paired["seeds"], [0, 1, 2])
        self.assertEqual((paired["wins"], paired["losses"], paired["ties"]), (1, 1, 1))
        self.assertAlmostEqual(paired["mean"], 0)
        self.assertAlmostEqual(paired["sample_sd"], .1)
        self.assertEqual(summary["features"]["512"]["f1"]["n"], 3)
        self.assertAlmostEqual(summary["features"]["512"]["f1"]["sample_sd"], .1)

    def test_capacity_changes_only_features_and_fits_training_text_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            feature_fixture(dataset)
            frozen = root / "frozen.json"
            original_data, original = freeze(dataset, frozen)
            fits = []
            original_fit = runner.TfidfVectorizer.fit
            def capture(vectorizer, texts, *args, **kwargs):
                texts = list(texts)
                fits.append(texts)
                return original_fit(vectorizer, texts, *args, **kwargs)
            with patch.object(runner.TfidfVectorizer, "fit", capture), \
                    patch.object(runner, "train_test_split", side_effect=AssertionError("resplit")):
                expanded, manifest = runner.prepare_data(
                    dataset, batch_size=4, cohort_manifest=frozen, tfidf_max_features=2048)
            self.assertEqual(len(fits), 1)
            expected_texts = []
            for filename in original["train_files_in_order"]:
                document = json.loads((dataset / filename).read_text())
                expected_texts.extend([document["source"]["content"]] +
                                      [comment["content"] for comment in document["comment"]])
            self.assertEqual(fits[0], expected_texts)
            for filename in original["validation_files_in_order"]:
                index = int(filename.removeprefix("cascade-").removesuffix(".json"))
                self.assertNotIn(chr(0x4E00 + index), manifest["tfidf"]["vocabulary"])
            self.assertEqual(original_data["feature_dim"], 512)
            self.assertEqual(expanded["feature_dim"], 2048)
            self.assertTrue(manifest["frozen_cohort"]["intentional_feature_capacity_change"])
            self.assertFalse(manifest["frozen_cohort"]["prior_original_input_fingerprints_match"])
            for key in ("files", "train_files_in_order", "validation_files_in_order", "train_batch_file_order"):
                self.assertEqual(manifest[key], original[key])
            for partition in ("train", "val"):
                for (before, batch, labels), (after, new_batch, new_labels) in zip(
                        original_data[partition + "_batches"], expanded[partition + "_batches"]):
                    self.assertEqual(before.x.size(0), after.x.size(0))
                    self.assertEqual((before.x.size(1), after.x.size(1)), (512, 2048))
                    for field in ("reply_edge_index", "duplication_edge_index", "duplication_freq",
                                  "timestamps", "propagation_depth", "root_mask"):
                        torch.testing.assert_close(getattr(before, field), getattr(after, field), rtol=0, atol=0)
                    torch.testing.assert_close(batch, new_batch, rtol=0, atol=0)
                    torch.testing.assert_close(labels, new_labels, rtol=0, atol=0)
            _, default = runner.prepare_data(dataset, batch_size=4, cohort_manifest=frozen)
            self.assertEqual(default["tfidf"], original["tfidf"])
            self.assertEqual(default["input_fingerprints"], original["input_fingerprints"])
            self.assertFalse(default["frozen_cohort"]["intentional_feature_capacity_change"])

    def test_capacity_override_keeps_raw_hash_guards_and_default_fingerprints_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            feature_fixture(dataset)
            frozen = root / "frozen.json"
            _, manifest = freeze(dataset, frozen)
            changed = dataset / manifest["files"][0]["file"]
            original_bytes = changed.read_bytes()
            changed.write_bytes(original_bytes + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner.prepare_data(dataset, batch_size=4, cohort_manifest=frozen, tfidf_max_features=2048)
            changed.write_bytes(original_bytes)
            corrupted = deepcopy(manifest)
            corrupted["input_fingerprints"]["train"][0] = "corrupted"
            frozen.write_text(json.dumps({"dataset": corrupted}))
            with self.assertRaisesRegex(ValueError, "graph/features"):
                runner.prepare_data(dataset, batch_size=4, cohort_manifest=frozen)
            frozen.write_text(json.dumps({"dataset": manifest}))
            with self.assertRaisesRegex(ValueError, "512"):
                runner.run_comparisons(dataset, root / "bad-matching", models=["matching_exact"],
                                       cohort_manifest=frozen, tfidf_max_features=2048)
            for invalid in (0, -1, 2.5):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                    runner.prepare_data(dataset, tfidf_max_features=invalid)

    def test_tiny_actual_comparison_saves_width_specific_inputs_and_recomputable_results(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            feature_fixture(dataset)
            frozen = root / "frozen.json"
            _, original = freeze(dataset, frozen)
            output = root / "experiment"
            arguments = dict(dataset_dir=dataset, output_dir=output, cohort_manifest=frozen,
                             seeds=[3], pretrain_epochs=1, finetune_epochs=1, patience=1,
                             batch_size=4, num_threads=1)
            rows, summary = entrypoint.run_feature_comparison(**arguments)
            self.assertEqual([row["tfidf_max_features"] for row in rows], [512, 2048])
            self.assertEqual(len({row["initial_state_sha256"] for row in rows}), 2)
            configs = []
            for row in rows:
                width = row["tfidf_max_features"]
                config = dict(row["model_config"])
                self.assertEqual(config.pop("in_dim"), width)
                configs.append(config)
                self.assertEqual(row["model"], "lambda_0p25")
                self.assertEqual(row["model_config"]["lam"], .25)
                self.assertEqual(row["model_config"]["duplicate_message_direction"], "copy_to_reference")
                folder = output / f"features_{width}"
                run_dir = folder / "seed_3/lambda_0p25"
                predictions = json.loads((run_dir / "predictions.json").read_text())
                self.assertEqual([p["file"] for p in predictions], original["validation_files_in_order"])
                self.assertEqual(runner.asdict(runner.compute_metrics(
                    [p["label"] for p in predictions], [p["prediction"] for p in predictions])), row["metrics"])
                manifest = json.loads((folder / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "completed")
                self.assertEqual(manifest["dataset"]["tfidf"]["max_features"], width)
                self.assertEqual(manifest["dataset"]["tfidf"]["feature_dim"], width)
                self.assertTrue(manifest["dataset"]["frozen_cohort"]["source_hashes_verified"])
                checkpoint = torch.load(run_dir / "best_checkpoint.pt", weights_only=True)
                self.assertEqual(checkpoint["model_config"], row["model_config"])
                self.assertEqual(runner.state_fingerprint(checkpoint["model_state_dict"]), row["trained_state_sha256"])
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(rows[0]["rng_stage_seeds"], rows[1]["rng_stage_seeds"])
            self.assertEqual(summary, entrypoint.summarize(rows))
            self.assertIsNone(summary["paired_f1"]["sample_sd"])
            self.assertEqual(json.loads((output / "feature_seed_metrics.json").read_text()), rows)
            self.assertEqual(json.loads((output / "feature_summary.json").read_text()), summary)
            protocol = json.loads((output / "feature_protocol.json").read_text())
            self.assertEqual(protocol["status"], "completed")
            self.assertEqual(protocol["feature_capacities"], [512, 2048])
            with self.assertRaises(FileExistsError):
                entrypoint.run_feature_comparison(**arguments)


if __name__ == "__main__":
    unittest.main()
