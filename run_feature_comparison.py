"""Compare 512 versus 2048 training-fitted text features on the frozen cohort.

Exact content matching, lambda=0.25, copy-to-reference messages and the same
training budget are retained. This changes vocabulary and the encoder's input
dimension/parameter count. Seeds are paired, but initial tensors and augmented
feature tensors cannot be identical across dimensions. Validation is reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import statistics

from run_matched_comparisons import run_comparisons, sha256_file, write_json

FEATURE_CAPACITIES = (512, 2048)


def summarize(rows):
    summary = {"features": {}, "paired_f1": {},
               "interpretation": "Development validation only; input vocabulary and parameter count change together."}
    by_width = {}
    for width in FEATURE_CAPACITIES:
        selected = [row for row in rows if row["tfidf_max_features"] == width]
        by_width[width] = {row["seed"]: row for row in selected}
        summary["features"][str(width)] = {
            metric: {"mean": statistics.mean(row["metrics"][metric] for row in selected),
                     "sample_sd": statistics.stdev(row["metrics"][metric] for row in selected)
                     if len(selected) > 1 else None, "n": len(selected)}
            for metric in ("f1", "accuracy", "precision", "recall")}
    seeds = sorted(set(by_width[512]) & set(by_width[2048]))
    deltas = [by_width[2048][seed]["metrics"]["f1"] - by_width[512][seed]["metrics"]["f1"]
              for seed in seeds]
    summary["paired_f1"] = {"direction": "2048 minus 512", "seeds": seeds,
                             "differences": deltas, "mean": statistics.mean(deltas),
                             "sample_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
                             "wins": sum(x > 1e-12 for x in deltas),
                             "losses": sum(x < -1e-12 for x in deltas),
                             "ties": sum(abs(x) <= 1e-12 for x in deltas)}
    return summary


def run_feature_comparison(dataset_dir, output_dir, cohort_manifest,
                           seeds=(0, 1, 2, 3, 4), split_seed=0,
                           pretrain_epochs=15, finetune_epochs=25, patience=6,
                           batch_size=8, device="cpu", num_threads=1):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running", "feature_capacities": list(FEATURE_CAPACITIES),
        "seeds": list(seeds), "lambda": .25,
        "matching": "original normalized exact matching; no new filter or containment",
        "direction": "copy_to_reference",
        "cohort_manifest": str(Path(cohort_manifest).resolve()),
        "cohort_manifest_sha256": sha256_file(Path(cohort_manifest)),
        "controlled": "Same saved ordered 400/100 cohort, stage seeds, hyperparameters, augmentation policy, training budget and checkpoint selection. TF-IDF fitted only on training text.",
        "changed": "TF-IDF vocabulary capacity, input feature values/dimension, input-layer parameter count and initialization shapes. Hidden width remains 64.",
        "limits": "This follows a negative matching-policy comparison on a reused validation set. It is a predeclared two-capacity development check, not independent confirmation or a pure duplicate-link effect.",
    }
    write_json(output_dir / "feature_protocol.json", protocol)
    rows = []
    try:
        for width in FEATURE_CAPACITIES:
            results, _ = run_comparisons(
                dataset_dir=dataset_dir, output_dir=output_dir / f"features_{width}",
                cohort_manifest=cohort_manifest, tfidf_max_features=width,
                models=["lambda_0p25"], seeds=seeds, split_seed=split_seed,
                pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                patience=patience, batch_size=batch_size, device=device,
                num_threads=num_threads)
            rows.extend({**row, "tfidf_max_features": width} for row in results)
            write_json(output_dir / "feature_seed_metrics.json", rows)
        summary = summarize(rows)
        write_json(output_dir / "feature_summary.json", summary)
        protocol["status"] = "completed"
        write_json(output_dir / "feature_protocol.json", protocol)
        for width, metrics in summary["features"].items():
            print(f"Features {width}: mean validation F1={metrics['f1']['mean']:.6f}", flush=True)
        return rows, summary
    except BaseException as error:
        protocol.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        write_json(output_dir / "feature_protocol.json", protocol)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/storage/weibo1"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / ("feature-capacity-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    parser.add_argument("--cohort-manifest", type=Path,
                        default=Path(__file__).resolve().parent / "experiment_protocols/matching_cohort.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_feature_comparison(**vars(parse_args()))
