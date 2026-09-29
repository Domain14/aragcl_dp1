"""Reproducible, validation-only comparisons of internal ARAGCL-DP variants.

The split and training-only TF-IDF fit are shared across every training seed.
This estimates variability from optimization/augmentation on ONE fixed split;
it is not cross-validation and does not provide independent test performance.

Example:
    python run_matched_comparisons.py --dataset-dir /storage/weibo1 \
        --output-dir results/matched-masking --device cpu --num-threads 1
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import re
import statistics
import sys
import time
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sklearn
import torch
import torch_geometric

from data.duplication_graph import build_duplication_graph, build_static_graph
from data.weibo_json_loader import load_cascade_json
from augmentation.diagnostics import AugmentationRecorder, AugmentationTraceRecorder
from models.aragcl_dp import make_ablation2, make_aragcl_dp
from run_real_data import make_minibatches
from train.metrics import compute_metrics
from train.trainer import TrainConfig, contrastive_pretrain, evaluation_mode, finetune

LAMBDA_VARIANTS = {
    "lambda_0": 0.0,
    "lambda_0p1": 0.1,
    "lambda_0p25": 0.25,
    "lambda_0p5": 0.5,
}
VARIANTS = {
    "full": "ARAGCL-DP",
    "no_dup_edges": "ARAGCL-DP (no duplicate-edge aggregation; duplicate metadata retained)",
    "static": "ARAGCL-DP (static reply graph; duplicate metadata removed)",
    "copy_to_reference": "ARAGCL-DP (duplicate messages from later copies to earliest reference)",
    "filtered_copy_to_reference": "ARAGCL-DP (copies to reference; exact Weibo marker excluded from duplicate links)",
    "filtered_no_dup_edges": "ARAGCL-DP (marker-filtered duplicate metadata; no duplicate-edge aggregation)",
    **{name: f"ARAGCL-DP (copies to reference; lambda={value:g})"
       for name, value in LAMBDA_VARIANTS.items()},
}
DEFAULT_MODELS = ("full", "no_dup_edges", "static")
FILTERED_MODELS = ("filtered_copy_to_reference", "filtered_no_dup_edges")
MARKER_FILTER_POLICY = "exclude_weibo_marker"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "fbeta")
PROTOCOL_NOTE = (
    "Validation-only: each checkpoint is selected by maximum validation F1 "
    "(first epoch wins ties), then scored on the same validation partition. "
    "No independent test set. Seeds vary optimization/augmentation, not the "
    "fixed stratified 80/20 split. This is not five-fold cross-validation. "
    "Identical text identifies inferred content duplicates, not verified reshares."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_fingerprint(state) -> str:
    """Hash names, types, shapes and bytes, independent of torch.save metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def stage_seed(seed: int, stage: str) -> int:
    # Stable across processes; Python's salted hash() must not be used here.
    key = f"aragcl-matched-v1:{seed}:{stage}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graph_to_device(graph, device):
    """Create a graph copy with all tensor fields on the requested device."""
    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value
    return replace(graph, **{field.name: move(getattr(graph, field.name))
                             for field in fields(graph)})


def prepare_data(dataset_dir, batch_size=8, split_seed=0, device="cpu",
                 max_cascades=None, sample_seed=0, include_filtered=False):
    """Load once; split once; fit a single shared, training-only vectorizer."""
    paths = sorted(Path(dataset_dir).resolve().glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No *.json cascade files in {dataset_dir}")
    inventory = []
    for path in paths:
        digest = sha256_file(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        label = int(document["source"]["label"])
        if label not in (0, 1):
            raise ValueError(f"Expected binary label in {path}, got {label}")
        inventory.append({"file": path.name, "path": str(path), "sha256": digest,
                          "cascade_id": str(document["source"]["tweet id"]),
                          "label": label, "node_count": len(document["comment"]) + 1,
                          "normalized_root_text_sha256": hashlib.sha256(
                              document["source"]["content"].strip().lower().encode()).hexdigest()
                              if document["source"]["content"].strip() else None})
    selected_indices = list(range(len(paths)))
    if max_cascades is not None:
        if max_cascades < 4 or max_cascades > len(paths):
            raise ValueError(f"max_cascades must be between 4 and {len(paths)}")
        if max_cascades < len(paths):
            selected_indices, _ = train_test_split(
                selected_indices, train_size=max_cascades,
                stratify=[item["label"] for item in inventory], random_state=sample_seed)
            selected_indices = sorted(selected_indices)
    records = [inventory[index] for index in selected_indices]
    raw, labels, centralities = [], [], []
    for record in records:
        path = Path(record["path"])
        posts, label, centrality = load_cascade_json(str(path))
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Dataset file changed while being loaded: {path}")
        raw.append(posts)
        labels.append(label)
        centralities.append(centrality)

    train_indices, val_indices = train_test_split(
        list(range(len(raw))), test_size=0.2, stratify=labels,
        random_state=split_seed,
    )
    root_groups = {}
    for index, record in enumerate(records):
        root_hash = record["normalized_root_text_sha256"]
        if root_hash is not None:
            root_groups.setdefault(root_hash, []).append(index)
    train_set, validation_set = set(train_indices), set(val_indices)
    overlap_groups = [{
        "normalized_root_text_sha256": root_hash,
        "train_files": [records[index]["file"] for index in group if index in train_set],
        "validation_files": [records[index]["file"] for index in group if index in validation_set],
    } for root_hash, group in root_groups.items()
        if train_set.intersection(group) and validation_set.intersection(group)]
    train_cascades = [raw[index] for index in train_indices]
    vectorizer = TfidfVectorizer(max_features=512, sublinear_tf=True,
                                  analyzer="char", ngram_range=(1, 2))
    vectorizer.fit([post.text for posts in train_cascades for post in posts])
    # Assignment uses the fitted training vocabulary for both partitions.
    for posts in raw:
        matrix = vectorizer.transform([post.text for post in posts])
        features = torch.tensor(matrix.toarray(), dtype=torch.float32)
        for index, post in enumerate(posts):
            post.feature = features[index]

    graphs, static_graphs, filtered_graphs = [], [], []
    for posts, centrality in zip(raw, centralities):
        graph = build_duplication_graph(posts)
        graph.precomputed_centrality.update(centrality)
        graphs.append(graph_to_device(graph, device))
        static_graphs.append(graph_to_device(build_static_graph(posts), device))
        if include_filtered:
            filtered = build_duplication_graph(posts, duplicate_policy=MARKER_FILTER_POLICY)
            filtered.precomputed_centrality.update(centrality)
            filtered_graphs.append(graph_to_device(filtered, device))

    def batches(all_graphs, indices, size):
        items = make_minibatches([all_graphs[index] for index in indices],
                                 [labels[index] for index in indices], size)
        return [(graph_to_device(graph, device), batch.to(device), y.to(device))
                for graph, batch, y in items]

    # Validation is one batch, matching previous runs; no shuffled epoch order.
    train = batches(graphs, train_indices, batch_size)
    validation = batches(graphs, val_indices, len(val_indices))
    static_train = batches(static_graphs, train_indices, batch_size)
    static_validation = batches(static_graphs, val_indices, len(val_indices))
    data = {
        "feature_dim": len(vectorizer.vocabulary_),
        "train_batches": train, "val_batches": validation,
        "pretrain_batches": [(graph, batch) for graph, batch, _ in train],
        "static_train_batches": static_train,
        "static_val_batches": static_validation,
        "static_pretrain_batches": [(graph, batch) for graph, batch, _ in static_train],
    }
    if include_filtered:
        filtered_train = batches(filtered_graphs, train_indices, batch_size)
        data.update({
            "filtered_train_batches": filtered_train,
            "filtered_val_batches": batches(filtered_graphs, val_indices, len(val_indices)),
            "filtered_pretrain_batches": [(graph, batch) for graph, batch, _ in filtered_train],
        })
    fingerprint_batches = [("train", "train_batches"), ("validation", "val_batches"),
                           ("static_train", "static_train_batches"),
                           ("static_validation", "static_val_batches")]
    if include_filtered:
        fingerprint_batches.extend([("filtered_train", "filtered_train_batches"),
                                    ("filtered_validation", "filtered_val_batches")])
    provenance = {
        "files": records,
        "sampling": {
            "max_cascades": max_cascades, "sample_seed": sample_seed,
            "source_file_count": len(inventory), "selected_file_count": len(records),
            "algorithm": "Select train_size=max_cascades with sklearn.train_test_split, stratify by source label, random_state=sample_seed; sort selected file indices before splitting",
            "selected_source_indices": selected_indices,
            "source_inventory": inventory,
            "cohort_note": "Cross-run comparability must be checked using recorded file hashes, split membership, features, and settings. This runner does not automatically establish that its cohort matches an earlier run.",
        },
        "normalized_root_overlap": {
            "normalization": "strip whitespace, lowercase; exclude blank roots",
            "cross_partition_group_count": len(overlap_groups),
            "cross_partition_file_count": sum(len(group["train_files"]) + len(group["validation_files"])
                                               for group in overlap_groups),
            "groups": overlap_groups,
            "interpretation": "Diagnostic of exact repeated root content only; zero overlap does not exclude paraphrases, related claims or other leakage.",
        },
        "split_seed": split_seed,
        "train_indices": train_indices, "validation_indices": val_indices,
        "train_files_in_order": [records[index]["file"] for index in train_indices],
        "validation_files_in_order": [records[index]["file"] for index in val_indices],
        "train_batch_file_order": [
            [records[index]["file"] for index in train_indices[start:start + batch_size]]
            for start in range(0, len(train_indices), batch_size)
        ],
        "batch_order": "Fixed original split order, shared by every variant, seed and epoch",
        "train_label_counts": {str(label): sum(labels[index] == label for index in train_indices)
                               for label in (0, 1)},
        "validation_label_counts": {str(label): sum(labels[index] == label for index in val_indices)
                                    for label in (0, 1)},
        "tfidf": {
            "fit_count": 1, "fit_partition": "train", "max_features": 512,
            "analyzer": "char", "ngram_range": [1, 2], "sublinear_tf": True,
            "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
            "idf": vectorizer.idf_.tolist(),
            "feature_dim": len(vectorizer.vocabulary_),
        },
        "input_fingerprints": {
            partition: [state_fingerprint({
                "x": graph.x, "reply_edges": graph.reply_edge_index,
                "duplicate_edges": graph.duplication_edge_index,
                "duplicate_frequency": graph.duplication_freq,
                "depth": graph.propagation_depth, "root_mask": graph.root_mask,
                "batch": batch, "labels": y,
            }) for graph, batch, y in data[key]]
            for partition, key in fingerprint_batches
        },
    }
    if include_filtered:
        provenance["duplicate_filter_audit"] = audit_duplicate_filter(
            records, graphs, filtered_graphs, train_indices, val_indices)
    return data, provenance


def audit_duplicate_filter(records, original_graphs, filtered_graphs,
                           train_indices, val_indices):
    """Record the frozen filter's effect without removing any observations."""
    train_set = set(train_indices)
    rows = []
    for index, (record, before, after) in enumerate(zip(records, original_graphs, filtered_graphs)):
        for name in ("x", "reply_edge_index", "timestamps", "propagation_depth", "root_mask"):
            if not torch.equal(getattr(before, name), getattr(after, name)):
                raise RuntimeError(f"Duplicate filtering unexpectedly changed {name}: {record['file']}")
        before_edges = before.duplication_edge_index.size(1)
        after_edges = after.duplication_edge_index.size(1)
        if after_edges > before_edges:
            raise RuntimeError("Excluding a marker must not add duplicate links")
        rows.append({
            "file": record["file"], "partition": "train" if index in train_set else "validation",
            "nodes": before.x.size(0), "reply_edges": before.reply_edge_index.size(1),
            "duplicate_edges_before": before_edges, "duplicate_edges_after": after_edges,
            "duplicate_edges_removed": before_edges - after_edges,
            "nodes_with_changed_duplication_frequency": int(
                (before.duplication_freq != after.duplication_freq).sum().item()),
        })
    totals = {}
    for partition, indices in (("train", train_indices), ("validation", val_indices)):
        selected = [rows[index] for index in indices]
        totals[partition] = {
            "cascades": len(selected),
            **{key: sum(row[key] for row in selected) for key in (
                "nodes", "reply_edges", "duplicate_edges_before", "duplicate_edges_after",
                "duplicate_edges_removed", "nodes_with_changed_duplication_frequency")},
            "cascades_with_removed_edges": sum(row["duplicate_edges_removed"] > 0 for row in selected),
            "cascades_with_duplicates_before": sum(row["duplicate_edges_before"] > 0 for row in selected),
            "cascades_with_duplicates_after": sum(row["duplicate_edges_after"] > 0 for row in selected),
        }
    return {
        "duplicate_policy": MARKER_FILTER_POLICY,
        "normalization": "strip whitespace and lowercase; exact whole-text match only",
        "excluded_texts": ["转发微博"],
        "rule_origin": "Fixed before this experiment from the prior training-partition audit; not learned from validation outcomes",
        "interpretation": "All nodes, features and reply edges are preserved. Removing duplicate links also changes duplicate frequency and augmentation importance; this tests the full filter intervention.",
        "unchanged_fields_verified": ["x", "reply_edge_index", "timestamps", "propagation_depth", "root_mask"],
        "summary": totals, "files": rows,
    }


def graph_construction_for_variant(variant):
    return {"representation": "static" if variant == "static" else "duplication",
            "duplicate_policy": MARKER_FILTER_POLICY if variant in FILTERED_MODELS else "none"}


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def summary_statistics(rows):
    """Use sample SD (ddof=1); a single observation has no sample SD."""
    def stats(values):
        return {"n": len(values), "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
    models = [model for model in VARIANTS if any(row["model"] == model for row in rows)]
    summary = {"protocol_note": PROTOCOL_NOTE, "models": {}, "paired_f1_differences": []}
    by_model = {}
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        by_model[model] = {row["seed"]: row["metrics"]["f1"] for row in selected}
        summary["models"][model] = {
            name: stats([row["metrics"][name] for row in selected]) for name in METRIC_NAMES
        }
    for position, left in enumerate(models):
        for right in models[position + 1:]:
            common_seeds = sorted(set(by_model[left]) & set(by_model[right]))
            if not common_seeds:
                continue
            differences = [by_model[left][seed] - by_model[right][seed] for seed in common_seeds]
            summary["paired_f1_differences"].append({
                "direction": f"{left} minus {right}", "seeds": common_seeds,
                "differences": differences, **stats(differences),
            })
    return summary


def write_aggregate(output_dir, rows):
    write_json(output_dir / "seed_metrics.json", rows)
    write_json(output_dir / "summary.json", summary_statistics(rows))
    with (output_dir / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "seed", "split_seed", "best_epoch",
                                                     *METRIC_NAMES, "validation_loss"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ("model", "seed", "split_seed", "best_epoch")}
                            | row["metrics"] | {"validation_loss": row["validation_loss"]})


def source_files():
    root = Path(__file__).resolve().parent
    paths = list(root.glob("*.py"))
    for directory in ("augmentation", "data", "eval", "losses", "models", "train"):
        paths.extend((root / directory).rglob("*.py"))
    return {str(path.relative_to(root)): path for path in sorted(paths)}


def run_comparisons(dataset_dir, output_dir, seeds=(0, 1, 2, 3, 4), split_seed=0,
                    pretrain_epochs=15, finetune_epochs=25, patience=6, batch_size=8,
                    device="cpu", num_threads=1, models=DEFAULT_MODELS,
                    max_cascades=None, sample_seed=0):
    seeds, models = list(seeds), list(models)
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if not models or len(models) != len(set(models)) or any(model not in VARIANTS for model in models):
        raise ValueError(f"Models must be distinct choices from {list(VARIANTS)}")
    if pretrain_epochs < 0 or min(finetune_epochs, patience, batch_size, num_threads) < 1:
        raise ValueError("Finetune epochs, patience, batch size and threads must be positive")
    include_filtered = any(model in FILTERED_MODELS for model in models)
    lambda_experiment = any(model in LAMBDA_VARIANTS for model in models)
    if lambda_experiment and (include_filtered or "static" in models):
        raise ValueError("Lambda comparisons require the same unfiltered duplication graph for every variant")
    record_augmentation = include_filtered or lambda_experiment
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if selected_device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory; existing results preserved: {output_dir}")
    if output_dir == Path(dataset_dir).resolve():
        raise ValueError("Output directory must differ from the raw dataset directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(epochs=finetune_epochs, patience=patience,
                      pretrain_epochs=pretrain_epochs, device=str(selected_device))
    sources = source_files()
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    with zipfile.ZipFile(output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sources.items():
            archive.write(path, name)
    manifest = {
        "status": "preparing", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_note": PROTOCOL_NOTE, "dataset_dir": str(Path(dataset_dir).resolve()),
        "training_seeds": seeds, "split_seed": split_seed,
        "max_cascades": max_cascades, "sample_seed": sample_seed,
        "variants": {model: VARIANTS[model] for model in models},
        "graph_construction_by_variant": {model: graph_construction_for_variant(model) for model in models},
        "augmentation_usage_recorded": record_augmentation,
        "augmentation_usage_note": "For marker-filter and lambda experiments, actual counts are recorded across both views and all pretraining batches, without additional random draws. Equal base rates need not yield equal realized corruption. Edge removal includes node deletion and edge dropout.",
        "duplicate_direction_note": "full retains reference_to_copy messages; copy_to_reference reverses only encoder duplicate messages. Stored links, duplicate metadata, augmentation rules and lambda remain unchanged. This is a modeling ablation, not a verified reshare reconstruction.",
        "training_config": asdict(cfg), "batch_size": batch_size,
        "num_threads": num_threads, "deterministic_algorithms": True,
        "rng_policy": "Same initialization seed and separately reset pretrain/finetune/evaluation stage seeds per variant; model execution order cannot advance another model's RNG",
        "rng_limitation": "Equal seeds do not guarantee identical masks if variants execute different numbers or shapes of random operations; exact input order and initial weights are verified",
        "source_sha256": hashes,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "torch": torch.__version__,
                        "torch_geometric": torch_geometric.__version__,
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                        "cuda_runtime": torch.version.cuda, "device": str(selected_device)},
        "command_argv": sys.argv, "runs": [],
    }
    if lambda_experiment:
        manifest.update({
            "lambda_by_variant": {name: LAMBDA_VARIANTS[name] for name in models if name in LAMBDA_VARIANTS},
            "lambda_comparison_note": "Lambda variants all retain unfiltered duplicate links and metadata, use_duplication=True, and copy_to_reference messages. Only lambda differs. Exact ordered augmented-view fingerprints must match within each seed before each variant is fine-tuned. F1 selection and the classification decision rule remain unchanged.",
            "augmentation_traces_match_within_seed": {},
        })
    write_json(output_dir / "manifest.json", manifest)
    rows = []
    try:
        data, dataset = prepare_data(dataset_dir, batch_size, split_seed, selected_device,
                                     max_cascades, sample_seed, include_filtered=include_filtered)
        manifest.update({"status": "running", "dataset": dataset})
        if include_filtered:
            write_json(output_dir / "duplicate_filter_audit.json", dataset["duplicate_filter_audit"])
        write_json(output_dir / "manifest.json", manifest)
        print(PROTOCOL_NOTE, flush=True)
        print(dataset["sampling"]["cohort_note"], flush=True)
        print("Exact normalized root-text groups crossing train/validation: "
              f"{dataset['normalized_root_overlap']['cross_partition_group_count']}", flush=True)
        print(f"Fixed split: {len(dataset['train_indices'])} train / {len(dataset['validation_indices'])} validation; seeds {seeds}", flush=True)
        if include_filtered:
            for partition, audit in dataset["duplicate_filter_audit"]["summary"].items():
                print(f"Marker filter ({partition}): duplicate links "
                      f"{audit['duplicate_edges_before']} -> {audit['duplicate_edges_after']}; "
                      f"{audit['cascades_with_removed_edges']} cascades affected; all posts retained", flush=True)
        for seed in seeds:
            expected_trace = None
            if lambda_experiment:
                manifest["augmentation_traces_match_within_seed"][str(seed)] = False
                write_json(output_dir / "manifest.json", manifest)
            seeds_for_stage = {stage: stage_seed(seed, stage)
                               for stage in ("initialization", "pretrain", "finetune", "evaluation")}
            reset_rng(seeds_for_stage["initialization"])
            reference = make_aragcl_dp(data["feature_dim"])
            initial_state = {key: value.detach().cpu().clone() for key, value in reference.state_dict().items()}
            expected_hash = state_fingerprint(initial_state)
            del reference
            for variant in models:
                started = time.monotonic()
                run_dir = output_dir / f"seed_{seed}" / variant
                run_dir.mkdir(parents=True)
                reset_rng(seeds_for_stage["initialization"])
                factory = make_ablation2 if variant in ("no_dup_edges", "filtered_no_dup_edges") else make_aragcl_dp
                direction = ("copy_to_reference" if variant in (
                    "copy_to_reference", "filtered_copy_to_reference", "filtered_no_dup_edges")
                             or variant in LAMBDA_VARIANTS
                             else "reference_to_copy")
                model = factory(data["feature_dim"], duplicate_message_direction=direction,
                                lam=LAMBDA_VARIANTS.get(variant, 0.5))
                model.load_state_dict(initial_state, strict=True)
                actual_hash = state_fingerprint(model.state_dict())
                if actual_hash != expected_hash:
                    raise RuntimeError("Matched initialization check failed")
                model.to(selected_device)
                recorder = (AugmentationTraceRecorder() if lambda_experiment
                            else AugmentationRecorder() if record_augmentation else None)
                trace_metadata = {}
                model.augmentation_observer = recorder
                graph_construction = graph_construction_for_variant(variant)
                prefix = ("filtered_" if variant in FILTERED_MODELS
                          else "static_" if variant == "static" else "")
                train = data[prefix + "train_batches"]
                validation = data[prefix + "val_batches"]
                pretrain = data[prefix + "pretrain_batches"]
                text_log = io.StringIO()
                with (run_dir / "training.log").open("w", encoding="utf-8") as log_file, \
                        redirect_stdout(Tee(sys.stdout, text_log, log_file)):
                    print(f"Seed {seed}, model {variant}, initial state {actual_hash}")
                    reset_rng(seeds_for_stage["pretrain"])
                    model, pretrain_history = contrastive_pretrain(VARIANTS[variant], model, pretrain, cfg)
                    if lambda_experiment:
                        trace = recorder.trace_dict()
                        write_json(run_dir / "augmentation_trace.json", trace)
                        write_json(run_dir / "augmentation_usage.json", recorder.to_dict())
                        if expected_trace is None:
                            expected_trace = trace
                        elif trace != expected_trace:
                            raise RuntimeError(f"Augmentation trace mismatch for seed {seed}, variant {variant}; comparison rejected")
                        trace_metadata = {"augmentation_trace_sha256": trace["sequence_sha256"]}
                    reset_rng(seeds_for_stage["finetune"])
                    finetune_history = finetune(VARIANTS[variant], model, train, validation, cfg)
                best_epoch = max(range(len(finetune_history)), key=lambda index: finetune_history[index].f1)
                reset_rng(seeds_for_stage["evaluation"])
                prediction_rows, true, predicted, loss_sum, observations = [], [], [], 0.0, 0
                val_records = [dataset["files"][index] for index in dataset["validation_indices"]]
                offset = 0
                with evaluation_mode(model):
                    for graph, batch, y in validation:
                        logits = model.classify(graph, batch=batch)
                        probabilities = logits.softmax(dim=1).cpu().tolist()
                        predictions = logits.argmax(dim=1).cpu().tolist()
                        labels = y.cpu().tolist()
                        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
                        observations += len(labels)
                        for record, label, pred, probs in zip(val_records[offset:], labels, predictions, probabilities):
                            prediction_rows.append({"file": record["file"], "cascade_id": record["cascade_id"],
                                                    "label": label, "prediction": pred,
                                                    "probability_0": probs[0], "probability_1": probs[1]})
                        offset += len(labels)
                        true.extend(labels)
                        predicted.extend(predictions)
                metrics = compute_metrics(true, predicted)
                if metrics != finetune_history[best_epoch]:
                    raise RuntimeError("Restored model metrics do not match the selected best epoch")
                losses = {int(epoch): (float(train_loss), float(val_loss)) for epoch, train_loss, val_loss in
                          re.findall(r"^Epoch (\d+), Train Loss ([\d.eE+-]+), Val Loss ([\d.eE+-]+),",
                                     text_log.getvalue(), flags=re.MULTILINE)}
                history = {
                    "pretrain_loss": pretrain_history,
                    "finetune": [{"epoch": index, **asdict(item),
                                  "train_loss_logged": losses.get(index, (None, None))[0],
                                  "validation_loss_logged": losses.get(index, (None, None))[1]}
                                 for index, item in enumerate(finetune_history)],
                    "loss_precision_note": "Fine-tuning losses are parsed from trainer logging (4 decimal places); per-epoch classification metrics and pretraining losses retain returned precision.",
                    "selected_epoch_zero_based": best_epoch,
                }
                row = {"model": variant, "label": VARIANTS[variant], "seed": seed,
                       "split_seed": split_seed, "metrics": asdict(metrics),
                       "validation_loss": loss_sum / observations, "best_epoch": best_epoch,
                       "epochs_completed": len(finetune_history),
                       "initial_state_sha256": actual_hash,
                       "trained_state_sha256": state_fingerprint(model.state_dict()),
                       "rng_stage_seeds": seeds_for_stage, "model_config": asdict(model.cfg),
                       "graph_construction": graph_construction,
                       "runtime_seconds": time.monotonic() - started,
                       "protocol_note": PROTOCOL_NOTE, **trace_metadata}
                write_json(run_dir / "metrics.json", row)
                write_json(run_dir / "history.json", history)
                write_json(run_dir / "predictions.json", prediction_rows)
                if recorder is not None:
                    write_json(run_dir / "augmentation_usage.json", recorder.to_dict())
                torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "model_config": asdict(model.cfg), "training_config": asdict(cfg),
                            "graph_construction": graph_construction,
                            "selected_epoch_zero_based": best_epoch,
                            "seed": seed, "split_seed": split_seed, "variant": variant,
                            "initial_state_sha256": actual_hash, "validation_metrics": asdict(metrics),
                            **trace_metadata},
                           run_dir / "best_checkpoint.pt")
                rows.append(row)
                manifest["runs"].append({"seed": seed, "model": variant,
                                         "directory": str(run_dir.relative_to(output_dir)),
                                         "initial_state_sha256": actual_hash, **trace_metadata})
                write_aggregate(output_dir, rows)
                write_json(output_dir / "manifest.json", manifest)
                print(f"Completed seed={seed} {variant}: validation F1={metrics.f1:.6f}", flush=True)
                del model
            if lambda_experiment:
                manifest["augmentation_traces_match_within_seed"][str(seed)] = True
                write_json(output_dir / "manifest.json", manifest)
                print(f"Verified identical ordered augmentation views across variants for seed {seed}", flush=True)
        # Guard against changing code/data halfway through a long experiment.
        if hashes != {name: sha256_file(path) for name, path in sources.items()}:
            raise RuntimeError("Source files changed during the experiment; do not combine these runs")
        if any(sha256_file(Path(record["path"])) != record["sha256"] for record in dataset["files"]):
            raise RuntimeError("Dataset files changed during the experiment")
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "manifest.json", manifest)
        summary = summary_statistics(rows)
        for model, metrics in summary["models"].items():
            item = metrics["f1"]
            spread = f"{item['sample_sd']:.6f}" if item["sample_sd"] is not None else "undefined (n=1)"
            print(f"{model}: validation F1 mean={item['mean']:.6f}, sample SD={spread}", flush=True)
        return rows, summary
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(output_dir / "manifest.json", manifest)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-cascades", type=int, default=None,
                        help="Optional stratified subset size; default uses all files")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda[:index], or mps")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=list(VARIANTS), default=list(DEFAULT_MODELS),
                        help="Default: full no_dup_edges static. Direction experiment: copy_to_reference. Marker-filter experiment: filtered_copy_to_reference and filtered_no_dup_edges.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_comparisons(**vars(parse_args()))
