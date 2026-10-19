"""Run the pre-specified PHEME leave-one-event-out classification test.

The primary comparison is the same ARAGCL-DP pipeline on reply-only graphs
versus the same graphs augmented with inferred exact-text links. The held-out
event is used only once, after checkpoint selection on the other events.

Example on Paperspace:
    python run_pheme_event_aware.py \
        --dataset-dir /storage/all-rnr-annotated-threads \
        --output-dir results/pheme-loeo-v1 --device cuda

Before the full run, `--audit-only` validates and inventories the dataset
without training. A training run is intentionally expensive: 9 events x 4
seeds x 2 matched conditions.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
import torch
import torch_geometric

from data.duplication_graph import build_duplication_graph, build_static_graph
from data.pheme_loader import load_pheme_dataset, fit_and_attach_tfidf
from models.aragcl_dp import ARAGCL_DP, ARAGCL_DP_Config
from run_real_data import make_minibatches
from train.trainer import TrainConfig, contrastive_pretrain, evaluation_mode


CONDITIONS = ("reply_only", "reply_plus_inferred_duplicates")
PROTOCOL_PATH = Path(__file__).resolve().parent / "experiment_protocols/pheme_event_aware.json"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, event: str, stage: str) -> int:
    key = f"pheme-loeo-v1:{seed}:{event}:{stage}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def reset_rng(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def graph_to_device(graph, device):
    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value
    return replace(graph, **{field.name: move(getattr(graph, field.name))
                             for field in fields(graph)})


def event_fold_plan(conversations, *, validation_fraction=0.15,
                    validation_seed=20261019):
    """Create immutable leave-one-event-out partitions and inner validation."""
    events = sorted({item.event for item in conversations})
    plan = []
    for heldout_event in events:
        candidates = [item for item in conversations if item.event != heldout_event]
        test = [item for item in conversations if item.event == heldout_event]
        if not test:
            continue
        labels = [item.label for item in candidates]
        if set(labels) != {0, 1}:
            raise ValueError(f"Training events excluding {heldout_event} do not contain both labels")
        train, validation = train_test_split(
            candidates, test_size=validation_fraction, stratify=labels,
            random_state=validation_seed)
        if {item.label for item in validation} != {0, 1}:
            raise ValueError(f"Inner validation for {heldout_event} lacks one label")
        plan.append({"heldout_event": heldout_event,
                     "train": sorted(train, key=lambda x: (x.event, x.thread_id)),
                     "validation": sorted(validation, key=lambda x: (x.event, x.thread_id)),
                     "test": sorted(test, key=lambda x: (x.event, x.thread_id))})
    if len(plan) != len(events):
        raise RuntimeError("Every event must appear exactly once as the held-out event")
    test_threads = [(item.event, item.thread_id) for fold in plan for item in fold["test"]]
    if len(test_threads) != len(set(test_threads)) or set(test_threads) != {
            (item.event, item.thread_id) for item in conversations}:
        raise RuntimeError("LOEO test folds must cover every thread exactly once")
    return plan


def _condition_batches(conversations, batch_size, device):
    labels = [item.label for item in conversations]
    duplicate_graphs = [graph_to_device(build_duplication_graph(item.posts), device)
                        for item in conversations]
    reply_graphs = [graph_to_device(build_static_graph(item.posts), device)
                    for item in conversations]

    def make(graphs):
        batches = make_minibatches(graphs, labels, batch_size=batch_size)
        return batches, [(graph, batch) for graph, batch, _ in batches]

    dup_batches, dup_pretrain = make(duplicate_graphs)
    reply_batches, reply_pretrain = make(reply_graphs)
    return {
        "reply_only": (reply_batches, reply_pretrain),
        "reply_plus_inferred_duplicates": (dup_batches, dup_pretrain),
    }


def _macro_scores(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "rumour_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "nonrumour_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "rumour_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "rumour_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _predictions(model, batches, conversations):
    rows, offset = [], 0
    with evaluation_mode(model):
        for graph, batch, labels in batches:
            logits = model.classify(graph, batch=batch)
            probabilities = logits.softmax(dim=1).cpu().tolist()
            predicted = logits.argmax(dim=1).cpu().tolist()
            true = labels.cpu().tolist()
            for conversation, label, pred, probs in zip(
                    conversations[offset:offset + len(true)], true, predicted, probabilities):
                rows.append({"event": conversation.event,
                             "thread_id": conversation.thread_id,
                             "label": int(label), "prediction": int(pred),
                             "probability_nonrumour": float(probs[0]),
                             "probability_rumour": float(probs[1])})
            offset += len(true)
    if offset != len(conversations):
        raise RuntimeError("Prediction count does not match conversation count")
    return rows


def _evaluate_validation(model, batches):
    truth, pred = [], []
    with evaluation_mode(model):
        for graph, batch, labels in batches:
            logits = model.classify(graph, batch=batch)
            truth.extend(labels.cpu().tolist())
            pred.extend(logits.argmax(dim=1).cpu().tolist())
    return _macro_scores(truth, pred)


def _finetune_macro(model, train_batches, validation_batches, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    best_score, best_state, best_epoch = -1.0, None, None
    patience_left, history = config.patience, []
    for epoch in range(config.epochs):
        model.train()
        for graph, batch, labels in train_batches:
            optimizer.zero_grad()
            logits = model.classify(graph, batch=batch)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        scores = _evaluate_validation(model, validation_batches)
        history.append({"epoch": epoch, **scores})
        if scores["macro_f1"] > best_score:
            best_score, best_state, best_epoch = scores["macro_f1"], deepcopy(model.state_dict()), epoch
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    return history, best_epoch, best_score


def _summarize_predictions(rows):
    return _macro_scores([row["label"] for row in rows],
                         [row["prediction"] for row in rows])


def summarize_seed_rows(rows):
    result = {condition: {} for condition in CONDITIONS}
    metric_names = list(rows[0]["metrics"].keys()) if rows else []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        for metric in metric_names:
            values = [row["metrics"][metric] for row in selected]
            result[condition][metric] = {
                "n_seeds": len(values),
                "mean": statistics.mean(values) if values else None,
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            }
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        by_condition = {row["condition"]: row for row in rows if row["seed"] == seed}
        if set(by_condition) != set(CONDITIONS):
            continue
        paired.append({"seed": seed, **{
            metric: by_condition[CONDITIONS[1]]["metrics"][metric] -
                    by_condition[CONDITIONS[0]]["metrics"][metric]
            for metric in metric_names}})
    result["paired_difference_duplication_minus_reply_only"] = paired
    for metric in metric_names:
        values = [row[metric] for row in paired]
        result.setdefault("paired_difference_summary", {})[metric] = {
            "n_seeds": len(values),
            "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        }
    return result


def _inventory(folds):
    return [{"heldout_event": fold["heldout_event"],
             "train": [{"event": c.event, "thread_id": c.thread_id, "label": c.label,
                        "source_sha256": c.source_sha256}
                       for c in fold["train"]],
             "validation": [{"event": c.event, "thread_id": c.thread_id, "label": c.label,
                             "source_sha256": c.source_sha256}
                            for c in fold["validation"]],
             "test": [{"event": c.event, "thread_id": c.thread_id, "label": c.label,
                       "source_sha256": c.source_sha256}
                      for c in fold["test"]],
             "label_counts": {
                 name: {str(label): sum(c.label == label for c in fold[name])
                        for label in (0, 1)}
                 for name in ("train", "validation", "test")}}
            for fold in folds]


def run_experiment(dataset_dir, output_dir, seeds=(0, 1, 2, 3), *,
                   device="cpu", num_threads=1, pretrain_epochs=15,
                   finetune_epochs=25, patience=6, batch_size=8,
                   tfidf_max_features=2048, audit_only=False):
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if min(num_threads, finetune_epochs, patience, batch_size, tfidf_max_features) < 1:
        raise ValueError("Threads, epochs, patience, batch size, and feature limit must be positive")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if selected_device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(num_threads)
    # PyG message passing may use CUDA scatter/index-add kernels without a
    # deterministic implementation. Requiring strict deterministic kernels
    # can abort an otherwise valid GPU run; control randomness with explicit
    # per-stage seeds and deterministic cuDNN settings instead.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    output_dir = Path(output_dir).resolve()
    if not audit_only:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Use an empty output directory; existing results preserved: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    conversations, audit = load_pheme_dataset(dataset_dir)
    folds = event_fold_plan(conversations)
    if audit_only:
        print(json.dumps({"audit": audit, "folds": [
            {"heldout_event": fold["heldout_event"],
             "train": len(fold["train"]), "validation": len(fold["validation"]),
             "test": len(fold["test"]),
             "test_labels": {str(label): sum(c.label == label for c in fold["test"])
                             for label in (0, 1)}} for fold in folds]}, indent=2))
        return audit, folds

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["actual_run"] = {
        "seeds": seeds, "device": str(selected_device), "num_threads": num_threads,
        "pretrain_epochs": pretrain_epochs, "finetune_epochs": finetune_epochs,
        "patience": patience, "batch_size": batch_size,
        "tfidf_max_features": tfidf_max_features,
        "inner_validation_seed": 20261019,
        "training_config_matches_pre_specified_defaults":
            seeds == [0, 1, 2, 3] and pretrain_epochs == 15 and finetune_epochs == 25 and
            patience == 6 and batch_size == 8 and tfidf_max_features == 2048,
    }
    project_root = Path(__file__).resolve().parent
    source_paths = [
        project_root / "data/pheme_loader.py",
        project_root / "data/duplication_graph.py",
        project_root / "run_pheme_event_aware.py",
        project_root / "run_real_data.py",
        project_root / "models/aragcl_dp.py",
        project_root / "models/encoders.py",
        project_root / "augmentation/views.py",
        project_root / "train/trainer.py",
        project_root / "train/metrics.py",
        PROTOCOL_PATH,
    ]
    source_hashes = {str(path.relative_to(project_root)): sha256_file(path)
                     for path in source_paths}
    manifest = {
        "status": "preparing", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_audit": audit, "protocol": protocol,
        "environment": {"python": sys.version, "platform": sys.platform,
                        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                        "cuda_runtime": torch.version.cuda, "device": str(selected_device)},
        "folds": _inventory(folds), "source_sha256": source_hashes, "runs": [],
    }
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "dataset_audit.json", audit)
    rows, oof_predictions = [], {seed: {condition: [] for condition in CONDITIONS} for seed in seeds}
    config = TrainConfig(epochs=finetune_epochs, patience=patience,
                         pretrain_epochs=pretrain_epochs, device=str(selected_device))
    try:
        for fold in folds:
            event = fold["heldout_event"]
            train, validation, test = fold["train"], fold["validation"], fold["test"]
            vectorizer = fit_and_attach_tfidf(
                train, other_sets=[validation, test], max_features=tfidf_max_features)
            feature_dim = len(vectorizer.vocabulary_)
            manifest.setdefault("fold_feature_spaces", {})[event] = {
                "fit_partition": "inner_train_only", "fit_post_count": sum(len(c.posts) for c in train),
                "vocabulary_size": feature_dim,
                "max_features": tfidf_max_features, "analyzer": "char",
                "ngram_range": [1, 2]}
            del vectorizer

            partitions = {}
            for name, conversations in (("train", train), ("validation", validation), ("test", test)):
                partitions[name] = _condition_batches(conversations, batch_size, selected_device)

            for seed in seeds:
                fold_seeds = {stage: stable_seed(seed, event, stage)
                              for stage in ("initialization", "pretrain", "finetune")}
                model_config = ARAGCL_DP_Config(
                    in_dim=feature_dim, use_duplication=True, view_strategy="random",
                    lam=0.25, duplicate_message_direction="copy_to_reference")
                reset_rng(fold_seeds["initialization"])
                reference = ARAGCL_DP(model_config)
                initial_state = {key: value.detach().cpu().clone()
                                 for key, value in reference.state_dict().items()}
                initial_hash = state_fingerprint(initial_state)
                del reference

                for condition in CONDITIONS:
                    started = time.monotonic()
                    run_dir = output_dir / event / f"seed_{seed}" / condition
                    run_dir.mkdir(parents=True, exist_ok=False)
                    reset_rng(fold_seeds["initialization"])
                    model = ARAGCL_DP(model_config)
                    model.load_state_dict(initial_state, strict=True)
                    if state_fingerprint(model.state_dict()) != initial_hash:
                        raise RuntimeError("Matched initial weights check failed")
                    model.to(selected_device)
                    train_batches, train_pretrain = partitions["train"][condition]
                    val_batches, _ = partitions["validation"][condition]
                    test_batches, _ = partitions["test"][condition]
                    with (run_dir / "training.log").open("w", encoding="utf-8") as log:
                        with redirect_stdout(log):
                            reset_rng(fold_seeds["pretrain"])
                            model, pretrain_history = contrastive_pretrain(
                                condition, model, train_pretrain, config)
                            reset_rng(fold_seeds["finetune"])
                            history, best_epoch, best_macro = _finetune_macro(
                                model, train_batches, val_batches, config)

                    predictions = _predictions(model, test_batches, test)
                    metrics = _summarize_predictions(predictions)
                    variant_row = {
                        "event": event, "seed": seed, "condition": condition,
                        "metrics": metrics, "test_conversation_count": len(predictions),
                        "best_epoch": best_epoch, "best_validation_macro_f1": best_macro,
                        "initial_state_sha256": initial_hash,
                        "trained_state_sha256": state_fingerprint(model.state_dict()),
                        "rng_stage_seeds": fold_seeds,
                        "model_config": asdict(model_config),
                        "runtime_seconds": time.monotonic() - started,
                        "test_prediction_file": str((run_dir / "predictions.json").relative_to(output_dir)),
                    }
                    write_json(run_dir / "metrics.json", variant_row)
                    write_json(run_dir / "validation_history.json", history)
                    write_json(run_dir / "pretrain_history.json", pretrain_history)
                    write_json(run_dir / "predictions.json", predictions)
                    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                "model_config": asdict(model_config),
                                "training_config": asdict(config), "event": event,
                                "seed": seed, "condition": condition,
                                "best_epoch": best_epoch, "best_validation_macro_f1": best_macro,
                                "validation_metrics": history[best_epoch]},
                               run_dir / "best_checkpoint.pt")
                    rows.append(variant_row)
                    oof_predictions[seed][condition].extend(predictions)
                    manifest["runs"].append({"event": event, "seed": seed,
                                             "condition": condition,
                                             "directory": str(run_dir.relative_to(output_dir)),
                                             "initial_state_sha256": initial_hash})
                    write_json(output_dir / "seed_fold_metrics.json", rows)
                    del model
                    if selected_device.type == "cuda":
                        torch.cuda.empty_cache()

            del partitions
            write_json(output_dir / "manifest.json", manifest)
            print(f"Completed held-out event: {event}", flush=True)

        seed_summary = []
        for seed in seeds:
            for condition in CONDITIONS:
                predictions = sorted(oof_predictions[seed][condition],
                                     key=lambda row: (row["event"], row["thread_id"]))
                if len(predictions) != len(conversations):
                    raise RuntimeError(f"OOF coverage mismatch: seed={seed}, condition={condition}")
                write_json(output_dir / f"oof_predictions_seed_{seed}_{condition}.json", predictions)
                seed_summary.append({"seed": seed, "condition": condition,
                                     "metrics": _summarize_predictions(predictions)})
        write_json(output_dir / "oof_seed_summary.json", seed_summary)
        write_json(output_dir / "summary.json", summarize_seed_rows(seed_summary))
        if source_hashes != {str(path.relative_to(project_root)): sha256_file(path)
                            for path in source_paths}:
            raise RuntimeError("Source files changed during training; do not combine these results")
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["oof_coverage"] = {str(seed): {
            condition: len(oof_predictions[seed][condition]) for condition in CONDITIONS}
            for seed in seeds}
        write_json(output_dir / "manifest.json", manifest)
        return rows, summarize_seed_rows(seed_summary)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(output_dir / "manifest.json", manifest)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Path to all-rnr-annotated-threads")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/pheme-loeo-v1"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--device", default="cpu", help="cpu, cuda[:index], or mps")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tfidf-max-features", type=int, default=2048)
    parser.add_argument("--audit-only", action="store_true",
                        help="Validate labels, graph files and event folds without training")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_experiment(**vars(parse_args()))
