"""Loader for the PHEME ``all-rnr-annotated-threads`` layout.

The archive stores one folder per source tweet under an event and a binary
label directory. ``structure.json`` is a nested parent-to-children mapping;
tweet JSONs live in ``source-tweets`` for the root and ``reactions`` for
replies. Rumours map to 1 and non-rumours to 0. Unclear or structurally
incomplete threads are excluded and recorded in the returned audit.

Text matching creates inferred within-thread relationships only. It does not
recover verified retweet provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from collections import Counter
from pathlib import Path
from typing import Iterable
import json

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .duplication_graph import RawPost


LABELS = {"rumour": 1, "nonrumour": 0}
DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


@dataclass
class PhemeConversation:
    event: str
    thread_id: str
    label: int
    posts: list[RawPost]
    source_sha256: str = ""


def _walk_structure(tree, parent_id=None, result=None):
    if result is None:
        result = []
    if isinstance(tree, dict):
        for node_id, children in tree.items():
            node_id = str(node_id)
            result.append((node_id, parent_id))
            _walk_structure(children, node_id, result)
    elif isinstance(tree, list):
        for child in tree:
            _walk_structure(child, parent_id, result)
    else:
        raise ValueError(f"Unexpected PHEME structure node: {type(tree).__name__}")
    return result


def _parse_twitter_time(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, DATE_FORMAT).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _tweet_path(thread_dir: Path, tweet_id: str, is_root: bool) -> Path:
    primary = thread_dir / ("source-tweets" if is_root else "reactions") / f"{tweet_id}.json"
    if primary.exists():
        return primary
    # A few exports place a root under reactions or vice versa. Resolve by ID
    # only; never search outside this thread's folder.
    alternate = thread_dir / ("reactions" if is_root else "source-tweets") / f"{tweet_id}.json"
    return alternate


def load_pheme_dataset(root: str | Path, *, include_unclear: bool = False):
    """Load PHEME conversations and a completeness/label audit.

    One incomplete thread is excluded rather than inventing text for a node
    present in the reply tree but absent from the archive. Set
    ``include_unclear=True`` to raise on the non-binary annotation instead of
    silently excluding it; this project uses the binary rumor task.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PHEME dataset directory not found: {root}")
    event_dirs = sorted(path for path in root.iterdir()
                        if path.is_dir() and not path.name.startswith("."))
    if not event_dirs:
        raise FileNotFoundError(f"No PHEME event folders found in {root}")

    conversations = []
    audit = {
        "dataset_root": str(root),
        "label_mapping": {"rumour": 1, "nonrumour": 0},
        "excluded_unclear": [], "excluded_incomplete": [],
        "event_counts": {}, "label_summary": {}, "totals": {},
        "timestamp_fallback_threads": [],
        "note": "Source labels are binary rumor/non-rumor; structure links are observed replies. Duplicate-content links are inferred separately.",
    }
    counts = {"loaded_threads": 0, "loaded_rumour": 0, "loaded_nonrumour": 0,
              "nodes": 0, "reply_edges": 0, "single_node_threads": 0,
              "threads_with_replies": 0, "threads_with_multiple_components": 0,
              "threads_with_inferred_exact_links": 0, "inferred_exact_links": 0,
              "posts_in_exact_duplicate_groups": 0}
    for label_name in LABELS:
        audit["label_summary"][label_name] = {
            "threads": 0, "threads_with_inferred_exact_links": 0,
            "inferred_exact_links": 0, "posts_in_exact_duplicate_groups": 0}

    for event_dir in event_dirs:
        event_name = event_dir.name.removesuffix("-all-rnr-threads")
        event_counts = {"rumour": 0, "nonrumour": 0, "unclear": 0,
                        "incomplete": 0, "loaded": 0,
                        "threads_with_inferred_exact_links": 0,
                        "inferred_exact_links": 0,
                        "posts_in_exact_duplicate_groups": 0}
        for class_dir in sorted(path for path in event_dir.iterdir()
                                if path.is_dir() and path.name in ("rumours", "non-rumours")):
            for thread_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
                annotation_path = thread_dir / "annotation.json"
                structure_path = thread_dir / "structure.json"
                if not annotation_path.is_file() or not structure_path.is_file():
                    audit["excluded_incomplete"].append({
                        "event": event_name, "thread_id": thread_dir.name,
                        "reason": "annotation_or_structure_missing"})
                    event_counts["incomplete"] += 1
                    continue
                try:
                    annotation_text = annotation_path.read_text(encoding="utf-8")
                    structure_text = structure_path.read_text(encoding="utf-8")
                    annotation = json.loads(annotation_text)
                    label_name = annotation.get("is_rumour")
                    tree = json.loads(structure_text)
                except (OSError, json.JSONDecodeError) as error:
                    audit["excluded_incomplete"].append({
                        "event": event_name, "thread_id": thread_dir.name,
                        "reason": f"invalid_json:{type(error).__name__}"})
                    event_counts["incomplete"] += 1
                    continue

                if label_name not in LABELS:
                    event_counts["unclear"] += 1
                    if include_unclear:
                        raise ValueError(f"Non-binary PHEME label {label_name!r}: {thread_dir}")
                    audit["excluded_unclear"].append({
                        "event": event_name, "thread_id": thread_dir.name,
                        "label": label_name})
                    continue
                expected_directory = "rumours" if label_name == "rumour" else "non-rumours"
                if class_dir.name != expected_directory:
                    raise ValueError(f"PHEME directory/annotation label mismatch: {thread_dir}")

                nodes = _walk_structure(tree)
                root_ids = [node_id for node_id, parent in nodes if parent is None]
                if not root_ids or thread_dir.name not in root_ids:
                    raise ValueError(f"Expected the source tweet among structure roots: {thread_dir}")
                node_ids = [node_id for node_id, _ in nodes]
                if len(node_ids) != len(set(node_ids)):
                    raise ValueError(f"Duplicate tweet ID in PHEME structure: {thread_dir}")

                raw_by_id, missing = {}, []
                thread_digest = sha256()
                thread_digest.update(annotation_text.encode("utf-8"))
                thread_digest.update(structure_text.encode("utf-8"))
                for node_id, parent_id in nodes:
                    path = _tweet_path(thread_dir, node_id, parent_id is None)
                    if not path.is_file():
                        missing.append(node_id)
                        continue
                    try:
                        raw_text = path.read_text(encoding="utf-8")
                        raw = json.loads(raw_text)
                    except (OSError, json.JSONDecodeError):
                        missing.append(node_id)
                        continue
                    if str(raw.get("id", node_id)) != node_id:
                        raise ValueError(f"Tweet JSON ID does not match structure ID: {path}")
                    thread_digest.update(node_id.encode("utf-8"))
                    thread_digest.update(raw_text.encode("utf-8"))
                    raw_by_id[node_id] = raw

                if missing:
                    event_counts["incomplete"] += 1
                    audit["excluded_incomplete"].append({
                        "event": event_name, "thread_id": thread_dir.name,
                        "reason": "tweet_json_missing", "missing_tweet_ids": missing})
                    continue

                raw_root = raw_by_id[thread_dir.name]
                absolute_times = {node_id: _parse_twitter_time(raw.get("created_at"))
                                  for node_id, raw in raw_by_id.items()}
                has_complete_timestamps = all(value is not None
                                              for value in absolute_times.values())
                root_time = absolute_times[thread_dir.name]
                if not has_complete_timestamps or root_time is None:
                    audit["timestamp_fallback_threads"].append({
                        "event": event_name, "thread_id": thread_dir.name,
                        "reason": "missing_or_invalid_created_at"})
                index_by_id = {node_id: index for index, (node_id, _) in enumerate(nodes)}
                cascade_key = f"{event_name}:{thread_dir.name}".encode("utf-8")
                cascade_id = int.from_bytes(sha256(cascade_key).digest()[:8], "big") & ((1 << 63) - 1)
                posts = []
                for order, (node_id, parent_id) in enumerate(nodes):
                    raw = raw_by_id[node_id]
                    absolute_time = absolute_times[node_id]
                    if has_complete_timestamps and root_time is not None:
                        timestamp = absolute_time - root_time
                    else:
                        # Stable input order is a deterministic fallback and is
                        # documented in provenance; it is not a measured time.
                        timestamp = float(order)
                    parent_index = None if parent_id is None else index_by_id[parent_id]
                    posts.append(RawPost(
                        post_id=order, parent_id=parent_index, cascade_id=cascade_id,
                        timestamp=timestamp, text=str(raw.get("text") or ""),
                        feature=torch.zeros(1, dtype=torch.float32)))

                label = LABELS[label_name]
                conversation = PhemeConversation(event_name, thread_dir.name, label, posts,
                                                thread_digest.hexdigest())
                content_counts = Counter(post.text.strip().lower()
                                         for post in posts if post.text.strip())
                duplicate_groups = [count for count in content_counts.values() if count > 1]
                duplicate_links = sum(count - 1 for count in duplicate_groups)
                duplicate_posts = sum(duplicate_groups)
                conversations.append(conversation)
                event_counts[label_name] += 1
                event_counts["loaded"] += 1
                counts["loaded_threads"] += 1
                counts["loaded_rumour" if label else "loaded_nonrumour"] += 1
                counts["nodes"] += len(posts)
                counts["reply_edges"] += sum(post.parent_id is not None for post in posts)
                counts["single_node_threads"] += len(posts) == 1
                counts["threads_with_replies"] += any(post.parent_id is not None for post in posts)
                counts["threads_with_multiple_components"] += len(root_ids) > 1
                counts["threads_with_inferred_exact_links"] += duplicate_links > 0
                counts["inferred_exact_links"] += duplicate_links
                counts["posts_in_exact_duplicate_groups"] += duplicate_posts
                label_summary = audit["label_summary"][label_name]
                label_summary["threads"] += 1
                label_summary["threads_with_inferred_exact_links"] += duplicate_links > 0
                label_summary["inferred_exact_links"] += duplicate_links
                label_summary["posts_in_exact_duplicate_groups"] += duplicate_posts
                event_counts["threads_with_inferred_exact_links"] += duplicate_links > 0
                event_counts["inferred_exact_links"] += duplicate_links
                event_counts["posts_in_exact_duplicate_groups"] += duplicate_posts
        audit["event_counts"][event_name] = event_counts

    audit["totals"] = counts
    if not conversations:
        raise ValueError(f"No usable binary PHEME conversations found in {root}")
    return conversations, audit


def fit_and_attach_tfidf(train: Iterable[PhemeConversation],
                         other_sets: Iterable[Iterable[PhemeConversation]] = (),
                         *, max_features: int = 2048):
    """Fit a character TF-IDF vocabulary on training posts only, then transform."""
    train = list(train)
    other_sets = [list(items) for items in other_sets]
    train_texts = [post.text for conversation in train for post in conversation.posts]
    if not train_texts or not any(text.strip() for text in train_texts):
        raise ValueError("Training partition has no non-empty tweet text")
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True,
                                 analyzer="char", ngram_range=(1, 2))
    vectorizer.fit(train_texts)
    for conversations in [train, *other_sets]:
        for conversation in conversations:
            texts = [post.text for post in conversation.posts]
            features = vectorizer.transform(texts).toarray()
            for post, feature in zip(conversation.posts, features):
                post.feature = torch.as_tensor(feature, dtype=torch.float32)
    return vectorizer
