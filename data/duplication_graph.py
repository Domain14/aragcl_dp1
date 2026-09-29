"""
RQ1 — Duplication-Based Propagation for Graph Construction (Section 3.5.3).

This implementation links existing posts within a cascade using nonblank,
normalized exact text (strip whitespace and lowercase). It creates no new
nodes and preserves each post's supplied feature vector. Text equality does
not establish a verified repost or its original source: the available Weibo
data has text and reply relationships, without explicit reshare-source IDs.

Each group forms a directed star from its earliest matching reference to
the later matches (timestamp ties retain input order). duplication_sets[v]
contains these later matches only for the reference; it is not an all-peers
same-content set. duplication_freq is the size of that stored set, while
propagation_depth comes from the reply tree. The encoder separately chooses
whether messages follow or reverse these stored duplicate-content links.

Output is a single heterogeneous-but-flattened graph object with:
  x                    [N, d]   node features
  reply_edge_index     [2, E_r] original propagation/reply edges
  duplication_edge_index [2, E_d] earliest reference -> later match edges
  timestamps           [N]      float, arbitrary time unit
  propagation_depth    [N]      int, depth of node in the reply tree
  duplication_freq     [N]      int, |D_v| for each node
  root_mask            [N]      bool, True at each cascade's root
"""
from dataclasses import dataclass, field
from typing import List, Optional
import torch


DUPLICATE_POLICIES = ("none", "exclude_weibo_marker")


@dataclass
class RawPost:
    post_id: int
    parent_id: Optional[int]     # None for the cascade root
    cascade_id: int
    timestamp: float
    text: str
    feature: torch.Tensor        # pre-computed content embedding, [d]


@dataclass
class DuplicationGraph:
    x: torch.Tensor
    reply_edge_index: torch.Tensor
    duplication_edge_index: torch.Tensor
    timestamps: torch.Tensor
    propagation_depth: torch.Tensor
    duplication_freq: torch.Tensor
    root_mask: torch.Tensor
    duplication_sets: List[List[int]] = field(default_factory=list)
    # Optional: node-level centrality scores computed OUTSIDE this
    # pipeline and attached at load time (e.g. a dataset that ships
    # pre-computed Degree/PageRank/Eigenvector/Betweenness per node).
    # Keyed by metric name -> [N] tensor aligned with node index.
    # augmentation/centrality.py's "dataset" strategy reads from here
    # instead of recomputing centrality via networkx.
    precomputed_centrality: dict = field(default_factory=dict)

    @property
    def edge_index(self) -> torch.Tensor:
        """Combined edge index (reply + duplication) for models that
        don't need the two edge types separately, e.g. plain GCN."""
        return torch.cat([self.reply_edge_index, self.duplication_edge_index], dim=1)


def _is_duplicate(text_a: str, text_b: str, exact_only: bool = True,
                   near_dup_threshold: float = 0.95) -> bool:
    """
    Normalized exact-content matching, not verification of resharing.
    Near-duplicate matching is not implemented. In particular, generic
    repeated replies can match under this rule without sharing a source.

    Empty/whitespace-only content is never treated as a duplicate of
    other empty content -- blank comments (image-only replies,
    deleted-text placeholders) are common in real data, and matching
    on "" would otherwise silently inflate N_r for any cascade with
    more than one blank comment.
    """
    if not text_a.strip() or not text_b.strip():
        return False
    if exact_only:
        return text_a.strip().lower() == text_b.strip().lower()
    raise NotImplementedError(
        "Near-duplicate detection not wired up -- implement a "
        "similarity function here (e.g. cosine on `feature`) if your "
        "dataset requires it, and set exact_only=False."
    )


def build_duplication_graph(posts: List[RawPost],
                             exact_only: bool = True, *,
                             duplicate_policy: str = "none") -> DuplicationGraph:
    """
    posts: all posts across one cascade, in any order (parent_id=None
    marks the root). Call once per cascade, then batch the resulting
    DuplicationGraph objects (e.g. via torch_geometric.data.Batch)
    for training.

    duplicate_policy="none" preserves the original matching rule.
    "exclude_weibo_marker" suppresses duplicate links only for content
    equal to "转发微博" after strip/lower normalization. All posts and
    reply connections remain; duplicate sets and frequencies reflect the
    retained links. This frozen rule does not infer reshare provenance.
    """
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise ValueError(
            f"duplicate_policy must be one of {DUPLICATE_POLICIES}; "
            f"got {duplicate_policy!r}"
        )
    posts_by_id = {p.post_id: p for p in posts}
    root = next(p for p in posts if p.parent_id is None)

    # --- propagation depth via BFS from root over reply edges -------
    children = {p.post_id: [] for p in posts}
    for p in posts:
        if p.parent_id is not None:
            children[p.parent_id].append(p.post_id)

    depth = {root.post_id: 0}
    frontier = [root.post_id]
    while frontier:
        nxt = []
        for pid in frontier:
            for c in children[pid]:
                depth[c] = depth[pid] + 1
                nxt.append(c)
        frontier = nxt

    # --- duplicate detection: group posts by (near-)identical text --
    groups: List[List[int]] = []
    seen = set()
    for p in posts:
        if p.post_id in seen:
            continue
        group = [p.post_id]
        seen.add(p.post_id)
        for q in posts:
            if q.post_id in seen:
                continue
            if _is_duplicate(p.text, q.text, exact_only=exact_only):
                group.append(q.post_id)
                seen.add(q.post_id)
        groups.append(group)

    # The earliest match is a reference, not a verified reshare source.
    # Store reference -> copy links among existing posts. Timestamp ties
    # retain the input ordering; no causal provenance is inferred from ties.
    duplication_sets = {p.post_id: [] for p in posts}
    dup_edges = []
    for group in groups:
        if (duplicate_policy == "exclude_weibo_marker"
                and posts_by_id[group[0]].text.strip().lower() == "转发微博"):
            continue
        group_sorted = sorted(group, key=lambda pid: posts_by_id[pid].timestamp)
        source_id = group_sorted[0]
        for dup_id in group_sorted[1:]:
            duplication_sets[source_id].append(dup_id)
            dup_edges.append((source_id, dup_id))

    duplication_freq = {pid: len(duplication_sets[pid]) for pid in duplication_sets}

    # --- assemble tensors ---------------------------------------------
    ids = [p.post_id for p in posts]
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    x = torch.stack([posts_by_id[pid].feature for pid in ids])
    timestamps = torch.tensor([posts_by_id[pid].timestamp for pid in ids],
                               dtype=torch.float)
    prop_depth = torch.tensor([depth.get(pid, 0) for pid in ids], dtype=torch.float)
    dup_freq = torch.tensor([duplication_freq.get(pid, 0) for pid in ids],
                             dtype=torch.float)
    root_mask = torch.tensor([pid == root.post_id for pid in ids], dtype=torch.bool)

    reply_edges = [(id_to_idx[p.parent_id], id_to_idx[p.post_id])
                   for p in posts if p.parent_id is not None]
    reply_edge_index = (torch.tensor(reply_edges, dtype=torch.long).t()
                         if reply_edges else torch.zeros((2, 0), dtype=torch.long))
    dup_edge_index = (torch.tensor([(id_to_idx[s], id_to_idx[d]) for s, d in dup_edges],
                                    dtype=torch.long).t()
                       if dup_edges else torch.zeros((2, 0), dtype=torch.long))

    duplication_sets_by_idx = [
        [id_to_idx[d] for d in duplication_sets[pid]] for pid in ids
    ]

    return DuplicationGraph(
        x=x,
        reply_edge_index=reply_edge_index,
        duplication_edge_index=dup_edge_index,
        timestamps=timestamps,
        propagation_depth=prop_depth,
        duplication_freq=dup_freq,
        root_mask=root_mask,
        duplication_sets=duplication_sets_by_idx,
    )


def build_static_graph(posts: List[RawPost]) -> DuplicationGraph:
    """
    The 'traditional static graph' baseline for Lens 1 / Ablation 1
    (RQ1 plan, Section 4 Step 2 and Step 5): the same posts and the
    same reply tree, but with NO duplication tracking at all --
    duplication_edge_index is empty and duplication_freq is zero for
    every node, exactly matching what a standard GCN/Vanilla-GCN input
    graph looks like.

    This is deliberately NOT just "call build_duplication_graph then
    zero out the duplication fields" -- it independently reconstructs
    the graph from posts so that Lens 1's structural-fidelity check
    (eval/structural_fidelity.py) is comparing two genuinely separate
    construction pipelines, the way your plan's Step 2 describes:
    "Generate standard static graphs (no node duplication)" as its own
    pipeline step alongside "Construct ARAGCL-DP graphs".

    Use this (rather than build_duplication_graph with
    encoder use_duplication=False) for Ablation 1 specifically --
    Ablation 1 removes duplication at the DATA/graph-construction
    level, so it also removes the duplication-aware centrality signal
    used by RQ2's augmentation. Ablation 2 (removing only the
    temporal/duplication EDGES from the encoder's aggregation while
    keeping the duplication_freq/propagation_depth metadata intact for
    centrality scoring) is achieved instead via
    ARAGCL_DP_Config(use_duplication=False) on the full
    build_duplication_graph() output -- see models/aragcl_dp.py.
    """
    posts_by_id = {p.post_id: p for p in posts}
    root = next(p for p in posts if p.parent_id is None)

    children = {p.post_id: [] for p in posts}
    for p in posts:
        if p.parent_id is not None:
            children[p.parent_id].append(p.post_id)

    depth = {root.post_id: 0}
    frontier = [root.post_id]
    while frontier:
        nxt = []
        for pid in frontier:
            for c in children[pid]:
                depth[c] = depth[pid] + 1
                nxt.append(c)
        frontier = nxt

    ids = [p.post_id for p in posts]
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    x = torch.stack([posts_by_id[pid].feature for pid in ids])
    timestamps = torch.tensor([posts_by_id[pid].timestamp for pid in ids], dtype=torch.float)
    prop_depth = torch.tensor([depth.get(pid, 0) for pid in ids], dtype=torch.float)
    root_mask = torch.tensor([pid == root.post_id for pid in ids], dtype=torch.bool)

    reply_edges = [(id_to_idx[p.parent_id], id_to_idx[p.post_id])
                   for p in posts if p.parent_id is not None]
    reply_edge_index = (torch.tensor(reply_edges, dtype=torch.long).t()
                         if reply_edges else torch.zeros((2, 0), dtype=torch.long))

    return DuplicationGraph(
        x=x,
        reply_edge_index=reply_edge_index,
        duplication_edge_index=torch.zeros((2, 0), dtype=torch.long),
        timestamps=timestamps,
        propagation_depth=prop_depth,
        duplication_freq=torch.zeros(len(ids), dtype=torch.float),
        root_mask=root_mask,
        duplication_sets=[[] for _ in ids],
    )
