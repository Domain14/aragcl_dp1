"""
Augmented view generation (Section 3.5.4 / Eq 3.2-3.4, 3.5, 3.6).

Given a graph and a per-node importance score (from
augmentation.centrality), produces one augmented view by:
  1. dropping nodes with probability p^n_v, removing their incident
     reply and duplication edges and recording the surviving nodes
  2. dropping edges with probability p^e_uv (derived from an edge
     weight -- here, the min of the two endpoint importance scores,
     a common surrogate when explicit edge weights aren't available)
  3. masking node attributes with probability p^m_v
  4. (if timestamps are supplied) applying temporal jitter, Eq 3.5:
         t_v' = t_v + delta,  delta ~ U(-epsilon, epsilon)

This single function is deliberately shared across every contrastive
model (RAGCL, ARAGCL-DP, and the GraphCL/GRACE/GACL baselines) --
per your proposal's own "Consistency" requirement (Section 3.6): the
same backbone and augmentation *mechanism* is used everywhere, and
only the *strategy* argument changes which centrality function feeds
it. This is what makes the RQ2 comparison a controlled experiment
rather than a confound between architecture and augmentation.
"""
from dataclasses import dataclass
from typing import Optional
import torch

from .centrality import (
    degree_centrality,
    pagerank_centrality,
    duplication_aware_centrality,
    importance_scores_to_drop_probs,
    DuplicationCentralityConfig,
    _validate_probability,
)

STRATEGIES = ("random", "degree", "pagerank", "duplication_aware", "ragcl", "dataset")


@dataclass
class AugmentedView:
    """An index-aligned view; ``keep_node`` defines its actual node set.

    Stable indices make node metadata and duplicate pairs reusable. Encoders
    must receive ``keep_node`` so deleted nodes cannot reappear via biases,
    self-loops, or pooling. Three-value unpacking remains available for older
    feature/edge inspection code, but loses the information needed to encode
    a node-dropped view correctly.
    """
    x: torch.Tensor
    reply_edge_index: torch.Tensor
    timestamps: Optional[torch.Tensor]
    duplication_edge_index: torch.Tensor
    keep_node: torch.Tensor
    # Store the sampled decision explicitly: an already-zero feature row
    # cannot tell us whether attribute masking was actually selected.
    # Appended with a default to preserve existing five-argument callers.
    attribute_mask: Optional[torch.Tensor] = None

    def __iter__(self):
        return iter((self.x, self.reply_edge_index, self.timestamps))

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return (self.x, self.reply_edge_index, self.timestamps)[index]


@dataclass
class ViewConfig:
    strategy: str = "duplication_aware"
    p_n: float = 0.2       # base node-drop rate
    p_e: float = 0.2       # base edge-drop rate
    p_m: float = 0.2       # base attribute-mask rate
    epsilon: float = 0.0   # temporal jitter range (0 = disabled)
    dup_cfg: DuplicationCentralityConfig = None
    centrality_key: str = "Degree"  # which metric to read for strategy="dataset"

    def __post_init__(self):
        for name in ("p_n", "p_e", "p_m"):
            _validate_probability(getattr(self, name), name)
        if self.dup_cfg is None:
            self.dup_cfg = DuplicationCentralityConfig(p_base=self.p_n)


def _importance_scores(x, edge_index, num_nodes, cfg: ViewConfig,
                        duplication_freq=None, propagation_depth=None,
                        precomputed_centrality=None):
    if cfg.strategy == "random":
        # GraphCL-style: uniform augmentation, no centrality guidance.
        return x.new_ones(num_nodes)
    if cfg.strategy == "degree":
        # GRACE/GCA-style baseline: degree-based importance.
        return degree_centrality(edge_index, num_nodes)
    if cfg.strategy == "pagerank":
        return pagerank_centrality(edge_index, num_nodes)
    if cfg.strategy == "dataset":
        # Uses centrality shipped WITH the dataset (e.g. the
        # Degree/Pagerank/Eigenvector/Betweenness arrays in the Weibo
        # JSON format -- see data/weibo_json_loader.py) instead of
        # recomputing it, so results match whatever graph the original
        # dataset authors used to compute it.
        if precomputed_centrality is None or cfg.centrality_key not in precomputed_centrality:
            raise ValueError(
                f"strategy='dataset' requires precomputed_centrality with key "
                f"'{cfg.centrality_key}' -- pass graph.precomputed_centrality "
                f"(see data/duplication_graph.py::DuplicationGraph) through."
            )
        return precomputed_centrality[cfg.centrality_key]
    if cfg.strategy in ("duplication_aware",):
        if duplication_freq is None or propagation_depth is None:
            raise ValueError(
                "duplication_aware strategy requires duplication_freq "
                "and propagation_depth tensors from the RQ1 graph "
                "construction step (data/duplication_graph.py)."
            )
        return duplication_aware_centrality(duplication_freq,
                                             propagation_depth, cfg.dup_cfg)
    if cfg.strategy == "ragcl":
        # Eq 3.2 base case: plain centrality (degree here as a stand-in
        # for phi_c) without the duplication extension -- this is your
        # RAGCL baseline, distinct from ARAGCL-DP.
        return degree_centrality(edge_index, num_nodes)
    raise ValueError(f"Unknown augmentation strategy: {cfg.strategy}")


def generate_view(x: torch.Tensor, edge_index: torch.Tensor,
                   cfg: ViewConfig,
                   duplication_freq: Optional[torch.Tensor] = None,
                   propagation_depth: Optional[torch.Tensor] = None,
                   root_mask: Optional[torch.Tensor] = None,
                   timestamps: Optional[torch.Tensor] = None,
                   precomputed_centrality: Optional[dict] = None,
                   duplication_edge_index: Optional[torch.Tensor] = None):
    """
    Returns an AugmentedView with both augmented edge relations and keep_node.
    Legacy ``x_aug, edge_index_aug, timestamps_aug = view`` unpacking is
    supported, but model callers must also use view.keep_node and the view's
    duplication_edge_index.

    root_mask: bool tensor, True for the root/source post of each
    cascade. RAGCL's first augmentation principle is "exempt root
    nodes" -- roots are never dropped or masked regardless of score.
    Root-incident edges remain eligible for edge dropout. Node indices remain
    stable; the encoder excludes dropped nodes from computation and pooling.
    Edge dropout is sampled independently for each relation using the same
    configured base rate and endpoint-importance rule.

    precomputed_centrality: pass graph.precomputed_centrality when
    cfg.strategy == "dataset" (see data/weibo_json_loader.py).
    """
    num_nodes = x.size(0)
    scores = _importance_scores(x, edge_index, num_nodes, cfg,
                                 duplication_freq, propagation_depth,
                                 precomputed_centrality)
    scores = scores.to(device=x.device, dtype=x.dtype)
    if scores.shape != (num_nodes,):
        raise ValueError("Importance scores must contain one value per node")

    if cfg.strategy == "random":
        # Uniform random augmentation uses its configured rates directly;
        # it must not depend on a centrality normalization convention.
        for name in ("p_n", "p_e", "p_m"):
            _validate_probability(getattr(cfg, name), name)
        node_drop_p = torch.full_like(scores, cfg.p_n)
        attr_mask_p = torch.full_like(scores, cfg.p_m)
    else:
        node_drop_p = importance_scores_to_drop_probs(scores, cfg.p_n)
        attr_mask_p = importance_scores_to_drop_probs(scores, cfg.p_m)

    if root_mask is not None:
        root_mask = root_mask.to(device=x.device, dtype=torch.bool)
        if root_mask.shape != (num_nodes,):
            raise ValueError("root_mask must contain one value per node")
        node_drop_p = node_drop_p.masked_fill(root_mask, 0.0)
        attr_mask_p = attr_mask_p.masked_fill(root_mask, 0.0)

    keep_node = torch.bernoulli(1.0 - node_drop_p).bool()

    def augment_edges(edges):
        # Min-endpoint importance is the edge-weight surrogate for Eq 3.3.
        src, dst = edges
        if cfg.strategy == "random":
            edge_drop_p = scores.new_full((src.numel(),), cfg.p_e)
        else:
            edge_weight = torch.minimum(scores[src], scores[dst])
            edge_drop_p = importance_scores_to_drop_probs(edge_weight, cfg.p_e)
        keep_edge = torch.bernoulli(1.0 - edge_drop_p).bool()
        keep_edge &= keep_node[src] & keep_node[dst]
        return edges[:, keep_edge]

    edge_index_aug = augment_edges(edge_index)

    x_aug = x.clone()
    mask = torch.bernoulli(attr_mask_p).bool()
    x_aug[mask] = 0.0
    # Preserve indices for metadata; keep_node removes these rows from the
    # encoder computation and readout, rather than relying on zero features.
    x_aug[~keep_node] = 0.0

    timestamps_aug = None
    if timestamps is not None and cfg.epsilon > 0:
        delta = (torch.rand_like(timestamps) * 2 - 1) * cfg.epsilon  # Eq 3.5
        timestamps_aug = timestamps + delta

    if duplication_edge_index is None:
        duplication_edge_index = edge_index.new_empty((2, 0))
    duplication_edge_index_aug = augment_edges(duplication_edge_index)

    return AugmentedView(x_aug, edge_index_aug, timestamps_aug,
                         duplication_edge_index_aug, keep_node, mask)


def generate_two_views(x, edge_index, cfg: ViewConfig, **kwargs):
    """Convenience wrapper: produces (view1, view2) for the contrastive
    objective (Eq 3.8), typically with the SAME strategy but independent
    stochastic draws."""
    v1 = generate_view(x, edge_index, cfg, **kwargs)
    v2 = generate_view(x, edge_index, cfg, **kwargs)
    return v1, v2
