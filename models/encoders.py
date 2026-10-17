"""
Duplication-aware graph encoder (Section 3.5.5, Eq 3.7):

    h_v^(l+1) = sigma( W^(l) * ( sum_{u in N(v)} h_u^(l)
                                 + lambda * sum_{d in D_v} h_d^(l) ) )

N(v)  = incoming reply neighbors, plus a self-loop
D_v   = duplicate-message neighbors under the selected message direction
lambda = influence weight of duplicated nodes (self.lam)

The graph stores earliest-reference -> later-copy links. The legacy
reference_to_copy mode follows these links; copy_to_reference aggregates
the builder's stored duplication_sets[v] into v. This resolves the
direction explicitly without claiming either reading is uniquely specified
by the proposal. Reversing a star does not create all-pairs duplication sets.
The GCN sums both message flows before its linear transform; the GAT is a
separate attention-based variant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.utils import add_self_loops


DUPLICATE_MESSAGE_DIRECTIONS = ("reference_to_copy", "copy_to_reference")


class DuplicationAwareConv(MessagePassing):
    """GCN-flavored version of Eq 3.7 (sum aggregation)."""

    def __init__(self, in_dim: int, out_dim: int, lam: float = 0.5):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_dim, out_dim)
        self.lam = lam

    def forward(self, x, reply_edge_index, duplication_edge_index):
        n = x.size(0)
        reply_edge_index, _ = add_self_loops(reply_edge_index, num_nodes=n)

        neighbor_msg = self.propagate(reply_edge_index, x=x)
        if duplication_edge_index.numel() > 0:
            dup_msg = self.propagate(duplication_edge_index, x=x)
        else:
            dup_msg = torch.zeros_like(neighbor_msg)

        out = neighbor_msg + self.lam * dup_msg
        return F.relu(self.lin(out))

    def message(self, x_j):
        return x_j


class DuplicationAwareGATConv(nn.Module):
    """
    GAT-flavored version of Eq 3.7: your RQ1 plan names 'Standard GCN
    or GAT architectures modified with a duplication influence
    hyperparameter (lambda)' as the required backbone options, so both
    need to exist, not just GCN. Runs attention separately over the
    reply-neighborhood and the duplication-set, then combines with the
    same additive lambda-weighted form as the GCN version (rather than
    attention's usual softmax-normalized combination), so that lambda
    keeps the same interpretation across both backbones and the RQ2
    lambda-sensitivity ablation doesn't need separate tuning per
    backbone.
    """

    def __init__(self, in_dim: int, out_dim: int, lam: float = 0.5, heads: int = 2):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.reply_gat = GATConv(in_dim, out_dim // heads, heads=heads, add_self_loops=True)
        self.dup_gat = GATConv(in_dim, out_dim // heads, heads=heads, add_self_loops=False)
        self.lam = lam

    def forward(self, x, reply_edge_index, duplication_edge_index):
        neighbor_msg = self.reply_gat(x, reply_edge_index)
        if duplication_edge_index.numel() > 0:
            dup_msg = self.dup_gat(x, duplication_edge_index)
        else:
            dup_msg = torch.zeros_like(neighbor_msg)
        return F.elu(neighbor_msg + self.lam * dup_msg)


class DuplicationAwareEncoder(nn.Module):
    """Stack of DuplicationAwareConv layers + graph-level readout.
    This is the shared backbone for RAGCL, ARAGCL-DP and all the
    contrastive baselines (GraphCL/GRACE/GACL) -- per your proposal's
    own consistency requirement, only the augmentation strategy that
    feeds this encoder should differ between experiments.

    dropout: applied after every layer's activation, training-only
    (inactive at eval time automatically via nn.Module's .training
    flag). Added because early real-data runs showed heavy overfitting
    -- train loss collapsing toward 0 while val loss climbed 2-3x from
    its minimum within a handful of epochs. Default 0.3 is a
    reasonable starting point for a small dataset; treat it as a
    hyperparameter to tune, not a fixed constant."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 lam: float = 0.5, use_duplication: bool = True,
                 backbone: str = "gcn", heads: int = 2, dropout: float = 0.3,
                 duplicate_message_direction: str = "reference_to_copy"):
        super().__init__()
        if duplicate_message_direction not in DUPLICATE_MESSAGE_DIRECTIONS:
            raise ValueError("duplicate_message_direction must be one of "
                             f"{DUPLICATE_MESSAGE_DIRECTIONS}, got {duplicate_message_direction!r}")
        self.duplicate_message_direction = duplicate_message_direction
        self.use_duplication = use_duplication
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * num_layers
        self.output_dim = dims[-1]
        if backbone == "gcn":
            self.convs = nn.ModuleList([
                DuplicationAwareConv(dims[i], dims[i + 1], lam=lam)
                for i in range(num_layers)
            ])
        elif backbone == "gat":
            self.convs = nn.ModuleList([
                DuplicationAwareGATConv(dims[i], dims[i + 1], lam=lam, heads=heads)
                for i in range(num_layers)
            ])
        else:
            raise ValueError(f"backbone must be 'gcn' or 'gat', got {backbone!r}")

    def forward(self, x, reply_edge_index, duplication_edge_index, batch=None,
                keep_node=None):
        """Encode surviving nodes and pool them in the original graph order.

        ``keep_node`` represents node deletion, whereas a zero feature vector
        alone represents attribute masking. Compaction before message passing
        keeps dropped nodes out of both relations, self-loops, layer biases,
        and graph pooling. Returned node embeddings retain original indices,
        with zero rows for deleted nodes. A graph with no retained nodes gets
        a zero graph embedding rather than disappearing from the batch.
        """
        num_nodes = x.size(0)
        num_graphs = 1 if batch is None else (
            int(batch.max().item()) + 1 if batch.numel() else 0)
        if keep_node is not None:
            keep_node = keep_node.to(device=x.device, dtype=torch.bool)
            if keep_node.shape != (num_nodes,):
                raise ValueError("keep_node must contain one value per node")
            node_ids = keep_node.nonzero(as_tuple=False).flatten()
            old_to_new = torch.full((num_nodes,), -1, dtype=torch.long,
                                     device=x.device)
            old_to_new[node_ids] = torch.arange(node_ids.numel(), device=x.device)

            def compact_edges(edges):
                surviving_edges = keep_node[edges[0]] & keep_node[edges[1]]
                return old_to_new[edges[:, surviving_edges]]

            reply_edge_index = compact_edges(reply_edge_index)
            duplication_edge_index = compact_edges(duplication_edge_index)
            x = x[keep_node]
            if batch is not None:
                batch = batch[keep_node]

        empty = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        dup_ei = duplication_edge_index if self.use_duplication else empty
        if self.duplicate_message_direction == "copy_to_reference":
            # Only encoder messages reverse. Stored links and augmentation
            # metadata retain their temporal reference -> copy convention.
            dup_ei = dup_ei.flip(0)
        h = x
        if h.size(0):
            for i, conv in enumerate(self.convs):
                h = conv(h, reply_edge_index, dup_ei)
                if i < len(self.convs) - 1:  # no dropout after the final layer
                    h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            # Keep autograd connected even when a rootless graph loses all its
            # nodes. Empty tensors must not enter GAT or a mean reduction.
            h = x.new_zeros((0, self.output_dim)) + x.sum() * 0.0

        if batch is None:
            graph_repr = (h.mean(dim=0, keepdim=True) if h.size(0)
                          else h.sum(dim=0, keepdim=True))
        else:
            from torch_geometric.nn import global_mean_pool
            graph_repr = global_mean_pool(h, batch, size=num_graphs)
        if keep_node is not None:
            h = h.new_zeros((num_nodes, self.output_dim)).index_copy(0, node_ids, h)
        return h, graph_repr
