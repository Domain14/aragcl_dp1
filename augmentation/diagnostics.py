"""Count actual augmentation decisions without changing stochastic draws.

Attach an ``AugmentationRecorder`` to ``model.augmentation_observer`` during
pretraining. It retains only integer counters, never graphs or tensors. Counts
are exposures across views/batches/epochs, not counts of unique dataset posts.
"""
import hashlib
import json
import sys

import torch


class AugmentationRecorder:
    """Callable observer of ``(original_graph, view1, view2)``.

    ``to_dict()`` is JSON serializable. Each rate identifies its counted
    numerator and denominator; a zero denominator produces ``None``. Legacy
    hand-built views without ``attribute_mask`` still contribute structural
    counts, but are excluded from attribute-mask rate denominators.
    """

    def __init__(self):
        self.counts = dict.fromkeys((
            "view_count", "original_nodes", "original_roots", "original_nonroots",
            "surviving_nodes", "surviving_roots", "surviving_nonroots",
            "dropped_nodes", "dropped_roots", "dropped_nonroots",
            "views_with_attribute_mask", "attribute_observed_original_nodes",
            "attribute_observed_original_roots", "attribute_observed_original_nonroots",
            "attribute_observed_surviving_nodes", "attribute_observed_surviving_nonroots",
            "attribute_masks_sampled", "attribute_masks_sampled_roots",
            "attribute_masks_sampled_nonroots", "surviving_masked_nodes",
            "surviving_masked_nonroots", "reply_edges_before", "reply_edges_after",
            "reply_edges_removed", "duplication_edges_before", "duplication_edges_after",
            "duplication_edges_removed",
        ), 0)

    def __call__(self, graph, view1, view2):
        self.record_view(graph, view1)
        self.record_view(graph, view2)

    def record_view(self, graph, view):
        keep = view.keep_node
        root = graph.root_mask
        num_nodes = int(graph.x.size(0))
        if keep.shape != (num_nodes,) or root.shape != (num_nodes,):
            raise ValueError("Augmentation diagnostics require node-aligned masks")
        mask = view.attribute_mask
        if mask is not None and mask.shape != (num_nodes,):
            raise ValueError("attribute_mask must contain one value per original node")

        count = self.counts
        count["view_count"] += 1
        count["original_nodes"] += num_nodes
        count["original_roots"] += int(root.sum().item())
        count["original_nonroots"] += int((~root).sum().item())
        count["surviving_nodes"] += int(keep.sum().item())
        count["surviving_roots"] += int((keep & root).sum().item())
        count["surviving_nonroots"] += int((keep & ~root).sum().item())
        count["dropped_nodes"] += int((~keep).sum().item())
        count["dropped_roots"] += int((~keep & root).sum().item())
        count["dropped_nonroots"] += int((~keep & ~root).sum().item())

        if mask is not None:
            count["views_with_attribute_mask"] += 1
            count["attribute_observed_original_nodes"] += num_nodes
            count["attribute_observed_original_roots"] += int(root.sum().item())
            count["attribute_observed_original_nonroots"] += int((~root).sum().item())
            count["attribute_observed_surviving_nodes"] += int(keep.sum().item())
            count["attribute_observed_surviving_nonroots"] += int((keep & ~root).sum().item())
            count["attribute_masks_sampled"] += int(mask.sum().item())
            count["attribute_masks_sampled_roots"] += int((mask & root).sum().item())
            count["attribute_masks_sampled_nonroots"] += int((mask & ~root).sum().item())
            count["surviving_masked_nodes"] += int((mask & keep).sum().item())
            count["surviving_masked_nonroots"] += int((mask & keep & ~root).sum().item())

        for relation in ("reply", "duplication"):
            before = int(getattr(graph, f"{relation}_edge_index").size(1))
            after = int(getattr(view, f"{relation}_edge_index").size(1))
            count[f"{relation}_edges_before"] += before
            count[f"{relation}_edges_after"] += after
            count[f"{relation}_edges_removed"] += before - after

    def to_dict(self):
        def rate(numerator, denominator):
            n, d = self.counts[numerator], self.counts[denominator]
            return {"numerator_count": numerator, "denominator_count": denominator,
                    "numerator": n, "denominator": d,
                    "value": n / d if d else None}

        return {
            "schema_version": 1,
            "count_units": "Pooled node/edge exposures across observed views, batches and epochs; not unique posts.",
            "attribute_mask_definition": "Actual sampled mask decisions after root protection, including decisions on dropped nodes; surviving_masked counts exclude dropped nodes. Zero input features alone do not imply masking.",
            "edge_removal_definition": "Combined removal from node deletion and relation-specific edge dropout, relative to each original graph relation.",
            "counts": dict(self.counts),
            "rates": {
                "node_drop_all_nodes": rate("dropped_nodes", "original_nodes"),
                "node_drop_nonroots": rate("dropped_nonroots", "original_nonroots"),
                "node_drop_roots": rate("dropped_roots", "original_roots"),
                "attribute_mask_sampled_all_nodes": rate("attribute_masks_sampled", "attribute_observed_original_nodes"),
                "attribute_mask_sampled_nonroots": rate("attribute_masks_sampled_nonroots", "attribute_observed_original_nonroots"),
                "attribute_mask_sampled_roots": rate("attribute_masks_sampled_roots", "attribute_observed_original_roots"),
                "attribute_mask_surviving_all_nodes": rate("surviving_masked_nodes", "attribute_observed_surviving_nodes"),
                "attribute_mask_surviving_nonroots": rate("surviving_masked_nonroots", "attribute_observed_surviving_nonroots"),
                "reply_edge_removal": rate("reply_edges_removed", "reply_edges_before"),
                "duplication_edge_removal": rate("duplication_edges_removed", "duplication_edges_before"),
            },
        }


class AugmentationTraceRecorder(AugmentationRecorder):
    """Record ordered fingerprints of exact views, alongside usage counts.

    The inherited callable observes view 1 then view 2 for every batch. Hashes
    include each field's name, dtype, shape and actual bytes (or an explicit
    None marker). Storage layout and device do not enter the fingerprint.
    Canonical payloads use little-endian bytes and length-prefixed metadata.
    Only hexadecimal hashes and integer counters are retained; this observer
    does not keep tensors, alter inputs or consume any random draws.
    """

    _FIELDS = ("x", "reply_edge_index", "duplication_edge_index", "keep_node",
               "attribute_mask", "timestamps")

    def __init__(self):
        super().__init__()
        self._ordered_view_sha256 = []

    @classmethod
    def _view_sha256(cls, view):
        digest = hashlib.sha256(b"aragcl-augmentation-view-v1\0")
        for name in cls._FIELDS:
            value = getattr(view, name)
            if value is None:
                metadata = {"name": name, "kind": "none"}
                payload = b""
            else:
                value = value.detach().cpu().contiguous()
                metadata = {"name": name, "kind": "tensor", "dtype": str(value.dtype),
                            "shape": list(value.shape), "byte_order": "little"}
                # Viewing as bytes also supports dtypes such as bfloat16 that
                # cannot be converted directly to a NumPy scalar dtype.
                payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
                if sys.byteorder != "little":
                    width = value.element_size() // (2 if value.is_complex() else 1)
                    payload = b"".join(payload[i:i + width][::-1]
                                       for i in range(0, len(payload), width))
            header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    def record_view(self, graph, view):
        fingerprint = self._view_sha256(view)
        super().record_view(graph, view)
        self._ordered_view_sha256.append(fingerprint)

    def trace_dict(self):
        """Return an independent JSON-serializable copy of the ordered trace."""
        ordered = list(self._ordered_view_sha256)
        digest = hashlib.sha256(b"aragcl-augmentation-sequence-v1\0")
        for fingerprint in ordered:
            # SHA-256 digests have a fixed width, making order unambiguous.
            digest.update(bytes.fromhex(fingerprint))
        return {"schema_version": 1, "view_count": len(ordered),
                "ordered_view_sha256": ordered, "sequence_sha256": digest.hexdigest()}
