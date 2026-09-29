"""Exact-view traces distinguish changes hidden by aggregate usage counts."""
import copy
import json
import unittest

import torch

from augmentation.diagnostics import AugmentationRecorder, AugmentationTraceRecorder
from augmentation.views import AugmentedView
from data.duplication_graph import DuplicationGraph
from models.aragcl_dp import ARAGCL_DP, ARAGCL_DP_Config


def fixture():
    graph = DuplicationGraph(
        x=torch.tensor([[1., 2.], [0., 0.], [0., 0.], [3., 4.]]),
        reply_edge_index=torch.tensor([[0, 0, 2], [1, 2, 3]]),
        duplication_edge_index=torch.tensor([[1, 1], [2, 3]]),
        timestamps=torch.arange(4, dtype=torch.float32),
        propagation_depth=torch.tensor([0., 1., 1., 2.]),
        duplication_freq=torch.tensor([0., 2., 0., 0.]),
        root_mask=torch.tensor([True, False, False, False]),
    )
    view = AugmentedView(
        x=graph.x.clone(), reply_edge_index=graph.reply_edge_index.clone(),
        duplication_edge_index=graph.duplication_edge_index.clone(),
        timestamps=graph.timestamps.clone(),
        keep_node=torch.ones(4, dtype=torch.bool),
        attribute_mask=torch.tensor([False, True, False, False]),
    )
    return graph, view


def one_view_trace(graph, view):
    recorder = AugmentationTraceRecorder()
    recorder.record_view(graph, view)
    return recorder


class AugmentationTraceTests(unittest.TestCase):
    def test_identical_values_have_identical_trace_without_tensor_retention(self):
        graph, view = fixture()
        first = one_view_trace(graph, view)
        other = copy.deepcopy(view)
        other.x = other.x.T.contiguous().T  # Same values, noncontiguous layout.
        self.assertFalse(other.x.is_contiguous())
        second = one_view_trace(graph, other)
        self.assertEqual(first.trace_dict(), second.trace_dict())
        self.assertEqual(first.to_dict(), second.to_dict())
        before = first.trace_dict()
        view.x[0, 0] = -123  # Previously observed values cannot change the trace.
        self.assertEqual(first.trace_dict(), before)
        before["ordered_view_sha256"].clear()
        self.assertEqual(first.trace_dict()["view_count"], 1)
        self.assertEqual(len(first.trace_dict()["ordered_view_sha256"]), 1)
        self.assertTrue(all(isinstance(item, str) for item in first._ordered_view_sha256))
        self.assertTrue(all(isinstance(item, int) for item in first.counts.values()))
        self.assertEqual(json.loads(json.dumps(first.trace_dict())), first.trace_dict())

    def test_mask_changes_on_zero_features_are_visible_despite_identical_counts(self):
        graph, view = fixture()
        other = copy.deepcopy(view)
        other.attribute_mask = torch.tensor([False, False, True, False])
        first = one_view_trace(graph, view)
        second = one_view_trace(graph, other)
        torch.testing.assert_close(view.x, other.x, rtol=0, atol=0)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(first.trace_dict()["ordered_view_sha256"],
                            second.trace_dict()["ordered_view_sha256"])

    def test_each_actual_field_and_metadata_affect_the_hash(self):
        graph, view = fixture()
        expected = one_view_trace(graph, view).trace_dict()["ordered_view_sha256"][0]
        changes = {
            "feature": lambda v: v.x.__setitem__((3, 1), 4.01),
            "reply_edge": lambda v: v.reply_edge_index.__setitem__((0, 0), 1),
            "duplicate_edge": lambda v: v.duplication_edge_index.__setitem__((0, 0), 0),
            "keep_node": lambda v: v.keep_node.__setitem__(1, False),
            "timestamp": lambda v: v.timestamps.__setitem__(0, .01),
            "timestamp_none": lambda v: setattr(v, "timestamps", None),
            "attribute_mask_none": lambda v: setattr(v, "attribute_mask", None),
            "dtype": lambda v: setattr(v, "x", v.x.to(torch.float64)),
            "shape": lambda v: setattr(v, "x", v.x.reshape(2, 4)),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                other = copy.deepcopy(view)
                change(other)
                actual = one_view_trace(graph, other).trace_dict()["ordered_view_sha256"][0]
                self.assertNotEqual(actual, expected)

    def test_two_view_order_and_repetitions_are_preserved(self):
        graph, view = fixture()
        other = copy.deepcopy(view)
        other.timestamps[0] += .1
        first, second = AugmentationTraceRecorder(), AugmentationTraceRecorder()
        first(graph, view, other)
        second(graph, other, view)
        a, b = first.trace_dict(), second.trace_dict()
        self.assertEqual(a["view_count"], 2)
        self.assertEqual(a["ordered_view_sha256"], b["ordered_view_sha256"][::-1])
        self.assertNotEqual(a["sequence_sha256"], b["sequence_sha256"])
        self.assertEqual(first.to_dict(), second.to_dict())
        first(graph, view, other)
        c = first.trace_dict()
        self.assertEqual(c["ordered_view_sha256"], a["ordered_view_sha256"] * 2)
        self.assertEqual(c["view_count"], first.to_dict()["counts"]["view_count"])
        self.assertNotEqual(c["sequence_sha256"], a["sequence_sha256"])

    def test_empty_optional_and_bfloat_fields_are_supported(self):
        graph, view = fixture()
        view.x = view.x.to(torch.bfloat16)
        view.duplication_edge_index = torch.empty((2, 0), dtype=torch.long)
        view.timestamps = None
        view.attribute_mask = None
        first = one_view_trace(graph, view)
        self.assertEqual(first.trace_dict(), one_view_trace(graph, copy.deepcopy(view)).trace_dict())
        empty = AugmentationTraceRecorder().trace_dict()
        self.assertEqual(empty["view_count"], 0)
        self.assertEqual(empty["ordered_view_sha256"], [])
        self.assertEqual(len(empty["sequence_sha256"]), 64)

    def test_observer_preserves_rng_outputs_gradients_and_usage(self):
        graph, _ = fixture()
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                torch.manual_seed(17)
                base = ARAGCL_DP(ARAGCL_DP_Config(
                    in_dim=2, hidden_dim=4, backbone=backbone, dropout=.4,
                    duplicate_message_direction="copy_to_reference",
                    view_strategy="random", p_n=.4, p_e=.3, p_m=.5, epsilon=.1,
                )).train()
                count_model, trace_model = copy.deepcopy(base), copy.deepcopy(base)
                counter, tracer = AugmentationRecorder(), AugmentationTraceRecorder()
                count_model.augmentation_observer = counter
                trace_model.augmentation_observer = tracer
                outputs = []
                for model in (base, count_model, trace_model):
                    self.assertEqual(list(base.state_dict()), list(model.state_dict()))
                    torch.manual_seed(57)
                    result = model.contrastive_forward(graph)
                    sum(value.square().sum() for value in result).backward()
                    outputs.append((result, torch.get_rng_state().clone(),
                                    [p.grad.clone() if p.grad is not None else None
                                     for p in model.parameters()]))
                for observed in outputs[1:]:
                    for expected, actual in zip(outputs[0][0], observed[0]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(torch.equal(outputs[0][1], observed[1]))
                    for expected, actual in zip(outputs[0][2], observed[2]):
                        if expected is None:
                            self.assertIsNone(actual)
                        else:
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(counter.to_dict(), tracer.to_dict())
                self.assertEqual(tracer.trace_dict()["view_count"], 2)


if __name__ == "__main__":
    unittest.main()
