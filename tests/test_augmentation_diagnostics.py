"""Actual augmentation counts must be accurate and observational only."""
import copy
import json
import unittest

import torch

from augmentation.diagnostics import AugmentationRecorder
from augmentation.views import AugmentedView, ViewConfig, generate_view
from data.duplication_graph import DuplicationGraph
from models.aragcl_dp import ARAGCL_DP, ARAGCL_DP_Config


def fixture():
    return DuplicationGraph(
        # Two originally-zero rows: neither zero row implies a sampled mask.
        x=torch.tensor([[0., 0.], [0., 0.], [2., 2.], [3., 3.]]),
        reply_edge_index=torch.tensor([[0, 0, 2], [1, 2, 3]]),
        duplication_edge_index=torch.tensor([[1, 1], [2, 3]]),
        timestamps=torch.arange(4, dtype=torch.float32),
        propagation_depth=torch.tensor([0., 1., 1., 2.]),
        duplication_freq=torch.tensor([0., 2., 0., 0.]),
        root_mask=torch.tensor([True, False, False, False]),
    )


class AugmentationRecorderTests(unittest.TestCase):
    def test_exact_counts_distinguish_mask_draws_from_deleted_and_zero_rows(self):
        graph = fixture()
        first = AugmentedView(
            x=torch.zeros_like(graph.x),
            reply_edge_index=torch.tensor([[0], [1]]),
            timestamps=None,
            duplication_edge_index=torch.tensor([[1], [2]]),
            keep_node=torch.tensor([True, True, True, False]),
            attribute_mask=torch.tensor([False, False, True, True]),
        )
        second = generate_view(
            graph.x, graph.reply_edge_index,
            ViewConfig(strategy="random", p_n=1, p_e=0, p_m=1),
            root_mask=graph.root_mask,
            duplication_edge_index=graph.duplication_edge_index,
        )
        torch.testing.assert_close(second.keep_node, graph.root_mask)
        torch.testing.assert_close(second.attribute_mask, ~graph.root_mask)
        recorder = AugmentationRecorder()
        recorder(graph, first, second)
        report = recorder.to_dict()
        expected = {
            "view_count": 2, "original_nodes": 8, "original_roots": 2,
            "original_nonroots": 6, "surviving_nodes": 4, "surviving_roots": 2,
            "surviving_nonroots": 2, "dropped_nodes": 4, "dropped_roots": 0,
            "dropped_nonroots": 4, "views_with_attribute_mask": 2,
            "attribute_observed_original_nodes": 8, "attribute_observed_original_roots": 2,
            "attribute_observed_original_nonroots": 6, "attribute_observed_surviving_nodes": 4,
            "attribute_observed_surviving_nonroots": 2, "attribute_masks_sampled": 5,
            "attribute_masks_sampled_roots": 0, "attribute_masks_sampled_nonroots": 5,
            "surviving_masked_nodes": 1, "surviving_masked_nonroots": 1,
            "reply_edges_before": 6, "reply_edges_after": 1, "reply_edges_removed": 5,
            "duplication_edges_before": 4, "duplication_edges_after": 1,
            "duplication_edges_removed": 3,
        }
        self.assertEqual(report["counts"], expected)
        self.assertEqual(report["rates"]["node_drop_nonroots"]["value"], 4 / 6)
        self.assertEqual(report["rates"]["attribute_mask_sampled_all_nodes"]["value"], 5 / 8)
        self.assertEqual(report["rates"]["attribute_mask_surviving_all_nodes"]["value"], 1 / 4)
        self.assertEqual(report["rates"]["attribute_mask_surviving_nonroots"]["value"], 1 / 2)
        self.assertEqual(report["rates"]["node_drop_roots"]["value"], 0)
        self.assertEqual(report["rates"]["attribute_mask_sampled_roots"]["value"], 0)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertTrue(all(isinstance(value, int) for value in recorder.counts.values()))
        # Exporting does not give callers a mutable reference to the counters.
        report["counts"]["view_count"] = 100
        self.assertEqual(recorder.to_dict()["counts"]["view_count"], 2)

    def test_empty_rates_and_legacy_views_are_explicitly_unobserved(self):
        recorder = AugmentationRecorder()
        self.assertTrue(all(rate["value"] is None for rate in recorder.to_dict()["rates"].values()))
        graph = fixture()
        graph.duplication_edge_index = torch.empty((2, 0), dtype=torch.long)
        legacy = AugmentedView(graph.x, graph.reply_edge_index, None,
                               graph.duplication_edge_index, torch.ones(4, dtype=torch.bool))
        x, reply, timestamps = legacy
        self.assertIs(x, graph.x)
        self.assertIs(reply, graph.reply_edge_index)
        self.assertIsNone(timestamps)
        self.assertIsNone(legacy.attribute_mask)
        recorder.record_view(graph, legacy)
        report = recorder.to_dict()
        self.assertEqual(report["counts"]["view_count"], 1)
        self.assertEqual(report["counts"]["views_with_attribute_mask"], 0)
        self.assertEqual(report["rates"]["node_drop_all_nodes"]["value"], 0)
        self.assertIsNone(report["rates"]["attribute_mask_sampled_all_nodes"]["value"])
        self.assertIsNone(report["rates"]["duplication_edge_removal"]["value"])

    def test_model_observer_preserves_outputs_gradients_rng_and_state_keys(self):
        graph = fixture()
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                torch.manual_seed(71)
                baseline = ARAGCL_DP(ARAGCL_DP_Config(
                    in_dim=2, hidden_dim=4, backbone=backbone, dropout=.4,
                    duplicate_message_direction="copy_to_reference",
                    view_strategy="random", p_n=.4, p_e=.3, p_m=.5,
                    epsilon=.1,
                )).train()
                observed = copy.deepcopy(baseline)
                recorder = AugmentationRecorder()
                observed.augmentation_observer = recorder
                self.assertEqual(list(baseline.state_dict()), list(observed.state_dict()))
                results = []
                for model in (baseline, observed):
                    torch.manual_seed(125)
                    outputs = model.contrastive_forward(graph)
                    sum(output.square().sum() for output in outputs).backward()
                    results.append((outputs, torch.get_rng_state().clone(),
                                    [p.grad.clone() if p.grad is not None else None
                                     for p in model.parameters()]))
                for expected, actual in zip(results[0][0], results[1][0]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(results[0][1], results[1][1]))
                for expected, actual in zip(results[0][2], results[1][2]):
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(recorder.to_dict()["counts"]["view_count"], 2)
                self.assertEqual(recorder.to_dict()["counts"]["original_nodes"], 8)
                # Supervised clean inference does not add augmentation counts.
                observed.classify(graph)
                self.assertEqual(recorder.to_dict()["counts"]["view_count"], 2)


if __name__ == "__main__":
    unittest.main()
