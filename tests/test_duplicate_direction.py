"""Explicit duplicate message directions on artificial graphs.

These checks establish message recipients and masking behaviour, not a
performance advantage. Stored graph edges continue to describe reference to
copy links; changing the encoder option must not mutate that representation.
"""
import unittest

import torch

from data.duplication_graph import DuplicationGraph
from models.aragcl_dp import ARAGCL_DP, ARAGCL_DP_Config
from models.encoders import DuplicationAwareEncoder


FORWARD = "reference_to_copy"
REVERSE = "copy_to_reference"


def fixture():
    x = torch.arange(1, 22, dtype=torch.float32).reshape(7, 3) / 10
    reply = torch.tensor([[0, 1, 0, 3, 4], [1, 2, 2, 4, 5]])
    duplicate = torch.tensor([[0, 1, 3, 4], [2, 2, 5, 5]])
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    return x, reply, duplicate, batch


def make_encoder(backbone, **kwargs):
    torch.manual_seed(419)
    return DuplicationAwareEncoder(
        in_dim=3, hidden_dim=4, num_layers=2, backbone=backbone,
        dropout=0, **kwargs).eval()


class DuplicateDirectionTests(unittest.TestCase):
    def test_hand_calculated_duplicate_messages_reach_intended_nodes(self):
        x = torch.tensor([[10.], [2.], [3.], [5.]])
        reply = torch.empty((2, 0), dtype=torch.long)
        duplicate = torch.tensor([[1, 1], [2, 3]])
        expected = {
            FORWARD: torch.tensor([[10.], [2.], [4.], [6.]]),
            REVERSE: torch.tensor([[10.], [6.], [3.], [5.]]),
        }
        for direction, target in expected.items():
            with self.subTest(direction=direction):
                encoder = DuplicationAwareEncoder(
                    1, 1, num_layers=1, lam=.5, dropout=0,
                    duplicate_message_direction=direction).eval()
                with torch.no_grad():
                    encoder.convs[0].lin.weight.fill_(1)
                    encoder.convs[0].lin.bias.zero_()
                nodes, graph = encoder(x, reply, duplicate)
                torch.testing.assert_close(nodes, target, rtol=0, atol=0)
                torch.testing.assert_close(graph, target.mean(0, keepdim=True),
                                           rtol=0, atol=0)

    def test_legacy_default_equals_explicit_reference_to_copy(self):
        data = fixture()
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                legacy = make_encoder(backbone)
                explicit = make_encoder(backbone,
                                        duplicate_message_direction=FORWARD)
                explicit.load_state_dict(legacy.state_dict())
                for expected, actual in zip(legacy(*data), explicit(*data)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reverse_mode_equals_flipping_only_duplicate_edges(self):
        x, reply, duplicate, batch = fixture()
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                forward = make_encoder(backbone,
                                       duplicate_message_direction=FORWARD)
                reverse = make_encoder(backbone,
                                       duplicate_message_direction=REVERSE)
                reverse.load_state_dict(forward.state_dict())
                expected = forward(x, reply, duplicate.flip(0), batch)
                actual = reverse(x, reply, duplicate, batch)
                for target, result in zip(expected, actual):
                    torch.testing.assert_close(result, target, rtol=0, atol=0)

    def test_encoding_does_not_mutate_stored_graph_or_mask(self):
        for direction in (FORWARD, REVERSE):
            for backbone in ("gcn", "gat"):
                with self.subTest(direction=direction, backbone=backbone):
                    x, reply, duplicate, batch = fixture()
                    keep = torch.tensor([True, False, True, True, False, True, True])
                    values = (x, reply, duplicate, batch, keep)
                    originals = [value.clone() for value in values]
                    make_encoder(backbone, duplicate_message_direction=direction)(
                        x, reply, duplicate, batch=batch, keep_node=keep)
                    for actual, expected in zip(values, originals):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_direction_has_no_effect_without_duplicate_contribution(self):
        x, reply, duplicate, batch = fixture()
        for backbone in ("gcn", "gat"):
            for case in ("empty_edges", "disabled", "zero_lambda"):
                with self.subTest(backbone=backbone, case=case):
                    kwargs = {"use_duplication": False} if case == "disabled" else {}
                    if case == "zero_lambda":
                        kwargs["lam"] = 0
                    edges = duplicate[:, :0] if case == "empty_edges" else duplicate
                    forward = make_encoder(backbone,
                                           duplicate_message_direction=FORWARD, **kwargs)
                    reverse = make_encoder(backbone,
                                           duplicate_message_direction=REVERSE, **kwargs)
                    reverse.load_state_dict(forward.state_dict())
                    for expected, actual in zip(forward(x, reply, edges, batch),
                                                reverse(x, reply, edges, batch)):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_mode_is_rejected_even_when_duplication_is_disabled(self):
        for use_duplication in (True, False):
            with self.subTest(use_duplication=use_duplication):
                with self.assertRaises(ValueError):
                    DuplicationAwareEncoder(
                        3, use_duplication=use_duplication,
                        duplicate_message_direction="backwards")
                with self.assertRaises(ValueError):
                    ARAGCL_DP(ARAGCL_DP_Config(
                        in_dim=3, use_duplication=use_duplication,
                        duplicate_message_direction="backwards"))

    def test_reverse_masking_matches_surviving_graph_and_preserves_batch_rows(self):
        x, reply, duplicate, batch = fixture()
        # Keep nodes 0 and 2 in graph 0 and singleton 6 in graph 2.
        # The middle graph is empty but must keep its position in pooled output.
        keep = torch.tensor([True, False, True, False, False, False, True])
        compact_reply = torch.tensor([[0], [1]])
        compact_duplicate = torch.tensor([[0], [1]])
        compact_batch = torch.tensor([0, 0, 2])
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                encoder = make_encoder(backbone, duplicate_message_direction=REVERSE)
                original = x.clone().requires_grad_()
                actual_nodes, actual_graphs = encoder(
                    original, reply, duplicate, batch, keep_node=keep)
                expected_nodes, expected_graphs = encoder(
                    x[keep], compact_reply, compact_duplicate, compact_batch)
                torch.testing.assert_close(actual_nodes[keep], expected_nodes)
                torch.testing.assert_close(actual_graphs, expected_graphs)
                self.assertEqual(actual_nodes[~keep].count_nonzero().item(), 0)
                self.assertEqual(actual_graphs.shape, (3, 4))
                self.assertEqual(actual_graphs[1].count_nonzero().item(), 0)
                torch.testing.assert_close(actual_graphs[2], actual_nodes[6])
                actual_graphs.square().sum().backward()
                self.assertEqual(original.grad[~keep].count_nonzero().item(), 0)
                self.assertTrue(torch.isfinite(original.grad).all().item())

    def test_model_config_selects_reverse_encoder_for_classification(self):
        x, reply, duplicate, batch = fixture()
        graph = DuplicationGraph(
            x=x, reply_edge_index=reply, duplication_edge_index=duplicate,
            timestamps=torch.arange(7, dtype=torch.float32),
            propagation_depth=torch.tensor([0., 1., 2., 0., 1., 2., 0.]),
            duplication_freq=torch.tensor([1., 1., 0., 1., 1., 0., 0.]),
            root_mask=torch.tensor([True, False, False, True, False, False, True]),
        )
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                model = ARAGCL_DP(ARAGCL_DP_Config(
                    in_dim=3, hidden_dim=4, dropout=0, backbone=backbone,
                    duplicate_message_direction=REVERSE)).eval()
                reference = make_encoder(backbone,
                                         duplicate_message_direction=FORWARD)
                reference.load_state_dict(model.encoder.state_dict())
                _, expected_graphs = reference(x, reply, duplicate.flip(0), batch)
                expected_logits = model.classifier(expected_graphs)
                torch.testing.assert_close(model.classify(graph, batch), expected_logits,
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
