"""Node deletion must be equivalent to encoding the induced surviving graph.

Fixtures are artificial and exercise both relation types, batching, roots,
layer biases, attention self-loops, pooling, and feature gradients.
"""
import unittest
from unittest.mock import patch

import torch

from augmentation.views import ViewConfig, generate_view, generate_two_views
from data.duplication_graph import DuplicationGraph
from models.aragcl_dp import ARAGCL_DP, ARAGCL_DP_Config
from models.encoders import DuplicationAwareEncoder


def fixture():
    # Three cascades, including a singleton. Nodes 1, 4 and 5 will be deleted.
    x = torch.arange(1, 22, dtype=torch.float32).reshape(7, 3) / 10
    reply = torch.tensor([[0, 1, 0, 3, 4], [1, 2, 2, 4, 5]])
    dup = torch.tensor([[1, 0, 4, 5], [2, 2, 3, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    root = torch.tensor([True, False, False, True, False, False, True])
    return x, reply, dup, batch, root


def make_graph():
    x, reply, dup, batch, root = fixture()
    graph = DuplicationGraph(
        x=x, reply_edge_index=reply, duplication_edge_index=dup,
        root_mask=root, timestamps=torch.arange(7, dtype=torch.float32),
        propagation_depth=torch.tensor([0., 1., 2., 0., 1., 2., 0.]),
        duplication_freq=torch.tensor([1., 1., 0., 0., 1., 1., 0.]),
    )
    return graph, batch


class AugmentedRelationTests(unittest.TestCase):
    def test_node_dropout_protects_each_root_and_filters_both_relations(self):
        x, reply, dup, batch, root = fixture()
        originals = [value.clone() for value in (x, reply, dup, root)]
        view = generate_view(
            x, reply, ViewConfig(strategy="random", p_n=1, p_e=0, p_m=1),
            root_mask=root, duplication_edge_index=dup)
        torch.testing.assert_close(view.keep_node, root)
        torch.testing.assert_close(view.x[root], x[root])
        self.assertEqual(view.reply_edge_index.shape, (2, 0))
        self.assertEqual(view.duplication_edge_index.shape, (2, 0))
        self.assertEqual(view.x[~root].count_nonzero().item(), 0)
        for original, actual in zip(originals, (x, reply, dup, root)):
            torch.testing.assert_close(original, actual, rtol=0, atol=0)
        # Existing inspection callers can still unpack three values.
        x_aug, edges_aug, times_aug = view
        self.assertIs(x_aug, view.x)
        self.assertIs(edges_aug, view.reply_edge_index)
        self.assertIsNone(times_aug)
        self.assertEqual(len(view), 3)
        self.assertIs(view[0], view.x)

    def test_edge_dropout_applies_to_duplicate_edges_even_between_roots(self):
        x, reply, dup, _, _ = fixture()
        view = generate_view(
            x, reply, ViewConfig(strategy="random", p_n=0, p_e=1, p_m=0),
            root_mask=torch.ones(len(x), dtype=torch.bool),
            duplication_edge_index=dup)
        self.assertTrue(view.keep_node.all().item())
        self.assertEqual(view.reply_edge_index.shape, (2, 0))
        self.assertEqual(view.duplication_edge_index.shape, (2, 0))
        torch.testing.assert_close(view.x, x)

    def test_attribute_masking_retains_nodes_and_both_relations(self):
        x, reply, dup, _, root = fixture()
        view = generate_view(
            x, reply, ViewConfig(strategy="random", p_n=0, p_e=0, p_m=1),
            root_mask=root, duplication_edge_index=dup)
        self.assertTrue(view.keep_node.all().item())
        torch.testing.assert_close(view.reply_edge_index, reply)
        torch.testing.assert_close(view.duplication_edge_index, dup)
        self.assertEqual(view.x[~root].count_nonzero().item(), 0)

    def test_duplication_edge_interior_rate_is_effective(self):
        count = 6001
        edges = torch.stack([torch.zeros(count - 1, dtype=torch.long),
                             torch.arange(1, count)])
        torch.manual_seed(125)
        view = generate_view(
            torch.ones(count, 1), edges.new_empty(2, 0),
            ViewConfig(strategy="random", p_n=0, p_e=.4, p_m=0),
            duplication_edge_index=edges)
        observed = 1 - view.duplication_edge_index.size(1) / edges.size(1)
        self.assertAlmostEqual(observed, .4, delta=.03)


class EncoderDeletionTests(unittest.TestCase):
    def make_encoder(self, backbone):
        torch.manual_seed(19)
        encoder = DuplicationAwareEncoder(3, 4, num_layers=2,
                                          backbone=backbone, dropout=0)
        # Nonzero biases expose the old "zero input means deleted" bug.
        with torch.no_grad():
            for name, parameter in encoder.named_parameters():
                if name.endswith("bias"):
                    parameter.fill_(.6)
        return encoder.eval()

    def test_masked_encoding_equals_induced_graph_for_gcn_and_gat(self):
        x, reply, dup, batch, _ = fixture()
        keep = torch.tensor([True, False, True, True, False, False, True])
        # Explicit reference: surviving original rows 0, 2, 3, 6.
        compact_reply = torch.tensor([[0], [1]])
        compact_dup = torch.tensor([[0], [1]])
        compact_batch = torch.tensor([0, 0, 1, 2])
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                encoder = self.make_encoder(backbone)
                h, g = encoder(x, reply, dup, batch=batch, keep_node=keep)
                expected_h, expected_g = encoder(x[keep], compact_reply,
                                                  compact_dup, batch=compact_batch)
                torch.testing.assert_close(h[keep], expected_h)
                torch.testing.assert_close(g, expected_g)
                self.assertEqual(h[~keep].count_nonzero().item(), 0)
                self.assertEqual(g.shape, (3, 4))
                # Singleton graph occupies exactly its original final row.
                torch.testing.assert_close(g[2], h[6])

    def test_deleted_features_and_incident_edges_cannot_affect_output_or_gradients(self):
        x, reply, dup, batch, _ = fixture()
        keep = torch.tensor([True, False, True, True, False, False, True])
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                encoder = self.make_encoder(backbone)
                original = x.clone().requires_grad_()
                _, expected = encoder(original, reply, dup, batch=batch, keep_node=keep)
                perturbed = x.clone()
                perturbed[~keep] = -1e6
                extra = torch.tensor([[1, 4, 5], [0, 3, 3]])
                _, actual = encoder(perturbed, torch.cat([reply, extra], 1),
                                     torch.cat([dup, extra], 1), batch=batch,
                                     keep_node=keep)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected.square().sum().backward()
                self.assertEqual(original.grad[~keep].count_nonzero().item(), 0)
                self.assertGreater(original.grad[keep].abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(original.grad).all().item())

    def test_empty_middle_graph_keeps_contrastive_batch_row(self):
        x, reply, dup, batch, _ = fixture()
        keep = torch.tensor([True, False, False, False, False, False, True])
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                _, g = self.make_encoder(backbone)(x, reply, dup, batch=batch,
                                                   keep_node=keep)
                self.assertEqual(g.shape, (3, 4))
                self.assertEqual(g[1].count_nonzero().item(), 0)
                self.assertTrue(torch.isfinite(g).all().item())

    def test_all_deleted_and_empty_inputs_have_defined_finite_outputs(self):
        x, reply, dup, batch, _ = fixture()
        for backbone in ("gcn", "gat"):
            for use_batch in (False, True):
                with self.subTest(backbone=backbone, use_batch=use_batch):
                    original = x.clone().requires_grad_()
                    h, g = self.make_encoder(backbone)(
                        original, reply, dup, batch=batch if use_batch else None,
                        keep_node=torch.zeros(len(x), dtype=torch.bool))
                    self.assertEqual(g.shape, (3 if use_batch else 1, 4))
                    self.assertEqual(g.count_nonzero().item(), 0)
                    self.assertEqual(h.count_nonzero().item(), 0)
                    g.sum().backward()
                    self.assertEqual(original.grad.count_nonzero().item(), 0)
            empty_edges = torch.empty(2, 0, dtype=torch.long)
            h, g = self.make_encoder(backbone)(x[:0], empty_edges, empty_edges)
            self.assertEqual(h.shape, (0, 4))
            self.assertEqual(g.shape, (1, 4))
            self.assertEqual(g.count_nonzero().item(), 0)

    def test_all_kept_mask_preserves_clean_encoding_exactly(self):
        x, reply, dup, batch, _ = fixture()
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                encoder = self.make_encoder(backbone)
                clean = encoder(x, reply, dup, batch=batch)
                masked = encoder(x, reply, dup, batch=batch,
                                  keep_node=torch.ones(len(x), dtype=torch.bool))
                for expected, actual in zip(clean, masked):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class ModelViewIntegrationTests(unittest.TestCase):
    def test_contrastive_forward_encodes_only_roots_when_every_other_node_is_dropped(self):
        graph, batch = make_graph()
        for backbone in ("gcn", "gat"):
            with self.subTest(backbone=backbone):
                model = ARAGCL_DP(ARAGCL_DP_Config(
                    in_dim=3, hidden_dim=4, backbone=backbone, dropout=0,
                    view_strategy="random", p_n=1, p_e=0, p_m=0)).eval()
                z1, z2 = model.contrastive_forward(graph, batch=batch)
                empty_edges = torch.empty(2, 0, dtype=torch.long)
                _, roots_only = model.encoder(graph.x[graph.root_mask], empty_edges,
                                               empty_edges, batch=torch.arange(3))
                expected = model.projection(roots_only)
                torch.testing.assert_close(z1, expected)
                torch.testing.assert_close(z2, expected)
                self.assertEqual(z1.shape, (3, 4))

    def test_full_and_no_duplicate_encoder_sample_identical_views_from_same_seed(self):
        graph, batch = make_graph()
        views = []

        def record_views(*args, **kwargs):
            result = generate_two_views(*args, **kwargs)
            views.append(result)
            return result

        for use_duplication in (True, False):
            model = ARAGCL_DP(ARAGCL_DP_Config(
                in_dim=3, hidden_dim=4, dropout=0, use_duplication=use_duplication,
                p_n=.5, p_e=.5, p_m=.5)).eval()
            torch.manual_seed(813)
            with patch("models.aragcl_dp.generate_two_views", side_effect=record_views):
                model.contrastive_forward(graph, batch=batch)
        for full, no_dup in zip(*views):
            for field in ("x", "keep_node", "reply_edge_index", "duplication_edge_index"):
                torch.testing.assert_close(getattr(full, field), getattr(no_dup, field),
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
