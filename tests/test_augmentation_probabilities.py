"""Regression checks for effective augmentation and degenerate graphs.

Run from the project directory with:
    python -m unittest discover -s tests -p 'test_augmentation_probabilities.py'
"""
import unittest

import torch

from augmentation.centrality import importance_scores_to_drop_probs
from augmentation.views import STRATEGIES, ViewConfig, generate_view


class ImportanceProbabilityTests(unittest.TestCase):
    def test_equal_scores_use_base_rate_at_every_scale(self):
        for value in (0.0, 1.0, 1e-8, 1e-30):
            with self.subTest(value=value):
                scores = torch.full((8,), value, dtype=torch.float64)
                probabilities = importance_scores_to_drop_probs(scores, 0.35)
                torch.testing.assert_close(probabilities, torch.full_like(scores, 0.35))
                self.assertEqual(probabilities.dtype, scores.dtype)
                self.assertEqual(probabilities.device, scores.device)

    def test_nonuniform_scores_preserve_importance_and_scale_invariance(self):
        scores = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        reference = importance_scores_to_drop_probs(scores, 0.6)
        self.assertTrue(torch.all(reference[:-1] > reference[1:]).item())
        self.assertAlmostEqual(reference[0].item(), 0.6)
        self.assertEqual(reference[-1].item(), 0.0)
        for scale in (1e-8, 1e-30, 1e20):
            with self.subTest(scale=scale):
                probabilities = importance_scores_to_drop_probs(scores * scale, 0.6)
                torch.testing.assert_close(probabilities, reference)

    def test_finite_extreme_scores_give_finite_probabilities(self):
        scores = torch.tensor([-3e38, 0.0, 3e38], dtype=torch.float32)
        probabilities = importance_scores_to_drop_probs(scores, 0.8)
        self.assertTrue(torch.isfinite(probabilities).all().item())
        torch.testing.assert_close(probabilities, torch.tensor([0.8, 0.4, 0.0]))

    def test_empty_scores_preserve_shape_and_dtype(self):
        scores = torch.empty(0, dtype=torch.float64)
        probabilities = importance_scores_to_drop_probs(scores, 0.2)
        self.assertEqual(probabilities.shape, scores.shape)
        self.assertEqual(probabilities.dtype, scores.dtype)

    def test_invalid_rates_are_rejected_in_config_and_probability_helper(self):
        for rate in (-0.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    importance_scores_to_drop_probs(torch.ones(3), rate)
                for field in ("p_n", "p_e", "p_m"):
                    with self.assertRaises(ValueError):
                        ViewConfig(**{field: rate})

    def test_nonfinite_scores_are_rejected(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    importance_scores_to_drop_probs(torch.tensor([0.0, score]), 0.2)


class EffectiveViewTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
        self.edges = torch.tensor([[0, 0, 2], [1, 2, 3]], dtype=torch.long)
        self.root = torch.tensor([True, False, False, False])

    def auxiliary_inputs(self, count):
        return {
            "duplication_freq": torch.zeros(count),
            "propagation_depth": torch.arange(count, dtype=torch.float32),
            "precomputed_centrality": {"Degree": torch.ones(count)},
        }

    def test_zero_rates_are_identity_for_every_strategy(self):
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                x_aug, edges_aug, times_aug = generate_view(
                    self.x, self.edges,
                    ViewConfig(strategy=strategy, p_n=0.0, p_e=0.0, p_m=0.0),
                    root_mask=self.root, **self.auxiliary_inputs(len(self.x)),
                )
                torch.testing.assert_close(x_aug, self.x)
                torch.testing.assert_close(edges_aug, self.edges)
                self.assertIsNone(times_aug)

    def test_random_node_rate_one_zeroes_nonroots_and_filters_incident_edges(self):
        x_aug, edges_aug, _ = generate_view(
            self.x, self.edges,
            ViewConfig(strategy="random", p_n=1.0, p_e=0.0, p_m=0.0),
            root_mask=self.root,
        )
        torch.testing.assert_close(x_aug[0], self.x[0])
        self.assertEqual(torch.count_nonzero(x_aug[1:]).item(), 0)
        self.assertEqual(edges_aug.shape, (2, 0))
        # Indices/pooling membership are deliberately not compacted by this API.
        self.assertEqual(x_aug.shape, self.x.shape)

    def test_random_attribute_rate_one_preserves_edges_and_root_features(self):
        x_aug, edges_aug, _ = generate_view(
            self.x, self.edges,
            ViewConfig(strategy="random", p_n=0.0, p_e=0.0, p_m=1.0),
            root_mask=self.root,
        )
        torch.testing.assert_close(x_aug[0], self.x[0])
        self.assertEqual(torch.count_nonzero(x_aug[1:]).item(), 0)
        torch.testing.assert_close(edges_aug, self.edges)

    def test_random_edge_rate_one_preserves_features_and_removes_all_edges(self):
        x_aug, edges_aug, _ = generate_view(
            self.x, self.edges,
            ViewConfig(strategy="random", p_n=0.0, p_e=1.0, p_m=0.0),
            root_mask=self.root,
        )
        torch.testing.assert_close(x_aug, self.x)
        # Root exemption protects root features, not its incident edges.
        self.assertEqual(edges_aug.shape, (2, 0))

    def test_random_interior_rates_produce_expected_frequencies(self):
        count = 6001
        x = torch.ones((count, 1))
        edges = torch.stack([torch.zeros(count - 1, dtype=torch.long),
                             torch.arange(1, count)])
        root = torch.arange(count) == 0
        for field, rate in (("p_n", 0.25), ("p_e", 0.5), ("p_m", 0.75)):
            with self.subTest(field=field):
                torch.manual_seed(140)
                rates = {"p_n": 0.0, "p_e": 0.0, "p_m": 0.0, field: rate}
                x_aug, edges_aug, _ = generate_view(
                    x, edges, ViewConfig(strategy="random", **rates), root_mask=root,
                )
                if field == "p_e":
                    observed = 1 - edges_aug.size(1) / edges.size(1)
                    torch.testing.assert_close(x_aug, x)
                else:
                    observed = (x_aug[1:] == 0).float().mean().item()
                self.assertAlmostEqual(observed, rate, delta=0.04)
                self.assertEqual(x_aug[0].item(), 1.0)

    def test_equal_dataset_centrality_actually_augments(self):
        x_aug, _, _ = generate_view(
            self.x, self.edges,
            ViewConfig(strategy="dataset", p_n=0.0, p_e=0.0, p_m=1.0),
            root_mask=self.root,
            precomputed_centrality={"Degree": torch.full((4,), 1e-8)},
        )
        torch.testing.assert_close(x_aug[0], self.x[0])
        self.assertEqual(torch.count_nonzero(x_aug[1:]).item(), 0)

    def test_empty_and_root_only_graphs_for_every_strategy(self):
        for count in (0, 1):
            for strategy in STRATEGIES:
                with self.subTest(count=count, strategy=strategy):
                    x = torch.ones((count, 3))
                    edges = torch.empty((2, 0), dtype=torch.long)
                    x_aug, edges_aug, _ = generate_view(
                        x, edges,
                        ViewConfig(strategy=strategy, p_n=1.0, p_e=1.0, p_m=1.0),
                        root_mask=torch.ones(count, dtype=torch.bool),
                        **self.auxiliary_inputs(count),
                    )
                    torch.testing.assert_close(x_aug, x)
                    self.assertEqual(edges_aug.shape, (2, 0))

    def test_dataset_scores_are_aligned_to_feature_dtype(self):
        x = self.x.to(dtype=torch.float64)
        x_aug, _, _ = generate_view(
            x, self.edges,
            ViewConfig(strategy="dataset", p_n=0.0, p_e=0.0, p_m=1.0),
            root_mask=self.root,
            precomputed_centrality={"Degree": torch.ones(4, dtype=torch.float32)},
        )
        self.assertEqual(x_aug.dtype, x.dtype)
        torch.testing.assert_close(x_aug[0], x[0])
        self.assertEqual(torch.count_nonzero(x_aug[1:]).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_augmentation_on_cuda_with_cpu_precomputed_scores(self):
        for strategy in ("random", "degree", "pagerank", "dataset"):
            with self.subTest(strategy=strategy):
                x = self.x.cuda()
                x_aug, edges_aug, _ = generate_view(
                    x, self.edges.cuda(),
                    ViewConfig(strategy=strategy, p_n=0.0, p_e=0.0, p_m=0.0),
                    root_mask=self.root.cuda(),
                    precomputed_centrality={"Degree": torch.ones(4)},
                )
                self.assertEqual(x_aug.device, x.device)
                self.assertEqual(edges_aug.device, x.device)
                torch.testing.assert_close(x_aug, x)


if __name__ == "__main__":
    unittest.main()
