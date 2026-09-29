"""Frozen Weibo marker exclusion changes duplicate links, never posts.

These graph-construction checks do not establish a performance advantage or
identify true reshare sources.
"""
import unittest

import torch

from data.duplication_graph import (
    DUPLICATE_POLICIES,
    RawPost,
    build_duplication_graph,
)


POLICY = "exclude_weibo_marker"
PRESERVED_FIELDS = (
    "x", "reply_edge_index", "timestamps", "propagation_depth", "root_mask",
)


def fixture():
    # Nonsequential IDs and shuffled marker timestamps exercise index mapping
    # and selection of the earliest reference independently of input order.
    values = [
        (40, None, 0, "source story"),
        (7, 40, 3, "转发微博"),
        (99, 40, 1, " \t转发微博\n"),
        (3, 99, 2, "转发微博"),
        (21, 7, 4, "Useful repeated claim"),
        (81, 21, 5, " useful REPEATED claim "),
    ]
    return [RawPost(post_id, parent_id, 17, timestamp, text,
                    torch.tensor([float(i), float(i + 1)]))
            for i, (post_id, parent_id, timestamp, text) in enumerate(values)]


def star(texts):
    return [RawPost(i, None if i == 0 else 0, 9, float(i), text,
                    torch.tensor([float(i), 1.]))
            for i, text in enumerate(texts)]


class DuplicateFilterTests(unittest.TestCase):
    def assert_preserved(self, before, after):
        for name in PRESERVED_FIELDS:
            with self.subTest(field=name):
                torch.testing.assert_close(getattr(after, name),
                                           getattr(before, name), rtol=0, atol=0)
        self.assertEqual(after.precomputed_centrality, before.precomputed_centrality)

    def test_default_is_unchanged_and_matches_explicit_none(self):
        posts = fixture()
        default = build_duplication_graph(posts)
        explicit = build_duplication_graph(posts, duplicate_policy="none")
        self.assert_preserved(default, explicit)
        expected_edges = torch.tensor([[2, 2, 4], [3, 1, 5]])
        expected_freq = torch.tensor([0., 0., 2., 0., 1., 0.])
        expected_sets = [[], [], [3, 1], [], [5], []]
        for graph in (default, explicit):
            torch.testing.assert_close(graph.duplication_edge_index, expected_edges)
            torch.testing.assert_close(graph.duplication_freq, expected_freq)
            self.assertEqual(graph.duplication_sets, expected_sets)

    def test_filter_preserves_posts_tree_features_and_meaningful_links(self):
        posts = fixture()
        original_features = [p.feature.clone() for p in posts]
        original_texts = [p.text for p in posts]
        before = build_duplication_graph(posts)
        after = build_duplication_graph(posts, duplicate_policy=POLICY)
        self.assert_preserved(before, after)
        torch.testing.assert_close(after.duplication_edge_index,
                                   torch.tensor([[4], [5]]))
        torch.testing.assert_close(after.duplication_freq,
                                   torch.tensor([0., 0., 0., 0., 1., 0.]))
        self.assertEqual(after.duplication_sets, [[], [], [], [], [5], []])
        self.assertEqual([p.text for p in posts], original_texts)
        for post, feature in zip(posts, original_features):
            torch.testing.assert_close(post.feature, feature, rtol=0, atol=0)

    def test_entire_marker_group_is_excluded_including_root(self):
        posts = star([" 转发微博 ", "转发微博", "\n转发微博\t"])
        before = build_duplication_graph(posts)
        after = build_duplication_graph(posts, duplicate_policy=POLICY)
        self.assert_preserved(before, after)
        self.assertEqual(before.duplication_edge_index.shape, (2, 2))
        self.assertEqual(after.duplication_edge_index.shape, (2, 0))
        self.assertEqual(after.duplication_edge_index.dtype, torch.long)
        self.assertEqual(after.duplication_sets, [[], [], []])
        torch.testing.assert_close(after.duplication_freq, torch.zeros(3))
        self.assertEqual(after.root_mask.tolist(), [True, False, False])

    def test_rule_is_exact_not_short_text_substring_or_fuzzy_filter(self):
        retained_texts = ["转发", "转发微博。", "请转发微博", "转发 微博", "顶", "OK"]
        for text in retained_texts:
            with self.subTest(text=text):
                posts = star(["source", text, " " + text.lower() + " "])
                before = build_duplication_graph(posts)
                after = build_duplication_graph(posts, duplicate_policy=POLICY)
                self.assert_preserved(before, after)
                torch.testing.assert_close(after.duplication_edge_index,
                                           torch.tensor([[1], [2]]))
                self.assertEqual(after.duplication_sets, [[], [2], []])
                torch.testing.assert_close(after.duplication_freq,
                                           torch.tensor([0., 1., 0.]))

    def test_blank_and_nonduplicate_content_have_no_links(self):
        for texts in (["source", "", "  "], ["source", "unique", "转发微博"],
                      ["转发微博"]):
            with self.subTest(texts=texts):
                posts = star(texts)
                before = build_duplication_graph(posts)
                after = build_duplication_graph(posts, duplicate_policy=POLICY)
                self.assert_preserved(before, after)
                self.assertEqual(after.duplication_edge_index.shape, (2, 0))
                self.assertEqual(after.duplication_sets, [[] for _ in posts])
                torch.testing.assert_close(after.duplication_freq, torch.zeros(len(posts)))

    def test_policy_is_frozen_keyword_only_and_invalid_values_are_rejected(self):
        self.assertEqual(DUPLICATE_POLICIES, ("none", "exclude_weibo_marker"))
        for invalid in ("", "exclude_short_text", "EXCLUDE_WEIBO_MARKER", None):
            with self.subTest(policy=invalid):
                with self.assertRaisesRegex(ValueError, "duplicate_policy"):
                    build_duplication_graph(fixture(), duplicate_policy=invalid)
        with self.assertRaises(TypeError):
            build_duplication_graph(fixture(), True, POLICY)


if __name__ == "__main__":
    unittest.main()
