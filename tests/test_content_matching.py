"""Shared-content matching contracts, not claims of verified reposting.

These fixtures distinguish direct textual evidence from transitive grouping,
exercise time/input ordering, and ensure new links never rewrite the reply
tree or feature vectors. Eligibility is an exploratory syntactic filter;
passing it does not establish informative content or reshare provenance.
"""
import unittest

import torch

from data.content_matching import (
    MATCHING_STRATEGIES,
    content_eligibility,
    match_posts,
    normalize_content,
)
from data.duplication_graph import RawPost, build_duplication_graph


PASSAGE = "最近看新闻得知黄晓明先生做了不少公益，感谢你对消防员的理解和支持。"
OTHER_PASSAGE = "不同的新闻报道需要独立检查原始资料才能判断事实是否真实可靠。"
PRESERVED_FIELDS = (
    "x", "reply_edge_index", "timestamps", "propagation_depth", "root_mask",
)


def posts_from_rows(rows):
    """Rows are (id, parent, timestamp, text); features identify input rows."""
    return [
        RawPost(pid, parent, 37, timestamp, text,
                torch.tensor([index + .125, -index - .5, index * 2.]))
        for index, (pid, parent, timestamp, text) in enumerate(rows)
    ]


def star(texts):
    return posts_from_rows([
        (index, None if index == 0 else 0, float(index), text)
        for index, text in enumerate(texts)
    ])


def tuples(links):
    return [(link["reference_id"], link["copy_id"], link["kind"])
            for link in links]


class ContentMatchingTests(unittest.TestCase):
    def test_exact_retains_legacy_groups_edge_order_and_marker_policy(self):
        # Group order differs from time order. IDs differ from row numbers.
        posts = posts_from_rows([
            (40, None, 0., "unmatched source"),
            (9, 40, 4., " Repeated claim "),
            (71, 40, 1., "转发微博"),
            (8, 9, 2., "REPEATED CLAIM"),
            (63, 71, 3., "转发微博"),
            (12, 8, 2., "repeated claim"),
            (90, 63, 5., ""),
            (91, 90, 6., " \t "),
        ])
        for policy in ("none", "exclude_weibo_marker"):
            with self.subTest(policy=policy):
                links = match_posts(posts, "exact", policy)
                expected = [(8, 12, "exact"), (8, 9, "exact")]
                if policy == "none":
                    expected.append((71, 63, "exact"))
                self.assertEqual(tuples(links), expected)
                # The legacy graph implementation is a separate path.
                graph = build_duplication_graph(posts, duplicate_policy=policy)
                ids = [post.post_id for post in posts]
                graph_edges = [(ids[a], ids[b], "exact")
                               for a, b in graph.duplication_edge_index.t().tolist()]
                self.assertEqual(tuples(links), graph_edges)

    def test_filter_retains_substantive_exact_text_and_excludes_short_common_text(self):
        posts = star([PASSAGE, " " + PASSAGE + "\n", "你好", "你好", "转发微博", "转发微博"])
        self.assertEqual(tuples(match_posts(posts, "filtered_exact")), [(0, 1, "exact")])
        self.assertEqual(tuples(match_posts(posts, "exact")),
                         [(0, 1, "exact"), (2, 3, "exact"), (4, 5, "exact")])

    def test_eligibility_rejects_repetitive_names_and_syntactic_metadata(self):
        examples = [
            "转发微博", "hello", "黄晓明" * 20,
            "@abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "https://example.org/abcdefghijklmnopqrstuvwxyz1234567890",
            "[abcdefghijklmnopqrstuvwxyz1234567890]",
            "［abcdefghijklmnopqrstuvwxyz1234567890］",
            "#abcdefghijklmnopqrstuvwxyz1234567890#",
            "@abcdefghijklmnopqrstuvwxyz 转发微博 [哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈]",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertFalse(content_eligibility(text)["eligible"])
                self.assertEqual(match_posts(star([text, text]), "filtered_containment"), [])

    def test_filter_boundaries_count_chinese_and_case_normalized_distinctness(self):
        exactly_twenty = "甲乙丙丁戊己庚辛壬癸" * 2
        result = content_eligibility(exactly_twenty)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["alphanumeric_count"], 20)
        self.assertEqual(result["distinct_alphanumeric_count"], 10)
        self.assertFalse(content_eligibility(exactly_twenty[:-1])["eligible"])
        self.assertFalse(content_eligibility("甲乙丙丁戊己庚辛壬" * 3)["eligible"])
        self.assertEqual(content_eligibility("AaBbCcDdEe" * 3)["distinct_alphanumeric_count"], 5)

    def test_attribution_removal_scores_only_and_keeps_substantive_quoted_text(self):
        text = "//@Someone：" + PASSAGE + " @Another [赞] #主题#"
        result = content_eligibility(text)
        self.assertTrue(result["eligible"])
        self.assertIn(PASSAGE, result["scoring_text"])
        for token in ("someone", "another", "赞", "主题"):
            self.assertNotIn(token, result["scoring_text"])
        posts = star([text, text])
        links = match_posts(posts, "filtered_exact")
        self.assertEqual(links[0]["shared_text"], normalize_content(text))
        self.assertEqual(posts[0].text, text)

    def test_scoring_normalization_does_not_create_false_text_matches(self):
        # Scoring copies become equal after metadata removal, but original
        # normalized posts do not. Neither whole post contains the other.
        a = "@alice " + PASSAGE + " #first#"
        b = "@bob " + PASSAGE + " #second#"
        self.assertEqual(content_eligibility(a)["scoring_text"],
                         content_eligibility(b)["scoring_text"])
        for strategy in MATCHING_STRATEGIES:
            with self.subTest(strategy=strategy):
                self.assertEqual(match_posts(star([a, b]), strategy), [])

    def test_literal_normalization_does_not_erase_internal_spaces_or_punctuation(self):
        variants = [PASSAGE.replace("，", ","), PASSAGE.replace("新闻", "新 闻")]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(match_posts(star([PASSAGE, variant]), "filtered_containment"), [])

    def test_both_prefix_and_suffix_commentary_are_allowed(self):
        wrapped = "我不同意这个说法：" + PASSAGE + "请先核实消息来源。"
        posts = star([PASSAGE, wrapped])
        self.assertEqual(match_posts(posts, "filtered_exact"), [])
        links = match_posts(posts, "filtered_containment")
        self.assertEqual(tuples(links), [(0, 1, "containment")])
        self.assertEqual(links[0]["shared_text"], PASSAGE)

    def test_earlier_longer_post_remains_reference_to_later_shorter_post(self):
        posts = posts_from_rows([
            (99, None, 0., "前面的评论：" + PASSAGE + "：后面的评论"),
            (5, 99, 2., PASSAGE),
        ])
        links = match_posts(posts, "filtered_containment")
        self.assertEqual(tuples(links), [(99, 5, "containment")])
        self.assertEqual(links[0]["shared_text"], PASSAGE)
        self.assertEqual((links[0]["reference_timestamp"], links[0]["copy_timestamp"]), (0., 2.))

    def test_nontransitive_containment_never_invents_direct_reference_link(self):
        # A contains B and C contains B, but A and C do not contain each
        # other. Unioning them as an equivalence group would invent A->C.
        posts = star(["甲方回应：" + PASSAGE, PASSAGE, PASSAGE + "：乙方回应"])
        links = match_posts(posts, "filtered_containment")
        self.assertEqual(tuples(links), [(0, 1, "containment"), (1, 2, "containment")])
        self.assertNotIn((0, 2, "containment"), tuples(links))

    def test_shared_internal_fragment_alone_is_not_whole_text_containment(self):
        posts = star(["甲方回应：" + PASSAGE + "甲方结论", "乙方回应：" + PASSAGE + "乙方结论"])
        self.assertEqual(match_posts(posts, "filtered_containment"), [])

    def test_tied_times_use_input_order_rather_than_numeric_ids(self):
        posts = posts_from_rows([
            (90, None, 1., PASSAGE),
            (2, 90, 1., PASSAGE + "附加说明"),
            (1, 2, 1., PASSAGE),
        ])
        self.assertEqual(tuples(match_posts(posts, "filtered_containment")),
                         [(90, 1, "exact"), (90, 2, "containment")])

    def test_exact_priority_avoids_second_incoming_containment_edge(self):
        posts = star([PASSAGE, "补充内容：" + PASSAGE, "补充内容：" + PASSAGE])
        links = match_posts(posts, "filtered_containment")
        self.assertEqual(tuples(links), [(1, 2, "exact"), (0, 1, "containment")])
        self.assertEqual(len({link["copy_id"] for link in links}), len(links))

    def test_earliest_direct_candidate_is_chosen_without_dense_fan_in(self):
        posts = star(["先前评论：" + PASSAGE, PASSAGE + "后续评论", PASSAGE])
        self.assertEqual(tuples(match_posts(posts, "filtered_containment")),
                         [(0, 2, "containment")])

    def test_graph_links_match_evidence_preserve_features_tree_and_metadata(self):
        posts = posts_from_rows([
            (44, None, 0., "根节点"),
            (13, 44, 3., "前缀：" + PASSAGE),
            (7, 44, 1., PASSAGE),
            (86, 13, 4., "前缀：" + PASSAGE),
            (29, 7, 2., "转发微博"),
            (3, 29, 5., "转发微博"),
            (8, 3, 6., OTHER_PASSAGE),
        ])
        original_texts = [post.text for post in posts]
        original_features = [post.feature.clone() for post in posts]
        base = build_duplication_graph(posts)
        indices = {post.post_id: index for index, post in enumerate(posts)}
        by_id = {post.post_id: post for post in posts}
        for strategy in MATCHING_STRATEGIES:
            with self.subTest(strategy=strategy):
                graph = build_duplication_graph(posts, matching_strategy=strategy)
                evidence = match_posts(posts, strategy)
                expected_edges = [(indices[link["reference_id"]], indices[link["copy_id"]])
                                  for link in evidence]
                self.assertEqual(graph.duplication_edge_index.t().tolist(),
                                 [list(edge) for edge in expected_edges])
                self.assertEqual(len({target for _, target in expected_edges}), len(expected_edges))
                for field in PRESERVED_FIELDS:
                    torch.testing.assert_close(getattr(graph, field), getattr(base, field), rtol=0, atol=0)
                self.assertEqual(graph.precomputed_centrality, base.precomputed_centrality)
                for index, targets in enumerate(graph.duplication_sets):
                    self.assertEqual(targets, [b for a, b in expected_edges if a == index])
                    self.assertEqual(graph.duplication_freq[index].item(), len(targets))
                for link in evidence:
                    reference, copy = by_id[link["reference_id"]], by_id[link["copy_id"]]
                    a, b = normalize_content(reference.text), normalize_content(copy.text)
                    self.assertLessEqual(reference.timestamp, copy.timestamp)
                    self.assertIn(link["shared_text"], a)
                    self.assertIn(link["shared_text"], b)
                    self.assertIn(link["shared_text"], (a, b))
                    if link["kind"] == "exact":
                        self.assertEqual(a, b)
                    else:
                        self.assertNotEqual(a, b)
                        self.assertTrue(content_eligibility(link["shared_text"])["eligible"])
                self.assertEqual([post.text for post in posts], original_texts)
                for post, expected in zip(posts, original_features):
                    torch.testing.assert_close(post.feature, expected, rtol=0, atol=0)

    def test_empty_blank_or_unmatched_inputs_produce_no_evidence(self):
        for posts in ([], star([""]), star(["", " "]), star([PASSAGE, OTHER_PASSAGE])):
            for strategy in MATCHING_STRATEGIES:
                with self.subTest(strategy=strategy, size=len(posts)):
                    self.assertEqual(match_posts(posts, strategy), [])

    def test_repeated_calls_and_iterable_input_are_deterministic(self):
        posts = star([PASSAGE, "前缀：" + PASSAGE, PASSAGE])
        expected = match_posts(posts, "filtered_containment")
        self.assertEqual(match_posts(iter(posts), "filtered_containment"), expected)
        self.assertEqual(match_posts(posts, "filtered_containment"), expected)

    def test_invalid_options_and_ambiguous_node_ids_are_rejected(self):
        for strategy in ("fuzzy", "", None):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(ValueError, "strategy"):
                    match_posts(star([PASSAGE]), strategy)
        with self.assertRaisesRegex(ValueError, "duplicate_policy"):
            match_posts(star([PASSAGE]), duplicate_policy="unknown")
        posts = star([PASSAGE, PASSAGE])
        posts[1].post_id = posts[0].post_id
        with self.assertRaisesRegex(ValueError, "unique"):
            match_posts(posts, "filtered_containment")
        with self.assertRaisesRegex(NotImplementedError, "near-duplicate"):
            build_duplication_graph(star([PASSAGE]), exact_only=False,
                                    matching_strategy="filtered_containment")


if __name__ == "__main__":
    unittest.main()
