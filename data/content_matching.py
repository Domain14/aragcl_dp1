"""Deterministic, inspectable shared-content links between existing posts.

These rules infer content relationships, never repost provenance. The v1
eligibility thresholds are a predeclared exploratory rule, not a validated
measure of informativeness. Text normalization stays deliberately compatible
with the legacy graph builder: strip outer whitespace and lowercase only.

The filtered strategies score content after removing URLs, @attribution names,
square-bracket emoji labels and paired #hashtags#. This scoring copy is NEVER
used for matching or model features. Thus two different posts do not become
identical merely because their mentions or hashtags are removed.

``match_posts`` accepts RawPost-compatible objects (post_id, timestamp, text)
and returns evidence dictionaries. Stored direction is earlier reference to
later copy, including when the *later* post contains the shorter text. A
reference is a computational convention, not a verified original author.
"""
from collections import OrderedDict
import re


MATCHING_STRATEGIES = ("exact", "filtered_exact", "filtered_containment")
CONTENT_RULE_VERSION = "shared-content-v1"
MIN_ALPHANUMERIC_CHARACTERS = 20
MIN_DISTINCT_ALPHANUMERIC_CHARACTERS = 10
_DUPLICATE_POLICIES = ("none", "exclude_weibo_marker")

_URL = re.compile(r"https?://[^\s]+", flags=re.IGNORECASE)
# A Weibo attribution commonly looks like //@name: or //@name：. Keep
# subsequent quoted content by stopping at either colon or another slash.
_MENTION = re.compile(r"(?:/{2})?@[^\s@：:/\[\]［］#]+[:：]?")
_EMOJI_LABEL = re.compile(r"\[[^\]\r\n]*\]|［[^］\r\n]*］")
_HASHTAG = re.compile(r"#[^#\r\n]*#")


def normalize_content(text: str) -> str:
    """The exact legacy normalization; does not remove internal punctuation."""
    return text.strip().lower()


def content_eligibility(text: str) -> dict:
    """Explain whether text passes the fixed exploratory content rule.

    Unicode ``isalnum`` counts Chinese characters as well as letters/digits.
    Distinctness is counted after lowercase normalization. URLs extend to the
    next whitespace; mentions extend to whitespace, slash, colon or a bracket.
    These conservative syntactic rules can discard useful content, so retained
    matches still need review. Counts are not evidence of verified resharing.
    """
    scoring_text = normalize_content(text)
    for pattern in (_URL, _MENTION, _EMOJI_LABEL, _HASHTAG):
        scoring_text = pattern.sub(" ", scoring_text)
    characters = [character for character in scoring_text if character.isalnum()]
    count = len(characters)
    distinct_count = len(set(characters))
    reasons = []
    if count < MIN_ALPHANUMERIC_CHARACTERS:
        reasons.append("too_few_alphanumeric_characters")
    if distinct_count < MIN_DISTINCT_ALPHANUMERIC_CHARACTERS:
        reasons.append("too_few_distinct_alphanumeric_characters")
    return {
        "eligible": not reasons,
        "alphanumeric_count": count,
        "distinct_alphanumeric_count": distinct_count,
        "reasons": reasons,
        "scoring_text": scoring_text,
        "rule_version": CONTENT_RULE_VERSION,
    }


def match_posts(posts, strategy: str = "exact", duplicate_policy: str = "none") -> list:
    """Return actual sparse graph links, with direct textual evidence.

    Each record contains reference_id, copy_id, kind (exact/containment),
    shared_text (normalized literal text), and both timestamps. Exact groups
    retain input-group order and an earliest-reference star. Timestamp ties
    retain input order. Blank text never matches.

    ``filtered_containment`` keeps filtered exact stars first. For each node
    without an incoming exact link, it adds at most one containment link from
    the earliest preceding directly matching post. The complete shorter raw
    normalized text must occur inside the longer, and only that shorter text
    is scored for eligibility. We do not infer transitive text relationships,
    union containment groups, or find shared internal fragments of two texts.
    """
    if strategy not in MATCHING_STRATEGIES:
        raise ValueError(f"strategy must be one of {MATCHING_STRATEGIES}; got {strategy!r}")
    if duplicate_policy not in _DUPLICATE_POLICIES:
        raise ValueError(
            f"duplicate_policy must be one of {_DUPLICATE_POLICIES}; got {duplicate_policy!r}"
        )
    posts = list(posts)
    if len({post.post_id for post in posts}) != len(posts):
        raise ValueError("post_id must be unique within one conversation")
    texts = [normalize_content(post.text) for post in posts]
    eligibility = [content_eligibility(text) for text in texts]
    groups = OrderedDict()
    for index, text in enumerate(texts):
        if not text:
            continue
        if duplicate_policy == "exclude_weibo_marker" and text == "转发微博":
            continue
        if strategy != "exact" and not eligibility[index]["eligible"]:
            continue
        groups.setdefault(text, []).append(index)

    def evidence(reference, copy, kind, shared):
        return {
            "reference_id": posts[reference].post_id,
            "copy_id": posts[copy].post_id,
            "kind": kind,
            "shared_text": shared,
            "reference_timestamp": posts[reference].timestamp,
            "copy_timestamp": posts[copy].timestamp,
        }

    links = []
    has_exact_incoming = set()
    for text, group in groups.items():
        ordered = sorted(group, key=lambda index: (posts[index].timestamp, index))
        for copy in ordered[1:]:
            links.append(evidence(ordered[0], copy, "exact", text))
            has_exact_incoming.add(copy)

    if strategy != "filtered_containment":
        return links

    chronological = sorted(range(len(posts)), key=lambda index: (posts[index].timestamp, index))
    for position, copy in enumerate(chronological):
        if copy in has_exact_incoming:
            continue
        for reference in chronological[:position]:
            a, b = texts[reference], texts[copy]
            if len(a) == len(b):
                continue
            shorter_index = reference if len(a) < len(b) else copy
            longer_index = copy if shorter_index == reference else reference
            shorter, longer = texts[shorter_index], texts[longer_index]
            if (eligibility[shorter_index]["eligible"] and shorter in longer):
                links.append(evidence(reference, copy, "containment", shorter))
                break
    return links
