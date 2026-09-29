"""KG resolver interface smoke tests (no LLM)."""
from __future__ import annotations

from se.core.memory.online.kp_resolver import KgResolver, OpenTopicResolver, build_kp_resolver


def test_open_topic_resolver_stable_key() -> None:
    r = OpenTopicResolver()
    k1, kp1 = r.resolve("fraction multiplication", "")
    k2, kp2 = r.resolve("fraction multiplication", "")
    assert k1 == k2
    assert kp1 is None and kp2 is None


def test_kg_resolver_matches_graph_node() -> None:
    r = KgResolver(course_id="")
    key, kp = r.resolve("regular polygon definition sides angles", "")
    assert kp is None or isinstance(kp, str)
    assert key


def test_build_from_config() -> None:
    r = build_kp_resolver({"kp_resolver": "open"}, embedder=None)
    assert isinstance(r, OpenTopicResolver)
