"""Deterministic consolidation validation (no LLM)."""
from __future__ import annotations

import uuid
from pathlib import Path

from se.core.memory.online.consolidate import apply_operations
from se.core.memory.online.kp_resolver import OpenTopicResolver
from se.core.memory.online.layered_store import LayeredStore
from se.core.memory.online.models import Observation, TopicCard


def test_mastered_requires_correct_after_error(tmp_path: Path) -> None:
    store = LayeredStore(store_dir=tmp_path, learner_id="L1")
    card = TopicCard(
        card_id="c1",
        learner_id="L1",
        topic_key="k1",
        label="subtraction",
        status="struggling",
        summary="",
        evidence_obs_ids=["o1"],
    )
    store.topics.append(card)
    o1 = Observation(
        obs_id="o1",
        learner_id="L1",
        session_id="s",
        topic_hint="subtraction",
        topic_key="k1",
        outcome="incorrect",
        content="Student computed 32-7 as 20",
        evidence_turn_ids=["s0:t1"],
        turn_order=1,
        independent=True,
    )
    o2 = Observation(
        obs_id="o2",
        learner_id="L1",
        session_id="s",
        topic_hint="subtraction",
        topic_key="k1",
        outcome="correct",
        content="Student computed 32-7 as 25",
        evidence_turn_ids=["s0:t3"],
        turn_order=3,
        independent=True,
    )
    store.observations.append(o1)
    audit = apply_operations(
        store=store,
        operations=[
            {
                "op": "update",
                "card_id": "c1",
                "status": "mastered",
                "summary": "Can subtract",
                "evidence_obs_ids": ["o2"],
            }
        ],
        new_obs=[o2],
        kp_resolver=OpenTopicResolver(),
    )
    assert audit.get("mastered_rejected", 0) == 0
    assert store.topic_by_id("c1").status == "mastered"
    store.persist()
    store2 = LayeredStore(store_dir=tmp_path, learner_id="L1")
    assert store2.topic_by_id("c1").status == "mastered"
