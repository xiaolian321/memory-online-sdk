"""Layered retrieval index (topics, traits, observations)."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from se.core.memory.online.index import _Bm25, _minmax
from se.core.memory.online.layered_store import LayeredStore, LayeredStoreManager
from se.core.memory.online.models import LayeredHit, LayeredRetrievalResult, Observation, TopicCard, LearnerTrait


def _card_text(c: TopicCard) -> str:
    active = "; ".join(m.text for m in c.active_misconceptions[:3])
    resolved = "; ".join(m.text for m in c.resolved_misconceptions[:2])
    return f"{c.label} {c.summary} status={c.status} active={active} resolved={resolved}"


def _trait_text(t: LearnerTrait) -> str:
    return t.content


def _obs_text(o: Observation) -> str:
    return f"{o.topic_hint} {o.content} {o.outcome} {o.error_cause}"


def _format_prompt_block(
    traits: list[LayeredHit],
    topics: list[LayeredHit],
    observations: list[LayeredHit],
) -> str:
    lines: list[str] = []
    if traits:
        lines.append("【学习者特质】")
        for i, h in enumerate(traits[:3], 1):
            lines.append(f"{i}. {h.payload.get('content', '')}")
    if topics:
        lines.append("【相关知识点】")
        for i, h in enumerate(topics[:3], 1):
            p = h.payload
            st = p.get("status", "")
            summary = p.get("summary", "")
            active = p.get("active_misconceptions") or []
            resolved = p.get("resolved_misconceptions") or []
            lines.append(f"{i}. [{st}] {p.get('label', '')}: {summary}")
            if active:
                lines.append(f"   当前误解: {'; '.join(active[:2])}")
            if resolved:
                lines.append(f"   已纠正: {'; '.join(resolved[:2])}")
    if observations:
        lines.append("【最近证据】")
        for i, h in enumerate(observations[:2], 1):
            lines.append(f"{i}. {h.payload.get('content', '')}")
    return "\n".join(lines)


class LayeredLearnerIndex:
    def __init__(self, store: LayeredStore, embedder: Any, cfg: dict[str, Any]) -> None:
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self._topic_vecs: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._trait_vecs: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._obs_vecs: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._topic_bm25: _Bm25 | None = None
        self._trait_bm25: _Bm25 | None = None
        self._obs_bm25: _Bm25 | None = None
        self._lock = threading.Lock()
        self.rebuild()

    def rebuild(self) -> None:
        topics = self.store.active_topics()
        traits = self.store.active_traits()
        obs = self.store.observations[-200:]
        if topics:
            self._topic_vecs = self.embedder.embed_passages([_card_text(c) for c in topics])
        else:
            self._topic_vecs = np.zeros((0, 0), dtype=np.float32)
        if traits:
            self._trait_vecs = self.embedder.embed_passages([_trait_text(t) for t in traits])
        else:
            self._trait_vecs = np.zeros((0, 0), dtype=np.float32)
        if obs:
            self._obs_vecs = self.embedder.embed_passages([_obs_text(o) for o in obs])
        else:
            self._obs_vecs = np.zeros((0, 0), dtype=np.float32)
        self._topic_bm25 = _Bm25([_card_text(c) for c in topics]) if topics else None
        self._trait_bm25 = _Bm25([_trait_text(t) for t in traits]) if traits else None
        self._obs_bm25 = _Bm25([_obs_text(o) for o in obs]) if obs else None
        self._topics = topics
        self._traits = traits
        self._obs = obs

    def _search_pool(
        self,
        qvec: np.ndarray,
        query_text: str,
        texts: list[str],
        vecs: np.ndarray,
        bm25: _Bm25 | None,
        items: list[Any],
        *,
        k: int,
        score_fn: Any | None = None,
    ) -> list[LayeredHit]:
        if not items or vecs.size == 0:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        dense = np.clip(vecs @ q, 0.0, 1.0)
        bm = _minmax(bm25.score(query_text) if bm25 else [0.0] * len(items))
        weights = self.cfg.get("score_weights") or {}
        w_dense = float(weights.get("dense", 0.55))
        w_bm25 = float(weights.get("bm25", 0.2))
        hits: list[LayeredHit] = []
        for i, item in enumerate(items):
            score = w_dense * float(dense[i]) + w_bm25 * float(bm[i])
            if score_fn:
                score = score_fn(item, score)
            hits.append(
                LayeredHit(
                    layer="topic",
                    ref_id=getattr(item, "card_id", getattr(item, "trait_id", getattr(item, "obs_id", ""))),
                    score=float(score),
                    payload=self._payload(item),
                    breakdown={"dense": float(dense[i]), "bm25": float(bm[i])},
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def _payload(self, item: Any) -> dict[str, Any]:
        if isinstance(item, TopicCard):
            return {
                "card_id": item.card_id,
                "label": item.label,
                "status": item.status,
                "summary": item.summary,
                "active_misconceptions": [m.text for m in item.active_misconceptions],
                "resolved_misconceptions": [m.text for m in item.resolved_misconceptions],
                "needs_review": item.needs_review,
            }
        if isinstance(item, LearnerTrait):
            return {"trait_id": item.trait_id, "content": item.content}
        if isinstance(item, Observation):
            return {
                "obs_id": item.obs_id,
                "content": item.content,
                "outcome": item.outcome,
                "topic_hint": item.topic_hint,
            }
        return {}

    def search(
        self,
        qvec: np.ndarray,
        query_text: str,
        *,
        k_topics: int = 3,
        k_traits: int = 3,
        k_obs: int = 2,
        session_id: str = "",
    ) -> LayeredRetrievalResult:
        soft = float(self.cfg.get("soft_floor", 0.25))
        session_boost = float(self.cfg.get("session_recency_boost", 0.08))

        def topic_score(card: TopicCard, base: float) -> float:
            s = base
            if card.status == "struggling":
                s += 0.06
            if card.needs_review:
                s += 0.05
            if card.status == "mastered":
                s *= 0.92
            return s

        traits = self._search_pool(
            qvec,
            query_text,
            [],
            self._trait_vecs,
            self._trait_bm25,
            self._traits,
            k=k_traits,
        )
        for h in traits:
            h.layer = "trait"
        topics = self._search_pool(
            qvec,
            query_text,
            [],
            self._topic_vecs,
            self._topic_bm25,
            self._topics,
            k=k_topics,
            score_fn=topic_score,
        )
        for h in topics:
            h.layer = "topic"
        obs = self._search_pool(
            qvec,
            query_text,
            [],
            self._obs_vecs,
            self._obs_bm25,
            self._obs,
            k=k_obs,
        )
        for h in obs:
            h.layer = "observation"
        if topics and topics[0].score < soft and len(topics) == 1:
            topics = [topics[0]]
        block = _format_prompt_block(traits, topics, obs)
        n_cand = len(traits) + len(topics) + len(obs)
        return LayeredRetrievalResult(
            intent="task",
            need_memory=bool(block),
            traits=traits,
            topics=topics,
            observations=obs,
            prompt_block=block,
            timings_ms={},
            n_candidates=n_cand,
        )

    def bump_use(self, trait_ids: list[str], card_ids: list[str]) -> None:
        for t in self.store.traits:
            if t.trait_id in trait_ids:
                t.use_count += 1
        for c in self.store.topics:
            if c.card_id in card_ids:
                c.use_count += 1
        self.store.persist()


class LayeredIndexManager:
    def __init__(self, *, store_dir: Path, embedder: Any, cfg: dict[str, Any]) -> None:
        self.stores = LayeredStoreManager(store_dir)
        self.embedder = embedder
        self.cfg = cfg
        self._indexes: dict[str, LayeredLearnerIndex] = {}
        self._lock = threading.Lock()

    def get(self, learner_id: str) -> LayeredLearnerIndex:
        with self._lock:
            idx = self._indexes.get(learner_id)
            store = self.stores.get(learner_id)
            if idx is None:
                idx = LayeredLearnerIndex(store, self.embedder, self.cfg)
                self._indexes[learner_id] = idx
            return idx

    def rebuild(self, learner_id: str) -> None:
        self.get(learner_id).rebuild()
