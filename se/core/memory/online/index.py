"""Per-learner vector + BM25 index with copy-on-write writes."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from se.core.memory.online.models import MemoryItem, RetrievalHit
from se.core.memory.semantic_retrieval import tokenize

UNRESOLVED_TYPES = frozenset({"misconception", "error_event", "knowledge_gap"})
CROSS_DEDUP_TYPES = frozenset({"misconception", "error_event"})


class _Bm25:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(d) for d in documents]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.n if self.n else 0.0
        df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.idf = {term: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for term, c in df.items()}

    def score(self, query: str) -> list[float]:
        if not self.docs:
            return []
        q_terms = tokenize(query)
        scores = [0.0] * self.n
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, doc in enumerate(self.docs):
                tf = doc.count(term)
                if tf <= 0:
                    continue
                dl = len(doc)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _item_text(item: MemoryItem) -> str:
    topic = (item.topic or "").strip()
    content = (item.content or "").strip()
    return f"{content} | {topic}" if topic else content


def _norm_content(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _types_compatible(a: str, b: str) -> bool:
    if a == b:
        return True
    return a in CROSS_DEDUP_TYPES and b in CROSS_DEDUP_TYPES


def _learner_dir(store_dir: Path, learner_id: str) -> Path:
    digest = hashlib.sha1(learner_id.encode("utf-8")).hexdigest()
    return store_dir / digest


class LearnerIndex:
    def __init__(
        self,
        *,
        learner_id: str,
        store_dir: Path,
        embedder: Any,
        cfg: dict[str, Any],
    ) -> None:
        self.learner_id = learner_id
        self.store_dir = _learner_dir(Path(store_dir), learner_id)
        self.embedder = embedder
        self.cfg = cfg
        self.items: list[MemoryItem] = []
        self.vecs: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._bm25: _Bm25 | None = None
        self._write_lock = threading.Lock()
        self.encode_latencies_ms: list[float] = []
        self._load()

    def _rebuild_bm25(self, items: list[MemoryItem]) -> _Bm25:
        return _Bm25([_item_text(it) for it in items])

    def _meta_path(self) -> Path:
        return self.store_dir / "meta.json"

    def _items_path(self) -> Path:
        return self.store_dir / "items.jsonl"

    def _vecs_path(self) -> Path:
        return self.store_dir / "vecs.npy"

    def _current_meta(self) -> dict[str, Any]:
        return {
            "model_id": self.embedder.model_id,
            "model_path": self.embedder.info().get("model_path"),
            "dim": int(self.vecs.shape[1]) if self.vecs.size else 0,
            "prompt_version": self.cfg.get("prompt_version", ""),
            "learner_id": self.learner_id,
        }

    def _persist(self, items: list[MemoryItem], vecs: np.ndarray) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        with self._items_path().open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
        np.save(self._vecs_path(), vecs)
        self._meta_path().write_text(
            json.dumps(self._current_meta(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self._items_path().is_file():
            return
        items = [
            MemoryItem.from_dict(json.loads(line))
            for line in self._items_path().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vecs = np.load(self._vecs_path()) if self._vecs_path().is_file() else np.zeros((0, 0), dtype=np.float32)
        meta = {}
        if self._meta_path().is_file():
            meta = json.loads(self._meta_path().read_text(encoding="utf-8"))
        need_reembed = (
            str(meta.get("model_id") or "") != str(self.embedder.model_id)
            or len(items) != int(vecs.shape[0] if vecs.ndim == 2 else 0)
        )
        if need_reembed and items:
            started = time.perf_counter()
            vecs = self.embedder.embed_passages([_item_text(it) for it in items])
            self.encode_latencies_ms.append((time.perf_counter() - started) * 1000.0)
        self.items = items
        self.vecs = np.asarray(vecs, dtype=np.float32)
        self._bm25 = self._rebuild_bm25(items)
        if need_reembed and items:
            self._persist(items, self.vecs)

    def _apply_supersedes(self, items: list[MemoryItem], ids: list[str], audit: dict[str, int]) -> None:
        want = set(ids)
        for old in items:
            if old.item_id in want and old.status == "active":
                old.status = "resolved"
                audit["resolved"] = audit.get("resolved", 0) + 1

    def bump_use_counts(self, item_ids: Iterable[str]) -> int:
        want = set(item_ids)
        if not want:
            return 0
        n = 0
        with self._write_lock:
            items = list(self.items)
            vecs = np.array(self.vecs, copy=True) if self.vecs.size else self.vecs
            for old in items:
                if old.item_id in want:
                    old.use_count = int(old.use_count) + 1
                    n += 1
            if n:
                self.items = items
                self._persist(items, vecs)
        return n

    def add(self, new_items: Iterable[MemoryItem]) -> dict[str, int]:
        incoming = [it for it in new_items if it and (it.content or "").strip()]
        audit = {"added": 0, "merged": 0, "resolved": 0, "pref_replaced": 0, "exact_merged": 0}
        if not incoming:
            return audit
        started = time.perf_counter()
        new_vecs = self.embedder.embed_passages([_item_text(it) for it in incoming])
        self.encode_latencies_ms.append((time.perf_counter() - started) * 1000.0)

        dedup = float(self.cfg.get("dedup_cos", 0.85))
        resolve_cos = float(self.cfg.get("resolve_cos", 0.75))
        with self._write_lock:
            items = list(self.items)
            if self.vecs.size == 0:
                vecs = np.zeros((0, new_vecs.shape[1]), dtype=np.float32)
            else:
                vecs = np.array(self.vecs, copy=True)
            for item, vec in zip(incoming, new_vecs):
                vec = np.asarray(vec, dtype=np.float32)
                if item.supersedes_ids:
                    self._apply_supersedes(items, item.supersedes_ids, audit)
                merged = False
                norm_new = _norm_content(item.content)
                for i, old in enumerate(items):
                    if old.status == "deleted":
                        continue
                    if _norm_content(old.content) == norm_new:
                        evidence = list(dict.fromkeys(list(old.evidence_turn_ids) + list(item.evidence_turn_ids)))
                        old.evidence_turn_ids = evidence
                        old.use_count = int(old.use_count) + 1
                        old.last_seen_turn = max(int(old.last_seen_turn), int(item.last_seen_turn))
                        if old.status == "resolved" and item.type in UNRESOLVED_TYPES:
                            old.status = "active"
                        if float(item.confidence) > float(old.confidence):
                            old.content = item.content
                            old.topic = item.topic
                            old.confidence = item.confidence
                            vecs[i] = vec
                        merged = True
                        audit["exact_merged"] += 1
                        audit["merged"] += 1
                        break
                if merged:
                    continue
                for i, old in enumerate(items):
                    if old.status not in {"active", "resolved"}:
                        continue
                    if not _types_compatible(old.type, item.type):
                        continue
                    old_vec = vecs[i]
                    denom = float(np.linalg.norm(old_vec) * np.linalg.norm(vec))
                    cos = float(np.dot(old_vec, vec) / denom) if denom > 0 else 0.0
                    if cos < dedup:
                        continue
                    evidence = list(dict.fromkeys(list(old.evidence_turn_ids) + list(item.evidence_turn_ids)))
                    old.evidence_turn_ids = evidence
                    old.use_count = int(old.use_count) + 1
                    old.last_seen_turn = max(int(old.last_seen_turn), int(item.last_seen_turn))
                    if old.status == "resolved" and item.type in UNRESOLVED_TYPES:
                        old.status = "active"
                    if float(item.confidence) > float(old.confidence):
                        old.content = item.content
                        old.topic = item.topic
                        old.confidence = item.confidence
                        vecs[i] = vec
                    merged = True
                    audit["merged"] += 1
                    break
                if item.type == "mastered":
                    for i, old in enumerate(items):
                        if old.status != "active" or old.type not in UNRESOLVED_TYPES:
                            continue
                        old_vec = vecs[i]
                        denom = float(np.linalg.norm(old_vec) * np.linalg.norm(vec))
                        cos = float(np.dot(old_vec, vec) / denom) if denom > 0 else 0.0
                        if cos >= resolve_cos:
                            old.status = "resolved"
                            audit["resolved"] += 1
                if item.type == "preference" and not merged:
                    for i, old in enumerate(items):
                        if old.status != "active" or old.type != "preference":
                            continue
                        old_vec = vecs[i]
                        denom = float(np.linalg.norm(old_vec) * np.linalg.norm(vec))
                        cos = float(np.dot(old_vec, vec) / denom) if denom > 0 else 0.0
                        if cos >= 0.85:
                            old.status = "deleted"
                            audit["pref_replaced"] += 1
                if merged:
                    continue
                items.append(item)
                vecs = np.vstack([vecs, vec.reshape(1, -1)]) if vecs.size else vec.reshape(1, -1)
                audit["added"] += 1
            self.items = items
            self.vecs = vecs
            self._bm25 = self._rebuild_bm25(items)
            self._persist(items, vecs)
        return audit

    def snapshot(self) -> list[MemoryItem]:
        return [it for it in self.items if it.status in {"active", "resolved"}]

    def all_items(self) -> list[MemoryItem]:
        return list(self.items)

    def search(
        self,
        query_vec: np.ndarray,
        query_text: str,
        *,
        k: int = 5,
        now_turn: int = 0,
        session_id: str = "",
        mode: str = "hybrid",
        apply_floor: bool = False,
        soft_floor: float | None = None,
        min_k: int = 0,
    ) -> list[RetrievalHit]:
        items = self.items
        vecs = self.vecs
        if not items or vecs.size == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.size != vecs.shape[1]:
            raise RuntimeError(f"query dim {q.size} != index dim {vecs.shape[1]}")
        dense_all = np.clip(vecs @ q, 0.0, 1.0)
        bm25_raw = (self._bm25.score(query_text) if self._bm25 is not None else [0.0] * len(items))
        if len(bm25_raw) != len(items):
            bm25_raw = [0.0] * len(items)
        bm25_all = _minmax(bm25_raw)
        weights = self.cfg.get("score_weights") or {}
        w_dense = float(weights.get("dense", 0.55))
        w_bm25 = float(weights.get("bm25", 0.20))
        w_rec = float(weights.get("recency", 0.10))
        w_rel = float(weights.get("reliability", 0.10))
        w_unr = float(weights.get("unresolved", 0.05))
        half_life = float(self.cfg.get("half_life_turns", 20)) or 20.0
        floor = float(self.cfg.get("score_floor", 0.0))
        soft = float(soft_floor if soft_floor is not None else self.cfg.get("soft_floor", 0.25))
        resolved_penalty = float(self.cfg.get("resolved_penalty", 0.55))
        session_boost = float(self.cfg.get("session_recency_boost", 0.08))
        all_hits: list[RetrievalHit] = []
        hits: list[RetrievalHit] = []
        for i, item in enumerate(items):
            if item.status not in {"active", "resolved"}:
                continue
            dense = float(dense_all[i])
            bm25 = float(bm25_all[i])
            recency = 0.5 ** ((max(0, now_turn - int(item.last_seen_turn))) / half_life)
            reliability = float(item.confidence) * min(1.0, 0.6 + 0.1 * float(item.use_count))
            unresolved = (
                1.0 if item.type in UNRESOLVED_TYPES and item.status == "active" else 0.0
            )
            hybrid = (
                w_dense * dense
                + w_bm25 * bm25
                + w_rec * recency
                + w_rel * reliability
                + w_unr * unresolved
            )
            if session_id and item.session_id == session_id:
                hybrid += session_boost
            if item.status == "resolved":
                hybrid *= resolved_penalty
            if mode == "recency":
                score = float(item.last_seen_turn)
            elif mode == "bm25":
                score = bm25
            elif mode == "dense":
                score = dense
            else:
                score = hybrid
            hit = RetrievalHit(
                item=item,
                score=float(score),
                breakdown={
                    "dense": dense,
                    "bm25": bm25,
                    "recency": float(recency),
                    "reliability": float(reliability),
                    "unresolved": float(unresolved),
                    "hybrid": float(hybrid),
                },
            )
            all_hits.append(hit)
            if apply_floor and mode in {"hybrid", "hybrid_floor"} and score < floor:
                continue
            hits.append(hit)
        hits.sort(key=lambda h: h.score, reverse=True)
        top = hits[: max(0, int(k))]
        if apply_floor and min_k > 0 and len(top) < min_k:
            all_hits.sort(key=lambda h: h.score, reverse=True)
            relaxed = [h for h in all_hits if h.score >= soft]
            if not relaxed:
                relaxed = all_hits
            seen = {h.item.item_id for h in top}
            for h in relaxed:
                if h.item.item_id in seen:
                    continue
                top.append(h)
                seen.add(h.item.item_id)
                if len(top) >= min_k:
                    break
            top.sort(key=lambda h: h.score, reverse=True)
            top = top[: max(int(k), min_k)]
        return top[: max(0, int(k))]


class IndexManager:
    def __init__(self, *, store_dir: Path, embedder: Any, cfg: dict[str, Any]) -> None:
        self.store_dir = Path(store_dir)
        self.embedder = embedder
        self.cfg = cfg
        self._indexes: dict[str, LearnerIndex] = {}
        self._lock = threading.Lock()

    def get(self, learner_id: str) -> LearnerIndex:
        with self._lock:
            idx = self._indexes.get(learner_id)
            if idx is None:
                idx = LearnerIndex(
                    learner_id=learner_id,
                    store_dir=self.store_dir,
                    embedder=self.embedder,
                    cfg=self.cfg,
                )
                self._indexes[learner_id] = idx
            return idx
