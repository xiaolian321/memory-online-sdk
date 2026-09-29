"""Online memory service: layered observe/consolidate + fast retrieval."""
from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from se.core.memory.online.config import (
    load_memory_online_config,
    repo_path,
    resolve_embedding_model_dir,
)
from se.core.memory.online.consolidate import consolidate_window, pick_related_cards
from se.core.memory.online.embedder import MemoryEmbedder
from se.core.memory.online.extract_window import extract_window
from se.core.memory.online.index import IndexManager
from se.core.memory.online.intent import IntentGate
from se.core.memory.online.kp_resolver import KpResolver, build_kp_resolver
from se.core.memory.online.layered_index import LayeredIndexManager
from se.core.memory.online.models import LayeredHit, MemoryItem, RetrievalResult, Turn
from se.core.memory.online.observe_window import observe_window
from se.core.memory.online.profile import profile_session


@dataclass
class _SessionState:
    turns: list[Turn] = field(default_factory=list)
    extracted_upto: int = 0
    student_since: int = 0
    question_text: str = ""
    window_summary: str = ""
    profiled: bool = False


@dataclass
class _LearnerState:
    now_turn: int = 0
    obs_turn_order: int = 0
    sessions: dict[str, _SessionState] = field(default_factory=dict)
    session_index: dict[str, int] = field(default_factory=dict)


class MemoryService:
    def __init__(
        self,
        config_path: str | Path | None = None,
        store_dir: str | Path | None = None,
        *,
        extract_every_n: int | None = None,
        max_workers: int = 4,
    ) -> None:
        self.cfg = load_memory_online_config(config_path)
        if extract_every_n is not None:
            self.cfg["extract_every_n"] = int(extract_every_n)
        self.memory_mode = str(self.cfg.get("memory_mode") or "layered").strip().lower()
        model_dir = resolve_embedding_model_dir(self.cfg)
        self.embedder = MemoryEmbedder(
            model_id=str(self.cfg["embedding_model_id"]),
            model_dir=model_dir,
            device=str(self.cfg.get("embedding_device") or "cuda:6"),
            use_fp16=bool(self.cfg.get("embedding_use_fp16", True)),
            batch_size=int(self.cfg.get("embedding_batch_size") or 32),
        )
        self.embedder.warmup()
        resolved_store = Path(store_dir) if store_dir else repo_path(str(self.cfg.get("store_dir") or "data/memory_online/store"))
        if not resolved_store.is_absolute():
            resolved_store = repo_path(str(resolved_store))
        self.store_dir = resolved_store
        self.index = IndexManager(store_dir=self.store_dir, embedder=self.embedder, cfg=self.cfg)
        self.layered = LayeredIndexManager(store_dir=self.store_dir, embedder=self.embedder, cfg=self.cfg)
        self.intent = IntentGate(
            self.embedder,
            threshold=float(self.cfg.get("intent_proto_threshold", 0.3)),
        )
        self._kp_resolvers: dict[str, KpResolver] = {}
        self._learners: dict[str, _LearnerState] = {}
        self._learner_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._extract_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future[Any]] = []
        self._stats_lock = threading.Lock()
        self._extract_count = 0
        self._extract_fail = 0
        self._extract_latencies: list[float] = []
        self._drop_counts: dict[str, int] = defaultdict(int)
        self._cache_hits = 0
        self._consolidate_latencies: list[float] = []
        self._profile_latencies: list[float] = []

    def close(self) -> None:
        self.flush()
        self._pool.shutdown(wait=True)

    def _kp_resolver(self, learner_id: str) -> KpResolver:
        r = self._kp_resolvers.get(learner_id)
        if r is None:
            r = build_kp_resolver(self.cfg, embedder=self.embedder)
            self._kp_resolvers[learner_id] = r
        return r

    def _learner(self, learner_id: str) -> _LearnerState:
        state = self._learners.get(learner_id)
        if state is None:
            state = _LearnerState()
            self._learners[learner_id] = state
        return state

    def _session(self, learner_id: str, session_id: str) -> _SessionState:
        learner = self._learner(learner_id)
        sess = learner.sessions.get(session_id)
        if sess is None:
            sess = _SessionState()
            learner.sessions[session_id] = sess
        return sess

    def _build_window(self, sess: _SessionState) -> list[Turn]:
        start = sess.extracted_upto
        window = [Turn(**{**t.to_dict(), "is_context": False}) for t in sess.turns[start:]]
        context_src = sess.turns[max(0, start - 2) : start]
        context = [
            Turn(
                turn_id=t.turn_id,
                session_id=t.session_id,
                role=t.role,
                text=t.text,
                ts=t.ts,
                is_context=True,
            )
            for t in context_src
        ]
        return context + window

    def _submit_window(self, learner_id: str, session_id: str, *, use_cache: bool = True) -> None:
        sess = self._session(learner_id, session_id)
        if sess.extracted_upto >= len(sess.turns):
            return
        window = self._build_window(sess)
        now_turn = self._learner(learner_id).now_turn
        prev_summary = sess.window_summary
        question = sess.question_text
        window_id = f"{session_id}:w{sess.extracted_upto}"
        sess.extracted_upto = len(sess.turns)
        sess.student_since = 0
        if self.memory_mode == "flat":
            fut = self._pool.submit(
                self._run_extract_flat,
                learner_id,
                session_id,
                window,
                now_turn,
                prev_summary,
                question,
                use_cache,
            )
        else:
            fut = self._pool.submit(
                self._run_extract_layered,
                learner_id,
                session_id,
                window,
                window_id,
                question,
                use_cache,
            )
        self._futures.append(fut)

    def _run_extract_flat(
        self,
        learner_id: str,
        session_id: str,
        window: list[Turn],
        now_turn: int,
        prev_summary: str,
        question: str,
        use_cache: bool,
    ) -> None:
        with self._extract_locks[learner_id]:
            existing = self.index.get(learner_id).snapshot()
            existing.sort(key=lambda it: (it.last_seen_turn, it.confidence), reverse=True)
            result = extract_window(
                turns=window,
                learner_id=learner_id,
                session_id=session_id,
                existing_items=existing[:20],
                window_summary_prev=prev_summary,
                question_text=question,
                extract_model=str(self.cfg["extract_model"]),
                prompt_version=str(self.cfg.get("prompt_version") or "extract_v2"),
                use_cache=use_cache,
                now_turn=now_turn,
                retries=int(self.cfg.get("extract_retries") or 2),
            )
            if result.status == "ok" and result.items:
                self.index.get(learner_id).add(result.items)
            self._record_extract_stats(result)
            sess = self._session(learner_id, session_id)
            if result.window_summary:
                sess.window_summary = result.window_summary

    def _run_extract_layered(
        self,
        learner_id: str,
        session_id: str,
        window: list[Turn],
        window_id: str,
        question: str,
        use_cache: bool,
    ) -> None:
        with self._extract_locks[learner_id]:
            learner = self._learner(learner_id)
            store = self.layered.stores.get(learner_id)
            obs_result = observe_window(
                turns=window,
                learner_id=learner_id,
                session_id=session_id,
                window_id=window_id,
                extract_model=str(self.cfg["extract_model"]),
                turn_order_base=learner.obs_turn_order,
                use_cache=use_cache,
                retries=int(self.cfg.get("extract_retries") or 2),
            )
            learner.obs_turn_order += len(obs_result.observations)
            self._record_extract_stats(obs_result)
            if obs_result.status != "ok" or not obs_result.observations:
                sess = self._session(learner_id, session_id)
                if obs_result.window_summary:
                    sess.window_summary = obs_result.window_summary
                return
            related = pick_related_cards(store, obs_result.observations)
            cons = consolidate_window(
                store=store,
                new_obs=obs_result.observations,
                related_cards=related,
                extract_model=str(self.cfg["extract_model"]),
                kp_resolver=self._kp_resolver(learner_id),
                use_cache=use_cache,
                retries=int(self.cfg.get("extract_retries") or 2),
            )
            with self._stats_lock:
                self._consolidate_latencies.append(float(cons.latency_ms))
                if cons.status != "ok":
                    self._extract_fail += 1
                for k, v in (cons.audit or {}).items():
                    self._drop_counts[f"cons_{k}"] += int(v)
            self.layered.rebuild(learner_id)
            sess = self._session(learner_id, session_id)
            if obs_result.window_summary:
                sess.window_summary = obs_result.window_summary

    def _run_profile(self, learner_id: str, session_id: str, use_cache: bool) -> None:
        with self._extract_locks[learner_id]:
            sess = self._session(learner_id, session_id)
            if sess.profiled:
                return
            store = self.layered.stores.get(learner_id)
            prof = profile_session(
                store=store,
                session_id=session_id,
                session_turns=sess.turns,
                extract_model=str(self.cfg["extract_model"]),
                use_cache=use_cache,
            )
            sess.profiled = True
            with self._stats_lock:
                self._profile_latencies.append(float(prof.latency_ms))
            self.layered.rebuild(learner_id)

    def _record_extract_stats(self, result: Any) -> None:
        with self._stats_lock:
            self._extract_count += 1
            self._extract_latencies.append(float(result.latency_ms))
            if result.status not in {"ok", "empty"}:
                self._extract_fail += 1
            dropped = (result.audit or {}).get("dropped") or result.audit or {}
            if isinstance(dropped, dict):
                for reason, n in dropped.items():
                    if isinstance(n, int):
                        self._drop_counts[str(reason)] += n

    def on_turn(
        self,
        learner_id: str,
        session_id: str,
        role: str,
        text: str,
        *,
        question_text: str = "",
        use_cache: bool = True,
    ) -> str:
        with self._learner_locks[learner_id]:
            learner = self._learner(learner_id)
            sess = self._session(learner_id, session_id)
            if question_text:
                sess.question_text = question_text
            learner.now_turn += 1
            if session_id not in learner.session_index:
                learner.session_index[session_id] = len(learner.session_index)
            turn_id = f"s{learner.session_index[session_id]}:t{len(sess.turns)}"
            sess.turns.append(
                Turn(
                    turn_id=turn_id,
                    session_id=session_id,
                    role=role,
                    text=text,
                    is_context=False,
                )
            )
            if role == "student":
                sess.student_since += 1
                if sess.student_since >= int(self.cfg.get("extract_every_n") or 3):
                    self._submit_window(learner_id, session_id, use_cache=use_cache)
            return turn_id

    def end_session(self, learner_id: str, session_id: str, *, use_cache: bool = True) -> None:
        with self._learner_locks[learner_id]:
            self._submit_window(learner_id, session_id, use_cache=use_cache)
            if self.memory_mode != "flat":
                fut = self._pool.submit(self._run_profile, learner_id, session_id, use_cache)
                self._futures.append(fut)

    def flush(self, learner_id: str | None = None, timeout: float | None = None) -> None:
        del learner_id
        pending = [f for f in self._futures if not f.done()]
        if pending:
            wait(pending, timeout=timeout)
        errors = [f.exception() for f in self._futures if f.done() and f.exception()]
        if errors:
            raise RuntimeError(f"background extract failed: {errors[0]}") from errors[0]

    def _build_query_text(self, learner_id: str, session_id: str, query: str, question_text: str = "") -> str:
        n_ctx = int(self.cfg.get("retrieve_context_turns") or 2)
        sess = self._session(learner_id, session_id)
        qtext = (question_text or sess.question_text or "").strip()
        recent = [t.text.strip() for t in sess.turns[-n_ctx:] if (t.text or "").strip()]
        parts = [p for p in [qtext, *recent, query.strip()] if p]
        return "\n".join(parts) if parts else query

    def record_use(self, learner_id: str, item_ids: list[str]) -> int:
        if self.memory_mode == "flat":
            return self.index.get(learner_id).bump_use_counts(item_ids)
        trait_ids = [x for x in item_ids if x.startswith("trait:")]
        card_ids = [x.replace("topic:", "") for x in item_ids if x.startswith("topic:")]
        self.layered.get(learner_id).bump_use(
            [x.replace("trait:", "") for x in trait_ids],
            card_ids,
        )
        return len(trait_ids) + len(card_ids)

    def _layered_hits_to_legacy(self, hits: list[LayeredHit]) -> list[Any]:
        from se.core.memory.online.models import RetrievalHit

        out: list[RetrievalHit] = []
        for h in hits:
            p = h.payload
            content = p.get("content") or p.get("summary") or ""
            typ = h.layer
            status = p.get("status") or "active"
            item = MemoryItem(
                item_id=f"{h.layer}:{h.ref_id}",
                learner_id="",
                session_id="",
                type=typ,
                content=str(content),
                topic=str(p.get("label") or p.get("topic_hint") or ""),
                status=str(status),
            )
            out.append(RetrievalHit(item=item, score=h.score, breakdown=h.breakdown))
        return out

    def retrieve(
        self,
        learner_id: str,
        query: str,
        session_id: str,
        k: int = 5,
        force: bool = False,
        *,
        question_text: str = "",
        mode: str = "hybrid_floor",
        apply_floor: bool | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        query_text = self._build_query_text(learner_id, session_id, query, question_text=question_text)
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query_text)
        embed_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        intent, need_memory = self.intent.classify(query, qvec)
        intent_ms = (time.perf_counter() - t1) * 1000.0

        if self.memory_mode != "flat":
            t2 = time.perf_counter()
            layered = self.layered.get(learner_id).search(
                qvec,
                query_text,
                k_topics=min(3, k),
                k_traits=3,
                k_obs=2,
                session_id=session_id,
            )
            layered.intent = intent
            layered.need_memory = force or need_memory
            hits = self._layered_hits_to_legacy(layered.traits + layered.topics + layered.observations)
            score_ms = (time.perf_counter() - t2) * 1000.0
            total_ms = (time.perf_counter() - started) * 1000.0
            return RetrievalResult(
                intent=intent,
                need_memory=layered.need_memory,
                items=hits[:k] if (force or need_memory) else [],
                timings_ms={
                    "embed": embed_ms,
                    "intent": intent_ms,
                    "score": score_ms,
                    "total": total_ms,
                },
                n_candidates=layered.n_candidates,
                prompt_block=layered.prompt_block,
            )

        use_floor = bool(self.cfg.get("score_floor", 0)) if apply_floor is None else apply_floor
        if mode == "hybrid_floor":
            use_floor = True
            search_mode = "hybrid"
        elif mode == "hybrid":
            use_floor = False if apply_floor is None else use_floor
            search_mode = "hybrid"
        else:
            search_mode = mode
            if apply_floor is None:
                use_floor = False
        soft_floor = float(self.cfg.get("soft_floor", 0.25))
        min_k = int(self.cfg.get("retrieve_min_k") or 1) if (use_floor and (force or need_memory)) else 0
        t2 = time.perf_counter()
        items = []
        n_cand = 0
        if force or need_memory:
            now_turn = self._learner(learner_id).now_turn
            hits = self.index.get(learner_id).search(
                qvec,
                query_text,
                k=k,
                now_turn=now_turn,
                session_id=session_id,
                mode=search_mode,
                apply_floor=use_floor,
                soft_floor=soft_floor,
                min_k=min_k,
            )
            n_cand = len(hits)
            items = hits
        score_ms = (time.perf_counter() - t2) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalResult(
            intent=intent,
            need_memory=need_memory,
            items=items,
            timings_ms={
                "embed": embed_ms,
                "intent": intent_ms,
                "score": score_ms,
                "total": total_ms,
            },
            n_candidates=n_cand,
        )

    def snapshot(self, learner_id: str) -> list[MemoryItem]:
        if self.memory_mode == "flat":
            return self.index.get(learner_id).snapshot()
        store = self.layered.stores.get(learner_id)
        out: list[MemoryItem] = []
        for c in store.active_topics():
            out.append(
                MemoryItem(
                    item_id=c.card_id,
                    learner_id=learner_id,
                    session_id="",
                    type="topic_card",
                    content=c.summary or c.label,
                    topic=c.label,
                    status=c.status,
                )
            )
        return out

    def snapshot_layers(self, learner_id: str) -> dict[str, Any]:
        return self.layered.stores.get(learner_id).snapshot()

    def stats(self) -> dict[str, Any]:
        encode_ms: list[float] = []
        for idx in self.index._indexes.values():
            encode_ms.extend(idx.encode_latencies_ms)
        with self._stats_lock:
            return {
                "extract_count": self._extract_count,
                "extract_fail": self._extract_fail,
                "extract_latencies_ms": list(self._extract_latencies),
                "consolidate_latencies_ms": list(self._consolidate_latencies),
                "profile_latencies_ms": list(self._profile_latencies),
                "drop_counts": dict(self._drop_counts),
                "encode_latencies_ms": encode_ms,
                "cache_hits": self._cache_hits,
                "memory_mode": self.memory_mode,
            }
