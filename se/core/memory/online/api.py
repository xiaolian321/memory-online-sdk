"""Process-local SDK for online memory: retrieve + commit_turn with persisted counters."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from se.core.memory.online.models import MemoryItem, RetrievalHit
from se.core.memory.online.service import MemoryService
from se.core.memory.online.session_store import (
    load_session,
    payload_from_state,
    save_session,
    turns_from_payload,
)


@dataclass
class MemoryItemView:
    item_id: str
    type: str
    content: str
    status: str
    score: float
    topic: str = ""
    layer: str = "legacy"
    breakdown: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_hit(cls, hit: RetrievalHit) -> MemoryItemView:
        it = hit.item
        layer = it.type if it.type in {"trait", "topic", "observation", "topic_card"} else "legacy"
        if it.item_id.startswith("trait:"):
            layer = "trait"
        elif it.item_id.startswith("topic:"):
            layer = "topic"
        elif it.item_id.startswith("observation:"):
            layer = "observation"
        return cls(
            item_id=it.item_id,
            type=it.type,
            content=it.content,
            status=it.status,
            score=float(hit.score),
            topic=it.topic or "",
            layer=layer,
            breakdown=dict(hit.breakdown or {}),
        )


@dataclass
class RetrieveResponse:
    trace_id: str
    intent: str
    need_memory: bool
    items: list[MemoryItemView]
    prompt_block: str
    timings_ms: dict[str, float]
    n_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "intent": self.intent,
            "need_memory": self.need_memory,
            "items": [item.__dict__ for item in self.items],
            "prompt_block": self.prompt_block,
            "timings_ms": self.timings_ms,
            "n_candidates": self.n_candidates,
        }


@dataclass
class CommitResponse:
    turn_index: int
    student_turn_id: str
    tutor_turn_id: str
    counter: int
    extract_scheduled: bool
    pending_jobs: int
    duplicate_msg: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "student_turn_id": self.student_turn_id,
            "tutor_turn_id": self.tutor_turn_id,
            "counter": self.counter,
            "extract_scheduled": self.extract_scheduled,
            "pending_jobs": self.pending_jobs,
            "duplicate_msg": self.duplicate_msg,
        }


def _format_prompt_block(items: list[MemoryItemView]) -> str:
    if not items:
        return ""
    lines = ["Relevant learner memories:"]
    for i, it in enumerate(items, start=1):
        lines.append(f"{i}. [{it.type}] {it.content}")
    return "\n".join(lines)


class MemoryClient:
    """High-level API for streaming tutoring integrations."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        store_dir: str | Path | None = None,
        *,
        extract_every_n: int | None = None,
    ) -> None:
        self._svc = MemoryService(
            config_path=config_path,
            store_dir=store_dir,
            extract_every_n=extract_every_n,
        )
        self._msg_ids: dict[tuple[str, str], set[str]] = {}
        self._last_activity: dict[tuple[str, str], float] = {}

    @property
    def store_dir(self) -> Path:
        return self._svc.store_dir

    @property
    def service(self) -> MemoryService:
        return self._svc

    def close(self) -> None:
        self._svc.close()

    def _msg_key(self, learner_id: str, session_id: str) -> tuple[str, str]:
        return learner_id, session_id

    def _ensure_session_loaded(self, learner_id: str, session_id: str) -> None:
        data = load_session(self._svc.store_dir, learner_id, session_id)
        if not data:
            return
        learner = self._svc._learner(learner_id)
        if session_id not in learner.sessions:
            sess = self._svc._session(learner_id, session_id)
            sess.turns = turns_from_payload(data, session_id)
            sess.extracted_upto = int(data.get("extracted_upto") or 0)
            sess.student_since = int(data.get("student_since") or 0)
            sess.question_text = str(data.get("question_text") or "")
            sess.window_summary = str(data.get("window_summary") or "")
            sess.profiled = bool(data.get("profiled"))
            if session_id not in learner.session_index:
                learner.session_index[session_id] = len(learner.session_index)
            learner.now_turn = max(learner.now_turn, len(sess.turns))
        key = self._msg_key(learner_id, session_id)
        self._msg_ids.setdefault(key, set()).update(data.get("msg_ids") or [])
        self._last_activity[key] = float(data.get("last_activity_ts") or time.time())

    def _persist_session(self, learner_id: str, session_id: str) -> None:
        sess = self._svc._session(learner_id, session_id)
        key = self._msg_key(learner_id, session_id)
        now = time.time()
        self._last_activity[key] = now
        payload = payload_from_state(
            turns=sess.turns,
            extracted_upto=sess.extracted_upto,
            student_since=sess.student_since,
            question_text=sess.question_text,
            window_summary=sess.window_summary,
            msg_ids=self._msg_ids.get(key, set()),
            last_activity_ts=now,
        )
        payload["profiled"] = sess.profiled
        save_session(self._svc.store_dir, learner_id, session_id, payload)

    def _maybe_idle_flush(self, learner_id: str, session_id: str) -> None:
        idle_s = float(self._svc.cfg.get("idle_flush_s") or 600)
        key = self._msg_key(learner_id, session_id)
        last = self._last_activity.get(key)
        if last is None or time.time() - last < idle_s:
            return
        sess = self._svc._session(learner_id, session_id)
        if sess.extracted_upto < len(sess.turns):
            self._svc._submit_window(learner_id, session_id, use_cache=True)
            self._persist_session(learner_id, session_id)

    def retrieve(
        self,
        learner_id: str,
        session_id: str,
        query: str,
        *,
        question_text: str = "",
        k: int = 5,
        force: bool = False,
    ) -> RetrieveResponse:
        self._ensure_session_loaded(learner_id, session_id)
        self._maybe_idle_flush(learner_id, session_id)
        result = self._svc.retrieve(
            learner_id,
            query,
            session_id,
            k=k,
            force=force,
            question_text=question_text,
        )
        views = [MemoryItemView.from_hit(h) for h in result.items]
        block = result.prompt_block or _format_prompt_block(views)
        return RetrieveResponse(
            trace_id=str(uuid.uuid4()),
            intent=result.intent,
            need_memory=result.need_memory,
            items=views,
            prompt_block=block,
            timings_ms=dict(result.timings_ms),
            n_candidates=result.n_candidates,
        )

    def commit_turn(
        self,
        learner_id: str,
        session_id: str,
        query: str,
        reply: str,
        *,
        msg_id: str,
        question_text: str = "",
        used_item_ids: tuple[str, ...] | list[str] = (),
    ) -> CommitResponse:
        if not msg_id:
            raise ValueError("msg_id is required for idempotent commit_turn")
        self._ensure_session_loaded(learner_id, session_id)
        key = self._msg_key(learner_id, session_id)
        seen = self._msg_ids.setdefault(key, set())
        if msg_id in seen:
            sess = self._svc._session(learner_id, session_id)
            pending = sum(1 for f in self._svc._futures if not f.done())
            return CommitResponse(
                turn_index=len(sess.turns),
                student_turn_id="",
                tutor_turn_id="",
                counter=sess.student_since,
                extract_scheduled=False,
                pending_jobs=pending,
                duplicate_msg=True,
            )

        student_since_before = self._svc._session(learner_id, session_id).student_since
        student_id = self._svc.on_turn(
            learner_id,
            session_id,
            "student",
            query,
            question_text=question_text,
            use_cache=True,
        )
        tutor_id = self._svc.on_turn(
            learner_id,
            session_id,
            "tutor",
            reply,
            question_text=question_text,
            use_cache=True,
        )
        if used_item_ids:
            self._svc.record_use(learner_id, list(used_item_ids))

        sess = self._svc._session(learner_id, session_id)
        extract_every = int(self._svc.cfg.get("extract_every_n") or 3)
        extract_scheduled = student_since_before + 1 >= extract_every and sess.student_since == 0

        seen.add(msg_id)
        self._persist_session(learner_id, session_id)
        pending = sum(1 for f in self._svc._futures if not f.done())
        counter = sess.student_since if sess.student_since else (0 if extract_scheduled else student_since_before + 1)
        return CommitResponse(
            turn_index=len(sess.turns),
            student_turn_id=student_id,
            tutor_turn_id=tutor_id,
            counter=counter,
            extract_scheduled=extract_scheduled,
            pending_jobs=pending,
            duplicate_msg=False,
        )

    def end_session(self, learner_id: str, session_id: str) -> None:
        self._ensure_session_loaded(learner_id, session_id)
        self._svc.end_session(learner_id, session_id, use_cache=True)
        self._persist_session(learner_id, session_id)

    def wait_pending(self, learner_id: str | None = None, timeout: float | None = 120.0) -> None:
        del learner_id
        self._svc.flush(timeout=timeout)

    def snapshot(self, learner_id: str) -> list[MemoryItem]:
        return self._svc.snapshot(learner_id)

    def snapshot_layers(self, learner_id: str) -> dict[str, Any]:
        return self._svc.snapshot_layers(learner_id)
