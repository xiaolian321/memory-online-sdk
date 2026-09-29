"""L0 observation extraction from dialogue windows."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from se.core.memory.online.extract_window import (
    _format_turns,
    _load_prompt,
    _normalize_student_subject,
    _resolve_turn_id,
    _student_turn_ids,
    _window_turn_ids,
)
from se.core.memory.online.models import OBS_OUTCOMES, Observation, ObserveResult, Turn

PROMPT_VERSION = "observe_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def observe_window(
    *,
    turns: list[Turn],
    learner_id: str,
    session_id: str,
    window_id: str,
    extract_model: str,
    turn_order_base: int,
    use_cache: bool = True,
    retries: int = 2,
) -> ObserveResult:
    from se.core.memory.online.llm import chat_json_for_model

    audit: dict[str, Any] = {"dropped": {}}
    if not turns:
        return ObserveResult([], "", "empty", 0.0, audit=audit)

    system = _load_prompt(PROMPT_VERSION)
    user = f"Dialogue window:\n{_format_turns(turns)}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    started = time.perf_counter()
    last_err = ""
    for attempt in range(retries + 1):
        try:
            data = chat_json_for_model(
                extract_model,
                messages,
                temperature=0.0,
                max_tokens=900,
                use_cache=use_cache,
                cache_kind="memory_online_observe_v1",
            )
            raw = data.get("observations") if isinstance(data, dict) else None
            if not isinstance(raw, list):
                raise ValueError("observations must be a list")
            dropped: dict[str, int] = {}
            kept: list[Observation] = []
            win_ids = _window_turn_ids(turns)
            student_ids = _student_turn_ids(turns)
            order = turn_order_base
            for row in raw:
                if not isinstance(row, dict):
                    dropped["not_dict"] = dropped.get("not_dict", 0) + 1
                    continue
                outcome = str(row.get("outcome") or "").strip().lower()
                if outcome not in OBS_OUTCOMES:
                    dropped["bad_outcome"] = dropped.get("bad_outcome", 0) + 1
                    continue
                content = str(row.get("content") or "").strip()
                if not content:
                    dropped["empty"] = dropped.get("empty", 0) + 1
                    continue
                content, _ = _normalize_student_subject(content)
                evidence = [_resolve_turn_id(str(x), turns) for x in (row.get("evidence_turn_ids") or [])]
                evidence = [e for e in evidence if e]
                if not evidence or any(e not in win_ids for e in evidence):
                    dropped["bad_evidence"] = dropped.get("bad_evidence", 0) + 1
                    continue
                if not any(e in student_ids for e in evidence):
                    dropped["no_student_evidence"] = dropped.get("no_student_evidence", 0) + 1
                    continue
                order += 1
                kept.append(
                    Observation(
                        obs_id=str(uuid.uuid4()),
                        learner_id=learner_id,
                        session_id=session_id,
                        topic_hint=str(row.get("topic_hint") or "").strip(),
                        outcome=outcome,
                        error_cause=str(row.get("error_cause") or "").strip(),
                        content=content[:160],
                        evidence_turn_ids=evidence,
                        turn_order=order,
                        independent=bool(row.get("independent", True)),
                        window_id=window_id,
                        created_at=_utc_now(),
                    )
                )
            summary = str(data.get("window_summary") or "").strip()
            latency_ms = (time.perf_counter() - started) * 1000.0
            audit["dropped"] = dropped
            audit["kept_count"] = len(kept)
            return ObserveResult(kept, summary, "ok", latency_ms, audit=audit)
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                time.sleep(1.0 if attempt == 0 else 3.0)
                continue
            break
    latency_ms = (time.perf_counter() - started) * 1000.0
    return ObserveResult([], "", "observe_failed", latency_ms, error=last_err, audit=audit)
