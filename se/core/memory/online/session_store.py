"""Persist per-session turn buffers and extract counters across process restarts."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from se.core.memory.online.models import Turn

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_name(name: str) -> str:
    return _SAFE.sub("_", name)[:120] or "session"


def _learner_root(store_dir: Path, learner_id: str) -> Path:
    digest = hashlib.sha1(learner_id.encode("utf-8")).hexdigest()
    return Path(store_dir) / digest / "sessions"


def session_path(store_dir: Path, learner_id: str, session_id: str) -> Path:
    return _learner_root(store_dir, learner_id) / f"{_safe_name(session_id)}.json"


def load_session(store_dir: Path, learner_id: str, session_id: str) -> dict[str, Any] | None:
    path = session_path(store_dir, learner_id, session_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(store_dir: Path, learner_id: str, session_id: str, payload: dict[str, Any]) -> None:
    path = session_path(store_dir, learner_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def payload_from_state(
    *,
    turns: list[Turn],
    extracted_upto: int,
    student_since: int,
    question_text: str,
    window_summary: str,
    msg_ids: set[str],
    last_activity_ts: float,
) -> dict[str, Any]:
    return {
        "session_id": turns[0].session_id if turns else "",
        "turns": [t.to_dict() for t in turns],
        "extracted_upto": extracted_upto,
        "student_since": student_since,
        "question_text": question_text,
        "window_summary": window_summary,
        "msg_ids": sorted(msg_ids),
        "last_activity_ts": last_activity_ts,
    }


def turns_from_payload(data: dict[str, Any], session_id: str) -> list[Turn]:
    out = []
    for row in data.get("turns") or []:
        out.append(
            Turn(
                turn_id=str(row.get("turn_id") or ""),
                session_id=str(row.get("session_id") or session_id),
                role=str(row.get("role") or ""),
                text=str(row.get("text") or ""),
                ts=str(row.get("ts") or ""),
                is_context=bool(row.get("is_context")),
            )
        )
    return out
