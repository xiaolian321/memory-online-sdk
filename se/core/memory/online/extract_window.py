"""Window-based memory extraction with closed-set validation."""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from se.core.memory.online.models import (
    MEMORY_TYPES,
    ExtractResult,
    MemoryItem,
    Turn,
)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT_VERSION = "extract_v2"

NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
ALLOWED_NAMES = frozenset({"Student", "Tutor"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_prompt(prompt_version: str) -> str:
    name = prompt_version if prompt_version.endswith(".txt") else f"{prompt_version}.txt"
    path = PROMPT_DIR / name
    if not path.is_file():
        path = PROMPT_DIR / "extract_v1.txt"
    return path.read_text(encoding="utf-8")


def _format_turns(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        tag = "context" if t.is_context else "window"
        lines.append(f"[{tag}] {t.turn_id} {t.role}: {t.text}")
    return "\n".join(lines)


def _format_existing(items: list[MemoryItem], *, include_ids: bool) -> str:
    if not items:
        return "(none)"
    lines = []
    for it in items[:20]:
        if include_ids:
            lines.append(f"- [id={it.item_id}][{it.type}] {it.content}")
        else:
            lines.append(f"- [{it.type}] {it.content}")
    return "\n".join(lines)


def _student_turn_ids(turns: list[Turn]) -> set[str]:
    return {t.turn_id for t in turns if t.role == "student" and not t.is_context}


def _window_turn_ids(turns: list[Turn]) -> set[str]:
    return {t.turn_id for t in turns if not t.is_context}


def _resolve_turn_id(raw: str, window_turns: list[Turn]) -> str:
    token = (raw or "").strip()
    if not token:
        return ""
    ids = [t.turn_id for t in window_turns if not t.is_context]
    if token in ids:
        return token
    compact = token.replace(" ", "")
    if compact in ids:
        return compact
    for tid in ids:
        if tid.endswith(f":{token}") or tid.endswith(f":t{token}") or tid == f"t{token}":
            return tid
        if token.endswith(tid) or tid.endswith(token):
            return tid
    return token


def _normalize_student_subject(content: str) -> tuple[str, bool]:
    """Replace capitalized personal names with Student."""
    changed = False
    out = content
    for match in NAME_RE.finditer(content):
        name = match.group(0)
        if name in ALLOWED_NAMES:
            continue
        out = out.replace(name, "Student")
        changed = True
    return out.strip(), changed


def _validate_item(
    raw: dict[str, Any],
    *,
    window_turns: list[Turn],
    learner_id: str,
    session_id: str,
    extractor_model: str,
    prompt_version: str,
    now_turn: int,
    audit: dict[str, int],
    existing_ids: set[str],
    enforce_student_subject: bool,
) -> MemoryItem | None:
    typ = str(raw.get("type") or "").strip().lower()
    if typ not in MEMORY_TYPES:
        audit["bad_type"] = audit.get("bad_type", 0) + 1
        return None
    scope = str(raw.get("scope") or "standing").strip().lower()
    if scope == "turn_only":
        audit["turn_only"] = audit.get("turn_only", 0) + 1
        return None
    content = str(raw.get("content") or "").strip()
    if not content:
        audit["empty_content"] = audit.get("empty_content", 0) + 1
        return None
    if enforce_student_subject:
        content, renamed = _normalize_student_subject(content)
        if renamed:
            audit["name_rewritten"] = audit.get("name_rewritten", 0) + 1
        if NAME_RE.search(content):
            for m in NAME_RE.finditer(content):
                if m.group(0) not in ALLOWED_NAMES:
                    audit["name_dropped"] = audit.get("name_dropped", 0) + 1
                    return None
    if len(content) > 200:
        content = content[:200]
        audit["truncated"] = audit.get("truncated", 0) + 1
    evidence = [_resolve_turn_id(str(x), window_turns) for x in (raw.get("evidence_turn_ids") or [])]
    evidence = [e for e in evidence if e]
    win_ids = _window_turn_ids(window_turns)
    student_ids = _student_turn_ids(window_turns)
    if not evidence:
        audit["no_evidence"] = audit.get("no_evidence", 0) + 1
        return None
    if any(e not in win_ids for e in evidence):
        audit["bad_evidence_turn"] = audit.get("bad_evidence_turn", 0) + 1
        return None
    if not any(e in student_ids for e in evidence):
        audit["no_student_evidence"] = audit.get("no_student_evidence", 0) + 1
        return None
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    supersedes = [
        str(x).strip()
        for x in (raw.get("supersedes") or [])
        if str(x).strip() in existing_ids
    ]
    return MemoryItem(
        item_id=str(uuid.uuid4()),
        learner_id=learner_id,
        session_id=session_id,
        type=typ,
        content=content,
        topic=str(raw.get("topic") or "").strip(),
        evidence_turn_ids=evidence,
        confidence=confidence,
        status="active",
        created_turn=now_turn,
        last_seen_turn=now_turn,
        use_count=1,
        created_at=_utc_now(),
        extractor_model=extractor_model,
        prompt_version=prompt_version,
        supersedes_ids=supersedes,
    )


def extract_window(
    *,
    turns: list[Turn],
    learner_id: str,
    session_id: str,
    existing_items: list[MemoryItem],
    window_summary_prev: str = "",
    question_text: str = "",
    extract_model: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    use_cache: bool = True,
    now_turn: int = 0,
    retries: int = 2,
) -> ExtractResult:
    from se.core.memory.online.llm import chat_json_for_model

    audit: dict[str, Any] = {"dropped": {}}
    if not turns:
        return ExtractResult([], "", "empty", 0.0, audit=audit)

    pv = prompt_version or DEFAULT_PROMPT_VERSION
    include_ids = pv.startswith("extract_v2")
    enforce_student = include_ids
    existing_ids = {it.item_id for it in existing_items}
    system = _load_prompt(pv)
    user = (
        f"Existing memories:\n{_format_existing(existing_items, include_ids=include_ids)}\n\n"
        f"Previous window summary:\n{window_summary_prev or '(none)'}\n\n"
        f"Question (optional):\n{question_text or '(none)'}\n\n"
        f"Dialogue window:\n{_format_turns(turns)}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    cache_kind = "memory_online_extract_v2" if include_ids else "memory_online_extract_v1"

    last_err = ""
    started = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            data = chat_json_for_model(
                extract_model,
                messages,
                temperature=0.0,
                max_tokens=800,
                use_cache=use_cache,
                cache_kind=cache_kind,
            )
            raw_items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(raw_items, list):
                raise ValueError("items must be a list")
            dropped: dict[str, int] = {}
            kept: list[MemoryItem] = []
            for raw in raw_items:
                if not isinstance(raw, dict):
                    dropped["not_dict"] = dropped.get("not_dict", 0) + 1
                    continue
                item = _validate_item(
                    raw,
                    window_turns=turns,
                    learner_id=learner_id,
                    session_id=session_id,
                    extractor_model=extract_model,
                    prompt_version=pv,
                    now_turn=now_turn,
                    audit=dropped,
                    existing_ids=existing_ids,
                    enforce_student_subject=enforce_student,
                )
                if item is not None:
                    kept.append(item)
            summary = str(data.get("window_summary") or "").strip()
            latency_ms = (time.perf_counter() - started) * 1000.0
            audit["dropped"] = dropped
            audit["raw_count"] = len(raw_items)
            audit["kept_count"] = len(kept)
            return ExtractResult(
                items=kept,
                window_summary=summary,
                status="ok",
                latency_ms=latency_ms,
                audit=audit,
            )
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                time.sleep(1.0 if attempt == 0 else 4.0)
                continue
            break
    latency_ms = (time.perf_counter() - started) * 1000.0
    return ExtractResult(
        items=[],
        window_summary=window_summary_prev,
        status="extract_failed",
        latency_ms=latency_ms,
        error=last_err,
        audit=audit,
    )
