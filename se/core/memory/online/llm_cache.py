"""Disk cache for online-memory LLM calls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_ROOT = Path(os.environ.get("MEMORY_ONLINE_LLM_CACHE") or ".cache/memory_online_llm")


def cache_key(**parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_file(kind: str, key: str) -> Path:
    return CACHE_ROOT / kind / f"{key}.json"


def load_cached(kind: str, key: str) -> dict[str, Any] | None:
    path = cache_file(kind, key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_cached(kind: str, key: str, data: dict[str, Any]) -> Path:
    path = cache_file(kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
