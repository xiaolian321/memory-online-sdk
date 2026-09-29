"""Load online memory config."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = _PKG / "config" / "memory_online.json"


def load_memory_online_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG
    if not cfg_path.is_file():
        example = _PKG / "config" / "memory_online.example.json"
        if example.is_file():
            cfg_path = example
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data["_config_path"] = str(cfg_path)
    return data


def find_hf_snapshot(model_id: str) -> Path | None:
    slug = "models--" + model_id.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / slug / "snapshots"
    if not root.is_dir():
        return None
    found = [snap for snap in root.iterdir() if (snap / "config.json").is_file()]
    if not found:
        return None
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0]


def resolve_embedding_model_dir(cfg: dict[str, Any]) -> str:
    explicit = str(cfg.get("embedding_model_dir") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if (path / "config.json").is_file():
            return str(path.resolve())
    env_dir = os.environ.get("MEMORY_ONLINE_EMBEDDING_DIR", "").strip()
    if env_dir and (Path(env_dir) / "config.json").is_file():
        return str(Path(env_dir).resolve())
    model_id = str(cfg.get("embedding_model_id") or "").strip()
    snap = find_hf_snapshot(model_id) if model_id else None
    if snap is not None:
        return str(snap)
    raise FileNotFoundError(
        f"Embedding model not found: set embedding_model_dir or MEMORY_ONLINE_EMBEDDING_DIR "
        f"or HF cache for {model_id!r}."
    )


def repo_path(*parts: str) -> Path:
    return _PKG.joinpath(*parts)
