"""Embedding wrapper for online memory retrieval."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import numpy as np

_LOCK = threading.Lock()
_MODEL = None
_MODEL_PATH: str | None = None
_DEVICE: str | None = None

_BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def _assert_cuda_device(device: str) -> None:
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            f"Online memory requires a CUDA device, got {device!r}. CPU/BM25 fallback is forbidden."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for online memory embeddings") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; online memory refuses CPU fallback.")
    idx = int(device.split(":", 1)[1]) if ":" in device else 0
    if idx >= torch.cuda.device_count():
        raise RuntimeError(
            f"{device} is not present (cuda device count={torch.cuda.device_count()})"
        )


def _resolve_model_path(model_id: str, model_dir: str) -> Path:
    explicit = (model_dir or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if (path / "config.json").is_file():
            return path
        raise FileNotFoundError(f"Embedding model dir missing config.json: {path}")
    from se.core.memory.online.config import find_hf_snapshot

    snap = find_hf_snapshot(model_id)
    if snap is not None:
        return snap
    raise FileNotFoundError(f"Embedding model not found for id={model_id!r}")


def _load_model(*, model_id: str, model_dir: str, device: str, use_fp16: bool):
    global _MODEL, _MODEL_PATH, _DEVICE
    _assert_cuda_device(device)
    path = _resolve_model_path(model_id, model_dir)
    path_str = str(path)
    with _LOCK:
        if _MODEL is not None and _MODEL_PATH == path_str and _DEVICE == device:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for online memory retrieval"
            ) from exc
        model = SentenceTransformer(path_str, device=device)
        if use_fp16:
            model.half()
        _MODEL = model
        _MODEL_PATH = path_str
        _DEVICE = device
        return _MODEL


def _is_e5(model_path: str) -> bool:
    return "e5" in model_path.lower()


def _is_bge(model_path: str) -> bool:
    return "bge" in model_path.lower()


def _prefix_query(text: str, model_path: str) -> str:
    t = (text or "").strip()
    if _is_e5(model_path):
        return t if t.startswith("query:") else f"query: {t}"
    if _is_bge(model_path) and t and not t.startswith(_BGE_QUERY_PREFIX):
        return f"{_BGE_QUERY_PREFIX}{t}"
    return t


def _prefix_passage(text: str, model_path: str) -> str:
    t = (text or "").strip()
    if _is_e5(model_path):
        return t if t.startswith("passage:") else f"passage: {t}"
    return t


class MemoryEmbedder:
    def __init__(
        self,
        *,
        model_id: str,
        model_dir: str,
        device: str = "cuda:6",
        use_fp16: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.model_dir = model_dir
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self._model_path = str(_resolve_model_path(model_id, model_dir))
        self._warmed = False
        self._dim = 0

    def warmup(self) -> None:
        if self._warmed:
            return
        vec = self.embed_query("warmup")
        self._dim = int(vec.shape[0])
        self._warmed = True

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        items = [str(t or "").strip() for t in texts]
        if not items:
            return np.zeros((0, self._dim), dtype=np.float32)
        model = _load_model(
            model_id=self.model_id,
            model_dir=self.model_dir,
            device=self.device,
            use_fp16=self.use_fp16,
        )
        raw = model.encode(
            items,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(raw, dtype=np.float32)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [_prefix_passage(t, self._model_path) for t in texts]
        return self._encode(prepared)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([_prefix_query(text, self._model_path)])[0]

    def info(self) -> dict[str, str | bool | int]:
        return {
            "model_id": self.model_id,
            "model_path": self._model_path,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "warmed": self._warmed,
            "dim": self._dim,
        }
