"""Topic key resolution: open labels or course KG nodes."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol

_NORM = re.compile(r"\s+")


def _norm_topic_key(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]


class KpResolver(Protocol):
    def resolve(self, topic_hint: str, context: str = "") -> tuple[str, str | None]: ...


class OpenTopicResolver:
    """Learner-internal open topic keys from LLM topic_hint."""

    def __init__(self, embedder: Any | None = None, merge_cos: float = 0.88) -> None:
        self.embedder = embedder
        self.merge_cos = merge_cos
        self._keys: dict[str, str] = {}
        self._vecs: dict[str, Any] = {}

    def resolve(self, topic_hint: str, context: str = "") -> tuple[str, str | None]:
        hint = (topic_hint or context or "general").strip()
        label = hint[:80]
        if self.embedder is not None and hint:
            import numpy as np

            vec = np.asarray(self.embedder.embed_query(hint), dtype=np.float32)
            for key, prev in self._vecs.items():
                denom = float(np.linalg.norm(vec) * np.linalg.norm(prev))
                cos = float(np.dot(vec, prev) / denom) if denom > 0 else 0.0
                if cos >= self.merge_cos:
                    return key, None
            key = _norm_topic_key(label)
            self._vecs[key] = vec
            self._keys[key] = label
            return key, None
        key = _norm_topic_key(label)
        self._keys.setdefault(key, label)
        return key, None

    def label_for(self, topic_key: str) -> str:
        return self._keys.get(topic_key, topic_key)


class KgResolver:
    """Bind topic_hint to a course KG node when match is confident."""

    def __init__(self, course_id: str = "") -> None:
        from se.core.kg.kp_index import get_kp_index

        self.index = get_kp_index(course_id)
        self.open = OpenTopicResolver()

    def resolve(self, topic_hint: str, context: str = "") -> tuple[str, str | None]:
        text = f"{topic_hint} {context}".strip()
        matches = self.index.match_kp(text, top_k=1)
        if matches:
            node = matches[0]
            valid = self.index.validate_nodes([node])
            if valid:
                return _norm_topic_key(node), valid[0]
        return self.open.resolve(topic_hint, context)


def build_kp_resolver(cfg: dict[str, Any], embedder: Any | None = None) -> KpResolver:
    spec = str(cfg.get("kp_resolver") or "open").strip()
    if spec == "open":
        return OpenTopicResolver(embedder=embedder)
    if spec.startswith("kg:"):
        course = spec.split(":", 1)[1]
        return KgResolver(course_id=course)
    return OpenTopicResolver(embedder=embedder)
