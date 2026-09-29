"""Rule + prototype-vector intent gate. No LLM."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from se.core.memory.online.models import INTENT_TYPES

PROTOTYPE_PATH = Path(__file__).resolve().parent / "prompts" / "intent_prototypes.json"

_TASK_RE = re.compile(
    r"(\d|[+\-*/=×÷]|答案|\banswer\b|\bi got\b)",
    re.IGNORECASE,
)
_INSTRUCTION_RE = re.compile(
    r"(以后|from now on|\balways\b|每次)",
    re.IGNORECASE,
)
_CHITCHAT_EXACT = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "thx",
    "good morning",
    "good night",
    "see you later",
    "ok",
    "okay",
    "好的",
    "好",
    "fine",
    "great",
    "hello how are you",
    "hello how are you?",
    "how are you",
    "how are you?",
}


def rule_intent(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return "chitchat"
    if _TASK_RE.search(raw):
        return "task"
    if _INSTRUCTION_RE.search(raw):
        return "instruction"
    compact = re.sub(r"\s+", " ", raw).strip().lower().rstrip("!.")
    if compact in _CHITCHAT_EXACT:
        return "chitchat"
    if len(raw) <= 12 and compact in {"hi", "hello", "thanks", "thank you", "好的", "ok"}:
        return "chitchat"
    return None


class IntentGate:
    def __init__(self, embedder: Any, *, threshold: float = 0.3) -> None:
        self.embedder = embedder
        self.threshold = threshold
        data = json.loads(PROTOTYPE_PATH.read_text(encoding="utf-8"))
        self._labels: list[str] = []
        texts: list[str] = []
        for label, examples in data.items():
            if label not in INTENT_TYPES:
                continue
            for ex in examples:
                self._labels.append(label)
                texts.append(str(ex))
        self._vecs = embedder.embed_passages(texts) if texts else np.zeros((0, 0), dtype=np.float32)

    def classify(self, text: str, query_vec: np.ndarray | None = None) -> tuple[str, bool]:
        ruled = rule_intent(text)
        if ruled is not None:
            return ruled, ruled != "chitchat"
        if self._vecs.size == 0:
            return "task", True
        q = query_vec if query_vec is not None else self.embedder.embed_query(text)
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        sims = self._vecs @ q
        best = int(np.argmax(sims))
        score = float(sims[best])
        label = self._labels[best] if score >= self.threshold else "task"
        if label not in INTENT_TYPES:
            label = "task"
        return label, label != "chitchat"
