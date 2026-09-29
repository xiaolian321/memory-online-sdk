"""Data models for online memory service."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

MEMORY_TYPES = frozenset(
    {
        "misconception",
        "error_event",
        "knowledge_gap",
        "mastered",
        "preference",
        "profile",
    }
)
MEMORY_STATUSES = frozenset({"active", "resolved", "deleted"})
INTENT_TYPES = frozenset({"task", "concept", "instruction", "chitchat"})
OBS_OUTCOMES = frozenset({"correct", "incorrect", "unsure"})
TOPIC_STATUSES = frozenset({"struggling", "learning", "mastered"})
TRAIT_STATUSES = frozenset({"active", "retired"})
MEMORY_LAYERS = frozenset({"trait", "topic", "observation", "legacy"})


class Role(str, Enum):
    STUDENT = "student"
    TUTOR = "tutor"


@dataclass
class Turn:
    turn_id: str
    session_id: str
    role: str
    text: str
    ts: str = ""
    is_context: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryItem:
    item_id: str
    learner_id: str
    session_id: str
    type: str
    content: str
    topic: str = ""
    evidence_turn_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "active"
    created_turn: int = 0
    last_seen_turn: int = 0
    use_count: int = 1
    created_at: str = ""
    extractor_model: str = ""
    prompt_version: str = ""
    supersedes_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("supersedes_ids"):
            d.pop("supersedes_ids", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryItem:
        return cls(
            item_id=str(d["item_id"]),
            learner_id=str(d["learner_id"]),
            session_id=str(d.get("session_id") or ""),
            type=str(d.get("type") or ""),
            content=str(d.get("content") or ""),
            topic=str(d.get("topic") or ""),
            evidence_turn_ids=list(d.get("evidence_turn_ids") or []),
            confidence=float(d.get("confidence") or 0.0),
            status=str(d.get("status") or "active"),
            created_turn=int(d.get("created_turn") or 0),
            last_seen_turn=int(d.get("last_seen_turn") or 0),
            use_count=int(d.get("use_count") or 1),
            created_at=str(d.get("created_at") or ""),
            extractor_model=str(d.get("extractor_model") or ""),
            prompt_version=str(d.get("prompt_version") or ""),
            supersedes_ids=list(d.get("supersedes_ids") or []),
        )


@dataclass
class MisconceptionEntry:
    text: str
    evidence_obs_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MisconceptionEntry:
        return cls(
            text=str(d.get("text") or ""),
            evidence_obs_ids=list(d.get("evidence_obs_ids") or []),
        )


@dataclass
class Observation:
    obs_id: str
    learner_id: str
    session_id: str
    topic_hint: str
    topic_key: str = ""
    kp_id: str = ""
    outcome: str = "unsure"
    error_cause: str = ""
    content: str = ""
    evidence_turn_ids: list[str] = field(default_factory=list)
    turn_order: int = 0
    independent: bool = True
    window_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Observation:
        return cls(
            obs_id=str(d["obs_id"]),
            learner_id=str(d["learner_id"]),
            session_id=str(d.get("session_id") or ""),
            topic_hint=str(d.get("topic_hint") or ""),
            topic_key=str(d.get("topic_key") or ""),
            kp_id=str(d.get("kp_id") or ""),
            outcome=str(d.get("outcome") or "unsure"),
            error_cause=str(d.get("error_cause") or ""),
            content=str(d.get("content") or ""),
            evidence_turn_ids=list(d.get("evidence_turn_ids") or []),
            turn_order=int(d.get("turn_order") or 0),
            independent=bool(d.get("independent", True)),
            window_id=str(d.get("window_id") or ""),
            created_at=str(d.get("created_at") or ""),
        )


@dataclass
class TopicCard:
    card_id: str
    learner_id: str
    topic_key: str
    label: str = ""
    kp_id: str = ""
    status: str = "learning"
    summary: str = ""
    active_misconceptions: list[MisconceptionEntry] = field(default_factory=list)
    resolved_misconceptions: list[MisconceptionEntry] = field(default_factory=list)
    evidence_obs_ids: list[str] = field(default_factory=list)
    needs_review: bool = False
    note: str = ""
    last_obs_turn_order: int = 0
    use_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["active_misconceptions"] = [m.to_dict() for m in self.active_misconceptions]
        d["resolved_misconceptions"] = [m.to_dict() for m in self.resolved_misconceptions]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TopicCard:
        return cls(
            card_id=str(d["card_id"]),
            learner_id=str(d["learner_id"]),
            topic_key=str(d.get("topic_key") or ""),
            label=str(d.get("label") or ""),
            kp_id=str(d.get("kp_id") or ""),
            status=str(d.get("status") or "learning"),
            summary=str(d.get("summary") or ""),
            active_misconceptions=[
                MisconceptionEntry.from_dict(x) for x in (d.get("active_misconceptions") or [])
            ],
            resolved_misconceptions=[
                MisconceptionEntry.from_dict(x) for x in (d.get("resolved_misconceptions") or [])
            ],
            evidence_obs_ids=list(d.get("evidence_obs_ids") or []),
            needs_review=bool(d.get("needs_review")),
            note=str(d.get("note") or ""),
            last_obs_turn_order=int(d.get("last_obs_turn_order") or 0),
            use_count=int(d.get("use_count") or 0),
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
        )


@dataclass
class LearnerTrait:
    trait_id: str
    learner_id: str
    content: str
    evidence_card_ids: list[str] = field(default_factory=list)
    evidence_obs_ids: list[str] = field(default_factory=list)
    status: str = "active"
    use_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearnerTrait:
        return cls(
            trait_id=str(d["trait_id"]),
            learner_id=str(d["learner_id"]),
            content=str(d.get("content") or ""),
            evidence_card_ids=list(d.get("evidence_card_ids") or []),
            evidence_obs_ids=list(d.get("evidence_obs_ids") or []),
            status=str(d.get("status") or "active"),
            use_count=int(d.get("use_count") or 0),
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
        )


@dataclass
class SessionDigest:
    session_id: str
    learner_id: str
    summary: str
    seq: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionDigest:
        return cls(
            session_id=str(d.get("session_id") or ""),
            learner_id=str(d.get("learner_id") or ""),
            summary=str(d.get("summary") or ""),
            seq=int(d.get("seq") or 0),
            created_at=str(d.get("created_at") or ""),
        )


@dataclass
class ObserveResult:
    observations: list[Observation]
    window_summary: str
    status: str
    latency_ms: float
    error: str = ""
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidateResult:
    operations: list[dict[str, Any]]
    status: str
    latency_ms: float
    error: str = ""
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileResult:
    session_summary: str
    trait_ops: list[dict[str, Any]]
    status: str
    latency_ms: float
    error: str = ""
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayeredHit:
    layer: str
    ref_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class LayeredRetrievalResult:
    intent: str
    need_memory: bool
    traits: list[LayeredHit]
    topics: list[LayeredHit]
    observations: list[LayeredHit]
    prompt_block: str
    timings_ms: dict[str, float]
    n_candidates: int = 0


@dataclass
class ExtractResult:
    items: list[MemoryItem]
    window_summary: str
    status: str
    latency_ms: float
    error: str = ""
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    item: MemoryItem
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    intent: str
    need_memory: bool
    items: list[RetrievalHit]
    timings_ms: dict[str, float]
    n_candidates: int = 0
    prompt_block: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "need_memory": self.need_memory,
            "items": [
                {
                    "item": hit.item.to_dict(),
                    "score": hit.score,
                    "breakdown": hit.breakdown,
                }
                for hit in self.items
            ],
            "timings_ms": self.timings_ms,
            "n_candidates": self.n_candidates,
        }
