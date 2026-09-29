"""Persistent L0/L1/L2 memory per learner."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from se.core.memory.online.models import (
    LearnerTrait,
    Observation,
    SessionDigest,
    TopicCard,
)


def _learner_dir(store_dir: Path, learner_id: str) -> Path:
    digest = hashlib.sha1(learner_id.encode("utf-8")).hexdigest()
    return Path(store_dir) / digest


class LayeredStore:
    def __init__(self, *, store_dir: Path, learner_id: str) -> None:
        self.store_dir = _learner_dir(store_dir, learner_id)
        self.learner_id = learner_id
        self.observations: list[Observation] = []
        self.topics: list[TopicCard] = []
        self.traits: list[LearnerTrait] = []
        self.digests: list[SessionDigest] = []
        self.consolidate_log: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._load()

    def _obs_path(self) -> Path:
        return self.store_dir / "observations.jsonl"

    def _topics_path(self) -> Path:
        return self.store_dir / "topics.jsonl"

    def _traits_path(self) -> Path:
        return self.store_dir / "traits.jsonl"

    def _digests_path(self) -> Path:
        return self.store_dir / "session_digests.jsonl"

    def _log_path(self) -> Path:
        return self.store_dir / "consolidate_log.jsonl"

    def _read_jsonl(self, path: Path, factory: Any) -> list[Any]:
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(factory.from_dict(json.loads(line)))
        return out

    def _write_jsonl(self, path: Path, rows: list[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    def _load(self) -> None:
        self.observations = self._read_jsonl(self._obs_path(), Observation)
        self.topics = self._read_jsonl(self._topics_path(), TopicCard)
        self.traits = self._read_jsonl(self._traits_path(), LearnerTrait)
        self.digests = self._read_jsonl(self._digests_path(), SessionDigest)
        if self._log_path().is_file():
            self.consolidate_log = [
                json.loads(line)
                for line in self._log_path().read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            self.consolidate_log = []

    def persist(self) -> None:
        with self._lock:
            self._write_jsonl(self._obs_path(), self.observations)
            self._write_jsonl(self._topics_path(), self.topics)
            self._write_jsonl(self._traits_path(), self.traits)
            self._write_jsonl(self._digests_path(), self.digests)
            self.store_dir.mkdir(parents=True, exist_ok=True)
            with self._log_path().open("w", encoding="utf-8") as f:
                for row in self.consolidate_log:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_observations(self, obs: list[Observation]) -> None:
        with self._lock:
            self.observations.extend(obs)

    def append_log(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.consolidate_log.append(entry)

    def topic_by_id(self, card_id: str) -> TopicCard | None:
        for c in self.topics:
            if c.card_id == card_id:
                return c
        return None

    def topic_by_key(self, topic_key: str) -> TopicCard | None:
        for c in self.topics:
            if c.topic_key == topic_key and c.status != "merged":
                return c
        return None

    def obs_by_id(self, obs_id: str) -> Observation | None:
        for o in self.observations:
            if o.obs_id == obs_id:
                return o
        return None

    def active_topics(self) -> list[TopicCard]:
        return [c for c in self.topics if c.status != "merged"]

    def active_traits(self) -> list[LearnerTrait]:
        return [t for t in self.traits if t.status == "active"]

    def snapshot(self) -> dict[str, Any]:
        return {
            "observations": [o.to_dict() for o in self.observations],
            "topics": [c.to_dict() for c in self.topics],
            "traits": [t.to_dict() for t in self.traits],
            "digests": [d.to_dict() for d in self.digests],
            "consolidate_log": list(self.consolidate_log),
        }


class LayeredStoreManager:
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir)
        self._stores: dict[str, LayeredStore] = {}
        self._lock = threading.Lock()

    def get(self, learner_id: str) -> LayeredStore:
        with self._lock:
            st = self._stores.get(learner_id)
            if st is None:
                st = LayeredStore(store_dir=self.store_dir, learner_id=learner_id)
                self._stores[learner_id] = st
            return st
