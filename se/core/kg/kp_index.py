"""Course KG index — whitelist, BM25 match, prerequisite lookup."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
_DEFAULT_KG = REPO / "data" / "kg-former" / "CS2501-离散数学（图论与代数结构）"
GRAPH_THEORY_COURSE = Path(os.environ.get("MEMORY_ONLINE_KG_COURSE_DIR") or _DEFAULT_KG)


@lru_cache(maxsize=4)
def _load_course(course_dir: str) -> tuple[dict[str, dict], list[dict]]:
    base = Path(course_dir)
    ent = base / "entity_final.json"
    rel = base / "relations_final.json"
    if not ent.is_file():
        return {}, []
    entities = json.loads(ent.read_text(encoding="utf-8-sig"))
    relations = json.loads(rel.read_text(encoding="utf-8-sig")) if rel.is_file() else []
    by_name = {e["name"]: e for e in entities}
    return by_name, relations


class KpIndex:
    """In-memory KG node index for a single course directory."""

    def __init__(self, course_dir: Path | None = None) -> None:
        self.course_dir = Path(course_dir or GRAPH_THEORY_COURSE)
        self.entities, self.relations = _load_course(str(self.course_dir))
        self.names = list(self.entities.keys())
        self._prereq: dict[str, list[str]] = {}
        self._dependents: dict[str, list[str]] = {}
        for rel in self.relations:
            if rel.get("predicate") != "depend_on":
                continue
            subj = str(rel.get("subject") or "")
            obj = str(rel.get("object") or "")
            if subj not in self.entities or obj not in self.entities:
                continue
            self._prereq.setdefault(subj, []).append(obj)
            self._dependents.setdefault(obj, []).append(subj)

    def kp_whitelist(self) -> frozenset[str]:
        return frozenset(self.entities.keys())

    def definition(self, name: str) -> str:
        ent = self.entities.get(name) or {}
        return str(ent.get("definition") or "")

    def match_kp(self, text: str, *, top_k: int = 3) -> list[str]:
        """BM25 recall over KG node names + definitions."""
        if not text.strip() or not self.names:
            return []
        from se.core.memory.online.index import _Bm25, _minmax
        from se.core.memory.semantic_retrieval import tokenize

        index_texts = [f"{n} {self.definition(n)[:120]}" for n in self.names]
        bm25 = _Bm25(index_texts)
        scores = _minmax(bm25.score(text))
        ranked = sorted(range(len(self.names)), key=lambda i: scores[i], reverse=True)
        out: list[str] = []
        for i in ranked[: max(top_k, 1)]:
            if scores[i] <= 0 and out:
                break
            out.append(self.names[i])
        return out[:top_k]

    def prerequisites(self, node: str) -> list[str]:
        return list(self._prereq.get(node, []))

    def dependents(self, node: str) -> list[str]:
        return list(self._dependents.get(node, []))

    def validate_nodes(self, nodes: list[str]) -> list[str]:
        wl = self.kp_whitelist()
        return [n for n in nodes if n in wl]

    def to_dict(self) -> dict[str, Any]:
        return {
            "course_dir": str(self.course_dir),
            "node_count": len(self.entities),
            "relation_count": len(self.relations),
        }


@lru_cache(maxsize=1)
def get_default_kp_index() -> KpIndex:
    return KpIndex(GRAPH_THEORY_COURSE)


def match_kp(text: str, *, top_k: int = 3, course_id: str = "") -> list[str]:
    return get_kp_index(course_id).match_kp(text, top_k=top_k)


def prerequisites(node: str, *, course_id: str = "") -> list[str]:
    return get_kp_index(course_id).prerequisites(node)


def get_kp_index(course_id: str = ""):
    """Route KG index by course_id."""
    cid = (course_id or "").lower()
    if cid in {"elementary_math", "mathdial", "math", "bridge"}:
        from se.core.kg.math_index import get_math_kp_index

        return get_math_kp_index()
    return get_default_kp_index()
