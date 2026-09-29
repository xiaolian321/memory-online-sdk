"""Elementary math KP index from Bridge TEKS-style labels."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
BRIDGE_REPLAY = REPO / "data" / "skill_eval" / "real_replay" / "decision_points.jsonl"
BRIDGE_RAW = REPO / "data" / "skill_eval" / "raw" / "bridge"

COURSE_ID = "elementary_math"


def _teks_from_label(label: str) -> str:
    return str(label or "").strip()


def _load_teks_nodes() -> dict[str, dict]:
    teks_re = re.compile(r"^\d+\.\d+[A-Z]\.")
    nodes: dict[str, dict] = {}
    if BRIDGE_REPLAY.exists():
        for line in BRIDGE_REPLAY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("source") or "") == "local_student":
                continue
            kp = _teks_from_label(str(row.get("kp_focus") or ""))
            if kp and teks_re.match(kp) and kp not in nodes:
                nodes[kp] = {"name": kp, "definition": f"TEKS standard: {kp}"}
    if BRIDGE_RAW.exists():
        for path in BRIDGE_RAW.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("kp_focus", "z_what", "topic"):
                    kp = _teks_from_label(str(row.get(key) or ""))
                    if re.match(r"^\d+\.\d+[A-Z]\.", kp) and kp not in nodes:
                        nodes[kp] = {"name": kp, "definition": f"TEKS standard: {kp}"}
    # MathDial-friendly generic nodes when TEKS match fails
    for name, definition in (
        ("arithmetic_reasoning", "Multi-step arithmetic word problems and number operations"),
        ("fractions_decimals", "Fractions, decimals, and place value"),
        ("measurement", "Measurement, units, and conversion"),
        ("algebraic_reasoning", "Patterns, equations, and proportional reasoning"),
    ):
        nodes.setdefault(name, {"name": name, "definition": definition})
    teks_nodes = [n for n in nodes if teks_re.match(n)]
    assert not any(re.search(r"[\u4e00-\u9fff]", n) for n in teks_nodes), "TEKS nodes must be ASCII-only"
    return nodes


class MathKpIndex:
    """Lightweight math KP whitelist (Bridge TEKS + generic fallbacks)."""

    def __init__(self) -> None:
        self.entities = _load_teks_nodes()
        self.names = sorted(self.entities.keys())

    def kp_whitelist(self) -> frozenset[str]:
        return frozenset(self.entities.keys())

    def definition(self, name: str) -> str:
        ent = self.entities.get(name) or {}
        return str(ent.get("definition") or "")

    def match_kp(self, text: str, *, top_k: int = 3) -> list[str]:
        if not text.strip():
            return []
        text_l = text.lower()
        scored: list[tuple[float, str]] = []
        for name in self.names:
            score = 0.0
            code = name.split(".")[0] if "." in name else ""
            if code and code in text_l:
                score += 0.3
            for tok in re.findall(r"[A-Za-z]{4,}", name):
                if tok.lower() in text_l:
                    score += 0.5
            for tok in ("fraction", "decimal", "divide", "multiply", "mile", "hour", "percent"):
                if tok in text_l and tok in name.lower():
                    score += 0.4
            if score > 0:
                scored.append((score, name))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            return [n for _, n in scored[:top_k]]
        # generic fallback by keyword
        if any(w in text_l for w in ("fraction", "decimal", "percent")):
            return ["fractions_decimals"][:top_k]
        if any(w in text_l for w in ("mile", "hour", "minute", "inch", "foot")):
            return ["measurement"][:top_k]
        return ["arithmetic_reasoning"][:top_k]

    def validate_nodes(self, nodes: list[str]) -> list[str]:
        wl = self.kp_whitelist()
        return [n for n in nodes if n in wl]

    def prerequisites(self, node: str) -> list[str]:
        return []

    def dependents(self, node: str) -> list[str]:
        return []

    def to_dict(self) -> dict[str, Any]:
        return {"course_id": COURSE_ID, "node_count": len(self.entities)}


@lru_cache(maxsize=1)
def get_math_kp_index() -> MathKpIndex:
    return MathKpIndex()
