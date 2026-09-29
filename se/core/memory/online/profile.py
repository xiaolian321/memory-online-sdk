"""Session digest and cross-topic learner traits."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from se.core.memory.online.extract_window import _load_prompt
from se.core.memory.online.layered_store import LayeredStore
from se.core.memory.online.models import LearnerTrait, ProfileResult, SessionDigest, Turn

PROMPT_VERSION = "profile_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_cards(store: LayeredStore) -> str:
    lines = []
    for c in store.active_topics()[:25]:
        lines.append(f"- {c.card_id} {c.label} status={c.status} summary={c.summary}")
    return "\n".join(lines) or "(none)"


def _format_traits(store: LayeredStore) -> str:
    lines = [f"- {t.trait_id} {t.content}" for t in store.active_traits()]
    return "\n".join(lines) or "(none)"


def _format_session_turns(turns: list[Turn]) -> str:
    return "\n".join(f"{t.turn_id} {t.role}: {t.text[:200]}" for t in turns[-40:])


def _valid_trait_refs(store: LayeredStore, card_ids: list[str], obs_ids: list[str]) -> bool:
    cards = {c.card_id for c in store.active_topics()}
    obs = {o.obs_id for o in store.observations}
    ok_cards = sum(1 for x in card_ids if x in cards)
    ok_obs = sum(1 for x in obs_ids if x in obs)
    windows = {store.obs_by_id(x).window_id for x in obs_ids if store.obs_by_id(x)}
    if ok_cards >= 2:
        return True
    if ok_cards >= 1 and ok_obs >= 1:
        return True
    if len(windows) >= 2 and ok_obs >= 2:
        return True
    return ok_cards + ok_obs >= 2


def profile_session(
    *,
    store: LayeredStore,
    session_id: str,
    session_turns: list[Turn],
    extract_model: str,
    use_cache: bool = True,
    retries: int = 2,
) -> ProfileResult:
    from se.core.memory.online.llm import chat_json_for_model

    system = _load_prompt(PROMPT_VERSION)
    user = (
        f"Existing traits:\n{_format_traits(store)}\n\n"
        f"Topic cards:\n{_format_cards(store)}\n\n"
        f"Session turns:\n{_format_session_turns(session_turns)}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    started = time.perf_counter()
    last_err = ""
    audit: dict[str, int] = {}
    for attempt in range(retries + 1):
        try:
            data = chat_json_for_model(
                extract_model,
                messages,
                temperature=0.0,
                max_tokens=1000,
                use_cache=use_cache,
                cache_kind="memory_online_profile_v1",
            )
            summary = str(data.get("session_summary") or "").strip()
            ops = data.get("trait_operations") or []
            if summary:
                store.digests.append(
                    SessionDigest(
                        session_id=session_id,
                        learner_id=store.learner_id,
                        summary=summary[:500],
                        seq=len(store.digests) + 1,
                        created_at=_utc_now(),
                    )
                )
            if isinstance(ops, list):
                for raw in ops:
                    if not isinstance(raw, dict):
                        continue
                    op = str(raw.get("op") or "").lower()
                    content = str(raw.get("content") or "").strip()
                    card_ids = [str(x) for x in (raw.get("evidence_card_ids") or [])]
                    obs_ids = [str(x) for x in (raw.get("evidence_obs_ids") or [])]
                    if op == "create":
                        if not content or not _valid_trait_refs(store, card_ids, obs_ids):
                            audit["trait_rejected"] = audit.get("trait_rejected", 0) + 1
                            continue
                        store.traits.append(
                            LearnerTrait(
                                trait_id=str(uuid.uuid4()),
                                learner_id=store.learner_id,
                                content=content[:240],
                                evidence_card_ids=card_ids,
                                evidence_obs_ids=obs_ids,
                                created_at=_utc_now(),
                                updated_at=_utc_now(),
                            )
                        )
                        audit["trait_created"] = audit.get("trait_created", 0) + 1
                    elif op == "update":
                        tid = str(raw.get("trait_id") or "")
                        trait = next((t for t in store.traits if t.trait_id == tid), None)
                        if not trait or not content:
                            continue
                        if not _valid_trait_refs(store, card_ids or trait.evidence_card_ids, obs_ids or trait.evidence_obs_ids):
                            audit["trait_rejected"] = audit.get("trait_rejected", 0) + 1
                            continue
                        trait.content = content[:240]
                        if card_ids:
                            trait.evidence_card_ids = card_ids
                        if obs_ids:
                            trait.evidence_obs_ids = obs_ids
                        trait.updated_at = _utc_now()
                        audit["trait_updated"] = audit.get("trait_updated", 0) + 1
                    elif op == "retire":
                        tid = str(raw.get("trait_id") or "")
                        trait = next((t for t in store.traits if t.trait_id == tid), None)
                        if trait:
                            trait.status = "retired"
                            trait.updated_at = _utc_now()
                            audit["trait_retired"] = audit.get("trait_retired", 0) + 1
            store.persist()
            latency_ms = (time.perf_counter() - started) * 1000.0
            return ProfileResult(summary, ops if isinstance(ops, list) else [], "ok", latency_ms, audit=audit)
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                time.sleep(1.0 if attempt == 0 else 3.0)
                continue
            break
    latency_ms = (time.perf_counter() - started) * 1000.0
    return ProfileResult("", [], "profile_failed", latency_ms, error=last_err, audit=audit)
