"""LLM consolidation of observations into topic cards with code validation."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from se.core.memory.online.extract_window import _load_prompt
from se.core.memory.online.kp_resolver import KpResolver
from se.core.memory.online.layered_store import LayeredStore
from se.core.memory.online.models import (
    MisconceptionEntry,
    Observation,
    TOPIC_STATUSES,
    ConsolidateResult,
    TopicCard,
)

PROMPT_VERSION = "consolidate_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_cards(cards: list[TopicCard]) -> str:
    if not cards:
        return "(none)"
    lines = []
    for c in cards:
        active = [m.text for m in c.active_misconceptions]
        lines.append(
            f"- id={c.card_id} key={c.topic_key} status={c.status} label={c.label} "
            f"summary={c.summary} active_mis={active} note={c.note}"
        )
    return "\n".join(lines)


def _format_obs(obs: list[Observation]) -> str:
    if not obs:
        return "(none)"
    lines = []
    for o in obs:
        lines.append(
            f"- id={o.obs_id} topic_hint={o.topic_hint} outcome={o.outcome} "
            f"independent={o.independent} order={o.turn_order} content={o.content}"
        )
    return "\n".join(lines)


def _find_obs(store: LayeredStore, obs_id: str, new_obs: list[Observation]) -> Observation | None:
    o = store.obs_by_id(obs_id)
    if o is not None:
        return o
    for x in new_obs:
        if x.obs_id == obs_id:
            return x
    return None


def _last_incorrect_order(store: LayeredStore, card: TopicCard, new_obs: list[Observation]) -> int:
    last = 0
    for oid in card.evidence_obs_ids:
        o = _find_obs(store, oid, new_obs)
        if o and o.outcome == "incorrect":
            last = max(last, o.turn_order)
    for o in new_obs:
        if o.outcome == "incorrect":
            last = max(last, o.turn_order)
    return last


def _validate_mastered(
    store: LayeredStore,
    card: TopicCard,
    evidence_obs_ids: list[str],
    new_obs: list[Observation],
) -> bool:
    last_err = _last_incorrect_order(store, card, new_obs)
    ok_after = False
    for oid in evidence_obs_ids:
        o = _find_obs(store, oid, new_obs)
        if not o or o.outcome != "correct":
            continue
        if o.turn_order <= last_err and last_err > 0:
            continue
        if not o.independent:
            continue
        ok_after = True
        break
    if last_err == 0:
        return any(
            (o := _find_obs(store, oid, new_obs))
            and o.outcome == "correct"
            and o.independent
            for oid in evidence_obs_ids
        )
    return ok_after


def consolidate_window(
    *,
    store: LayeredStore,
    new_obs: list[Observation],
    related_cards: list[TopicCard],
    extract_model: str,
    kp_resolver: KpResolver,
    use_cache: bool = True,
    retries: int = 2,
) -> ConsolidateResult:
    from se.core.memory.online.llm import chat_json_for_model

    if not new_obs:
        return ConsolidateResult([], "empty", 0.0)

    for o in new_obs:
        key, kp_id = kp_resolver.resolve(o.topic_hint, o.content)
        o.topic_key = key
        if kp_id:
            o.kp_id = kp_id

    system = _load_prompt(PROMPT_VERSION)
    user = (
        f"Existing topic cards:\n{_format_cards(related_cards)}\n\n"
        f"New observations:\n{_format_obs(new_obs)}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    started = time.perf_counter()
    last_err = ""
    for attempt in range(retries + 1):
        try:
            data = chat_json_for_model(
                extract_model,
                messages,
                temperature=0.0,
                max_tokens=1200,
                use_cache=use_cache,
                cache_kind="memory_online_consolidate_v1",
            )
            ops = data.get("operations") if isinstance(data, dict) else None
            if not isinstance(ops, list):
                raise ValueError("operations must be a list")
            audit = apply_operations(
                store=store,
                operations=ops,
                new_obs=new_obs,
                kp_resolver=kp_resolver,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            store.append_log({"ops": ops, "audit": audit, "obs_ids": [o.obs_id for o in new_obs]})
            return ConsolidateResult(ops, "ok", latency_ms, audit=audit)
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                time.sleep(1.0 if attempt == 0 else 3.0)
                continue
            break
    latency_ms = (time.perf_counter() - started) * 1000.0
    return ConsolidateResult([], "consolidate_failed", latency_ms, error=last_err)


def _mis_from_raw(rows: Any) -> list[MisconceptionEntry]:
    out: list[MisconceptionEntry] = []
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        text = str(r.get("text") or "").strip()
        if text:
            out.append(
                MisconceptionEntry(
                    text=text[:200],
                    evidence_obs_ids=[str(x) for x in (r.get("evidence_obs_ids") or [])],
                )
            )
    return out


def apply_operations(
    *,
    store: LayeredStore,
    operations: list[Any],
    new_obs: list[Observation],
    kp_resolver: KpResolver,
) -> dict[str, int]:
    audit: dict[str, int] = {}
    obs_ids_new = {o.obs_id for o in new_obs}
    for raw in operations:
        if not isinstance(raw, dict):
            audit["bad_op"] = audit.get("bad_op", 0) + 1
            continue
        op = str(raw.get("op") or "").strip().lower()
        if op == "create":
            label = str(raw.get("label") or raw.get("topic_hint") or "").strip()
            if not label and new_obs:
                label = new_obs[0].topic_hint
            key, kp_id = kp_resolver.resolve(label, "")
            existing = store.topic_by_key(key)
            if existing:
                raw = {**raw, "card_id": existing.card_id}
                op = "update"
            else:
                status = str(raw.get("status") or "learning").lower()
                if status not in TOPIC_STATUSES:
                    status = "learning"
                if status == "mastered":
                    status = "learning"
                    audit["mastered_rejected"] = audit.get("mastered_rejected", 0) + 1
                card = TopicCard(
                    card_id=str(uuid.uuid4()),
                    learner_id=store.learner_id,
                    topic_key=key,
                    label=label or getattr(kp_resolver, "label_for", lambda k: k)(key),
                    kp_id=str(raw.get("kp_id") or kp_id or ""),
                    status=status,
                    summary=str(raw.get("summary") or "").strip(),
                    active_misconceptions=_mis_from_raw(raw.get("active_misconceptions")),
                    resolved_misconceptions=_mis_from_raw(raw.get("resolved_misconceptions")),
                    evidence_obs_ids=[x for x in (raw.get("evidence_obs_ids") or []) if x in obs_ids_new or store.obs_by_id(str(x))],
                    needs_review=bool(raw.get("needs_review")),
                    note=str(raw.get("note") or "")[:300],
                    last_obs_turn_order=max((o.turn_order for o in new_obs), default=0),
                    created_at=_utc_now(),
                    updated_at=_utc_now(),
                )
                store.topics.append(card)
                audit["created"] = audit.get("created", 0) + 1
                continue
        if op in {"update", "set_status"}:
            card_id = str(raw.get("card_id") or "")
            card = store.topic_by_id(card_id)
            if not card:
                audit["missing_card"] = audit.get("missing_card", 0) + 1
                continue
            status = str(raw.get("status") or card.status).lower()
            if status not in TOPIC_STATUSES:
                status = card.status
            ev = [str(x) for x in (raw.get("evidence_obs_ids") or [])]
            if status == "mastered" and not _validate_mastered(store, card, ev, new_obs):
                status = "learning"
                audit["mastered_rejected"] = audit.get("mastered_rejected", 0) + 1
            card.status = status
            if raw.get("summary"):
                card.summary = str(raw.get("summary"))[:300]
            if raw.get("active_misconceptions") is not None:
                card.active_misconceptions = _mis_from_raw(raw.get("active_misconceptions"))
            if raw.get("resolved_misconceptions") is not None:
                card.resolved_misconceptions = _mis_from_raw(raw.get("resolved_misconceptions"))
            if raw.get("note") is not None:
                card.note = str(raw.get("note") or "")[:300]
            card.needs_review = bool(raw.get("needs_review", card.needs_review))
            for oid in ev:
                if oid not in card.evidence_obs_ids:
                    card.evidence_obs_ids.append(oid)
            for o in new_obs:
                if o.topic_key == card.topic_key and o.obs_id not in card.evidence_obs_ids:
                    card.evidence_obs_ids.append(o.obs_id)
                if o.outcome == "incorrect" and card.status == "mastered":
                    card.needs_review = True
                    audit["mastered_regress_flagged"] = audit.get("mastered_regress_flagged", 0) + 1
            card.last_obs_turn_order = max(card.last_obs_turn_order, max((o.turn_order for o in new_obs), default=0))
            card.updated_at = _utc_now()
            audit["updated"] = audit.get("updated", 0) + 1
            continue
        if op == "merge":
            target = store.topic_by_id(str(raw.get("target_card_id") or ""))
            source = store.topic_by_id(str(raw.get("source_card_id") or ""))
            if not target or not source:
                audit["merge_failed"] = audit.get("merge_failed", 0) + 1
                continue
            target.evidence_obs_ids = list(dict.fromkeys(target.evidence_obs_ids + source.evidence_obs_ids))
            target.active_misconceptions.extend(source.active_misconceptions)
            target.resolved_misconceptions.extend(source.resolved_misconceptions)
            source.status = "merged"
            source.note = f"merged_into:{target.card_id}"
            audit["merged"] = audit.get("merged", 0) + 1
            continue
        if op == "resolve_misconception":
            card = store.topic_by_id(str(raw.get("card_id") or ""))
            if not card:
                continue
            text = str(raw.get("text") or "").strip()
            ev = [str(x) for x in (raw.get("evidence_obs_ids") or [])]
            moved = None
            for m in card.active_misconceptions:
                if m.text == text or not text:
                    moved = m
                    break
            if moved is None and card.active_misconceptions:
                moved = card.active_misconceptions.pop(0)
            elif moved:
                card.active_misconceptions.remove(moved)
            if moved:
                moved.evidence_obs_ids = list(dict.fromkeys(moved.evidence_obs_ids + ev))
                card.resolved_misconceptions.append(moved)
                audit["resolved_mc"] = audit.get("resolved_mc", 0) + 1
            continue
        audit["unknown_op"] = audit.get("unknown_op", 0) + 1

    # Attach orphan obs to cards by topic_key
    for o in new_obs:
        card = store.topic_by_key(o.topic_key)
        if card and o.obs_id not in card.evidence_obs_ids:
            card.evidence_obs_ids.append(o.obs_id)
            if o.outcome == "incorrect" and card.status == "mastered":
                card.needs_review = True
    store.append_observations(new_obs)
    store.persist()
    return audit


def pick_related_cards(store: LayeredStore, new_obs: list[Observation], limit: int = 8) -> list[TopicCard]:
    keys = {o.topic_key for o in new_obs if o.topic_key}
    related = [c for c in store.active_topics() if c.topic_key in keys]
    if len(related) >= limit:
        return related[:limit]
    rest = [c for c in store.active_topics() if c not in related]
    rest.sort(key=lambda c: c.last_obs_turn_order, reverse=True)
    related.extend(rest[: max(0, limit - len(related))])
    return related
