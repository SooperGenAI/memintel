"""
app/stores/decision.py
──────────────────────────────────────────────────────────────────────────────
DecisionStore — asyncpg-backed persistence for the `decisions` table.

Responsibilities
────────────────
1. record()             — INSERT one decision record; returns the assigned decision_id.
2. get()                — SELECT one decision record by decision_id.
3. list_for_entity()    — SELECT recent decisions for (entity_id, concept_id).

Recording is fire-and-forget — callers must wrap record() in try/except and
never propagate exceptions to the primary evaluation path.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from app.models.decision import DecisionRecord

log = logging.getLogger(__name__)

_TABLE = "decisions"


class DecisionStore:
    """Async store for the `decisions` table."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def record(self, decision: DecisionRecord) -> str:
        """
        INSERT a DecisionRecord and return the assigned decision_id (UUID string).

        Raises on DB error — callers must catch and log.
        """
        row = await self._pool.fetchrow(
            f"""
            INSERT INTO {_TABLE} (
                concept_id, concept_version,
                condition_id, condition_version,
                entity_id,
                fired,
                concept_value,
                threshold_applied,
                ir_hash,
                input_primitives,
                signal_errors,
                reason,
                action_ids_fired,
                dry_run
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7,
                $8::jsonb, $9,
                $10::jsonb, $11::jsonb,
                $12, $13::text[], $14
            )
            RETURNING decision_id::text
            """,
            decision.concept_id,
            decision.concept_version,
            decision.condition_id,
            decision.condition_version,
            decision.entity_id,
            decision.fired,
            decision.concept_value,
            json.dumps(decision.threshold_applied) if decision.threshold_applied is not None else None,
            decision.ir_hash,
            json.dumps(decision.input_primitives) if decision.input_primitives is not None else None,
            json.dumps(decision.signal_errors) if decision.signal_errors is not None else None,
            decision.reason,
            decision.action_ids_fired or [],
            decision.dry_run,
        )
        return str(row["decision_id"])

    async def get(self, decision_id: str) -> DecisionRecord | None:
        """
        Fetch one decision record by decision_id. Returns None when not found.
        """
        row = await self._pool.fetchrow(
            f"""
            SELECT
                decision_id::text,
                concept_id, concept_version,
                condition_id, condition_version,
                entity_id, evaluated_at,
                fired, concept_value,
                threshold_applied, ir_hash,
                input_primitives, signal_errors,
                reason, action_ids_fired, dry_run
            FROM {_TABLE}
            WHERE decision_id = $1::uuid
            """,
            decision_id,
        )
        if row is None:
            return None
        return _row_to_record(row)

    async def list_for_entity(
        self,
        entity_id: str,
        concept_id: str,
        limit: int = 50,
    ) -> list[DecisionRecord]:
        """
        Return the most recent decisions for (entity_id, concept_id), newest first.
        """
        rows = await self._pool.fetch(
            f"""
            SELECT
                decision_id::text,
                concept_id, concept_version,
                condition_id, condition_version,
                entity_id, evaluated_at,
                fired, concept_value,
                threshold_applied, ir_hash,
                input_primitives, signal_errors,
                reason, action_ids_fired, dry_run
            FROM {_TABLE}
            WHERE entity_id = $1 AND concept_id = $2
            ORDER BY evaluated_at DESC
            LIMIT $3
            """,
            entity_id,
            concept_id,
            limit,
        )
        return [_row_to_record(r) for r in rows]


# ── Row deserialisation ────────────────────────────────────────────────────────

def _parse_jsonb(raw: Any) -> Any:
    """Deserialise a JSONB column that may arrive as str or dict/list."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _row_to_record(row: Any) -> DecisionRecord:
    return DecisionRecord(
        decision_id=row["decision_id"],
        concept_id=row["concept_id"],
        concept_version=row["concept_version"],
        condition_id=row["condition_id"],
        condition_version=row["condition_version"],
        entity_id=row["entity_id"],
        evaluated_at=row.get("evaluated_at"),
        fired=row["fired"],
        concept_value=row.get("concept_value"),
        threshold_applied=_parse_jsonb(row.get("threshold_applied")),
        ir_hash=row.get("ir_hash"),
        input_primitives=_parse_jsonb(row.get("input_primitives")),
        signal_errors=_parse_jsonb(row.get("signal_errors")),
        reason=row.get("reason"),
        action_ids_fired=list(row.get("action_ids_fired") or []),
        dry_run=row["dry_run"],
    )
