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
import structlog
from datetime import datetime
from typing import Any

import asyncpg

from app.models.decision import DecisionRecord

log = structlog.get_logger(__name__)

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
            str(decision.concept_value) if decision.concept_value is not None else None,
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

    async def find_by_condition_entity_timestamp(
        self,
        condition_id: str,
        condition_version: str,
        entity_id: str,
        timestamp: str,
    ) -> "DecisionRecord | None":
        """
        Find a decision record by (condition_id, condition_version, entity_id, evaluated_at).
        Returns None when not found or when timestamp cannot be parsed.

        timestamp must be an ISO 8601 string; 'Z' suffix is treated as UTC.
        asyncpg requires a Python datetime — the string is parsed here.
        """
        try:
            ts: datetime = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

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
            WHERE condition_id = $1
              AND condition_version = $2
              AND entity_id = $3
              AND evaluated_at = $4
            """,
            condition_id,
            condition_version,
            entity_id,
            ts,
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


def _parse_concept_value(raw: str | None) -> bool | float | int | str | None:
    """
    Deserialise concept_value from the TEXT column.

    Serialisation contract (DecisionStore.record):
      None          → NULL
      True          → "True"
      False         → "False"
      float/int     → str(value)   e.g. "1.87", "42"
      str           → stored as-is e.g. "high_risk"

    Deserialisation order:
      1. NULL → None
      2. Try float() — covers "1.87", "42", "-0.5", etc.
      3. Check "True" / "False" — returns the corresponding bool
      4. Fall through → return the raw string unchanged
    """
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        if raw == "True":
            return True
        if raw == "False":
            return False
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
        concept_value=_parse_concept_value(row.get("concept_value")),
        threshold_applied=_parse_jsonb(row.get("threshold_applied")),
        ir_hash=row.get("ir_hash"),
        input_primitives=_parse_jsonb(row.get("input_primitives")),
        signal_errors=_parse_jsonb(row.get("signal_errors")),
        reason=row.get("reason"),
        action_ids_fired=list(row.get("action_ids_fired") or []),
        dry_run=row["dry_run"],
    )
