"""
app/stores/feedback.py
──────────────────────────────────────────────────────────────────────────────
FeedbackStore — asyncpg-backed persistence for the `feedback_records` table.

Deduplication contract
──────────────────────
One feedback record is allowed per decision, identified by the uniqueness key:
  (condition_id, condition_version, entity, decision_timestamp)

create() checks for an existing record before INSERT and raises ConflictError
with a clear message. The DB unique constraint `uq_feedback_decision` is the
final guard against races — asyncpg.UniqueViolationError is also caught and
converted to ConflictError.

PII note
────────
The `note` field may contain free-text PII. It is stored as-is but must
NEVER be logged — not in debug, info, or error log calls. Structured log
fields in this file intentionally omit `note`.

Column ↔ field mapping
──────────────────────
DB column           Python field (FeedbackRecord)
──────────────────  ───────────────────────────────────────────────
feedback_id         feedback_id         (DB default gen_random_uuid)
condition_id        condition_id
condition_version   condition_version
entity              entity
decision_timestamp  timestamp           (renamed at the Python boundary)
feedback            feedback
note                note                (PII — never log)
recorded_at         recorded_at
"""
from __future__ import annotations

import logging

import asyncpg

from app.models.calibration import FeedbackRecord, FeedbackValue
from app.models.errors import ConflictError

log = logging.getLogger(__name__)


class FeedbackStore:
    """Async store for the `feedback_records` table."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── create ────────────────────────────────────────────────────────────────

    async def create(self, record: FeedbackRecord) -> FeedbackRecord:
        """
        Persist a feedback record and return it with feedback_id and recorded_at
        populated from DB defaults.

        Raises ConflictError if a record already exists for the uniqueness key
        (condition_id, condition_version, entity, timestamp).
        """
        # Pre-flight duplicate check for a cleaner error message than the DB
        # constraint violation would produce.
        existing = await self.find(
            condition_id=record.condition_id,
            condition_version=record.condition_version,
            entity=record.entity,
            timestamp=record.timestamp,
        )
        if existing is not None:
            raise ConflictError(
                "Feedback already submitted for this decision.",
                location="(condition_id, condition_version, entity, timestamp)",
            )

        try:
            row = await self._pool.fetchrow(
                """
                INSERT INTO feedback_records (
                    condition_id, condition_version, entity,
                    decision_timestamp, feedback, note
                )
                VALUES ($1, $2, $3, $4::timestamptz, $5, $6)
                RETURNING
                    feedback_id, condition_id, condition_version, entity,
                    decision_timestamp, feedback, note, recorded_at
                """,
                record.condition_id,
                record.condition_version,
                record.entity,
                record.timestamp,
                record.feedback.value,
                record.note,
            )
        except asyncpg.UniqueViolationError:
            # Race between the pre-flight check and the INSERT — treat as conflict.
            raise ConflictError(
                "Feedback already submitted for this decision.",
                location="(condition_id, condition_version, entity, timestamp)",
            )

        log.info(
            "feedback_created",
            extra={
                "feedback_id": row["feedback_id"],
                "condition_id": record.condition_id,
                "condition_version": record.condition_version,
                "entity": record.entity,
                "feedback": record.feedback.value,
                # note intentionally omitted — may contain PII
            },
        )
        return _row_to_record(row)

    # ── get_by_condition ──────────────────────────────────────────────────────

    async def get_by_condition(
        self,
        condition_id: str,
        version: str,
    ) -> list[FeedbackRecord]:
        """
        Return all feedback records for a condition version, oldest-first.

        Ordered by recorded_at ASC to preserve the chronological signal used
        by CalibrationService.derive_direction() when computing majority vote.
        """
        rows = await self._pool.fetch(
            """
            SELECT
                feedback_id, condition_id, condition_version, entity,
                decision_timestamp, feedback, note, recorded_at
            FROM feedback_records
            WHERE condition_id = $1
              AND condition_version = $2
            ORDER BY recorded_at ASC
            """,
            condition_id,
            version,
        )
        return [_row_to_record(r) for r in rows]

    # ── find ──────────────────────────────────────────────────────────────────

    async def find(
        self,
        condition_id: str,
        condition_version: str,
        entity: str,
        timestamp: str,
    ) -> FeedbackRecord | None:
        """
        Look up a feedback record by its uniqueness key.

        Returns None if no record exists. Used by create() for the pre-flight
        duplicate check, and by the feedback route to verify idempotency.
        """
        row = await self._pool.fetchrow(
            """
            SELECT
                feedback_id, condition_id, condition_version, entity,
                decision_timestamp, feedback, note, recorded_at
            FROM feedback_records
            WHERE condition_id = $1
              AND condition_version = $2
              AND entity = $3
              AND decision_timestamp = $4::timestamptz
            """,
            condition_id,
            condition_version,
            entity,
            timestamp,
        )
        return _row_to_record(row) if row else None


# ── Row mapping helper ────────────────────────────────────────────────────────

def _row_to_record(row: asyncpg.Record) -> FeedbackRecord:
    """
    Convert an asyncpg Record from feedback_records into a FeedbackRecord.

    decision_timestamp (DB column) maps to timestamp (Python field) — the
    rename exists because 'timestamp' is clearer at the API boundary.
    recorded_at is returned as an ISO 8601 string to match the model's str type.
    """
    return FeedbackRecord(
        feedback_id=row["feedback_id"],
        condition_id=row["condition_id"],
        condition_version=row["condition_version"],
        entity=row["entity"],
        timestamp=row["decision_timestamp"].isoformat(),
        feedback=FeedbackValue(row["feedback"]),
        note=row["note"],
        recorded_at=row["recorded_at"].isoformat(),
    )
