"""
app/stores/job.py
──────────────────────────────────────────────────────────────────────────────
JobStore — asyncpg-backed persistence for the `jobs` table.

Status transition machine
─────────────────────────
Valid transitions (enforced in update_status() before every DB write):

  queued    → running    ✅
  queued    → cancelled  ✅
  running   → completed  ✅
  running   → failed     ✅
  running   → cancelled  ✅
  completed → *          ❌  terminal
  failed    → *          ❌  terminal
  cancelled → *          ❌  terminal

Any attempt to leave a terminal state raises ConflictError → HTTP 409.
The transition check uses VALID_JOB_TRANSITIONS from app.models.result, which
is the single source of truth for this machine — not redefined here.

Timestamp rules
───────────────
  started_at   — set on first transition to 'running'; never updated again.
  completed_at — set on transition to 'completed', 'failed', or 'cancelled'.
  updated_at   — set on every status change.

JSONB columns
─────────────
  request_body  — original ExecuteRequest dict; stored at enqueue time.
  result_body   — ConceptResult / FullPipelineResult dict; set on 'completed'.
  error_body    — ErrorResponse dict; set on 'failed'.

Column ↔ field mapping
──────────────────────
DB column        Python field (Job)
───────────────  ─────────────────────────────────────────────
job_id           job_id
job_type         job_type
status           status
request_body     request_body   (excluded from API)
result_body      result_body    (excluded from API)
error_body       error_body     (excluded from API)
poll_interval_s  poll_interval_seconds  (renamed at Python boundary)
enqueued_at      enqueued_at    (excluded from API)
started_at       started_at     (excluded from API)
completed_at     completed_at   (excluded from API)
updated_at       updated_at     (excluded from API)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from app.models.errors import ConflictError, ErrorType, MemintelError
from app.models.result import Job, JobStatus, VALID_JOB_TRANSITIONS

log = logging.getLogger(__name__)

#: Statuses that set completed_at when entered.
_COMPLETION_STATUSES: frozenset[JobStatus] = frozenset({
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
})


class JobStore:
    """Async store for the `jobs` table."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── enqueue ───────────────────────────────────────────────────────────────

    async def enqueue(self, request_body: dict[str, Any]) -> Job:
        """
        Create a new job with status='queued' and return it.

        request_body is the original ExecuteRequest (or equivalent) serialised
        as a dict. It is stored for audit and potential retry.
        """
        row = await self._pool.fetchrow(
            """
            INSERT INTO jobs (request_body)
            VALUES ($1)
            RETURNING
                job_id, job_type, status, request_body,
                result_body, error_body, poll_interval_s,
                enqueued_at, started_at, completed_at, updated_at
            """,
            json.dumps(request_body),
        )
        log.info(
            "job_enqueued",
            extra={"job_id": row["job_id"]},
        )
        return _row_to_job(row)

    # ── get ───────────────────────────────────────────────────────────────────

    async def get(self, job_id: str) -> Job | None:
        """Return a Job by job_id, or None if not found."""
        row = await self._pool.fetchrow(
            """
            SELECT
                job_id, job_type, status, request_body,
                result_body, error_body, poll_interval_s,
                enqueued_at, started_at, completed_at, updated_at
            FROM jobs
            WHERE job_id = $1
            """,
            job_id,
        )
        return _row_to_job(row) if row else None

    # ── update_status ─────────────────────────────────────────────────────────

    async def update_status(
        self,
        job_id: str,
        new_status: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> Job:
        """
        Transition a job to new_status, enforcing the state machine.

        Raises:
          MemintelError(NOT_FOUND) — job_id does not exist.
          ConflictError            — transition is invalid (e.g. terminal → any).

        Sets started_at on the first transition to 'running'.
        Sets completed_at on transition to 'completed', 'failed', or 'cancelled'.
        Stores result_body / error_body alongside the status update atomically.
        """
        job = await self.get(job_id)
        if job is None:
            raise MemintelError(
                ErrorType.NOT_FOUND,
                f"Job '{job_id}' not found.",
                location="job_id",
            )

        current = job.status
        target = JobStatus(new_status)

        if target not in VALID_JOB_TRANSITIONS.get(current, frozenset()):
            raise ConflictError(
                f"Cannot transition job '{job_id}' from '{current.value}' "
                f"to '{target.value}'.",
                location="status",
            )

        new_status_val = target.value
        result_json = json.dumps(result) if result is not None else None
        error_json = json.dumps(error) if error is not None else None

        row = await self._pool.fetchrow(
            """
            UPDATE jobs
            SET
                status       = $2,
                result_body  = COALESCE($3::jsonb, result_body),
                error_body   = COALESCE($4::jsonb, error_body),
                started_at   = CASE
                                 WHEN $2 = 'running' AND started_at IS NULL
                                 THEN NOW()
                                 ELSE started_at
                               END,
                completed_at = CASE
                                 WHEN $2 = ANY($5::text[])
                                 THEN NOW()
                                 ELSE completed_at
                               END,
                updated_at   = NOW()
            WHERE job_id = $1
            RETURNING
                job_id, job_type, status, request_body,
                result_body, error_body, poll_interval_s,
                enqueued_at, started_at, completed_at, updated_at
            """,
            job_id,
            new_status_val,
            result_json,
            error_json,
            [s.value for s in _COMPLETION_STATUSES],
        )

        log.info(
            "job_status_updated",
            extra={
                "job_id": job_id,
                "from_status": current.value,
                "to_status": new_status_val,
            },
        )
        return _row_to_job(row)

    # ── cancel ────────────────────────────────────────────────────────────────

    async def cancel(self, job_id: str) -> Job:
        """
        Cancel a job. Convenience wrapper around update_status('cancelled').

        Raises ConflictError if the job is already in a terminal state.
        """
        return await self.update_status(job_id, JobStatus.CANCELLED.value)


# ── Row mapping helper ────────────────────────────────────────────────────────

def _row_to_job(row: asyncpg.Record) -> Job:
    """
    Convert an asyncpg Record from the jobs table into a Job model.

    poll_interval_s (DB column) maps to poll_interval_seconds (Python field).
    JSONB columns (request_body, result_body, error_body) may be returned as
    dicts or strings depending on asyncpg codec registration — both handled.
    """
    def _parse_json(val: Any) -> Any:
        if val is None:
            return None
        return json.loads(val) if isinstance(val, str) else val

    return Job(
        job_id=row["job_id"],
        job_type=row["job_type"],
        status=JobStatus(row["status"]),
        poll_interval_seconds=row["poll_interval_s"],
        request_body=_parse_json(row["request_body"]),
        result_body=_parse_json(row["result_body"]),
        error_body=_parse_json(row["error_body"]),
        enqueued_at=row["enqueued_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        updated_at=row["updated_at"],
    )
