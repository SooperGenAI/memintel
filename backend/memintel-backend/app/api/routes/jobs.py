"""
app/api/routes/jobs.py
──────────────────────────────────────────────────────────────────────────────
Async job lifecycle endpoints.

Endpoints
─────────
  GET    /jobs/{job_id}    getJob      — poll job status and result
  DELETE /jobs/{job_id}    cancelJob   — cancel a queued or running job

Ownership rules
───────────────
Both endpoints are deterministic — no LLM involvement.

GET /jobs/{job_id}:
  Returns the current state of an async job created by POST /execute/async.
  Poll at the interval returned in poll_interval_seconds.

  result is populated only when status='completed'.
  error  is populated only when status='failed'.
  Both are None for queued, running, and cancelled states.

  HTTP 404 — job_id not found.

DELETE /jobs/{job_id}:
  Requests cancellation of a queued or running job.
  Returns the updated JobResult. Cancellation is best-effort:
    - queued  → immediately transitions to cancelled.
    - running → signals cancellation; the worker may still complete.
    - terminal (completed, failed, cancelled) → HTTP 409 (ConflictError).

  HTTP 404 — job_id not found.

Error handling
──────────────
MemintelError subclasses (NotFoundError, ConflictError) are caught globally
by the exception handler in main.py — routes do not catch them here.
"""
from __future__ import annotations

import structlog

import asyncpg
from fastapi import APIRouter, Depends

from app.models.errors import ErrorResponse, NotFoundError
from app.models.result import ConceptResult, FullPipelineResult, Job, JobResult, JobStatus
from app.persistence.db import get_db
from app.persistence.stores import get_job_store
from app.stores import JobStore

log = structlog.get_logger(__name__)

router = APIRouter(tags=["Jobs"])


def _job_to_result(job: Job) -> JobResult:
    """Convert a Job (internal store model) to a JobResult (API response model)."""
    result = None
    if job.status == JobStatus.COMPLETED and job.result_body is not None:
        try:
            # FullPipelineResult has a 'decision' field; try it first.
            result = FullPipelineResult.model_validate(job.result_body)
        except Exception:
            try:
                result = ConceptResult.model_validate(job.result_body)
            except Exception:
                log.warning("job_result_deserialization_failed", job_id=job.job_id, exc_info=True)

    error = None
    if job.status == JobStatus.FAILED and job.error_body is not None:
        try:
            error = ErrorResponse.model_validate(job.error_body)
        except Exception:
            log.warning("job_error_deserialization_failed", job_id=job.job_id, exc_info=True)

    return JobResult(
        job_id=job.job_id or "",
        status=job.status,
        poll_interval_seconds=job.poll_interval_seconds,
        result=result,
        error=error,
    )


# ── GET /jobs/{job_id} ─────────────────────────────────────────────────────────

@router.get(
    "/{job_id}",
    summary="Get async job status and result",
    response_model=JobResult,
    status_code=200,
)
async def get_job(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> JobResult:
    """
    Return the current status of an async job.

    Poll at the interval returned in poll_interval_seconds. Once the job
    reaches a terminal state (completed, failed, cancelled) no further
    transitions occur.

    result is populated only when status='completed'.
    error  is populated only when status='failed'.

    HTTP 404 — job not found.
    """
    log.info("get_job_request", job_id=job_id)
    job = await store.get(job_id)
    if job is None:
        raise NotFoundError(f"Job '{job_id}' not found.", location="job_id")
    return _job_to_result(job)


# ── DELETE /jobs/{job_id} ─────────────────────────────────────────────────────

@router.delete(
    "/{job_id}",
    summary="Cancel an async job",
    response_model=JobResult,
    status_code=200,
)
async def cancel_job(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> JobResult:
    """
    Request cancellation of a queued or running job.

    Cancellation is best-effort for running jobs — the worker may complete
    before the cancel signal is processed. Queued jobs are cancelled immediately.

    Terminal jobs (completed, failed, already cancelled) → HTTP 409.

    HTTP 404 — job not found.
    """
    log.info("cancel_job_request", job_id=job_id)
    job = await store.get(job_id)
    if job is None:
        raise NotFoundError(f"Job '{job_id}' not found.", location="job_id")
    cancelled = await store.cancel(job_id)
    return _job_to_result(cancelled)
