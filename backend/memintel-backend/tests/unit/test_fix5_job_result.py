"""
tests/unit/test_fix5_job_result.py
──────────────────────────────────────────────────────────────────────────────
FIX 5 tests: _job_to_result() populates result and error from result_body /
error_body on completed / failed jobs.

Before FIX 5, _job_to_result() always returned result=None and error=None
regardless of the job's actual outcome, making GET /jobs/{id} useless for
completed and failed jobs.
"""
from __future__ import annotations

from app.api.routes.jobs import _job_to_result
from app.models.result import (
    ConceptOutputType,
    ConceptResult,
    DecisionResult,
    DecisionType,
    FullPipelineResult,
    Job,
    JobStatus,
)
from app.models.errors import ErrorResponse, ErrorType


# ── Helpers ────────────────────────────────────────────────────────────────────

def _concept_result_dict() -> dict:
    return {
        "value": 0.82,
        "type": "float",
        "entity": "user_42",
        "version": "1.0",
        "deterministic": True,
        "timestamp": "2024-01-15T09:00:00Z",
    }


def _full_pipeline_result_dict() -> dict:
    return {
        "result": _concept_result_dict(),
        "decision": {
            "value": True,
            "type": "boolean",
            "condition_id": "high_churn",
            "condition_version": "1.0",
            "entity": "user_42",
            "timestamp": "2024-01-15T09:00:00Z",
            "actions_triggered": [],
        },
        "dry_run": False,
        "entity": "user_42",
        "timestamp": "2024-01-15T09:00:00Z",
    }


def _error_dict() -> dict:
    return {
        "error": {
            "type": "not_found",
            "message": "Concept not found",
            "location": "concept_id",
        }
    }


def _make_job(
    status: JobStatus,
    result_body: dict | None = None,
    error_body: dict | None = None,
) -> Job:
    return Job(
        job_id="job-001",
        status=status,
        result_body=result_body,
        error_body=error_body,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_completed_job_with_full_pipeline_result_body_populates_result():
    """
    A completed job with a FullPipelineResult body must have result populated.
    """
    job = _make_job(
        status=JobStatus.COMPLETED,
        result_body=_full_pipeline_result_dict(),
    )

    job_result = _job_to_result(job)

    assert job_result.result is not None, (
        "result must be populated for a completed job with result_body"
    )
    assert isinstance(job_result.result, FullPipelineResult), (
        f"result must be a FullPipelineResult, got {type(job_result.result).__name__}"
    )
    assert job_result.error is None


def test_completed_job_with_concept_result_body_populates_result():
    """
    A completed job with a ConceptResult body must have result populated.
    """
    job = _make_job(
        status=JobStatus.COMPLETED,
        result_body=_concept_result_dict(),
    )

    job_result = _job_to_result(job)

    assert job_result.result is not None, (
        "result must be populated for a completed job with result_body"
    )
    # ConceptResult doesn't have 'decision' field, so FullPipelineResult fails
    # and falls back to ConceptResult.
    assert isinstance(job_result.result, (ConceptResult, FullPipelineResult)), (
        f"result must be ConceptResult or FullPipelineResult, got {type(job_result.result).__name__}"
    )
    assert job_result.error is None


def test_failed_job_populates_error():
    """
    A failed job with an error_body must have error populated and result=None.
    """
    job = _make_job(
        status=JobStatus.FAILED,
        error_body=_error_dict(),
    )

    job_result = _job_to_result(job)

    assert job_result.error is not None, (
        "error must be populated for a failed job with error_body"
    )
    assert isinstance(job_result.error, ErrorResponse), (
        f"error must be an ErrorResponse, got {type(job_result.error).__name__}"
    )
    assert job_result.result is None


def test_queued_job_has_null_result_and_error():
    """
    A queued job must have result=None and error=None regardless of any bodies.
    """
    job = _make_job(status=JobStatus.QUEUED)

    job_result = _job_to_result(job)

    assert job_result.result is None
    assert job_result.error is None


def test_completed_job_with_null_result_body_has_null_result():
    """
    A completed job with no result_body must still return result=None.
    """
    job = _make_job(status=JobStatus.COMPLETED, result_body=None)

    job_result = _job_to_result(job)

    assert job_result.result is None


def test_job_to_result_basic_fields_always_populated():
    """
    job_id, status, and poll_interval_seconds must always be present.
    """
    job = _make_job(status=JobStatus.RUNNING)
    job_result = _job_to_result(job)

    assert job_result.job_id == "job-001"
    assert job_result.status == JobStatus.RUNNING
    assert job_result.poll_interval_seconds == 2
