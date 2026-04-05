"""
tests/unit/test_v7_models.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for all V7 Pydantic models (Session M-1).

Tests cover:
  - VocabularyContext per-list cap (500 IDs each list, independent)
  - ReasoningStep outcome Literal constraint
  - CompileToken.used default
  - CreateTaskRequest backward compatibility
  - CompileConceptResponse structure
  - V7 ErrorType enum values and HTTP status mappings
  - Typed V7 exception subclasses

Nothing here touches the database, service layer, or LLM.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.concept import VocabularyContext, MAX_VOCABULARY_IDS
from app.models.concept_compile import (
    CompileConceptRequest,
    CompileConceptResponse,
    CompiledConcept,
    CompileToken,
    CorStepEvent,
    CorCompleteEvent,
    CorErrorEvent,
    RegisterConceptRequest,
    RegisterConceptResponse,
    SignalBinding,
)
from app.models.errors import (
    ErrorType,
    CompileTokenConsumedError,
    CompileTokenExpiredError,
    CompileTokenNotFoundError,
    VocabularyContextTooLargeError,
    VocabularyMismatchError,
    http_status_for,
)
from app.models.task import (
    CreateTaskRequest,
    ReasoningStep,
    ReasoningTrace,
    Task,
    TaskStatus,
    DeliveryConfig,
    DeliveryType,
)
from datetime import datetime, timezone


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_delivery() -> DeliveryConfig:
    return DeliveryConfig(type=DeliveryType.WEBHOOK, endpoint="https://example.com/hook")


def _make_task() -> Task:
    return Task(
        intent="Alert when churn risk is high",
        concept_id="cpt_001",
        concept_version="1.0",
        condition_id="cnd_001",
        condition_version="1.0",
        action_id="act_001",
        action_version="1.0",
        entity_scope="users",
        delivery=_make_delivery(),
    )


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── VocabularyContext ─────────────────────────────────────────────────────────

class TestVocabularyContext:

    def test_501_concept_ids_raises(self):
        """501 concept IDs + 0 condition IDs → ValidationError (per-list cap)."""
        with pytest.raises(ValidationError) as exc_info:
            VocabularyContext(
                available_concept_ids=[f"cpt_{i}" for i in range(501)],
                available_condition_ids=[],
            )
        errors = exc_info.value.errors()
        assert any("available_concept_ids" in str(e) for e in errors)

    def test_501_condition_ids_raises(self):
        """0 concept IDs + 501 condition IDs → ValidationError (per-list cap)."""
        with pytest.raises(ValidationError) as exc_info:
            VocabularyContext(
                available_concept_ids=[],
                available_condition_ids=[f"cnd_{i}" for i in range(501)],
            )
        errors = exc_info.value.errors()
        assert any("available_condition_ids" in str(e) for e in errors)

    def test_499_plus_499_is_valid(self):
        """499 concept IDs + 499 condition IDs → valid (998 total, under per-list cap)."""
        vc = VocabularyContext(
            available_concept_ids=[f"cpt_{i}" for i in range(499)],
            available_condition_ids=[f"cnd_{i}" for i in range(499)],
        )
        assert len(vc.available_concept_ids) == 499
        assert len(vc.available_condition_ids) == 499

    def test_500_plus_500_is_valid(self):
        """500 concept IDs + 500 condition IDs → valid (at limit exactly)."""
        vc = VocabularyContext(
            available_concept_ids=[f"cpt_{i}" for i in range(500)],
            available_condition_ids=[f"cnd_{i}" for i in range(500)],
        )
        assert len(vc.available_concept_ids) == MAX_VOCABULARY_IDS
        assert len(vc.available_condition_ids) == MAX_VOCABULARY_IDS

    def test_empty_lists_are_valid_at_model_layer(self):
        """
        Empty lists (0 + 0) are valid at model layer.
        The semantic rejection (vocabulary_mismatch) happens in the service layer,
        not here. The model must not reject empty lists.
        """
        vc = VocabularyContext(
            available_concept_ids=[],
            available_condition_ids=[],
        )
        assert vc.available_concept_ids == []
        assert vc.available_condition_ids == []

    def test_cap_is_per_list_not_combined(self):
        """
        501 concept IDs + 0 condition IDs fails (concept list alone exceeded).
        499 concept IDs + 499 condition IDs passes (998 combined, each list under cap).
        """
        with pytest.raises(ValidationError):
            VocabularyContext(
                available_concept_ids=[f"cpt_{i}" for i in range(501)],
                available_condition_ids=[],
            )

        # The reverse: condition list exceeded, concept list fine
        with pytest.raises(ValidationError):
            VocabularyContext(
                available_concept_ids=[],
                available_condition_ids=[f"cnd_{i}" for i in range(501)],
            )


# ── ReasoningStep ─────────────────────────────────────────────────────────────

class TestReasoningStep:

    def test_outcome_accepted(self):
        step = ReasoningStep(
            step_index=1,
            label="Intent Parsing",
            summary="Parsed the user intent",
            outcome="accepted",
        )
        assert step.outcome == "accepted"

    def test_outcome_skipped(self):
        step = ReasoningStep(
            step_index=2,
            label="Concept Selection",
            summary="concept_id provided by caller — selection skipped",
            outcome="skipped",
        )
        assert step.outcome == "skipped"

    def test_outcome_failed(self):
        step = ReasoningStep(
            step_index=3,
            label="Condition Compilation",
            summary="No strategy matched",
            outcome="failed",
        )
        assert step.outcome == "failed"

    def test_invalid_outcome_raises(self):
        with pytest.raises(ValidationError):
            ReasoningStep(
                step_index=1,
                label="Intent Parsing",
                summary="...",
                outcome="invalid_outcome",
            )

    def test_candidates_defaults_to_none(self):
        step = ReasoningStep(
            step_index=1, label="Intent Parsing", summary="...", outcome="accepted"
        )
        assert step.candidates is None

    def test_candidates_can_be_set(self):
        step = ReasoningStep(
            step_index=2,
            label="Concept Selection",
            summary="Selected from vocabulary",
            candidates=["cpt_001", "cpt_002"],
            outcome="accepted",
        )
        assert step.candidates == ["cpt_001", "cpt_002"]


# ── ReasoningTrace ────────────────────────────────────────────────────────────

class TestReasoningTrace:

    def test_minimal_trace(self):
        trace = ReasoningTrace(steps=[])
        assert trace.steps == []
        assert trace.compilation_duration_ms is None

    def test_with_duration(self):
        trace = ReasoningTrace(
            steps=[
                ReasoningStep(step_index=1, label="Step", summary="...", outcome="accepted")
            ],
            compilation_duration_ms=1240,
        )
        assert trace.compilation_duration_ms == 1240


# ── CompileToken ──────────────────────────────────────────────────────────────

class TestCompileToken:

    def test_used_defaults_to_false(self):
        token = CompileToken(
            token_id="tok-uuid-001",
            token_string="opaque_token_abc",
            identifier="loan.repayment_ratio",
            ir_hash="sha256abcdef",
            expires_at=_utcnow(),
            created_at=_utcnow(),
        )
        assert token.used is False

    def test_used_can_be_set_true(self):
        token = CompileToken(
            token_id="tok-uuid-002",
            token_string="opaque_token_xyz",
            identifier="loan.repayment_ratio",
            ir_hash="sha256abcdef",
            expires_at=_utcnow(),
            used=True,
            created_at=_utcnow(),
        )
        assert token.used is True


# ── CreateTaskRequest backward compatibility ──────────────────────────────────

class TestCreateTaskRequestBackwardCompat:

    def test_existing_call_without_new_fields_still_validates(self):
        """
        A request with none of the V7 fields (concept_id, vocabulary_context,
        stream, return_reasoning) must still validate successfully.
        """
        req = CreateTaskRequest(
            intent="Alert when churn risk spikes",
            entity_scope="users",
            delivery=_make_delivery(),
        )
        assert req.concept_id is None
        assert req.vocabulary_context is None
        assert req.stream is False
        assert req.return_reasoning is False

    def test_vocabulary_context_with_empty_lists_is_valid(self):
        """
        vocabulary_context={available_concept_ids:[], available_condition_ids:[]}
        is valid at model layer. Empty-list semantic rejection is service-layer only.
        """
        req = CreateTaskRequest(
            intent="Alert when engagement drops",
            entity_scope="users",
            delivery=_make_delivery(),
            vocabulary_context=VocabularyContext(
                available_concept_ids=[],
                available_condition_ids=[],
            ),
        )
        assert req.vocabulary_context is not None
        assert req.vocabulary_context.available_concept_ids == []

    def test_concept_id_field_accepted(self):
        req = CreateTaskRequest(
            intent="Alert when loan is overdue",
            entity_scope="loans",
            delivery=_make_delivery(),
            concept_id="cpt_001",
        )
        assert req.concept_id == "cpt_001"

    def test_stream_and_return_reasoning_fields(self):
        req = CreateTaskRequest(
            intent="Alert when risk exceeds threshold",
            entity_scope="accounts",
            delivery=_make_delivery(),
            stream=True,
            return_reasoning=True,
        )
        assert req.stream is True
        assert req.return_reasoning is True

    def test_existing_dry_run_field_unaffected(self):
        req = CreateTaskRequest(
            intent="Test intent",
            entity_scope="users",
            delivery=_make_delivery(),
            dry_run=True,
        )
        assert req.dry_run is True


# ── Task reasoning_trace field ────────────────────────────────────────────────

class TestTaskReasoningTrace:

    def test_reasoning_trace_defaults_to_none(self):
        task = _make_task()
        assert task.reasoning_trace is None

    def test_reasoning_trace_can_be_set(self):
        trace = ReasoningTrace(
            steps=[
                ReasoningStep(step_index=1, label="Intent Parsing", summary="...", outcome="accepted"),
            ],
            compilation_duration_ms=500,
        )
        task = _make_task()
        task.reasoning_trace = trace
        assert task.reasoning_trace is not None
        assert len(task.reasoning_trace.steps) == 1


# ── CompileConceptResponse ────────────────────────────────────────────────────

class TestCompileConceptResponse:

    def _make_compiled_concept(self) -> CompiledConcept:
        return CompiledConcept(
            identifier="loan.repayment_ratio",
            output_type="float",
            formula_summary="payments_on_time / payments_due, 90-day rolling window",
            signal_bindings=[
                SignalBinding(signal_name="payments_on_time", role="numerator"),
                SignalBinding(signal_name="payments_due", role="denominator"),
            ],
        )

    def test_compiled_concept_is_nested_object(self):
        resp = CompileConceptResponse(
            compile_token="tok_abc",
            compiled_concept=self._make_compiled_concept(),
            expires_at=_utcnow(),
        )
        assert isinstance(resp.compiled_concept, CompiledConcept)
        assert len(resp.compiled_concept.signal_bindings) == 2

    def test_reasoning_trace_is_none_when_not_requested(self):
        """reasoning_trace defaults to None (absent) when return_reasoning=False."""
        resp = CompileConceptResponse(
            compile_token="tok_abc",
            compiled_concept=self._make_compiled_concept(),
            expires_at=_utcnow(),
        )
        assert resp.reasoning_trace is None

    def test_reasoning_trace_populated_when_requested(self):
        trace = ReasoningTrace(
            steps=[
                ReasoningStep(step_index=1, label="Interpret Concept Intent", summary="...", outcome="accepted"),
                ReasoningStep(step_index=2, label="Signal Identification", summary="...", outcome="accepted"),
                ReasoningStep(step_index=3, label="DAG Construction", summary="...", outcome="accepted"),
                ReasoningStep(step_index=4, label="Type Validation", summary="...", outcome="accepted"),
            ],
        )
        resp = CompileConceptResponse(
            compile_token="tok_xyz",
            compiled_concept=self._make_compiled_concept(),
            reasoning_trace=trace,
            expires_at=_utcnow(),
        )
        assert resp.reasoning_trace is not None
        assert len(resp.reasoning_trace.steps) == 4

    def test_signal_bindings_is_list(self):
        concept = self._make_compiled_concept()
        assert isinstance(concept.signal_bindings, list)
        assert concept.signal_bindings[0].signal_name == "payments_on_time"
        assert concept.signal_bindings[0].role == "numerator"


# ── CompileConceptRequest validators ─────────────────────────────────────────

class TestCompileConceptRequest:

    def test_valid_request(self):
        req = CompileConceptRequest(
            identifier="loan.repayment_ratio",
            description="Ratio of on-time payments",
            output_type="float",
            signal_names=["payments_on_time", "payments_due"],
        )
        assert req.output_type == "float"
        assert req.return_reasoning is False
        assert req.stream is False

    def test_empty_output_type_raises(self):
        with pytest.raises(ValidationError):
            CompileConceptRequest(
                identifier="test",
                description="Test concept",
                output_type="",
                signal_names=[],
            )

    def test_whitespace_output_type_raises(self):
        with pytest.raises(ValidationError):
            CompileConceptRequest(
                identifier="test",
                description="Test concept",
                output_type="   ",
                signal_names=[],
            )


# ── V7 ErrorType enum and HTTP status ─────────────────────────────────────────

class TestV7ErrorTypes:

    def test_vocabulary_mismatch_maps_to_422(self):
        assert http_status_for(ErrorType.VOCABULARY_MISMATCH) == 422

    def test_vocabulary_context_too_large_maps_to_422(self):
        assert http_status_for(ErrorType.VOCABULARY_CONTEXT_TOO_LARGE) == 422

    def test_compile_token_expired_maps_to_400(self):
        """
        CRITICAL: compile_token_expired must map to 400, NOT 410.
        Cross-team contract with Canvas — do not change this.
        """
        assert http_status_for(ErrorType.COMPILE_TOKEN_EXPIRED) == 400

    def test_compile_token_not_found_maps_to_404(self):
        assert http_status_for(ErrorType.COMPILE_TOKEN_NOT_FOUND) == 404

    def test_compile_token_consumed_maps_to_409(self):
        assert http_status_for(ErrorType.COMPILE_TOKEN_CONSUMED) == 409

    def test_error_type_values_are_snake_case(self):
        """All V7 error type string values must be snake_case."""
        v7_types = [
            ErrorType.VOCABULARY_MISMATCH,
            ErrorType.VOCABULARY_CONTEXT_TOO_LARGE,
            ErrorType.COMPILE_TOKEN_EXPIRED,
            ErrorType.COMPILE_TOKEN_NOT_FOUND,
            ErrorType.COMPILE_TOKEN_CONSUMED,
        ]
        for et in v7_types:
            assert et.value == et.value.lower(), f"{et} value is not lowercase"
            assert " " not in et.value, f"{et} value contains spaces"


# ── V7 typed exceptions ───────────────────────────────────────────────────────

class TestV7Exceptions:

    def test_vocabulary_mismatch_error_http_status(self):
        exc = VocabularyMismatchError()
        assert exc.http_status == 422
        assert exc.error_type == ErrorType.VOCABULARY_MISMATCH

    def test_vocabulary_context_too_large_http_status(self):
        exc = VocabularyContextTooLargeError()
        assert exc.http_status == 422
        assert exc.error_type == ErrorType.VOCABULARY_CONTEXT_TOO_LARGE

    def test_compile_token_expired_maps_to_400_not_410(self):
        """Verify the cross-team contract is enforced in the exception class."""
        exc = CompileTokenExpiredError()
        assert exc.http_status == 400
        assert exc.http_status != 410

    def test_compile_token_not_found_http_status(self):
        exc = CompileTokenNotFoundError()
        assert exc.http_status == 404

    def test_compile_token_consumed_http_status(self):
        exc = CompileTokenConsumedError()
        assert exc.http_status == 409


# ── SSE event models ──────────────────────────────────────────────────────────

class TestSSEEventModels:

    def test_cor_step_event(self):
        event = CorStepEvent(
            step_index=2,
            label="Concept Selection",
            summary="Selected loan.repayment_ratio from vocabulary",
            candidates=["loan.repayment_ratio", "loan.delinquency_score"],
            outcome="accepted",
        )
        assert event.step_index == 2
        assert len(event.candidates) == 2

    def test_cor_step_candidates_defaults_to_none(self):
        event = CorStepEvent(step_index=1, label="Intent Parsing", summary="...", outcome="accepted")
        assert event.candidates is None

    def test_cor_complete_all_none_defaults(self):
        event = CorCompleteEvent()
        assert event.task_id is None
        assert event.compile_token is None
        assert event.concept_id is None
        assert event.status is None

    def test_cor_complete_task_id(self):
        event = CorCompleteEvent(task_id="tsk_abc123", status="active")
        assert event.task_id == "tsk_abc123"
        assert event.compile_token is None

    def test_cor_complete_compile_token(self):
        event = CorCompleteEvent(compile_token="tok_xyz", status="success")
        assert event.compile_token == "tok_xyz"
        assert event.task_id is None

    def test_cor_error_event(self):
        event = CorErrorEvent(
            failure_reason="vocabulary_mismatch",
            failed_at_step=2,
            suggestion="Review the module vocabulary",
        )
        assert event.failure_reason == "vocabulary_mismatch"
        assert event.failed_at_step == 2

    def test_cor_error_optional_fields_default_none(self):
        event = CorErrorEvent(failure_reason="step_timed_out")
        assert event.failed_at_step is None
        assert event.suggestion is None
