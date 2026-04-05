"""
tests/unit/test_task_authoring_v7.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for M-5 V7 additions to TaskAuthoringService:
  1. vocabulary_context validation (Hard Rule 1)
  2. concept_id shortcut (Hard Rule 2)
  3. return_reasoning trace (Hard Rule 3)
  4. backward compatibility (Hard Rule 4)

All tests use in-memory mock stores and MockLLMClient — no DB, no HTTP stack.
The registry's versions() delegates to the store, so we control known/unknown
concept_ids by pre-seeding the _MockDefinitionStore.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from app.models.concept import VocabularyContext
from app.models.errors import (
    ConceptNotFoundError,
    VocabularyContextTooLargeError,
    VocabularyMismatchError,
)
from app.models.result import DryRunResult
from app.models.task import (
    CreateTaskRequest,
    DeliveryConfig,
    DeliveryType,
    ReasoningTrace,
    Task,
)
from app.registry.definitions import DefinitionRegistry
from app.services.task_authoring import TaskAuthoringService
from tests.mocks.mock_llm_client import MockLLMClient

# ── Shared in-memory stores ────────────────────────────────────────────────────
# Copied from test_task_authoring_with_mock.py — same contract.

from app.models.concept import DefinitionResponse, SearchResult, VersionSummary
from app.models.errors import ConflictError, NotFoundError
from app.models.task import Namespace


class _MockDefinitionStore:
    """In-memory DefinitionStore."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DefinitionResponse] = {}
        self._bodies: dict[tuple[str, str], dict] = {}
        self._insert_order: list[tuple[str, str]] = []

    async def register(
        self,
        definition_id: str,
        version: str,
        definition_type: str,
        namespace: str,
        body: dict[str, Any],
        meaning_hash: str | None = None,
        ir_hash: str | None = None,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key in self._rows:
            raise ConflictError(
                f"Definition '{definition_id}' version '{version}' already registered.",
                location=f"{definition_id}:{version}",
            )
        ts = datetime.now(timezone.utc)
        response = DefinitionResponse(
            definition_id=definition_id,
            version=version,
            definition_type=definition_type,
            namespace=Namespace(namespace),
            meaning_hash=meaning_hash,
            ir_hash=ir_hash,
            deprecated=False,
            created_at=ts,
            updated_at=ts,
        )
        self._rows[key] = response
        self._bodies[key] = body
        self._insert_order.append(key)
        return response

    async def get(self, definition_id: str, version: str) -> dict[str, Any] | None:
        return self._bodies.get((definition_id, version))

    async def get_metadata(
        self, definition_id: str, version: str
    ) -> DefinitionResponse | None:
        return self._rows.get((definition_id, version))

    async def versions(self, definition_id: str) -> list[VersionSummary]:
        ordered = [k for k in reversed(self._insert_order) if k[0] == definition_id]
        return [
            VersionSummary(
                version=k[1],
                created_at=self._rows[k].created_at,
                deprecated=self._rows[k].deprecated,
            )
            for k in ordered
        ]

    async def list(
        self,
        definition_type: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchResult:
        items = list(self._rows.values())
        if definition_type:
            items = [i for i in items if i.definition_type == definition_type]
        return SearchResult(items=items[:limit], has_more=False, total_count=len(items))

    async def deprecate(
        self,
        definition_id: str,
        version: str,
        replacement_version: str | None,
        reason: str,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key not in self._rows:
            raise NotFoundError(
                f"Definition '{definition_id}' version '{version}' not found."
            )
        updated = self._rows[key].model_copy(update={"deprecated": True})
        self._rows[key] = updated
        return updated

    def seed_concept(self, concept_id: str, version: str = "1.0.0") -> None:
        """Pre-seed a concept version so registry.versions() finds it."""
        ts = datetime.now(timezone.utc)
        key = (concept_id, version)
        self._rows[key] = DefinitionResponse(
            definition_id=concept_id,
            version=version,
            definition_type="concept",
            namespace=Namespace("personal"),
            deprecated=False,
            created_at=ts,
            updated_at=ts,
        )
        self._bodies[key] = {"concept_id": concept_id, "version": version}
        self._insert_order.append(key)


class _MockTaskStore:
    """In-memory TaskStore."""

    def __init__(self) -> None:
        self._tasks: list[Task] = []

    async def create(self, task: Task) -> Task:
        task = task.model_copy(update={"task_id": f"task-{len(self._tasks) + 1:04d}"})
        self._tasks.append(task)
        return task

    async def get(self, task_id: str) -> Task | None:
        for t in self._tasks:
            if t.task_id == task_id:
                return t
        return None

    @property
    def last_task(self) -> Task | None:
        return self._tasks[-1] if self._tasks else None


class _MockContextStore:
    def __init__(self, active_context=None) -> None:
        self._context = active_context

    async def get_active(self):
        return self._context


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_service(
    *,
    mock_llm: MockLLMClient | None = None,
    def_store: _MockDefinitionStore | None = None,
) -> tuple[_MockTaskStore, _MockDefinitionStore, MockLLMClient, TaskAuthoringService]:
    """Return (task_store, def_store, mock_llm, service) wired together."""
    if mock_llm is None:
        mock_llm = MockLLMClient()
    if def_store is None:
        def_store = _MockDefinitionStore()
    task_store = _MockTaskStore()
    registry = DefinitionRegistry(store=def_store)
    service = TaskAuthoringService(
        task_store=task_store,
        definition_registry=registry,
        guardrails=None,
        llm_client=mock_llm,
        max_retries=1,
        context_store=_MockContextStore(None),
        guardrails_store=None,
    )
    return task_store, def_store, mock_llm, service


def _req(
    intent: str = "Alert when churn risk is high",
    *,
    vocabulary_context: VocabularyContext | None = None,
    concept_id: str | None = None,
    return_reasoning: bool = False,
    dry_run: bool = False,
) -> CreateTaskRequest:
    return CreateTaskRequest(
        intent=intent,
        entity_scope="account",
        delivery=DeliveryConfig(
            type=DeliveryType.WEBHOOK, endpoint="https://example.com/hook"
        ),
        vocabulary_context=vocabulary_context,
        concept_id=concept_id,
        return_reasoning=return_reasoning,
        dry_run=dry_run,
    )


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── 1. vocabulary_context validation ──────────────────────────────────────────

class TestVocabularyContextValidation:
    """Pre-LLM vocabulary_context checks (Hard Rule 1)."""

    def test_absent_vocabulary_context_no_error(self):
        """No vocabulary_context → no error (global fallback)."""
        _, _, _, service = _make_service()
        result = run(service.create_task(_req()))
        assert isinstance(result, Task)

    def test_both_empty_raises_vocabulary_mismatch(self):
        """Both lists empty → VocabularyMismatchError raised before LLM call."""
        _, _, mock_llm, service = _make_service()
        vc = VocabularyContext(available_concept_ids=[], available_condition_ids=[])
        with pytest.raises(VocabularyMismatchError):
            run(service.create_task(_req(vocabulary_context=vc)))
        # LLM must NOT have been called (check happens pre-LLM)
        assert mock_llm.call_count == 0

    def test_501_concept_ids_raises_too_large(self):
        """501 available_concept_ids → VocabularyContextTooLargeError.

        model_construct() bypasses Pydantic validation to test the service-layer
        cap check, which is defense-in-depth against callers that bypass the
        HTTP request boundary (e.g., internal callers).
        """
        _, _, mock_llm, service = _make_service()
        vc = VocabularyContext.model_construct(
            available_concept_ids=[f"c.concept_{i}" for i in range(501)],
            available_condition_ids=[],
        )
        with pytest.raises(VocabularyContextTooLargeError):
            run(service.create_task(_req(vocabulary_context=vc)))
        assert mock_llm.call_count == 0

    def test_501_condition_ids_raises_too_large(self):
        """501 available_condition_ids → VocabularyContextTooLargeError.

        model_construct() bypasses Pydantic validation to test the service-layer
        cap check.
        """
        _, _, mock_llm, service = _make_service()
        vc = VocabularyContext.model_construct(
            available_concept_ids=["c.one"],
            available_condition_ids=[f"cond.{i}" for i in range(501)],
        )
        with pytest.raises(VocabularyContextTooLargeError):
            run(service.create_task(_req(vocabulary_context=vc)))
        assert mock_llm.call_count == 0

    def test_500_plus_500_ids_is_valid(self):
        """Exactly 500 concept IDs + 500 condition IDs → no cap error."""
        _, _, mock_llm, service = _make_service()
        # MockLLMClient generates concept_ids like "mock.concept_churn_NNNN"
        # Build 500-element list that includes the expected concept_id pattern.
        # Since we can't know the suffix, include a wildcard by using all 500
        # slots for generic IDs and one that matches the mock output.
        mock_llm_instance = MockLLMClient()
        # Peek what ID will be generated (counter will be 1)
        mock_llm_instance._counter = 0
        expected_cid = "mock.concept_churn_0001"
        vc = VocabularyContext(
            available_concept_ids=[expected_cid] + [f"extra.{i}" for i in range(499)],
            available_condition_ids=[f"cond.{i}" for i in range(500)],
        )
        _, _, _, service = _make_service(mock_llm=mock_llm_instance)
        result = run(service.create_task(_req(vocabulary_context=vc)))
        assert isinstance(result, Task)

    def test_non_empty_concept_ids_restricts_selection(self):
        """Non-empty available_concept_ids → LLM context receives vocabulary_context."""
        _, _, mock_llm, service = _make_service()
        vc = VocabularyContext(
            available_concept_ids=["mock.concept_churn_0001"],
            available_condition_ids=[],
        )
        run(service.create_task(_req(vocabulary_context=vc)))
        # vocabulary_context should be injected into the LLM context
        assert "vocabulary_context" in mock_llm.last_context

    def test_none_match_intent_raises_vocabulary_mismatch(self):
        """Vocabulary present but LLM selects a concept outside the list → VocabularyMismatchError."""
        _, _, _, service = _make_service()
        # MockLLMClient for churn intent returns "mock.concept_churn_0001"
        # Provide a vocabulary that does NOT include that concept_id
        vc = VocabularyContext(
            available_concept_ids=["some.other_concept"],
            available_condition_ids=[],
        )
        with pytest.raises(VocabularyMismatchError):
            run(service.create_task(_req(vocabulary_context=vc)))

    def test_both_empty_error_checked_before_size_cap(self):
        """both-empty raises VocabularyMismatchError even though an empty list is size 0 < 500."""
        _, _, mock_llm, service = _make_service()
        vc = VocabularyContext(available_concept_ids=[], available_condition_ids=[])
        with pytest.raises(VocabularyMismatchError):
            run(service.create_task(_req(vocabulary_context=vc)))


# ── 2. concept_id shortcut ────────────────────────────────────────────────────

class TestConceptIdShortcut:
    """concept_id provided → CoR Steps 1&2 skipped (Hard Rule 2)."""

    def test_provided_concept_id_does_not_raise(self):
        """Valid pre-compiled concept_id → task created without error."""
        def_store = _MockDefinitionStore()
        def_store.seed_concept("loan.repayment_ratio", "1.0.0")
        task_store, _, _, service = _make_service(def_store=def_store)
        result = run(service.create_task(_req(concept_id="loan.repayment_ratio")))
        assert isinstance(result, Task)
        assert task_store.last_task is not None

    def test_task_uses_pre_compiled_concept_id(self):
        """Task.concept_id equals the pre-compiled value, not LLM-generated."""
        def_store = _MockDefinitionStore()
        def_store.seed_concept("loan.repayment_ratio", "1.0.0")
        task_store, _, _, service = _make_service(def_store=def_store)
        run(service.create_task(_req(concept_id="loan.repayment_ratio")))
        assert task_store.last_task.concept_id == "loan.repayment_ratio"

    def test_task_uses_pre_compiled_concept_version(self):
        """Task.concept_version is resolved from registry, not LLM-generated."""
        def_store = _MockDefinitionStore()
        def_store.seed_concept("loan.repayment_ratio", "1.0.0")
        task_store, _, _, service = _make_service(def_store=def_store)
        run(service.create_task(_req(concept_id="loan.repayment_ratio")))
        assert task_store.last_task.concept_version == "1.0.0"

    def test_unknown_concept_id_raises_concept_not_found(self):
        """concept_id not in registry → ConceptNotFoundError (HTTP 404)."""
        _, _, mock_llm, service = _make_service()
        with pytest.raises(ConceptNotFoundError) as exc_info:
            run(service.create_task(_req(concept_id="does.not_exist")))
        assert exc_info.value.http_status == 404
        # LLM must NOT have been called
        assert mock_llm.call_count == 0

    def test_absent_concept_id_runs_full_cor(self):
        """No concept_id → full CoR pipeline uses LLM-generated concept."""
        task_store, _, mock_llm, service = _make_service()
        run(service.create_task(_req()))
        assert mock_llm.call_count == 1
        # Task concept_id should be LLM-generated (prefixed with mock.)
        assert task_store.last_task.concept_id.startswith("mock.")

    def test_concept_not_found_raised_before_llm_call(self):
        """concept_id not found error is raised before the LLM is called."""
        _, _, mock_llm, service = _make_service()
        with pytest.raises(ConceptNotFoundError):
            run(service.create_task(_req(concept_id="does.not_exist")))
        assert mock_llm.call_count == 0

    def test_reasoning_trace_with_concept_id_has_skipped_steps(self):
        """return_reasoning=True + concept_id → Steps 1&2 are skipped."""
        def_store = _MockDefinitionStore()
        def_store.seed_concept("loan.repayment_ratio", "1.0.0")
        task_store, _, _, service = _make_service(def_store=def_store)
        result = run(
            service.create_task(
                _req(concept_id="loan.repayment_ratio", return_reasoning=True)
            )
        )
        assert isinstance(result, Task)
        trace = result.reasoning_trace
        assert trace is not None
        steps = {s.step_index: s for s in trace.steps}
        assert steps[1].outcome == "skipped"
        assert steps[2].outcome == "skipped"
        assert "pre-compiled" in steps[1].summary
        assert "pre-compiled" in steps[2].summary


# ── 3. return_reasoning ────────────────────────────────────────────────────────

class TestReturnReasoning:
    """return_reasoning flag controls trace presence (Hard Rule 3)."""

    def test_return_reasoning_true_attaches_trace(self):
        """return_reasoning=True → Task.reasoning_trace is present with 4 steps."""
        _, _, _, service = _make_service()
        result = run(service.create_task(_req(return_reasoning=True)))
        assert isinstance(result, Task)
        assert result.reasoning_trace is not None
        assert isinstance(result.reasoning_trace, ReasoningTrace)
        assert len(result.reasoning_trace.steps) == 4

    def test_return_reasoning_false_trace_is_absent(self):
        """return_reasoning=False (default) → reasoning_trace is None."""
        _, _, _, service = _make_service()
        result = run(service.create_task(_req(return_reasoning=False)))
        assert isinstance(result, Task)
        assert result.reasoning_trace is None

    def test_default_return_reasoning_is_false(self):
        """Default CreateTaskRequest has return_reasoning=False."""
        assert _req().return_reasoning is False

    def test_trace_has_four_steps_without_concept_id(self):
        """Full CoR → trace has 4 steps all with outcome='accepted'."""
        _, _, _, service = _make_service()
        result = run(service.create_task(_req(return_reasoning=True)))
        trace = result.reasoning_trace
        assert len(trace.steps) == 4
        step_indices = [s.step_index for s in trace.steps]
        assert step_indices == [1, 2, 3, 4]
        for step in trace.steps:
            assert step.outcome == "accepted"

    def test_trace_with_concept_id_steps_3_and_4_accepted(self):
        """concept_id path → Steps 3&4 are accepted even when 1&2 skipped."""
        def_store = _MockDefinitionStore()
        def_store.seed_concept("loan.repayment_ratio", "1.0.0")
        _, _, _, service = _make_service(def_store=def_store)
        result = run(
            service.create_task(
                _req(concept_id="loan.repayment_ratio", return_reasoning=True)
            )
        )
        trace = result.reasoning_trace
        steps = {s.step_index: s for s in trace.steps}
        assert steps[3].outcome == "accepted"
        assert steps[4].outcome == "accepted"


# ── 4. backward compatibility ──────────────────────────────────────────────────

class TestBackwardCompatibility:
    """Existing callers without V7 fields are unaffected (Hard Rule 4)."""

    def test_existing_call_without_v7_fields_creates_task(self):
        """Plain CreateTaskRequest (no V7 fields) → task created normally."""
        task_store, _, _, service = _make_service()
        req = CreateTaskRequest(
            intent="Alert when churn risk is high",
            entity_scope="account",
            delivery=DeliveryConfig(
                type=DeliveryType.WEBHOOK,
                endpoint="https://example.com/hook",
            ),
        )
        result = run(service.create_task(req))
        assert isinstance(result, Task)
        assert task_store.last_task is not None
        assert task_store.last_task.concept_id.startswith("mock.")

    def test_task_is_persisted_with_v7_absent(self):
        """Without V7 fields the task persists normally with task_id assigned."""
        task_store, _, _, service = _make_service()
        req = CreateTaskRequest(
            intent="Alert when active user rate drops",
            entity_scope="account",
            delivery=DeliveryConfig(
                type=DeliveryType.WEBHOOK,
                endpoint="https://example.com/hook",
            ),
        )
        result = run(service.create_task(req))
        assert result.task_id is not None
        assert result.reasoning_trace is None

    def test_dry_run_without_v7_fields_returns_dry_run_result(self):
        """dry_run=True without V7 fields → DryRunResult returned."""
        _, _, _, service = _make_service()
        req = CreateTaskRequest(
            intent="Alert when churn risk is high",
            entity_scope="account",
            delivery=DeliveryConfig(
                type=DeliveryType.WEBHOOK,
                endpoint="https://example.com/hook",
            ),
            dry_run=True,
        )
        result = run(service.create_task(req))
        assert isinstance(result, DryRunResult)
