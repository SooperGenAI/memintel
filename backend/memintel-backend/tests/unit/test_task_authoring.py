"""
tests/unit/test_task_authoring.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for TaskAuthoringService.

Coverage:
  1. threshold_task.json creates a valid, version-pinned Task
  2. dry_run=True returns DryRunResult; nothing persisted
  3. Missing strategy.type in LLM output → immediate semantic_error
  4. LLM failure after MAX_RETRIES → semantic_error (HTTP 422)
  5. action_binding_failed when no action can be resolved
  6. Task is version-pinned (concept_id, condition_id immutable after create)

Test isolation: every test creates its own store / registry / service.
No shared mutable state between tests.
"""
from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app.llm.fixtures import LLMFixtureClient
from app.models.concept import ConceptDefinition, DefinitionResponse, SearchResult, VersionSummary
from app.models.errors import ConflictError, ErrorType, MemintelError, NotFoundError
from app.models.result import DryRunResult
from app.models.task import (
    CreateTaskRequest,
    DeliveryConfig,
    DeliveryType,
    IMMUTABLE_TASK_FIELDS,
    Namespace,
    Task,
    TaskStatus,
)
from app.registry.definitions import DefinitionRegistry

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "app" / "llm" / "fixtures"


# ── Mock stores ───────────────────────────────────────────────────────────────

class MockDefinitionStore:
    """
    In-memory DefinitionStore for testing.  Mirrors the one in test_registry.py.
    """

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

    @property
    def create_count(self) -> int:
        return len(self._insert_order)

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
        if namespace:
            items = [i for i in items if i.namespace.value == namespace]
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
            raise NotFoundError(f"Definition '{definition_id}' version '{version}' not found.")
        updated = self._rows[key].model_copy(update={"deprecated": True,
                                                      "replacement_version": replacement_version})
        self._rows[key] = updated
        return updated

    async def promote(
        self,
        definition_id: str,
        version: str,
        from_namespace: str,
        to_namespace: str,
        elevated_key: bool = False,
    ) -> DefinitionResponse:
        from app.models.errors import ErrorType
        if to_namespace == "global" and not elevated_key:
            raise MemintelError(
                ErrorType.AUTH_ERROR,
                f"Promoting to '{to_namespace}' requires elevated privileges.",
            )
        key = (definition_id, version)
        if key not in self._rows:
            raise NotFoundError(f"Definition '{definition_id}' version '{version}' not found.")
        updated = self._rows[key].model_copy(update={"namespace": Namespace(to_namespace)})
        self._rows[key] = updated
        return updated


class MockTaskStore:
    """
    In-memory TaskStore for testing.

    create() records the Task and assigns a synthetic task_id.
    create_count tracks how many times create() was called (used in dry_run tests).
    """

    def __init__(self) -> None:
        self._tasks: list[Task] = []

    async def create(self, task: Task) -> Task:
        task = task.model_copy(update={"task_id": f"task-{len(self._tasks) + 1:04d}"})
        self._tasks.append(task)
        return task

    @property
    def create_count(self) -> int:
        return len(self._tasks)

    @property
    def last_task(self) -> Task | None:
        return self._tasks[-1] if self._tasks else None


# ── Broken LLM clients ────────────────────────────────────────────────────────

class NoStrategyLLMClient:
    """Returns threshold fixture with 'strategy' removed from condition."""

    def generate_task(self, intent: str, context: dict) -> dict:
        data = json.loads((_FIXTURES_DIR / "threshold_task.json").read_text(encoding="utf-8"))
        data["condition"].pop("strategy", None)
        return data


class NoStrategyTypeLLMClient:
    """Returns threshold fixture with strategy.type removed."""

    def generate_task(self, intent: str, context: dict) -> dict:
        data = json.loads((_FIXTURES_DIR / "threshold_task.json").read_text(encoding="utf-8"))
        data["condition"]["strategy"].pop("type", None)
        return data


class AlwaysFailingLLMClient:
    """Always raises RuntimeError — simulates an unavailable LLM service."""

    def generate_task(self, intent: str, context: dict) -> dict:
        raise RuntimeError("LLM service unavailable")


class NoActionLLMClient:
    """Returns threshold fixture with action replaced by an empty dict."""

    def generate_task(self, intent: str, context: dict) -> dict:
        data = json.loads((_FIXTURES_DIR / "threshold_task.json").read_text(encoding="utf-8"))
        data["action"] = {}
        return data


class IncompleteActionLLMClient:
    """Returns threshold fixture with action missing 'config' and 'trigger'."""

    def generate_task(self, intent: str, context: dict) -> dict:
        data = json.loads((_FIXTURES_DIR / "threshold_task.json").read_text(encoding="utf-8"))
        data["action"] = {"action_id": "org.notify_team", "version": "1.0"}
        # config and trigger are absent — binding should fail without delivery fallback
        return data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_service(
    llm_client: Any = None,
    max_retries: int = 3,
) -> tuple[MockTaskStore, MockDefinitionStore, Any]:
    """Return (task_store, def_store, service) wired together."""
    task_store = MockTaskStore()
    def_store  = MockDefinitionStore()
    registry   = DefinitionRegistry(store=def_store)
    from app.services.task_authoring import TaskAuthoringService
    service = TaskAuthoringService(
        task_store=task_store,
        definition_registry=registry,
        llm_client=llm_client or LLMFixtureClient(),
        max_retries=max_retries,
    )
    return task_store, def_store, service


def _notification_request(intent: str = "alert when churn risk is high") -> CreateTaskRequest:
    """Minimal valid CreateTaskRequest with a notification delivery."""
    return CreateTaskRequest(
        intent=intent,
        entity_scope="user",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="slack-alerts"),
    )


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── 1. threshold_task.json creates valid Task ─────────────────────────────────

class TestCreateTask:
    def test_threshold_task_creates_valid_task(self):
        task_store, _, service = _make_service()
        request = _notification_request()

        result = run(service.create_task(request))

        assert isinstance(result, Task)
        assert result.task_id is not None
        assert result.concept_id   == "org.churn_risk_score"
        assert result.condition_id == "org.high_churn_risk"
        assert result.action_id    == "org.notify_team"
        assert result.status       == TaskStatus.ACTIVE
        assert result.entity_scope == "user"
        assert result.intent       == "alert when churn risk is high"

    def test_task_stored_in_task_store(self):
        task_store, _, service = _make_service()
        run(service.create_task(_notification_request()))
        assert task_store.create_count == 1

    def test_definitions_registered_in_store(self):
        task_store, def_store, service = _make_service()
        run(service.create_task(_notification_request()))

        # Concept, condition, and action should all be in the definition store.
        assert def_store.create_count >= 3

    def test_z_score_fixture_creates_task(self):
        """LLMFixtureClient routes 'anomaly' intent to z_score fixture."""
        task_store, _, service = _make_service()
        request = CreateTaskRequest(
            intent="detect anomaly in payment failure rate",
            entity_scope="account",
            delivery=DeliveryConfig(type=DeliveryType.WEBHOOK, endpoint="https://hooks.example.com"),
        )
        result = run(service.create_task(request))

        assert isinstance(result, Task)
        assert result.concept_id   == "org.payment_failure_rate"
        assert result.condition_id == "org.payment_failure_anomaly"


# ── 2. dry_run returns DryRunResult, nothing persisted ───────────────────────

class TestDryRun:
    def test_dry_run_returns_dry_run_result(self):
        task_store, _, service = _make_service()
        request = CreateTaskRequest(
            intent="alert when churn risk is high",
            entity_scope="user",
            delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="slack-alerts"),
            dry_run=True,
        )

        result = run(service.create_task(request))

        assert isinstance(result, DryRunResult)

    def test_dry_run_does_not_call_task_store(self):
        task_store, _, service = _make_service()
        request = CreateTaskRequest(
            intent="alert when churn risk is high",
            entity_scope="user",
            delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="slack-alerts"),
            dry_run=True,
        )

        run(service.create_task(request))

        assert task_store.create_count == 0

    def test_dry_run_does_not_register_definitions(self):
        task_store, def_store, service = _make_service()
        request = CreateTaskRequest(
            intent="alert when churn risk is high",
            entity_scope="user",
            delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="slack-alerts"),
            dry_run=True,
        )

        run(service.create_task(request))

        assert def_store.create_count == 0

    def test_dry_run_result_has_concept_and_condition(self):
        _, _, service = _make_service()
        request = CreateTaskRequest(
            intent="alert when churn risk is high",
            entity_scope="user",
            delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="slack-alerts"),
            dry_run=True,
        )

        result = run(service.create_task(request))

        assert isinstance(result.concept, dict)
        assert result.concept.get("concept_id") == "org.churn_risk_score"
        assert result.action_id    == "org.notify_team"
        assert result.action_version == "1.0"
        assert result.validation.valid is True

    def test_dry_run_has_no_task_id(self):
        """DryRunResult carries no task_id — it is not a Task."""
        _, _, service = _make_service()
        request = CreateTaskRequest(
            intent="alert when churn risk is high",
            entity_scope="user",
            delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="slack-alerts"),
            dry_run=True,
        )

        result = run(service.create_task(request))

        # DryRunResult has no task_id field.
        assert not hasattr(result, "task_id") or getattr(result, "task_id", None) is None


# ── 3. Missing strategy → immediate semantic_error ────────────────────────────

class TestStrategyValidation:
    def test_missing_strategy_raises_semantic_error(self):
        _, _, service = _make_service(llm_client=NoStrategyLLMClient())
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_missing_strategy_type_raises_semantic_error(self):
        _, _, service = _make_service(llm_client=NoStrategyTypeLLMClient())
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_missing_strategy_error_message_mentions_strategy(self):
        _, _, service = _make_service(llm_client=NoStrategyLLMClient())
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert "strategy" in exc_info.value.message.lower()

    def test_missing_strategy_location_is_condition_strategy(self):
        _, _, service = _make_service(llm_client=NoStrategyLLMClient())
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert "condition.strategy" in (exc_info.value.location or "")

    def test_missing_strategy_does_not_persist(self):
        task_store, def_store, service = _make_service(llm_client=NoStrategyLLMClient())
        with pytest.raises(MemintelError):
            run(service.create_task(_notification_request()))
        assert task_store.create_count == 0
        assert def_store.create_count == 0


# ── 4. LLM failure after MAX_RETRIES → semantic_error ────────────────────────

class TestLLMRetry:
    def test_llm_failure_after_max_retries_raises_semantic_error(self):
        _, _, service = _make_service(
            llm_client=AlwaysFailingLLMClient(),
            max_retries=3,
        )
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR

    def test_llm_failure_message_includes_attempt_count(self):
        _, _, service = _make_service(
            llm_client=AlwaysFailingLLMClient(),
            max_retries=3,
        )
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert "3 attempt" in exc_info.value.message

    def test_llm_failure_message_includes_last_error(self):
        _, _, service = _make_service(
            llm_client=AlwaysFailingLLMClient(),
            max_retries=2,
        )
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert "LLM service unavailable" in exc_info.value.message

    def test_llm_failure_does_not_persist(self):
        task_store, def_store, service = _make_service(
            llm_client=AlwaysFailingLLMClient(),
            max_retries=2,
        )
        with pytest.raises(MemintelError):
            run(service.create_task(_notification_request()))
        assert task_store.create_count == 0
        assert def_store.create_count == 0

    def test_max_retries_1_fails_after_single_attempt(self):
        _, _, service = _make_service(
            llm_client=AlwaysFailingLLMClient(),
            max_retries=1,
        )
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert "1 attempt" in exc_info.value.message


# ── 5. action_binding_failed when no action resolves ────────────────────────

class TestActionBinding:
    def test_empty_action_raises_binding_failed(self):
        _, _, service = _make_service(llm_client=NoActionLLMClient())
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert exc_info.value.error_type == ErrorType.ACTION_BINDING_FAILED

    def test_incomplete_action_without_trigger_raises_binding_failed(self):
        """No trigger → system default cannot fill in the condition binding."""
        _, _, service = _make_service(llm_client=IncompleteActionLLMClient())
        # The delivery config IS present but action has no trigger → still fails.
        with pytest.raises(MemintelError) as exc_info:
            run(service.create_task(_notification_request()))
        assert exc_info.value.error_type == ErrorType.ACTION_BINDING_FAILED

    def test_action_binding_failed_does_not_persist(self):
        task_store, def_store, service = _make_service(llm_client=NoActionLLMClient())
        with pytest.raises(MemintelError):
            run(service.create_task(_notification_request()))
        assert task_store.create_count == 0

    def test_complete_action_from_llm_does_not_raise(self):
        """Sanity: threshold fixture has a complete action — no binding failure."""
        task_store, _, service = _make_service()
        result = run(service.create_task(_notification_request()))
        assert isinstance(result, Task)
        assert result.action_id == "org.notify_team"


# ── 6. Task is version-pinned ─────────────────────────────────────────────────

class TestVersionPinning:
    def test_task_has_concept_id_from_fixture(self):
        _, _, service = _make_service()
        task = run(service.create_task(_notification_request()))
        assert task.concept_id == "org.churn_risk_score"

    def test_task_has_concept_version_from_fixture(self):
        _, _, service = _make_service()
        task = run(service.create_task(_notification_request()))
        assert task.concept_version == "1.0"

    def test_task_has_condition_id_from_fixture(self):
        _, _, service = _make_service()
        task = run(service.create_task(_notification_request()))
        assert task.condition_id == "org.high_churn_risk"

    def test_task_has_condition_version_from_fixture(self):
        _, _, service = _make_service()
        task = run(service.create_task(_notification_request()))
        assert task.condition_version == "1.0"

    def test_task_has_action_id_from_fixture(self):
        _, _, service = _make_service()
        task = run(service.create_task(_notification_request()))
        assert task.action_id == "org.notify_team"

    def test_task_has_action_version_from_fixture(self):
        _, _, service = _make_service()
        task = run(service.create_task(_notification_request()))
        assert task.action_version == "1.0"

    def test_concept_id_is_in_immutable_fields(self):
        assert "concept_id" in IMMUTABLE_TASK_FIELDS

    def test_concept_version_is_in_immutable_fields(self):
        assert "concept_version" in IMMUTABLE_TASK_FIELDS

    def test_condition_id_is_in_immutable_fields(self):
        assert "condition_id" in IMMUTABLE_TASK_FIELDS

    def test_action_id_is_in_immutable_fields(self):
        assert "action_id" in IMMUTABLE_TASK_FIELDS

    def test_action_version_is_in_immutable_fields(self):
        assert "action_version" in IMMUTABLE_TASK_FIELDS
