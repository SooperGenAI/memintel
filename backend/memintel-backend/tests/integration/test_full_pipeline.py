"""
tests/integration/test_full_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Integration tests — full task lifecycle using LLM fixtures.

Coverage:
  1. test_threshold_task_full_pipeline  — complete lifecycle with threshold fixture
  2. test_z_score_task_full_pipeline    — complete lifecycle with z_score fixture
  3. test_equals_task_full_pipeline     — complete lifecycle with equals fixture
  4. test_composite_task_full_pipeline  — complete lifecycle with composite fixture
  5. test_determinism_harness           — same inputs 3× → identical results
  6. test_error_injection               — type_error, semantic_error, execution_error,
                                         graph_error
  7. test_dry_run_propagation           — DryRunResult when dry_run=True; all actions
                                         'would_trigger' in execute dry_run
  8. test_definition_immutability       — same (id, version) → ConflictError (HTTP 409)

Test isolation: each test builds its own stores and service instances.
No shared mutable state between tests.
No database, LLM, or HTTP calls — all external dependencies are in-memory stubs.
"""
from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timezone
from typing import Any

import pytest

from app.models.calibration import (
    ApplyCalibrationRequest,
    CalibrationStatus,
    CalibrationToken,
    CalibrateRequest,
    FeedbackRecord,
    FeedbackRequest,
    FeedbackValue,
    NoRecommendationReason,
)
from app.models.concept import DefinitionResponse, SearchResult, VersionSummary
from app.models.condition import (
    BOOLEAN_STRATEGIES,
    CATEGORICAL_STRATEGIES,
    ConditionDefinition,
    DecisionType,
    StrategyType,
)
from app.models.errors import ConflictError, ErrorType, MemintelError, NotFoundError
from app.models.result import (
    ActionTriggered,
    ActionTriggeredStatus,
    ConceptOutputType,
    ConceptResult,
    DecisionResult,
    DryRunResult,
    FullPipelineResult,
)
from app.models.task import (
    CreateTaskRequest,
    DeliveryConfig,
    DeliveryType,
    IMMUTABLE_TASK_FIELDS,
    Namespace,
    Task,
    TaskStatus,
    TaskUpdateRequest,
)
from app.models.decision import DecisionRecord
from app.registry.definitions import DefinitionRegistry
from app.services.calibration import CalibrationService
from app.services.feedback import FeedbackService
from app.services.task_authoring import TaskAuthoringService


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(coro: Any) -> Any:
    """Run a coroutine synchronously — mirrors the helper in unit tests."""
    return asyncio.run(coro)


# ── PassthroughValidator ───────────────────────────────────────────────────────

class PassthroughValidator:
    """
    Validator stub that always passes.

    Bypasses the DefinitionRegistry._freeze_check() path for concepts so that
    tests can register fixture-produced concept dicts without a real compiler.
    Duck-typed to match app.compiler.validator.Validator's interface.
    """

    def validate_schema(self, definition: Any) -> None:  # noqa: ARG002
        pass

    def validate_types(self, definition: Any) -> None:  # noqa: ARG002
        pass


# ── InMemoryTaskStore ──────────────────────────────────────────────────────────

class InMemoryTaskStore:
    """In-memory task store with immutable-field guard and soft-delete support."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._counter = 0

    async def create(self, task: Task) -> Task:
        if task.status == TaskStatus.PREVIEW:
            raise ValueError("preview tasks must not be persisted.")
        self._counter += 1
        task_id = f"task-{self._counter}"
        stored = task.model_copy(update={
            "task_id": task_id,
            "created_at": datetime.now(timezone.utc),
        })
        self._tasks[task_id] = stored
        return stored

    async def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    async def update(self, task_id: str, updates: dict[str, Any]) -> Task:
        forbidden = IMMUTABLE_TASK_FIELDS & updates.keys()
        if forbidden:
            raise MemintelError(
                ErrorType.PARAMETER_ERROR,
                f"Cannot update immutable field(s): {sorted(forbidden)}.",
                location=", ".join(sorted(forbidden)),
            )
        task = self._tasks.get(task_id)
        if task is None:
            raise NotFoundError(f"Task '{task_id}' not found.", location="task_id")
        updated = task.model_copy(update=updates)
        self._tasks[task_id] = updated
        return updated

    async def find_by_condition_version(
        self, condition_id: str, version: str
    ) -> list[Task]:
        return [
            t for t in self._tasks.values()
            if t.condition_id == condition_id
            and t.condition_version == version
            and t.status != TaskStatus.DELETED
        ]


# ── InMemoryDefinitionStore ────────────────────────────────────────────────────

class InMemoryDefinitionStore:
    """
    In-memory definition store.

    Raises ConflictError on duplicate (definition_id, version) registration —
    same as DefinitionStore in production (DB unique constraint behaviour).
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DefinitionResponse] = {}
        self._bodies: dict[tuple[str, str], dict[str, Any]] = {}
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
                f"Definition '{definition_id}' version '{version}' is already registered. "
                "Definitions are immutable — create a new version instead.",
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


# ── InMemoryFeedbackStore ──────────────────────────────────────────────────────

class InMemoryFeedbackStore:
    """
    In-memory feedback store with duplicate detection.

    Uniqueness key: (condition_id, condition_version, entity, timestamp).
    Mirrors FeedbackStore.create()'s ConflictError on duplicate.
    """

    def __init__(self) -> None:
        self._by_condition: dict[tuple[str, str], list[FeedbackRecord]] = {}
        self._by_key: dict[tuple[str, str, str, str], FeedbackRecord] = {}
        self._counter = 0

    async def create(self, record: FeedbackRecord) -> FeedbackRecord:
        key = (
            record.condition_id,
            record.condition_version,
            record.entity,
            record.timestamp,
        )
        if key in self._by_key:
            raise ConflictError(
                "Feedback already submitted for this decision.",
                location="(condition_id, condition_version, entity, timestamp)",
            )
        self._counter += 1
        stored = record.model_copy(update={
            "feedback_id": f"fb-{self._counter}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
        cond_key = (record.condition_id, record.condition_version)
        self._by_condition.setdefault(cond_key, []).append(stored)
        self._by_key[key] = stored
        return stored

    async def get_by_condition(
        self, condition_id: str, version: str
    ) -> list[FeedbackRecord]:
        return self._by_condition.get((condition_id, version), [])

    async def find(
        self,
        condition_id: str,
        condition_version: str,
        entity: str,
        timestamp: str,
    ) -> FeedbackRecord | None:
        return self._by_key.get((condition_id, condition_version, entity, timestamp))


# ── _AlwaysFoundDecisionStore ─────────────────────────────────────────────────

class _AlwaysFoundDecisionStore:
    """
    Decision store stub that always returns a valid DecisionRecord.

    FeedbackService.submit() (FIX 1) validates that a decision record exists
    for (condition_id, condition_version, entity, timestamp). Integration tests
    use fake timestamps; this stub bypasses the lookup so they can focus on
    the calibration flow rather than decision record management.
    """

    async def find_by_condition_entity_timestamp(
        self,
        condition_id: str,
        condition_version: str,
        entity_id: str,
        timestamp: str,
    ) -> DecisionRecord:
        return DecisionRecord(
            decision_id="mock-decision-stub",
            concept_id="org.mock_concept",
            concept_version="1.0",
            condition_id=condition_id,
            condition_version=condition_version,
            entity_id=entity_id,
            fired=True,
            concept_value=0.85,
            threshold_applied={"direction": "above", "value": 0.80},
            input_primitives={"feature_a": 0.85},
        )


# ── InMemoryCalibrationTokenStore ─────────────────────────────────────────────

class InMemoryCalibrationTokenStore:
    """
    In-memory calibration token store.

    Tokens are single-use (resolve_and_invalidate() returns None on second call)
    and never expire in the test context (expires_at set to far future).
    """

    def __init__(self) -> None:
        self._tokens: dict[str, CalibrationToken] = {}
        self._used: set[str] = set()

    async def create(self, token: Any) -> str:
        token_string = secrets.token_urlsafe(16)
        self._tokens[token_string] = CalibrationToken(
            token_string=token_string,
            condition_id=token.condition_id,
            condition_version=token.condition_version,
            recommended_params=token.recommended_params,
            expires_at=datetime(9999, 12, 31, tzinfo=timezone.utc),
        )
        return token_string

    async def resolve_and_invalidate(self, token_string: str) -> Any | None:
        if token_string in self._used or token_string not in self._tokens:
            return None
        self._used.add(token_string)
        return self._tokens[token_string]


# ── MockGuardrailsStore ────────────────────────────────────────────────────────

class MockGuardrailsStore:
    """
    Guardrails store with wide-open bounds and clamp policy.

    Returns no bounds ({}), so CalibrationService.adjust_params() never clips
    or rejects a recommendation. on_bounds_exceeded='clamp' is harmless here
    since there are no bounds to exceed.
    """

    def get_guardrails(self) -> Any:
        class _Constraints:
            on_bounds_exceeded = "clamp"

        class _Guardrails:
            constraints = _Constraints()

        return _Guardrails()

    def get_threshold_bounds(self, strategy: str) -> dict[str, Any]:  # noqa: ARG002
        return {}


# ── BadStrategyLLMClient ───────────────────────────────────────────────────────

class BadStrategyLLMClient:
    """
    LLM stub that returns a condition dict with strategy.params missing.

    Used by test_error_injection to trigger MemintelError(SEMANTIC_ERROR) inside
    TaskAuthoringService._validate_strategy_presence().
    """

    def generate_task(self, intent: str, context: dict) -> dict:  # noqa: ARG002
        return {
            "concept": {
                "concept_id": "org.test_score",
                "version": "1.0",
                "namespace": "org",
                "output_type": "float",
                "primitives": {
                    "score": {"type": "float", "missing_data_policy": "zero"},
                },
                "features": {
                    "val": {
                        "op": "normalize",
                        "inputs": {"input": "score"},
                        "params": {},
                    },
                },
                "output_feature": "val",
            },
            "condition": {
                "condition_id": "org.test_condition",
                "version": "1.0",
                "concept_id": "org.test_score",
                "concept_version": "1.0",
                "namespace": "org",
                # "strategy" has type but is missing "params" — triggers SEMANTIC_ERROR
                "strategy": {"type": "threshold"},
            },
            "action": {
                "action_id": "org.test_action",
                "version": "1.0",
                "namespace": "org",
                "config": {"type": "notification", "channel": "test-channel"},
                "trigger": {
                    "fire_on": "true",
                    "condition_id": "org.test_condition",
                    "condition_version": "1.0",
                },
            },
        }


# ── MockExecuteService ────────────────────────────────────────────────────────

class MockExecuteService:
    """
    Mock executor with real strategy evaluation.

    Evaluates threshold, z_score, equals, and composite strategies against a
    static entity→value lookup table supplied at construction. The concept
    execution step is bypassed — the provided value is used directly as the
    concept output.

    entity_concept_values maps entity identifiers to concept output values:
      float  → used by threshold, z_score (as direct z-score), and composite
      str    → used by equals (categorical label)

    Type safety enforcement (reflects the compiler's validate_types() behaviour):
      Applying threshold/z_score to a categorical (str) value → TYPE_ERROR.
      Applying equals to a numeric value → TYPE_ERROR.

    Cycle detection for composite conditions:
      Tracks visited condition_ids through recursion; circular references → GRAPH_ERROR.
    """

    def __init__(
        self,
        registry: DefinitionRegistry,
        entity_concept_values: dict[str, float | str],
    ) -> None:
        self._registry = registry
        self._entity_values = entity_concept_values

    async def evaluate_full(
        self,
        concept_id: str,
        concept_version: str,
        condition_id: str,
        condition_version: str,
        entity: str,
        timestamp: str | None = None,
        dry_run: bool = False,
    ) -> FullPipelineResult:
        if entity not in self._entity_values:
            raise MemintelError(
                ErrorType.EXECUTION_ERROR,
                f"No data available for entity '{entity}'. "
                "Primitive unavailable or entity not found.",
            )

        concept_value = self._entity_values[entity]

        # Load condition from registry (raises NotFoundError if missing)
        condition_body = await self._registry.get(condition_id, condition_version)
        condition = ConditionDefinition.model_validate(condition_body)

        # Evaluate strategy; cycle detection starts here for composite
        decision_value = await self._evaluate_strategy(
            condition, entity, _visiting=frozenset()
        )

        strategy_type = condition.strategy.type
        decision_type = (
            DecisionType.CATEGORICAL
            if strategy_type in CATEGORICAL_STRATEGIES
            else DecisionType.BOOLEAN
        )

        concept_output_type = (
            ConceptOutputType.CATEGORICAL
            if isinstance(concept_value, str)
            else ConceptOutputType.FLOAT
        )

        concept_result = ConceptResult(
            value=concept_value,
            type=concept_output_type,
            entity=entity,
            version=concept_version,
            deterministic=(timestamp is not None),
            timestamp=timestamp,
        )

        # Build actions_triggered — only when the decision fires
        action_fires = bool(decision_value)
        if action_fires:
            action_status = (
                ActionTriggeredStatus.WOULD_TRIGGER
                if dry_run
                else ActionTriggeredStatus.TRIGGERED
            )
            actions_triggered = [
                ActionTriggered(
                    action_id=f"mock_action.{condition_id}",
                    action_version="1.0",
                    status=action_status,
                )
            ]
        else:
            actions_triggered = []

        decision_result = DecisionResult(
            value=decision_value,
            type=decision_type,
            entity=entity,
            condition_id=condition_id,
            condition_version=condition_version,
            timestamp=timestamp,
            actions_triggered=actions_triggered,
        )

        return FullPipelineResult(
            result=concept_result,
            decision=decision_result,
            dry_run=dry_run,
            entity=entity,
            timestamp=timestamp,
        )

    async def _evaluate_strategy(
        self,
        condition: ConditionDefinition,
        entity: str,
        _visiting: frozenset[str],
    ) -> bool | str:
        """
        Evaluate a condition strategy for an entity.

        _visiting tracks the condition_id chain through recursive composite
        evaluation so that circular references raise GRAPH_ERROR.
        """
        # Cycle detection — must happen before any work on this condition
        if condition.condition_id in _visiting:
            raise MemintelError(
                ErrorType.GRAPH_ERROR,
                f"Circular dependency detected: condition '{condition.condition_id}' "
                "references itself through its composite operands.",
                location=f"condition.{condition.condition_id}",
            )
        visiting = _visiting | {condition.condition_id}

        concept_value = self._entity_values.get(entity)
        strategy = condition.strategy

        if strategy.type == StrategyType.THRESHOLD:
            if isinstance(concept_value, str):
                raise MemintelError(
                    ErrorType.TYPE_ERROR,
                    f"Cannot apply threshold strategy to categorical value '{concept_value}'. "
                    "Threshold requires a numeric (float/int) concept output.",
                    location="condition.strategy.type",
                )
            val = float(concept_value)  # type: ignore[arg-type]
            if strategy.params.direction.value == "above":
                return val > strategy.params.value
            return val < strategy.params.value

        if strategy.type == StrategyType.Z_SCORE:
            if isinstance(concept_value, str):
                raise MemintelError(
                    ErrorType.TYPE_ERROR,
                    f"Cannot apply z_score strategy to categorical value '{concept_value}'.",
                    location="condition.strategy.type",
                )
            # concept_value is treated as the pre-computed z-score in this mock
            z = float(concept_value)  # type: ignore[arg-type]
            direction = strategy.params.direction.value
            threshold = strategy.params.threshold
            if direction == "above":
                return z > threshold
            if direction == "below":
                return z < threshold
            return abs(z) > threshold  # "any"

        if strategy.type == StrategyType.EQUALS:
            if isinstance(concept_value, (int, float)):
                raise MemintelError(
                    ErrorType.TYPE_ERROR,
                    f"Cannot apply equals strategy to numeric value '{concept_value}'. "
                    "Equals requires a categorical concept output.",
                    location="condition.strategy.type",
                )
            # Return matched label when equal; False when no match
            if concept_value == strategy.params.value:
                return str(concept_value)
            return False

        if strategy.type == StrategyType.COMPOSITE:
            operator = strategy.params.operator.value
            operand_results: list[bool] = []
            for sub_condition_id in strategy.params.operands:
                sub_body = await self._registry.get(sub_condition_id, "1.0")
                sub_condition = ConditionDefinition.model_validate(sub_body)
                sub_result = await self._evaluate_strategy(sub_condition, entity, visiting)
                operand_results.append(bool(sub_result))
            if operator == "AND":
                return all(operand_results)
            return any(operand_results)  # OR

        raise MemintelError(
            ErrorType.EXECUTION_ERROR,
            f"Strategy type '{strategy.type}' is not supported by MockExecuteService.",
        )


# ── Condition body helper ──────────────────────────────────────────────────────

def _make_threshold_condition_body(
    condition_id: str,
    version: str,
    concept_id: str,
    concept_version: str,
    threshold_value: float,
    direction: str = "above",
) -> dict[str, Any]:
    """Return a minimal condition body with threshold strategy."""
    return {
        "condition_id": condition_id,
        "version": version,
        "concept_id": concept_id,
        "concept_version": concept_version,
        "namespace": "org",
        "strategy": {
            "type": "threshold",
            "params": {"direction": direction, "value": threshold_value},
        },
    }


# ── Service factory ────────────────────────────────────────────────────────────

def _make_services(
    entity_concept_values: dict[str, float | str] | None = None,
) -> tuple[
    InMemoryDefinitionStore,
    InMemoryTaskStore,
    InMemoryFeedbackStore,
    InMemoryCalibrationTokenStore,
    DefinitionRegistry,
    TaskAuthoringService,
    FeedbackService,
    CalibrationService,
    MockExecuteService,
]:
    """
    Build a complete in-memory service stack for one test.

    All stores are fresh. Returns each component individually so tests can
    inspect internal state (e.g. def_store._bodies) or inject additional data.
    """
    def_store   = InMemoryDefinitionStore()
    task_store  = InMemoryTaskStore()
    fb_store    = InMemoryFeedbackStore()
    tok_store   = InMemoryCalibrationTokenStore()
    guardrails  = MockGuardrailsStore()

    registry = DefinitionRegistry(store=def_store, validator=PassthroughValidator())

    task_svc = TaskAuthoringService(
        task_store=task_store,
        definition_registry=registry,
    )
    fb_svc = FeedbackService(
        feedback_store=fb_store,
        definition_registry=registry,
        decision_store=_AlwaysFoundDecisionStore(),
    )
    cal_svc = CalibrationService(
        feedback_store=fb_store,
        token_store=tok_store,
        task_store=task_store,
        definition_registry=registry,
        guardrails_store=guardrails,
    )
    executor = MockExecuteService(
        registry=registry,
        entity_concept_values=entity_concept_values or {},
    )

    return (
        def_store, task_store, fb_store, tok_store,
        registry, task_svc, fb_svc, cal_svc, executor,
    )


def _submit_feedbacks(
    fb_svc: FeedbackService,
    condition_id: str,
    condition_version: str,
    feedback: FeedbackValue,
    count: int = 3,
) -> None:
    """Submit `count` unique feedback records to meet MIN_FEEDBACK_THRESHOLD."""
    for i in range(1, count + 1):
        resp = run(fb_svc.submit(FeedbackRequest(
            condition_id=condition_id,
            condition_version=condition_version,
            entity=f"entity_{i}",
            timestamp=f"2024-06-{i:02d}T12:00:00Z",
            feedback=feedback,
        )))
        assert resp.status == "recorded"
        assert resp.feedback_id


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_threshold_task_full_pipeline() -> None:
    """
    Full lifecycle for a threshold condition derived from the threshold fixture.

    Fixture routing: intent "churn risk" → threshold_task.json
    Concept: org.churn_risk_score (float, normalize)
    Condition: org.high_churn_risk (threshold above 0.80)
    Action: org.notify_team (notification)

    Pipeline:
      create_task → evaluate_full → feedback×3 → calibrate → apply → rebind
    """
    (
        def_store, task_store, _, _, registry,
        task_svc, fb_svc, cal_svc, executor,
    ) = _make_services(entity_concept_values={"user_001": 0.87})

    # 1. Create task — routes to threshold_task.json fixture
    request = CreateTaskRequest(
        intent="alert when churn risk is high above threshold",
        entity_scope="all_users",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="slack-alerts"),
    )
    task = run(task_svc.create_task(request))

    assert isinstance(task, Task)
    assert task.task_id is not None
    assert task.status == TaskStatus.ACTIVE
    assert task.concept_id    == "org.churn_risk_score"
    assert task.concept_version == "1.0"
    assert task.condition_id   == "org.high_churn_risk"
    assert task.condition_version == "1.0"
    assert task.action_id      == "org.notify_team"

    condition_id      = task.condition_id
    condition_version = task.condition_version

    # 2. Evaluate full pipeline with timestamp → deterministic=True
    result = run(executor.evaluate_full(
        concept_id=task.concept_id,
        concept_version=task.concept_version,
        condition_id=condition_id,
        condition_version=condition_version,
        entity="user_001",
        timestamp="2024-06-01T12:00:00Z",
    ))

    # Core-spec 1G alignment assertions
    assert result.result.deterministic is True          # timestamp provided
    assert result.decision.value is True                # 0.87 > 0.80
    assert len(result.decision.actions_triggered) == 1
    assert result.decision.actions_triggered[0].status == ActionTriggeredStatus.TRIGGERED
    # actionsTriggered is on decision — NOT at FullPipelineResult top level
    assert not hasattr(result, "actions_triggered")

    # 3. Submit 3 feedbacks — meets MIN_FEEDBACK_THRESHOLD = 3
    _submit_feedbacks(fb_svc, condition_id, condition_version, FeedbackValue.FALSE_POSITIVE)

    # 4. Calibrate — majority false_positive → tighten → increase threshold value
    cal_result = run(cal_svc.calibrate(CalibrateRequest(
        condition_id=condition_id,
        condition_version=condition_version,
    )))
    assert cal_result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    assert cal_result.calibration_token is not None
    assert cal_result.recommended_params is not None
    assert cal_result.recommended_params["value"] > 0.80   # tighten → higher threshold
    assert cal_result.current_params["value"] == 0.80

    # 5. Apply calibration — registers new immutable condition version
    applied = run(cal_svc.apply_calibration(ApplyCalibrationRequest(
        calibration_token=cal_result.calibration_token,
    )))
    assert applied.condition_id      == condition_id
    assert applied.previous_version  == condition_version
    assert applied.new_version       != condition_version
    assert applied.new_version       == "1.1"            # auto-incremented from "1.0"
    assert applied.params_applied["value"] > 0.80

    # Old version unchanged in registry
    old_body = def_store._bodies.get((condition_id, "1.0"))
    assert old_body is not None
    assert old_body["strategy"]["params"]["value"] == 0.80

    # New version persisted with updated params
    new_body = def_store._bodies.get((condition_id, "1.1"))
    assert new_body is not None
    assert new_body["strategy"]["params"]["value"] == applied.params_applied["value"]

    # Task still bound to old version (informational — service never rebinds)
    assert len(applied.tasks_pending_rebind) == 1
    assert applied.tasks_pending_rebind[0].task_id == task.task_id

    # 6. Rebind task to new condition version
    updated_task = run(task_svc.update_task(
        task.task_id,
        TaskUpdateRequest(condition_version=applied.new_version),
    ))
    assert updated_task.condition_version == applied.new_version
    assert updated_task.condition_id      == condition_id   # immutable — unchanged


def test_z_score_task_full_pipeline() -> None:
    """
    Full lifecycle for a z_score condition derived from the z_score fixture.

    Fixture routing: intent "payment failure spike" → z_score_task.json
    Concept: org.payment_failure_rate (float, time_series, mean)
    Condition: org.payment_failure_anomaly (z_score threshold=2.5 above window=30d)
    Action: org.page_oncall (webhook)

    Entity value 3.1 is treated as the pre-computed z-score.
    z_score calibration: tighten → increase threshold param.
    """
    (
        _, task_store, _, _, registry,
        task_svc, fb_svc, cal_svc, executor,
    ) = _make_services(entity_concept_values={"entity_zscore": 3.1})

    request = CreateTaskRequest(
        intent="page oncall when payment failure spike detected",
        entity_scope="all_merchants",
        delivery=DeliveryConfig(
            type=DeliveryType.WEBHOOK, endpoint="https://hooks.example.com/oncall"
        ),
    )
    task = run(task_svc.create_task(request))

    assert isinstance(task, Task)
    assert task.concept_id    == "org.payment_failure_rate"
    assert task.condition_id  == "org.payment_failure_anomaly"
    assert task.action_id     == "org.page_oncall"

    condition_id      = task.condition_id
    condition_version = task.condition_version

    # Evaluate — z_score 3.1 > threshold 2.5 → decision fires
    result = run(executor.evaluate_full(
        concept_id=task.concept_id,
        concept_version=task.concept_version,
        condition_id=condition_id,
        condition_version=condition_version,
        entity="entity_zscore",
        timestamp="2024-06-01T08:00:00Z",
    ))

    assert result.result.deterministic is True
    assert result.decision.value is True
    assert len(result.decision.actions_triggered) == 1

    # Submit 3 feedbacks then calibrate
    _submit_feedbacks(fb_svc, condition_id, condition_version, FeedbackValue.FALSE_POSITIVE)

    cal_result = run(cal_svc.calibrate(CalibrateRequest(
        condition_id=condition_id,
        condition_version=condition_version,
    )))
    assert cal_result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
    # z_score calibration uses 'threshold' param key (not 'value')
    assert "threshold" in cal_result.recommended_params
    assert cal_result.recommended_params["threshold"] > 2.5   # tighten
    assert cal_result.current_params["threshold"] == 2.5

    # Apply and rebind
    applied = run(cal_svc.apply_calibration(ApplyCalibrationRequest(
        calibration_token=cal_result.calibration_token,
    )))
    assert applied.new_version != condition_version
    assert applied.params_applied["threshold"] > 2.5

    updated_task = run(task_svc.update_task(
        task.task_id,
        TaskUpdateRequest(condition_version=applied.new_version),
    ))
    assert updated_task.condition_version == applied.new_version


def test_equals_task_full_pipeline() -> None:
    """
    Full lifecycle for an equals condition derived from the equals fixture.

    Fixture routing: intent "classify risk category" → equals_task.json
    Concept: org.risk_category (categorical, labels=[low_risk,medium_risk,high_risk])
    Condition: org.is_high_risk (equals value='high_risk')
    Action: org.trigger_intervention (notification)

    Calibration on equals strategy always returns no_recommendation
    with reason='not_applicable_strategy' (no numeric param to adjust).
    """
    (
        _, _, _, _, _,
        task_svc, fb_svc, cal_svc, executor,
    ) = _make_services(entity_concept_values={"customer_eq": "high_risk"})

    request = CreateTaskRequest(
        intent="classify risk category and trigger intervention for high_risk label",
        entity_scope="all_customers",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="risk-management"),
    )
    task = run(task_svc.create_task(request))

    assert isinstance(task, Task)
    assert task.concept_id   == "org.risk_category"
    assert task.condition_id == "org.is_high_risk"

    condition_id      = task.condition_id
    condition_version = task.condition_version

    # Evaluate — categorical value 'high_risk' matches equals value → decision fires
    result = run(executor.evaluate_full(
        concept_id=task.concept_id,
        concept_version=task.concept_version,
        condition_id=condition_id,
        condition_version=condition_version,
        entity="customer_eq",
        timestamp="2024-06-01T09:00:00Z",
    ))

    assert result.result.deterministic is True
    # For equals: decision.value is the matched label (str), not True
    assert result.decision.value == "high_risk"
    assert result.result.type == ConceptOutputType.CATEGORICAL
    assert len(result.decision.actions_triggered) == 1

    # Submit 3 feedbacks — equals calibration still returns no_recommendation
    _submit_feedbacks(fb_svc, condition_id, condition_version, FeedbackValue.FALSE_POSITIVE)

    cal_result = run(cal_svc.calibrate(CalibrateRequest(
        condition_id=condition_id,
        condition_version=condition_version,
    )))
    # Core-spec requirement: equals → always no_recommendation (not_applicable_strategy)
    assert cal_result.status == CalibrationStatus.NO_RECOMMENDATION
    assert cal_result.no_recommendation_reason == NoRecommendationReason.NOT_APPLICABLE_STRATEGY
    assert cal_result.calibration_token is None
    assert cal_result.recommended_params is None


def test_composite_task_full_pipeline() -> None:
    """
    Full lifecycle for a composite condition derived from the composite fixture.

    Fixture routing: intent "composite risk" → composite_task.json
    Concept: org.risk_score (float, normalize)
    Condition: org.high_value_at_risk (composite AND operands=[org.high_churn_risk,
               org.high_ltv_customer])
    Action: org.escalate_to_success (workflow)

    Sub-conditions must be pre-registered before evaluate_full because the
    composite executor recursively looks them up in the registry.

    Calibration on composite always returns no_recommendation with
    reason='not_applicable_strategy'.
    """
    (
        def_store, _, _, _, registry,
        task_svc, fb_svc, cal_svc, executor,
    ) = _make_services(entity_concept_values={"customer_vip": 0.87})

    # Pre-register sub-conditions referenced by the composite operands.
    # org.high_churn_risk: threshold above 0.80 (0.87 > 0.80 → True)
    run(registry.register(
        _make_threshold_condition_body(
            "org.high_churn_risk", "1.0",
            "org.churn_risk_score", "1.0",
            threshold_value=0.80,
        ),
        namespace="org",
        definition_type="condition",
    ))
    # org.high_ltv_customer: threshold above 0.50 (0.87 > 0.50 → True)
    run(registry.register(
        _make_threshold_condition_body(
            "org.high_ltv_customer", "1.0",
            "org.ltv_score", "1.0",
            threshold_value=0.50,
        ),
        namespace="org",
        definition_type="condition",
    ))

    request = CreateTaskRequest(
        intent="escalate composite high-value at-risk customers",
        entity_scope="enterprise_customers",
        delivery=DeliveryConfig(type=DeliveryType.WORKFLOW, workflow_id="customer_success_playbook"),
    )
    task = run(task_svc.create_task(request))

    assert isinstance(task, Task)
    assert task.concept_id   == "org.risk_score"
    assert task.condition_id == "org.high_value_at_risk"
    assert task.action_id    == "org.escalate_to_success"

    condition_id      = task.condition_id
    condition_version = task.condition_version

    # Evaluate — composite AND: both sub-conditions fire → True
    result = run(executor.evaluate_full(
        concept_id=task.concept_id,
        concept_version=task.concept_version,
        condition_id=condition_id,
        condition_version=condition_version,
        entity="customer_vip",
        timestamp="2024-06-01T10:00:00Z",
    ))

    assert result.result.deterministic is True
    assert result.decision.value is True
    assert len(result.decision.actions_triggered) == 1

    # Submit feedbacks — composite calibration still returns no_recommendation
    _submit_feedbacks(fb_svc, condition_id, condition_version, FeedbackValue.FALSE_POSITIVE)

    cal_result = run(cal_svc.calibrate(CalibrateRequest(
        condition_id=condition_id,
        condition_version=condition_version,
    )))
    # Core-spec requirement: composite → always no_recommendation (not_applicable_strategy)
    assert cal_result.status == CalibrationStatus.NO_RECOMMENDATION
    assert cal_result.no_recommendation_reason == NoRecommendationReason.NOT_APPLICABLE_STRATEGY
    assert cal_result.calibration_token is None


def test_determinism_harness() -> None:
    """
    Same execution inputs 3× → all outputs are identical.

    Determinism is guaranteed when a fixed timestamp is provided:
      - concept_result.value must be identical across all three calls
      - decision.value must be identical
      - result.deterministic must be True in all cases

    Note: the mock executor is inherently deterministic; this test verifies
    that the FullPipelineResult structure carries the deterministic flag
    correctly and that no random state leaks into results.
    """
    (_, _, _, _, _, task_svc, _, _, executor) = _make_services(
        entity_concept_values={"user_det": 0.87}
    )

    task = run(task_svc.create_task(CreateTaskRequest(
        intent="determinism test churn alert",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="test"),
    )))

    params = dict(
        concept_id=task.concept_id,
        concept_version=task.concept_version,
        condition_id=task.condition_id,
        condition_version=task.condition_version,
        entity="user_det",
        timestamp="2024-06-01T00:00:00Z",
    )

    results = [run(executor.evaluate_full(**params)) for _ in range(3)]

    for r in results:
        assert r.result.deterministic is True

    # All concept values identical
    values = [r.result.value for r in results]
    assert len(set(values)) == 1, f"Expected identical values; got {values}"

    # All decision values identical
    decisions = [r.decision.value for r in results]
    assert len(set(decisions)) == 1, f"Expected identical decisions; got {decisions}"

    # Full result structures are equal
    assert results[0].model_dump() == results[1].model_dump() == results[2].model_dump()


def test_error_injection() -> None:
    """
    Verify that all four error categories propagate correctly.

    Scenarios:
      1. type_error       — threshold strategy applied to a categorical (str) value
      2. semantic_error   — condition dict missing strategy.params (LLM output defect)
      3. execution_error  — entity not present in data store (primitive unavailable)
      4. graph_error      — composite condition with a self-referential circular operand
    """
    # ── Scenario 1: type_error ────────────────────────────────────────────────
    (_, _, _, _, _, task_svc, _, _, executor_1) = _make_services(
        entity_concept_values={"bad_entity": "high_risk"}  # str value, but threshold condition
    )
    # Register threshold condition targeting the "bad_entity" with categorical value
    run(task_svc.create_task(CreateTaskRequest(
        intent="churn risk threshold test",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="test"),
    )))
    # The threshold_task fixture routes here; concept is float but entity_concept_values
    # returns a str → TYPE_ERROR raised inside _evaluate_strategy
    (_, _, _, _, type_registry, type_task_svc, _, _, type_executor) = _make_services(
        entity_concept_values={"type_entity": "some_label"}  # categorical value
    )
    type_task = run(type_task_svc.create_task(CreateTaskRequest(
        intent="churn risk above threshold",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="test"),
    )))
    with pytest.raises(MemintelError) as exc_info:
        run(type_executor.evaluate_full(
            concept_id=type_task.concept_id,
            concept_version=type_task.concept_version,
            condition_id=type_task.condition_id,
            condition_version=type_task.condition_version,
            entity="type_entity",
            timestamp="2024-01-01T00:00:00Z",
        ))
    assert exc_info.value.error_type == ErrorType.TYPE_ERROR

    # ── Scenario 2: semantic_error ────────────────────────────────────────────
    (_, semantic_task_store, _, _, semantic_registry, _, _, _, _) = _make_services()
    bad_svc = TaskAuthoringService(
        task_store=semantic_task_store,
        definition_registry=semantic_registry,
        llm_client=BadStrategyLLMClient(),  # returns condition without strategy.params
    )
    with pytest.raises(MemintelError) as exc_info:
        run(bad_svc.create_task(CreateTaskRequest(
            intent="this intent is ignored by the bad client",
            entity_scope="all",
            delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="test"),
        )))
    assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR

    # ── Scenario 3: execution_error ───────────────────────────────────────────
    (_, _, _, _, _, exec_task_svc, _, _, exec_executor) = _make_services(
        entity_concept_values={}  # empty — no entity data available
    )
    exec_task = run(exec_task_svc.create_task(CreateTaskRequest(
        intent="churn risk execution error test",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="test"),
    )))
    with pytest.raises(MemintelError) as exc_info:
        run(exec_executor.evaluate_full(
            concept_id=exec_task.concept_id,
            concept_version=exec_task.concept_version,
            condition_id=exec_task.condition_id,
            condition_version=exec_task.condition_version,
            entity="missing_entity",  # not in entity_concept_values
        ))
    assert exc_info.value.error_type == ErrorType.EXECUTION_ERROR

    # ── Scenario 4: graph_error ───────────────────────────────────────────────
    (_, _, _, _, graph_registry, _, _, _, graph_executor) = _make_services(
        entity_concept_values={"graph_entity": 0.75}
    )
    # Register a composite condition that references itself → circular dependency
    circular_body = {
        "condition_id": "org.circular_condition",
        "version": "1.0",
        "concept_id": "org.risk_score",
        "concept_version": "1.0",
        "namespace": "org",
        "strategy": {
            "type": "composite",
            "params": {
                "operator": "AND",
                # Both operands are the same condition → self-referential cycle
                "operands": ["org.circular_condition", "org.circular_condition"],
            },
        },
    }
    run(graph_registry.register(
        circular_body, namespace="org", definition_type="condition"
    ))
    # Also register a stub concept so evaluate_full can get the concept_id
    run(graph_registry.register(
        {
            "concept_id": "org.risk_score",
            "version": "1.0",
            "namespace": "org",
            "output_type": "float",
            "primitives": {"sig": {"type": "float", "missing_data_policy": "zero"}},
            "features": {
                "val": {"op": "normalize", "inputs": {"input": "sig"}, "params": {}}
            },
            "output_feature": "val",
        },
        namespace="org",
        definition_type="concept",
    ))
    with pytest.raises(MemintelError) as exc_info:
        run(graph_executor.evaluate_full(
            concept_id="org.risk_score",
            concept_version="1.0",
            condition_id="org.circular_condition",
            condition_version="1.0",
            entity="graph_entity",
        ))
    assert exc_info.value.error_type == ErrorType.GRAPH_ERROR


def test_dry_run_propagation() -> None:
    """
    dry_run=True in task creation returns DryRunResult without registering
    any definitions or persisting a task.

    dry_run=True in evaluate_full returns all actionsTriggered with
    status='would_trigger' instead of 'triggered'.
    """
    # ── Part A: create_task dry_run=True → DryRunResult, nothing persisted ────
    dry_def_store   = InMemoryDefinitionStore()
    dry_task_store  = InMemoryTaskStore()
    dry_registry    = DefinitionRegistry(store=dry_def_store, validator=PassthroughValidator())
    dry_task_svc    = TaskAuthoringService(
        task_store=dry_task_store, definition_registry=dry_registry
    )

    dry_result = run(dry_task_svc.create_task(CreateTaskRequest(
        intent="churn risk dry run test",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="dry-channel"),
        dry_run=True,
    )))

    # Must return DryRunResult — NOT a Task
    assert isinstance(dry_result, DryRunResult)
    assert dry_result.action_id is not None
    assert dry_result.validation.valid is True

    # No definitions registered (dry_run bypasses _register_and_persist)
    assert len(dry_def_store._bodies) == 0

    # No task persisted
    assert len(dry_task_store._tasks) == 0

    # ── Part B: evaluate_full dry_run=True → would_trigger ────────────────────
    (_, _, _, _, _, task_svc_b, _, _, executor_b) = _make_services(
        entity_concept_values={"user_b": 0.87}
    )
    task_b = run(task_svc_b.create_task(CreateTaskRequest(
        intent="churn risk for dry run execute test",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="test"),
    )))

    base_params = dict(
        concept_id=task_b.concept_id,
        concept_version=task_b.concept_version,
        condition_id=task_b.condition_id,
        condition_version=task_b.condition_version,
        entity="user_b",
        timestamp="2024-06-01T00:00:00Z",
    )

    # dry_run=True — all triggered actions must have would_trigger status
    dry_exec_result = run(executor_b.evaluate_full(**base_params, dry_run=True))
    assert dry_exec_result.dry_run is True
    assert len(dry_exec_result.decision.actions_triggered) > 0
    for action in dry_exec_result.decision.actions_triggered:
        assert action.status == ActionTriggeredStatus.WOULD_TRIGGER

    # dry_run=False — same params → triggered status
    live_exec_result = run(executor_b.evaluate_full(**base_params, dry_run=False))
    assert live_exec_result.dry_run is False
    assert len(live_exec_result.decision.actions_triggered) > 0
    for action in live_exec_result.decision.actions_triggered:
        assert action.status == ActionTriggeredStatus.TRIGGERED


def test_definition_immutability() -> None:
    """
    Registering the same (definition_id, version) twice must raise ConflictError.

    Tested at two levels:
      1. Directly via DefinitionRegistry.register() — same condition body twice
      2. Via TaskAuthoringService.create_task() — same intent twice routes to
         the same LLM fixture, which produces the same (id, version) pair.
         The second call must raise ConflictError from the store layer.

    This enforces the immutability contract from DefinitionStore: once
    registered, a (definition_id, version) pair is permanent and immutable.
    """
    # ── Level 1: direct registry call ────────────────────────────────────────
    def_store  = InMemoryDefinitionStore()
    registry   = DefinitionRegistry(store=def_store, validator=PassthroughValidator())

    condition_body = _make_threshold_condition_body(
        "org.imm_test_condition", "1.0",
        "org.imm_test_concept", "1.0",
        threshold_value=0.70,
    )

    # First registration — must succeed
    run(registry.register(condition_body, namespace="org", definition_type="condition"))

    # Second registration of same (id, version) — must raise ConflictError
    with pytest.raises(ConflictError) as exc_info:
        run(registry.register(condition_body, namespace="org", definition_type="condition"))

    assert "already registered" in str(exc_info.value).lower()

    # ── Level 2: via task authoring service ──────────────────────────────────
    (_, _, _, _, _, task_svc, _, _, _) = _make_services()

    request = CreateTaskRequest(
        intent="immutability test churn risk high alert",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="test"),
    )

    # First call — creates task and registers definitions
    first_task = run(task_svc.create_task(request))
    assert isinstance(first_task, Task)

    # Second call with same intent → same fixture → same (id, version) pairs
    # The concept registration will fail with ConflictError
    with pytest.raises(ConflictError):
        run(task_svc.create_task(request))

    # ── Token reuse raises PARAMETER_ERROR (immutability of single-use tokens) ──
    (_, _, _, _, _, task_svc_t, fb_svc_t, cal_svc_t, _) = _make_services()
    token_task = run(task_svc_t.create_task(CreateTaskRequest(
        intent="token test churn risk",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="test"),
    )))
    _submit_feedbacks(
        fb_svc_t,
        token_task.condition_id,
        token_task.condition_version,
        FeedbackValue.FALSE_POSITIVE,
    )
    cal = run(cal_svc_t.calibrate(CalibrateRequest(
        condition_id=token_task.condition_id,
        condition_version=token_task.condition_version,
    )))
    token_str = cal.calibration_token

    # First use — succeeds
    run(cal_svc_t.apply_calibration(ApplyCalibrationRequest(calibration_token=token_str)))

    # Second use of same token — PARAMETER_ERROR (single-use guarantee)
    with pytest.raises(MemintelError) as exc_info:
        run(cal_svc_t.apply_calibration(ApplyCalibrationRequest(calibration_token=token_str)))

    assert exc_info.value.error_type == ErrorType.PARAMETER_ERROR
