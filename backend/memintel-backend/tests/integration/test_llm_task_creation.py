"""
tests/integration/test_llm_task_creation.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for LLM-driven task creation.

Exercises the TaskAuthoringService pipeline with a ScriptedLLMClient that
returns controlled outputs instead of calling a real LLM.  All tests inject
llm_client=ScriptedLLMClient(...) explicitly so no real LLM is called.

Coverage:
  1. test_llm_schema_conformance_each_strategy   — one create_task() call per
       strategy type (threshold/z_score/composite/equals); asserts LLM output
       schema conformance and version-pinned bundle in the returned Task.

  2. test_guardrails_unregistered_strategy_rejected — LLM returns a strategy
       type not in the StrategyType enum; assert MemintelError(SEMANTIC_ERROR)
       is raised and the invalid output is not silently passed through.

  3. test_guardrails_missing_strategy_params_hard_fail — LLM returns a
       condition with strategy.params absent; assert the hard-fail path in
       _validate_strategy_presence raises SEMANTIC_ERROR immediately with no
       retry (call_count == 1 after the exception).

  4. test_invalid_structure_retried_not_passed_through — first LLM call returns
       a structurally invalid dict (missing "condition" key); second call returns
       a valid threshold output; assert the service retried (call_count == 2)
       and task creation succeeded.

Test isolation: each test builds its own service stack.
No database, real LLM, or HTTP calls.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from typing import Any

import pytest

from app.models.calibration import CalibrationToken
from app.models.concept import ConceptDefinition, DefinitionResponse, SearchResult, VersionSummary
from app.models.condition import (
    ConditionDefinition,
    StrategyType,
)
from app.models.config import ApplicationContext
from app.models.errors import ConflictError, ErrorType, MemintelError, NotFoundError
from app.models.guardrails import (
    Guardrails,
    GuardrailConstraints,
    StrategyRegistryEntry,
    ThresholdBounds,
)
from app.models.task import (
    CreateTaskRequest,
    DeliveryConfig,
    DeliveryType,
    Task,
    TaskStatus,
)
from app.registry.definitions import DefinitionRegistry
from app.services.task_authoring import TaskAuthoringService


# ── Module-level env setup ─────────────────────────────────────────────────────

# All tests in this module inject a ScriptedLLMClient explicitly via llm_client=,
# so _select_llm_client() is never called.  No module-level env var is needed.


# ── Helpers ─────────────────────────────────────────────────────────────────────

def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── ScriptedLLMClient ─────────────────────────────────────────────────────────

class ScriptedLLMClient:
    """
    LLM client stub that returns outputs from a fixed, ordered queue.

    Supports both generate_task() (first call) and refine_task() (retry calls)
    so that the retry loop in TaskAuthoringService._generate_with_retries() can
    exercise the refinement path.

    call_count tracks total calls so tests can assert retry behaviour.
    """

    def __init__(self, outputs: list[dict]) -> None:
        self._queue: list[dict] = list(outputs)
        self.call_count: int = 0

    def generate_task(self, intent: str, context: dict) -> dict:
        self.call_count += 1
        if not self._queue:
            raise RuntimeError("ScriptedLLMClient: output queue exhausted")
        return self._queue.pop(0)

    def refine_task(
        self,
        intent: str,
        context: dict,
        previous_output: dict,
        errors: list[str],
    ) -> dict:
        self.call_count += 1
        if not self._queue:
            raise RuntimeError("ScriptedLLMClient: output queue exhausted")
        return self._queue.pop(0)


# ── PassthroughValidator ──────────────────────────────────────────────────────

class PassthroughValidator:
    """
    Validator stub that never raises.

    Bypasses the compiler's _freeze_check so that concept bodies produced by
    the fixture JSON files (including the categorical equals fixture) can be
    registered without triggering Rule 12 (labeled categorical).  The compiler
    layer is tested separately in test_compiler_pipeline.py.
    """

    def validate_schema(self, definition: Any) -> None:
        pass

    def validate_types(self, definition: Any) -> None:
        pass


# ── InMemoryDefinitionStore ───────────────────────────────────────────────────

class InMemoryDefinitionStore:
    """
    Minimal in-memory definition store.  Enforces (id, version) uniqueness.
    """

    def __init__(self) -> None:
        self._bodies: dict[tuple[str, str], dict] = {}
        self._meta:   dict[tuple[str, str], dict] = {}

    async def register(
        self,
        definition_id: str,
        version: str,
        body: dict,
        namespace: str,
        definition_type: str,
        meaning_hash: str | None = None,
        ir_hash: str | None = None,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key in self._bodies:
            raise ConflictError(
                f"Definition '{definition_id}' version '{version}' already exists."
            )
        self._bodies[key] = body
        self._meta[key] = {
            "definition_id": definition_id,
            "version":        version,
            "namespace":      namespace,
            "definition_type": definition_type,
        }
        return DefinitionResponse(
            definition_id=definition_id,
            version=version,
            namespace=namespace,
            definition_type=definition_type,
        )

    async def get(self, definition_id: str, version: str) -> dict | None:
        return self._bodies.get((definition_id, version))

    async def get_metadata(self, definition_id: str, version: str) -> dict | None:
        return self._meta.get((definition_id, version))

    async def versions(self, definition_id: str) -> list[VersionSummary]:
        return [
            VersionSummary(version=v, deprecated=False)
            for (did, v) in self._bodies
            if did == definition_id
        ]

    async def list(
        self,
        definition_type: str | None = None,
        namespace: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResult:
        return SearchResult(items=[], total=0)

    async def update(self, definition_id: str, version: str, patch: dict) -> dict:
        key = (definition_id, version)
        if key not in self._bodies:
            raise NotFoundError(f"Definition '{definition_id}' v'{version}' not found.")
        self._bodies[key] = {**self._bodies[key], **patch}
        return self._bodies[key]


# ── InMemoryTaskStore ─────────────────────────────────────────────────────────

class InMemoryTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._counter = 0

    async def create(self, task: Task) -> Task:
        self._counter += 1
        task = task.model_copy(update={"task_id": f"task_{self._counter:04d}"})
        self._tasks[task.task_id] = task  # type: ignore[index]
        return task

    async def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    async def update(self, task_id: str, patch: dict) -> Task:
        if task_id not in self._tasks:
            raise NotFoundError(f"Task '{task_id}' not found.")
        task = self._tasks[task_id]
        self._tasks[task_id] = task.model_copy(update=patch)
        return self._tasks[task_id]

    async def find_by_condition_version(
        self,
        condition_id: str,
        condition_version: str,
    ) -> list[Task]:
        return [
            t for t in self._tasks.values()
            if t.condition_id == condition_id
            and t.condition_version == condition_version
        ]


# ── Fixture outputs ───────────────────────────────────────────────────────────

# These are the exact fixture dicts that LLMFixtureClient loads from disk.
# They are inlined here so the test is self-contained and independent of the
# fixture files (removing the filesystem dependency from this test module).

_THRESHOLD_OUTPUT: dict = {
    "concept": {
        "concept_id": "org.churn_risk_score",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "description": "Normalised churn risk score derived from user engagement signal.",
        "primitives": {
            "engagement_score": {"type": "float", "missing_data_policy": "zero"}
        },
        "features": {
            "churn_score": {
                "op": "normalize",
                "inputs": {"input": "engagement_score"},
                "params": {},
            }
        },
        "output_feature": "churn_score",
    },
    "condition": {
        "condition_id": "org.high_churn_risk",
        "version": "1.0",
        "concept_id": "org.churn_risk_score",
        "concept_version": "1.0",
        "namespace": "org",
        "strategy": {
            "type": "threshold",
            "params": {"direction": "above", "value": 0.80},
        },
    },
    "action": {
        "action_id": "org.notify_team",
        "version": "1.0",
        "namespace": "org",
        "config": {
            "type": "notification",
            "channel": "slack-alerts",
            "message_template": "High churn risk for {entity}.",
        },
        "trigger": {
            "fire_on": "true",
            "condition_id": "org.high_churn_risk",
            "condition_version": "1.0",
        },
    },
}

_Z_SCORE_OUTPUT: dict = {
    "concept": {
        "concept_id": "org.payment_failure_rate",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "description": "Mean payment failure rate over recent events.",
        "primitives": {
            "failure_events": {
                "type": "time_series<float>",
                "missing_data_policy": "forward_fill",
            }
        },
        "features": {
            "failure_rate": {
                "op": "mean",
                "inputs": {"input": "failure_events"},
                "params": {},
            }
        },
        "output_feature": "failure_rate",
    },
    "condition": {
        "condition_id": "org.payment_failure_anomaly",
        "version": "1.0",
        "concept_id": "org.payment_failure_rate",
        "concept_version": "1.0",
        "namespace": "org",
        "strategy": {
            "type": "z_score",
            "params": {"threshold": 2.5, "direction": "above", "window": "30d"},
        },
    },
    "action": {
        "action_id": "org.page_oncall",
        "version": "1.0",
        "namespace": "org",
        "config": {
            "type": "webhook",
            "endpoint": "https://hooks.example.com/oncall",
        },
        "trigger": {
            "fire_on": "true",
            "condition_id": "org.payment_failure_anomaly",
            "condition_version": "1.0",
        },
    },
}

_COMPOSITE_OUTPUT: dict = {
    "concept": {
        "concept_id": "org.risk_score",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "description": "Composite risk score combining multiple signals.",
        "primitives": {
            "risk_signal": {"type": "float", "missing_data_policy": "zero"}
        },
        "features": {
            "risk_score": {
                "op": "normalize",
                "inputs": {"input": "risk_signal"},
                "params": {},
            }
        },
        "output_feature": "risk_score",
    },
    "condition": {
        "condition_id": "org.high_value_at_risk",
        "version": "1.0",
        "concept_id": "org.risk_score",
        "concept_version": "1.0",
        "namespace": "org",
        "strategy": {
            "type": "composite",
            "params": {
                "operator": "AND",
                "operands": ["org.high_churn_risk", "org.high_ltv_customer"],
            },
        },
    },
    "action": {
        "action_id": "org.escalate_to_success",
        "version": "1.0",
        "namespace": "org",
        "config": {
            "type": "workflow",
            "workflow_id": "customer_success_playbook",
        },
        "trigger": {
            "fire_on": "true",
            "condition_id": "org.high_value_at_risk",
            "condition_version": "1.0",
        },
    },
}

_EQUALS_OUTPUT: dict = {
    "concept": {
        "concept_id": "org.risk_category",
        "version": "1.0",
        "namespace": "org",
        "output_type": "categorical",
        "labels": ["low_risk", "medium_risk", "high_risk"],
        "description": "Risk category label from upstream classifier.",
        "primitives": {
            "risk_category_label": {
                "type": "categorical",
                "missing_data_policy": "zero",
                "labels": ["low_risk", "medium_risk", "high_risk"],
            }
        },
        "features": {
            "risk_label": {
                "op": "passthrough",
                "inputs": {"input": "risk_category_label"},
                "params": {},
            }
        },
        "output_feature": "risk_label",
    },
    "condition": {
        "condition_id": "org.is_high_risk",
        "version": "1.0",
        "concept_id": "org.risk_category",
        "concept_version": "1.0",
        "namespace": "org",
        "strategy": {
            "type": "equals",
            "params": {
                "value": "high_risk",
                "labels": ["low_risk", "medium_risk", "high_risk"],
            },
        },
    },
    "action": {
        "action_id": "org.trigger_intervention",
        "version": "1.0",
        "namespace": "org",
        "config": {
            "type": "notification",
            "channel": "risk-management",
            "message_template": "Entity {entity} classified as high_risk.",
        },
        "trigger": {
            "fire_on": "true",
            "condition_id": "org.is_high_risk",
            "condition_version": "1.0",
        },
    },
}


# ── Guardrails ────────────────────────────────────────────────────────────────

def _make_guardrails() -> Guardrails:
    """
    Build a minimal Guardrails object with all four strategy types registered.

    The strategy_registry is the authoritative list of allowed strategies.
    Constraints add hard bounds on threshold (0.0–1.0) and z_score (0.5–10.0).
    """
    return Guardrails(
        application_context=ApplicationContext(
            name="Test Domain",
            description="Integration test application context.",
            instructions=["Alert on high-risk conditions", "Prefer conservative thresholds"],
        ),
        strategy_registry={
            "threshold": StrategyRegistryEntry(
                version="1.0",
                description="Fixed numeric threshold",
                input_types=["float", "int"],
                output_type="decision<boolean>",
            ),
            "z_score": StrategyRegistryEntry(
                version="1.0",
                description="Z-score anomaly detection",
                input_types=["float"],
                output_type="decision<boolean>",
            ),
            "composite": StrategyRegistryEntry(
                version="1.0",
                description="Composite AND/OR over sub-conditions",
                input_types=[],
                output_type="decision<boolean>",
            ),
            "equals": StrategyRegistryEntry(
                version="1.0",
                description="Categorical label equality",
                input_types=["categorical"],
                output_type="decision<categorical>",
            ),
        },
        constraints=GuardrailConstraints(
            threshold_bounds={
                "threshold": ThresholdBounds(min=0.0, max=1.0),
                "z_score":   ThresholdBounds(min=0.5, max=10.0),
            },
        ),
    )


# ── Service factory ───────────────────────────────────────────────────────────

def _make_service(
    llm_client: Any,
    max_retries: int = 3,
) -> tuple[TaskAuthoringService, InMemoryDefinitionStore, InMemoryTaskStore]:
    """Build a complete in-memory service stack for one test."""
    def_store  = InMemoryDefinitionStore()
    task_store = InMemoryTaskStore()
    registry   = DefinitionRegistry(store=def_store, validator=PassthroughValidator())
    svc = TaskAuthoringService(
        task_store=task_store,
        definition_registry=registry,
        guardrails=_make_guardrails(),
        llm_client=llm_client,
        max_retries=max_retries,
    )
    return svc, def_store, task_store


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_llm_schema_conformance_each_strategy() -> None:
    """
    For each of the four strategy types (threshold, z_score, composite, equals):

      1. ScriptedLLMClient returns the corresponding LLM fixture output directly.
      2. TaskAuthoringService.create_task() runs the full pipeline.
      3. Asserts that:
         a. LLM output conforms to the expected schema (concept/condition/action
            are present and parseable as domain models).
         b. condition.strategy.type is the expected strategy.
         c. condition.strategy.params match the declared parameter schema.
         d. The returned Task is version-pinned:
            concept_id, concept_version, condition_id, condition_version,
            action_id, action_version are all non-empty strings.
         e. task.status == ACTIVE.

    Each test injects its own ScriptedLLMClient via llm_client= so the
    real AnthropicClient is never instantiated.
    """
    cases = [
        ("threshold",  _THRESHOLD_OUTPUT, {"direction": "above", "value": 0.80}),
        ("z_score",    _Z_SCORE_OUTPUT,   {"threshold": 2.5, "direction": "above", "window": "30d"}),
        ("composite",  _COMPOSITE_OUTPUT, {"operator": "AND", "operands": ["org.high_churn_risk", "org.high_ltv_customer"]}),
        ("equals",     _EQUALS_OUTPUT,    {"value": "high_risk", "labels": ["low_risk", "medium_risk", "high_risk"]}),
    ]

    for strategy_type, llm_output, expected_params in cases:
        scripted = ScriptedLLMClient([copy.deepcopy(llm_output)])
        svc, def_store, task_store = _make_service(scripted)

        request = CreateTaskRequest(
            intent=f"alert when {strategy_type} condition fires",
            entity_scope="all_users",
            delivery=DeliveryConfig(
                type=DeliveryType.NOTIFICATION,
                channel="alerts",
            ),
        )

        task = run(svc.create_task(request))

        # ── (a) LLM output conforms to schema — concept/condition/action parseable ──
        # If parsing failed, create_task() would have raised MemintelError before
        # returning; reaching here means pydantic accepted all three models.
        assert isinstance(task, Task), (
            f"{strategy_type}: expected Task, got {type(task).__name__}"
        )

        # ── (b) condition.strategy.type matches expected strategy ─────────────────
        raw_condition = def_store._bodies.get(
            (llm_output["condition"]["condition_id"], "1.0")
        )
        assert raw_condition is not None, (
            f"{strategy_type}: condition not registered in def_store"
        )
        assert raw_condition.get("strategy", {}).get("type") == strategy_type, (
            f"{strategy_type}: condition.strategy.type mismatch: "
            f"{raw_condition.get('strategy', {}).get('type')!r}"
        )

        # ── (c) condition.strategy.params conform to declared parameter schema ─────
        parsed_condition = ConditionDefinition.model_validate(raw_condition)
        actual_params = parsed_condition.strategy.params.model_dump(exclude_none=True)
        for key, expected_val in expected_params.items():
            assert key in actual_params, (
                f"{strategy_type}: expected param '{key}' missing from strategy params"
            )
            assert actual_params[key] == expected_val, (
                f"{strategy_type}: param '{key}' = {actual_params[key]!r}, "
                f"expected {expected_val!r}"
            )

        # ── (d) Task is version-pinned ─────────────────────────────────────────────
        assert task.concept_id       and isinstance(task.concept_id,       str)
        assert task.concept_version  and isinstance(task.concept_version,  str)
        assert task.condition_id     and isinstance(task.condition_id,     str)
        assert task.condition_version and isinstance(task.condition_version, str)
        assert task.action_id        and isinstance(task.action_id,        str)
        assert task.action_version   and isinstance(task.action_version,   str)

        # concept_id / condition_id match the LLM output
        assert task.concept_id   == llm_output["concept"]["concept_id"]
        assert task.condition_id == llm_output["condition"]["condition_id"]
        assert task.action_id    == llm_output["action"]["action_id"]

        # ── (e) Task is ACTIVE ────────────────────────────────────────────────────
        assert task.status == TaskStatus.ACTIVE, (
            f"{strategy_type}: expected ACTIVE, got {task.status}"
        )

        # ── LLM was called exactly once (no spurious retries on valid output) ──────
        assert scripted.call_count == 1, (
            f"{strategy_type}: expected 1 LLM call, got {scripted.call_count}"
        )


def test_guardrails_unregistered_strategy_rejected() -> None:
    """
    Guardrail: strategy type not in StrategyType enum → MemintelError(SEMANTIC_ERROR).

    The LLM returns a condition with strategy.type = "ml_score" which is not
    a registered strategy in the StrategyType enum.  TaskAuthoringService calls
    ConditionDefinition(**condition_dict) after the retry loop; pydantic rejects
    the unknown enum value and the service raises MemintelError(SEMANTIC_ERROR).

    This verifies "no strategy outside the registry is accepted" — the invalid
    output is never silently passed through to the definition store or task store.
    """
    bad_output = copy.deepcopy(_THRESHOLD_OUTPUT)
    bad_output["condition"]["strategy"]["type"] = "ml_score"   # not in StrategyType

    scripted = ScriptedLLMClient([bad_output])
    svc, def_store, task_store = _make_service(scripted, max_retries=1)

    request = CreateTaskRequest(
        intent="alert when ml_score fires",
        entity_scope="all_users",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="alerts"),
    )

    with pytest.raises(MemintelError) as exc_info:
        run(svc.create_task(request))

    assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR, (
        f"Expected SEMANTIC_ERROR, got {exc_info.value.error_type}"
    )

    # The invalid output was NOT silently passed: no concept/condition/action registered.
    assert len(def_store._bodies) == 0, (
        "def_store should be empty — invalid strategy must not be persisted"
    )
    assert len(task_store._tasks) == 0, (
        "task_store should be empty — invalid strategy must not produce a task"
    )


def test_guardrails_missing_strategy_params_hard_fail() -> None:
    """
    Guardrail: condition strategy without 'params' key → hard fail, no retry.

    The LLM returns a condition with strategy.type present but strategy.params
    absent.  TaskAuthoringService._validate_strategy_presence() catches this
    AFTER the retry loop as a hard semantic error — the service does NOT retry
    and does NOT pass the incomplete output downstream.

    Asserts:
      - MemintelError(SEMANTIC_ERROR) is raised.
      - error.location is "condition.strategy.params".
      - LLM was called exactly once (hard fail → no retry).
      - Nothing was registered or persisted.
    """
    bad_output = copy.deepcopy(_THRESHOLD_OUTPUT)
    del bad_output["condition"]["strategy"]["params"]   # remove params

    scripted = ScriptedLLMClient([bad_output])
    svc, def_store, task_store = _make_service(scripted, max_retries=3)

    request = CreateTaskRequest(
        intent="alert when threshold fires — params missing test",
        entity_scope="all_users",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="alerts"),
    )

    with pytest.raises(MemintelError) as exc_info:
        run(svc.create_task(request))

    # Must be a hard semantic error with a location pointing to the missing params.
    assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR
    assert exc_info.value.location == "condition.strategy.params", (
        f"Expected location='condition.strategy.params', got {exc_info.value.location!r}"
    )

    # Hard fail — no retry; LLM called exactly once.
    # _validate_strategy_presence() fires AFTER _generate_with_retries() returns,
    # so the retry loop ran (1 call) and then the hard-fail check caught it.
    assert scripted.call_count == 1, (
        f"Expected 1 LLM call (hard fail, no retry), got {scripted.call_count}"
    )

    # Clean state — nothing persisted.
    assert len(def_store._bodies) == 0
    assert len(task_store._tasks) == 0


def test_invalid_structure_retried_not_passed_through() -> None:
    """
    Retry: structurally invalid output is rejected and retried, not passed through.

    Sequence:
      Call 1 — returns {"concept": {...}, "action": {...}} (missing "condition")
               → _validate_llm_output_structure() catches the missing key.
               → retries with refine_task() (ScriptedLLMClient.refine_task).
      Call 2 — returns the valid threshold output.
               → task creation succeeds.

    Asserts:
      - LLM was called twice (initial + 1 retry).
      - The structurally invalid output from call 1 was NOT passed to pydantic
        (no partial concept registered in def_store from the bad attempt).
      - The final Task is ACTIVE and version-pinned.
    """
    # Call 1: missing "condition" key → fails _validate_llm_output_structure
    bad_output = {
        "concept": _THRESHOLD_OUTPUT["concept"],
        "action":  _THRESHOLD_OUTPUT["action"],
        # "condition" intentionally omitted
    }
    # Call 2: valid threshold output → success
    good_output = copy.deepcopy(_THRESHOLD_OUTPUT)

    scripted = ScriptedLLMClient([bad_output, good_output])
    svc, def_store, task_store = _make_service(scripted, max_retries=3)

    request = CreateTaskRequest(
        intent="alert when churn risk is high — retry test",
        entity_scope="all_users",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="alerts"),
    )

    task = run(svc.create_task(request))

    # ── Retry happened ─────────────────────────────────────────────────────────
    assert scripted.call_count == 2, (
        f"Expected 2 LLM calls (initial + 1 retry), got {scripted.call_count}"
    )

    # ── Task creation succeeded on second attempt ─────────────────────────────
    assert isinstance(task, Task)
    assert task.status == TaskStatus.ACTIVE
    assert task.concept_id   == "org.churn_risk_score"
    assert task.condition_id == "org.high_churn_risk"

    # ── Version-pinned bundle ─────────────────────────────────────────────────
    assert task.concept_id       and task.concept_version
    assert task.condition_id     and task.condition_version
    assert task.action_id        and task.action_version

    # ── Only valid definitions registered (from call 2, not call 1) ───────────
    # The incomplete bad_output from call 1 was NEVER passed to pydantic or the
    # registry; the concept/condition/action in def_store come exclusively from
    # the valid second attempt.
    assert ("org.churn_risk_score", "1.0") in def_store._bodies
    assert ("org.high_churn_risk",  "1.0") in def_store._bodies
    assert ("org.notify_team",      "1.0") in def_store._bodies
    assert len(def_store._bodies) == 3, (
        f"Expected exactly 3 entries in def_store, got {len(def_store._bodies)}: "
        f"{list(def_store._bodies.keys())}"
    )
