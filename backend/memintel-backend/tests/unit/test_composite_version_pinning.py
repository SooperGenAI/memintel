"""
tests/unit/test_composite_version_pinning.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for composite operand version pinning.

Covers the 4 spec requirements from the version-pinning fix:

  1. test_composite_operand_version_pinned_at_compile
     - task_authoring pins operand versions at create_task() time using the
       latest registered version of each string operand condition_id.

  2. test_composite_operand_version_in_definition_body
     - The persisted condition body contains OperandRef dicts
       (condition_id + condition_version), not plain strings.

  3. test_composite_version_change_requires_rebind
     - Calibrating an operand to a new version does NOT change the composite
       evaluation result — the old version stays pinned.

  4. test_composite_audit_trail_shows_pinned_versions
     - The stored composite condition body is readable as a ConditionDefinition;
       operands are OperandRef objects with the pinned condition_version.
"""
from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest

from app.models.concept import DefinitionResponse, SearchResult, VersionSummary
from app.models.condition import (
    ConditionDefinition,
    CompositeParams,
    OperandRef,
    StrategyType,
)
from app.models.errors import ConflictError, ErrorType, MemintelError, NotFoundError
from app.models.task import (
    CreateTaskRequest,
    DeliveryConfig,
    DeliveryType,
    Task,
)
from app.registry.definitions import DefinitionRegistry
from app.services.task_authoring import TaskAuthoringService


# ── Helpers ─────────────────────────────────────────────────────────────────

def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── In-memory stores ─────────────────────────────────────────────────────────

class InMemoryDefinitionStore:
    """Minimal in-memory store with versions() returning newest-first."""

    def __init__(self) -> None:
        self._bodies: dict[tuple[str, str], dict] = {}
        self._order: list[tuple[str, str]] = []   # insertion order (newest = last registered)

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
        self._order.append(key)
        return DefinitionResponse(
            definition_id=definition_id,
            version=version,
            namespace=namespace,
            definition_type=definition_type,
        )

    async def get(self, definition_id: str, version: str) -> dict | None:
        return self._bodies.get((definition_id, version))

    async def get_metadata(self, definition_id: str, version: str) -> dict | None:
        key = (definition_id, version)
        if key not in self._bodies:
            return None
        return {"definition_id": definition_id, "version": version}

    async def versions(self, definition_id: str) -> list[VersionSummary]:
        """Return versions newest-first (last registered = newest)."""
        matching = [
            VersionSummary(version=v, deprecated=False)
            for (did, v) in reversed(self._order)
            if did == definition_id
        ]
        return matching

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


class PassthroughValidator:
    def validate_schema(self, definition: Any) -> None:
        pass
    def validate_types(self, definition: Any) -> None:
        pass


class ScriptedLLMClient:
    def __init__(self, outputs: list[dict]) -> None:
        self._queue = list(outputs)
        self.call_count = 0

    def generate_task(self, intent: str, context: dict) -> dict:
        self.call_count += 1
        return self._queue.pop(0)

    def refine_task(self, intent: str, context: dict, previous_output: dict, errors: list[str]) -> dict:
        self.call_count += 1
        return self._queue.pop(0)


# ── Fixtures ─────────────────────────────────────────────────────────────────

_OPERAND_A_V1 = {
    "condition_id": "org.cond_a",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.score",
    "concept_version": "1.0",
    "strategy": {"type": "threshold", "params": {"direction": "above", "value": 0.5}},
}

_OPERAND_B_V1 = {
    "condition_id": "org.cond_b",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.score",
    "concept_version": "1.0",
    "strategy": {"type": "threshold", "params": {"direction": "below", "value": 0.8}},
}

_OPERAND_A_V2 = {
    "condition_id": "org.cond_a",
    "version": "2.0",
    "namespace": "org",
    "concept_id": "org.score",
    "concept_version": "1.0",
    "strategy": {"type": "threshold", "params": {"direction": "above", "value": 0.9}},
}

_COMPOSITE_LLM_OUTPUT = {
    "concept": {
        "concept_id": "org.composite_concept",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "description": "Composite test concept.",
        "primitives": {"sig": {"type": "float", "missing_data_policy": "zero"}},
        "features": {
            "val": {"op": "normalize", "inputs": {"input": "sig"}, "params": {}}
        },
        "output_feature": "val",
    },
    "condition": {
        "condition_id": "org.composite_cond",
        "version": "1.0",
        "concept_id": "org.composite_concept",
        "concept_version": "1.0",
        "namespace": "org",
        "strategy": {
            "type": "composite",
            "params": {
                "operator": "AND",
                # LLM emits plain strings — version pinning converts these.
                "operands": ["org.cond_a", "org.cond_b"],
            },
        },
    },
    "action": {
        "action_id": "org.notify",
        "version": "1.0",
        "namespace": "org",
        "config": {"type": "notification", "channel": "alerts", "message_template": "fired"},
        "trigger": {"fire_on": "true", "condition_id": "org.composite_cond", "condition_version": "1.0"},
    },
}


def _make_service_with_operands(
    llm_output: dict,
    operand_versions: list[tuple[str, str, dict]],
) -> tuple[TaskAuthoringService, InMemoryDefinitionStore]:
    """
    Build a service with pre-registered operand conditions.

    operand_versions: list of (definition_id, version, body) tuples.
    """
    def_store  = InMemoryDefinitionStore()
    task_store = InMemoryTaskStore()
    for op_id, op_ver, op_body in operand_versions:
        run(def_store.register(
            definition_id=op_id,
            version=op_ver,
            body=op_body,
            namespace="org",
            definition_type="condition",
        ))
    registry = DefinitionRegistry(store=def_store, validator=PassthroughValidator())
    svc = TaskAuthoringService(
        task_store=task_store,
        definition_registry=registry,
        llm_client=ScriptedLLMClient([copy.deepcopy(llm_output)]),
        max_retries=1,
    )
    return svc, def_store


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_composite_operand_version_pinned_at_compile():
    """
    PART C: task_authoring pins each string operand to the latest registered
    version at create_task() time.

    Setup: org.cond_a has v1.0 registered; org.cond_b has v1.0 registered.
    LLM returns operands=["org.cond_a", "org.cond_b"].
    After create_task(), the stored condition body must have operands as
    OperandRef dicts with condition_version="1.0".
    """
    svc, def_store = _make_service_with_operands(
        _COMPOSITE_LLM_OUTPUT,
        operand_versions=[
            ("org.cond_a", "1.0", _OPERAND_A_V1),
            ("org.cond_b", "1.0", _OPERAND_B_V1),
        ],
    )
    task = run(svc.create_task(CreateTaskRequest(
        intent="fire when composite AND fires",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="alerts"),
    )))
    assert isinstance(task, Task)

    stored_body = def_store._bodies.get(("org.composite_cond", "1.0"))
    assert stored_body is not None

    operands = stored_body["strategy"]["params"]["operands"]
    assert len(operands) == 2
    assert operands[0]["condition_id"]      == "org.cond_a"
    assert operands[0]["condition_version"] == "1.0"
    assert operands[1]["condition_id"]      == "org.cond_b"
    assert operands[1]["condition_version"] == "1.0"


def test_composite_operand_version_in_definition_body():
    """
    PART A/B: The persisted condition body is parseable as ConditionDefinition
    with OperandRef objects — not plain strings — in params.operands.
    """
    svc, def_store = _make_service_with_operands(
        _COMPOSITE_LLM_OUTPUT,
        operand_versions=[
            ("org.cond_a", "1.0", _OPERAND_A_V1),
            ("org.cond_b", "1.0", _OPERAND_B_V1),
        ],
    )
    run(svc.create_task(CreateTaskRequest(
        intent="composite condition test",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="alerts"),
    )))

    stored_body = def_store._bodies["org.composite_cond", "1.0"]
    parsed = ConditionDefinition.model_validate(stored_body)

    assert parsed.strategy.type == StrategyType.COMPOSITE
    assert isinstance(parsed.strategy.params, CompositeParams)
    operands = parsed.strategy.params.operands
    assert len(operands) == 2
    for op in operands:
        assert isinstance(op, OperandRef)
        assert isinstance(op.condition_id, str)
        assert isinstance(op.condition_version, str)


def test_composite_version_change_requires_rebind():
    """
    PART B: Calibrating an operand (registering a new version) does NOT
    silently change the composite definition body.  The pinned version in the
    stored body remains "1.0" even after "2.0" becomes the latest version.

    This simulates: after create_task() pins org.cond_a@1.0, a calibration
    step registers org.cond_a@2.0.  The stored composite body must still
    reference condition_version="1.0".
    """
    svc, def_store = _make_service_with_operands(
        _COMPOSITE_LLM_OUTPUT,
        operand_versions=[
            ("org.cond_a", "1.0", _OPERAND_A_V1),
            ("org.cond_b", "1.0", _OPERAND_B_V1),
        ],
    )
    run(svc.create_task(CreateTaskRequest(
        intent="test version stability",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="alerts"),
    )))

    # Simulate calibration: register a new version of org.cond_a.
    run(def_store.register(
        definition_id="org.cond_a",
        version="2.0",
        body=_OPERAND_A_V2,
        namespace="org",
        definition_type="condition",
    ))

    # The stored composite body must still pin org.cond_a@1.0 — not 2.0.
    stored_body = def_store._bodies["org.composite_cond", "1.0"]
    operands = stored_body["strategy"]["params"]["operands"]
    cond_a_ref = next(op for op in operands if op["condition_id"] == "org.cond_a")
    assert cond_a_ref["condition_version"] == "1.0", (
        f"Expected pinned version '1.0' but found '{cond_a_ref['condition_version']}'. "
        "Calibration must not silently update existing composite condition bodies."
    )


def test_composite_audit_trail_shows_pinned_versions():
    """
    Audit trail: reading back the stored condition body reveals exact
    (condition_id, condition_version) pairs that were pinned at compile time.

    Verifies the full round-trip: LLM output → version pinning → Pydantic parse
    → model_dump → same OperandRef dicts are retrievable and complete.
    """
    svc, def_store = _make_service_with_operands(
        _COMPOSITE_LLM_OUTPUT,
        operand_versions=[
            ("org.cond_a", "1.0", _OPERAND_A_V1),
            ("org.cond_b", "1.0", _OPERAND_B_V1),
        ],
    )
    run(svc.create_task(CreateTaskRequest(
        intent="audit trail test",
        entity_scope="all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="alerts"),
    )))

    stored_body = def_store._bodies["org.composite_cond", "1.0"]
    parsed = ConditionDefinition.model_validate(stored_body)
    dumped = parsed.strategy.params.model_dump()

    # The model_dump should round-trip the OperandRef dicts faithfully.
    expected_operands = [
        {"condition_id": "org.cond_a", "condition_version": "1.0"},
        {"condition_id": "org.cond_b", "condition_version": "1.0"},
    ]
    assert dumped["operands"] == expected_operands, (
        f"Audit trail mismatch.\n"
        f"  Expected: {expected_operands}\n"
        f"  Got:      {dumped['operands']}"
    )
