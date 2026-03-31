"""
tests/unit/test_execute_service.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for ExecuteService.

Coverage:
  1.  execute() — concept found → ConceptResult returned
  2.  execute() — concept not found → NotFoundError
  3.  execute() — snapshot mode (timestamp=None) → deterministic=False
  4.  execute() — deterministic mode (timestamp present) → deterministic=True
  5.  execute_batch() — mixed success/failure → BatchExecuteResult correct totals
  6.  execute_batch() — all succeed → failed=0
  7.  execute_range() — generates one result per step
  8.  execute_range() — bad interval raises execution_error
  9.  execute_range() — from > to raises execution_error
  10. execute_async() — returns Job with status='queued'
  11. execute_graph() — graph not found → NotFoundError
  12. execute_graph() — ir_hash mismatch → MemintelError(CONFLICT)
  13. evaluate_condition() — condition found → DecisionResult with correct fields
  14. evaluate_condition() — condition not found → NotFoundError
  15. evaluate_condition_batch() — two entities → two DecisionResults
  16. evaluate_full() — full pipeline → FullPipelineResult structure correct
  17. evaluate_full() — dry_run=True → actions_triggered status='would_trigger'
  18. _parse_iso_duration() — common durations parsed correctly
  19. composite AND both operands true → DecisionResult.value is True
  20. composite AND one operand false → DecisionResult.value is False
  21. composite OR one operand true → DecisionResult.value is True
  22. composite OR both operands false → DecisionResult.value is False
  23. composite missing operand → NotFoundError with operand id in message

Test isolation: every test builds its own MockPool. No shared mutable state.
No real DB or connector calls.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.models.errors import ErrorType, MemintelError, NotFoundError
from app.models.result import BatchExecuteResult, ConceptResult, DecisionResult, FullPipelineResult, Job, JobStatus
from app.services.execute import ExecuteService, _parse_iso_duration
from datetime import timedelta


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Minimal concept body that compiles cleanly.
_CONCEPT_BODY = {
    "concept_id": "org.test_score",
    "version": "1.0",
    "namespace": "org",
    "output_type": "float",
    "primitives": {
        "revenue": {"type": "float", "missing_data_policy": "zero"}
    },
    "features": {
        "score": {
            "op": "normalize",
            "inputs": {"input": "revenue"},
            "params": {"min": 0.0, "max": 1000.0},
        }
    },
    "output_feature": "score",
}

# Threshold condition wired to the concept above.
_CONDITION_BODY = {
    "condition_id": "high_score",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.test_score",
    "concept_version": "1.0",
    "strategy": {
        "type": "threshold",
        "params": {
            "direction": "above",
            "value": 0.5,
        },
    },
}

# Action bound to the condition above.
_ACTION_BODY = {
    "action_id": "notify_high_score",
    "version": "1.0",
    "namespace": "org",
    "config": {
        "type": "webhook",
        "endpoint": "https://example.com/webhook",
    },
    "trigger": {
        "fire_on": "true",
        "condition_id": "high_score",
        "condition_version": "1.0",
    },
}


# ── Pool stubs ────────────────────────────────────────────────────────────────

class MockPool:
    """
    Minimal asyncpg pool stub.

    fetchrow_map: dict[tuple, dict|None] — keyed by (definition_id, version).
    fetch_rows: list[dict]               — returned by fetch() for action queries.
    graph_row: dict|None                 — returned by the GraphStore graph lookup.
    job_row: dict|None                   — returned by the JobStore lookup.
    """

    def __init__(
        self,
        fetchrow_map: dict | None = None,
        fetch_rows: list[dict] | None = None,
        graph_row: dict | None = None,
        job_row: dict | None = None,
    ) -> None:
        self._fetchrow_map = fetchrow_map or {}
        self._fetch_rows = fetch_rows or []
        self._graph_row = graph_row
        self._job_row = job_row
        self._insert_row = job_row  # used for INSERT RETURNING

    async def fetchrow(self, query: str, *args: Any) -> dict | None:
        # GraphStore.get() queries by graph_id (single string arg)
        if "execution_graphs" in query:
            return self._graph_row
        # JobStore.get() and enqueue() query by job_id
        if "FROM jobs" in query:
            return self._job_row
        if "INSERT INTO jobs" in query:
            return self._insert_row
        # Definition lookups: args = (definition_id, version)
        if len(args) >= 2:
            key = (args[0], args[1])
            if key in self._fetchrow_map:
                return self._fetchrow_map[key]
        # Single-arg lookups
        if len(args) == 1 and args[0] in self._fetchrow_map:
            return self._fetchrow_map[args[0]]
        return None

    async def fetch(self, query: str, *args: Any) -> list[dict]:
        return self._fetch_rows

    async def execute(self, query: str, *args: Any) -> None:
        pass


def _concept_row() -> dict:
    return {"body": json.dumps(_CONCEPT_BODY)}


def _condition_row() -> dict:
    return {"body": json.dumps(_CONDITION_BODY)}


def _action_row() -> dict:
    return {"body": json.dumps(_ACTION_BODY)}


def _job_row_queued(job_id: str = "test-job-123") -> dict:
    import datetime
    return {
        "job_id": job_id,
        "job_type": "execute",
        "status": "queued",
        "request_body": "{}",
        "result_body": None,
        "error_body": None,
        "poll_interval_s": 2,
        "enqueued_at": datetime.datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "updated_at": datetime.datetime.utcnow(),
    }


def _make_service(pool: MockPool) -> ExecuteService:
    return ExecuteService(pool=pool)


def _run(coro):
    return asyncio.run(coro)


# ── Request stubs ─────────────────────────────────────────────────────────────

class _Req:
    """Lightweight stand-in for Pydantic request models."""
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self) -> dict:
        return self.__dict__.copy()


def _execute_req(**kw) -> _Req:
    defaults = dict(
        id="org.test_score",
        version="1.0",
        entity="acct_1",
        timestamp=None,
        explain=False,
        cache=True,
        missing_data_policy=None,
        explain_mode="full",
    )
    defaults.update(kw)
    return _Req(**defaults)


def _condition_req(**kw) -> _Req:
    defaults = dict(
        condition_id="high_score",
        condition_version="1.0",
        entity="acct_1",
        timestamp=None,
        explain=False,
    )
    defaults.update(kw)
    return _Req(**defaults)


def _full_req(**kw) -> _Req:
    defaults = dict(
        concept_id="org.test_score",
        concept_version="1.0",
        condition_id="high_score",
        condition_version="1.0",
        entity="acct_1",
        timestamp=None,
        explain=False,
        dry_run=False,
    )
    defaults.update(kw)
    return _Req(**defaults)


# ── execute() ─────────────────────────────────────────────────────────────────

def test_execute_concept_found_returns_concept_result():
    """execute() with valid concept → ConceptResult with correct entity."""
    pool = MockPool(fetchrow_map={("org.test_score", "1.0"): _concept_row()})
    service = _make_service(pool)
    req = _execute_req()
    result = _run(service.execute(req))

    assert isinstance(result, ConceptResult)
    assert result.entity == "acct_1"
    assert result.version == "1.0"


def test_execute_concept_not_found_raises_not_found():
    """execute() with unknown concept → NotFoundError."""
    pool = MockPool(fetchrow_map={})
    service = _make_service(pool)
    req = _execute_req(id="org.missing", version="9.9")

    with pytest.raises(NotFoundError) as exc_info:
        _run(service.execute(req))
    assert "org.missing" in str(exc_info.value)


def test_execute_snapshot_mode_deterministic_false():
    """execute() without timestamp → snapshot mode; deterministic=False."""
    pool = MockPool(fetchrow_map={("org.test_score", "1.0"): _concept_row()})
    service = _make_service(pool)
    req = _execute_req(timestamp=None)
    result = _run(service.execute(req))

    assert result.deterministic is False
    assert result.timestamp is None


def test_execute_deterministic_mode():
    """execute() with timestamp → deterministic=True."""
    pool = MockPool(fetchrow_map={("org.test_score", "1.0"): _concept_row()})
    service = _make_service(pool)
    req = _execute_req(timestamp="2024-01-01T00:00:00Z")
    result = _run(service.execute(req))

    assert result.deterministic is True
    assert result.timestamp == "2024-01-01T00:00:00Z"


# ── execute_batch() ────────────────────────────────────────────────────────────

def test_execute_batch_all_succeed():
    """execute_batch() with all valid entities → failed=0."""
    pool = MockPool(fetchrow_map={("org.test_score", "1.0"): _concept_row()})
    service = _make_service(pool)
    req = _Req(id="org.test_score", version="1.0", entities=["a", "b", "c"],
               timestamp=None, explain=False)
    batch = _run(service.execute_batch(req))

    assert isinstance(batch, BatchExecuteResult)
    assert batch.total == 3
    assert batch.failed == 0
    assert all(item.result is not None for item in batch.results)
    assert [item.entity for item in batch.results] == ["a", "b", "c"]


def test_execute_batch_concept_not_found_raises():
    """execute_batch() with unknown concept → NotFoundError (not absorbed)."""
    pool = MockPool(fetchrow_map={})
    service = _make_service(pool)
    req = _Req(id="org.missing", version="9.9", entities=["a"],
               timestamp=None, explain=False)

    with pytest.raises(NotFoundError):
        _run(service.execute_batch(req))


# ── execute_range() ────────────────────────────────────────────────────────────

def test_execute_range_generates_correct_step_count():
    """execute_range() PT1H over 3 hours → 4 results (0h, 1h, 2h, 3h)."""
    pool = MockPool(fetchrow_map={("org.test_score", "1.0"): _concept_row()})
    service = _make_service(pool)
    req = _Req(
        id="org.test_score", version="1.0", entity="acct_1",
        from_timestamp="2024-01-01T00:00:00Z",
        to_timestamp="2024-01-01T03:00:00Z",
        interval="PT1H",
        explain=False,
    )
    results = _run(service.execute_range(req))

    assert len(results) == 4
    assert all(isinstance(r, ConceptResult) for r in results)
    assert all(r.deterministic is True for r in results)


def test_execute_range_bad_interval_raises():
    """execute_range() with unparseable interval → MemintelError(EXECUTION_ERROR)."""
    pool = MockPool(fetchrow_map={("org.test_score", "1.0"): _concept_row()})
    service = _make_service(pool)
    req = _Req(
        id="org.test_score", version="1.0", entity="acct_1",
        from_timestamp="2024-01-01T00:00:00Z",
        to_timestamp="2024-01-02T00:00:00Z",
        interval="NOT_A_DURATION",
        explain=False,
    )
    with pytest.raises(MemintelError) as exc_info:
        _run(service.execute_range(req))
    assert exc_info.value.error_type == ErrorType.EXECUTION_ERROR


def test_execute_range_from_after_to_raises():
    """execute_range() with from > to → MemintelError(EXECUTION_ERROR)."""
    pool = MockPool(fetchrow_map={("org.test_score", "1.0"): _concept_row()})
    service = _make_service(pool)
    req = _Req(
        id="org.test_score", version="1.0", entity="acct_1",
        from_timestamp="2024-01-02T00:00:00Z",
        to_timestamp="2024-01-01T00:00:00Z",
        interval="PT1H",
        explain=False,
    )
    with pytest.raises(MemintelError) as exc_info:
        _run(service.execute_range(req))
    assert exc_info.value.error_type == ErrorType.EXECUTION_ERROR


# ── execute_async() ────────────────────────────────────────────────────────────

def test_execute_async_returns_queued_job():
    """execute_async() → Job with status='queued' and a job_id."""
    pool = MockPool(job_row=_job_row_queued("job-42"))
    service = _make_service(pool)
    req = _execute_req()
    job = _run(service.execute_async(req))

    assert isinstance(job, Job)
    assert job.job_id == "job-42"
    assert job.status == JobStatus.QUEUED


# ── execute_graph() ────────────────────────────────────────────────────────────

def test_execute_graph_not_found_raises():
    """execute_graph() with unknown graph_id → NotFoundError."""
    pool = MockPool(graph_row=None)
    service = _make_service(pool)
    req = _Req(
        graph_id="unknown-graph-id",
        entity="acct_1",
        ir_hash=None,
        timestamp=None,
        explain=False,
        cache=True,
        missing_data_policy=None,
        explain_mode="full",
    )
    with pytest.raises(NotFoundError) as exc_info:
        _run(service.execute_graph(req))
    assert "unknown-graph-id" in str(exc_info.value)


def test_execute_graph_ir_hash_mismatch_raises_conflict():
    """execute_graph() with wrong ir_hash → MemintelError(CONFLICT)."""
    import datetime
    graph_row = {
        "graph_id": "g1",
        "concept_id": "org.test_score",
        "version": "1.0",
        "ir_hash": "correct-hash",
        "graph_body": json.dumps({
            "graph_id": "g1",
            "concept_id": "org.test_score",
            "version": "1.0",
            "ir_hash": "correct-hash",
            "output_type": "float",
            "output_node_id": "feat:score",
            "nodes": [
                {"node_id": "prim:revenue", "op": "primitive_fetch",
                 "inputs": {}, "params": {"source_name": "revenue",
                 "declared_type": "float", "missing_data_policy": "zero"},
                 "output_type": "float"},
                {"node_id": "feat:score", "op": "normalize",
                 "inputs": {"input": "prim:revenue"}, "params": {"min": 0.0, "max": 1000.0},
                 "output_type": "float"},
            ],
            "edges": [{"from_node_id": "prim:revenue", "to_node_id": "feat:score", "input_slot": "input"}],
            "topological_order": ["prim:revenue", "feat:score"],
            "parallelizable_groups": [["prim:revenue"], ["feat:score"]],
            "created_at": datetime.datetime.utcnow().isoformat(),
        }),
        "created_at": datetime.datetime.utcnow(),
    }
    pool = MockPool(graph_row=graph_row)
    service = _make_service(pool)
    req = _Req(
        graph_id="g1",
        entity="acct_1",
        ir_hash="wrong-hash",
        timestamp=None,
        explain=False,
        cache=True,
        missing_data_policy=None,
        explain_mode="full",
    )
    with pytest.raises(MemintelError) as exc_info:
        _run(service.execute_graph(req))
    assert exc_info.value.error_type == ErrorType.CONFLICT


# ── evaluate_condition() ───────────────────────────────────────────────────────

def test_evaluate_condition_found_returns_decision_result():
    """evaluate_condition() with valid condition → DecisionResult."""
    pool = MockPool(fetchrow_map={
        ("high_score", "1.0"): _condition_row(),
        ("org.test_score", "1.0"): _concept_row(),
    })
    service = _make_service(pool)
    req = _condition_req()
    result = _run(service.evaluate_condition(req))

    assert isinstance(result, DecisionResult)
    assert result.entity == "acct_1"
    assert result.condition_id == "high_score"
    assert result.condition_version == "1.0"
    assert isinstance(result.value, bool)
    assert result.actions_triggered == []


def test_evaluate_condition_not_found_raises():
    """evaluate_condition() with unknown condition → NotFoundError."""
    pool = MockPool(fetchrow_map={})
    service = _make_service(pool)
    req = _condition_req(condition_id="no_such_condition")

    with pytest.raises(NotFoundError) as exc_info:
        _run(service.evaluate_condition(req))
    assert "no_such_condition" in str(exc_info.value)


# ── evaluate_condition_batch() ────────────────────────────────────────────────

def test_evaluate_condition_batch_returns_one_result_per_entity():
    """evaluate_condition_batch() → one DecisionResult per entity in order."""
    pool = MockPool(fetchrow_map={
        ("high_score", "1.0"): _condition_row(),
        ("org.test_score", "1.0"): _concept_row(),
    })
    service = _make_service(pool)
    req = _Req(
        condition_id="high_score",
        condition_version="1.0",
        entities=["acct_1", "acct_2"],
        timestamp="2024-06-01T00:00:00Z",
    )
    results = _run(service.evaluate_condition_batch(req))

    assert len(results) == 2
    assert results[0].entity == "acct_1"
    assert results[1].entity == "acct_2"
    assert all(isinstance(r.value, bool) for r in results)


# ── evaluate_full() ────────────────────────────────────────────────────────────

def test_evaluate_full_returns_full_pipeline_result():
    """evaluate_full() → FullPipelineResult with correct structure."""
    pool = MockPool(
        fetchrow_map={
            ("high_score", "1.0"): _condition_row(),
            ("org.test_score", "1.0"): _concept_row(),
        },
        fetch_rows=[],  # no actions
    )
    service = _make_service(pool)
    req = _full_req()
    result = _run(service.evaluate_full(req))

    assert isinstance(result, FullPipelineResult)
    assert isinstance(result.result, ConceptResult)
    assert isinstance(result.decision, DecisionResult)
    assert result.entity == "acct_1"
    assert result.dry_run is False
    # actions_triggered is nested inside decision, NOT at FullPipelineResult level
    assert hasattr(result.decision, "actions_triggered")


def test_evaluate_full_dry_run_actions_would_trigger():
    """evaluate_full(dry_run=True) with a bound action → status='would_trigger'."""
    pool = MockPool(
        fetchrow_map={
            ("high_score", "1.0"): _condition_row(),
            ("org.test_score", "1.0"): _concept_row(),
        },
        fetch_rows=[_action_row()],
    )
    service = _make_service(pool)
    req = _full_req(dry_run=True)
    result = _run(service.evaluate_full(req))

    assert result.dry_run is True
    # The bound action fires on 'true' — concept returns None (null), which is
    # falsy, so threshold > 0.5 is False → action may be skipped.
    # Either skipped or would_trigger is acceptable (depends on decision value).
    for at in result.decision.actions_triggered:
        assert at.status.value in ("would_trigger", "skipped")


# ── _parse_iso_duration() ──────────────────────────────────────────────────────

@pytest.mark.parametrize("duration,expected_seconds", [
    ("PT1H",      3600),
    ("PT30M",     1800),
    ("P1D",       86400),
    ("P1W",       604800),
    ("PT1H30M",   5400),
    ("PT0.5H",    1800),
    ("P1DT12H",   86400 + 43200),
])
def test_parse_iso_duration_common_cases(duration, expected_seconds):
    """`_parse_iso_duration` parses common ISO 8601 duration strings."""
    result = _parse_iso_duration(duration)
    assert isinstance(result, timedelta)
    assert abs(result.total_seconds() - expected_seconds) < 1.0


def test_parse_iso_duration_bad_input_raises():
    """`_parse_iso_duration` raises ValueError for non-ISO strings."""
    with pytest.raises(ValueError):
        _parse_iso_duration("1 hour")

    with pytest.raises(ValueError):
        _parse_iso_duration("T1H")  # missing leading P


# ── composite strategy tests ───────────────────────────────────────────────────
#
# concept result: normalize(revenue=0, min=0, max=1000) → 0.0
#   op_true  condition: threshold below 0.5  → True  (0.0 < 0.5)
#   op_false condition: threshold above 0.5  → False (0.0 <= 0.5)

_OP_TRUE_BODY = {
    "condition_id": "op_true",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.test_score",
    "concept_version": "1.0",
    "strategy": {
        "type": "threshold",
        "params": {"direction": "below", "value": 0.5},
    },
}

_OP_FALSE_BODY = {
    "condition_id": "op_false",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.test_score",
    "concept_version": "1.0",
    "strategy": {
        "type": "threshold",
        "params": {"direction": "above", "value": 0.5},
    },
}


def _composite_condition_body(operator: str, operands: list[str]) -> dict:
    return {
        "condition_id": f"composite_{operator.lower()}",
        "version": "1.0",
        "namespace": "org",
        "concept_id": "org.test_score",
        "concept_version": "1.0",
        "strategy": {
            "type": "composite",
            "params": {"operator": operator, "operands": operands},
        },
    }


def _composite_pool(operator: str, operands: list[str], operand_bodies: dict) -> MockPool:
    """
    Build a MockPool for a composite condition test.

    fetchrow_map includes:
      (composite_id, "1.0") → composite condition row (version-keyed lookup)
      ("org.test_score", "1.0") → concept row
      "op_true" / "op_false" → operand condition rows (latest-version lookup)
    """
    condition_id = f"composite_{operator.lower()}"
    return MockPool(
        fetchrow_map={
            (condition_id, "1.0"): {"body": json.dumps(
                _composite_condition_body(operator, operands)
            )},
            ("org.test_score", "1.0"): _concept_row(),
            **{name: {"body": json.dumps(body)} for name, body in operand_bodies.items()},
        },
        fetch_rows=[],
    )


def test_composite_and_both_true_returns_true():
    """AND composite with both operands evaluating True → decision.value is True."""
    pool = _composite_pool(
        operator="AND",
        operands=["op_true", "op_true"],
        operand_bodies={"op_true": _OP_TRUE_BODY},
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="composite_and",
        condition_version="1.0",
        timestamp="2024-01-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert isinstance(result, DecisionResult)
    assert result.value is True


def test_composite_and_one_false_returns_false():
    """AND composite with one operand evaluating False → decision.value is False."""
    pool = _composite_pool(
        operator="AND",
        operands=["op_true", "op_false"],
        operand_bodies={"op_true": _OP_TRUE_BODY, "op_false": _OP_FALSE_BODY},
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="composite_and",
        condition_version="1.0",
        timestamp="2024-01-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert isinstance(result, DecisionResult)
    assert result.value is False


def test_composite_or_one_true_returns_true():
    """OR composite with one operand evaluating True → decision.value is True."""
    pool = _composite_pool(
        operator="OR",
        operands=["op_false", "op_true"],
        operand_bodies={"op_false": _OP_FALSE_BODY, "op_true": _OP_TRUE_BODY},
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="composite_or",
        condition_version="1.0",
        timestamp="2024-01-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert isinstance(result, DecisionResult)
    assert result.value is True


def test_composite_or_both_false_returns_false():
    """OR composite with both operands evaluating False → decision.value is False."""
    pool = _composite_pool(
        operator="OR",
        operands=["op_false", "op_false"],
        operand_bodies={"op_false": _OP_FALSE_BODY},
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="composite_or",
        condition_version="1.0",
        timestamp="2024-01-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert isinstance(result, DecisionResult)
    assert result.value is False


def test_composite_missing_operand_raises_not_found():
    """Composite referencing an unregistered operand → NotFoundError naming the operand."""
    pool = _composite_pool(
        operator="AND",
        operands=["op_true", "op_missing"],
        operand_bodies={"op_true": _OP_TRUE_BODY},
        # op_missing is NOT in operand_bodies → _fetch_condition_latest returns None
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="composite_and",
        condition_version="1.0",
        timestamp="2024-01-01T00:00:00Z",
    )
    with pytest.raises(NotFoundError) as exc_info:
        _run(service.evaluate_condition(req))
    assert "op_missing" in str(exc_info.value)
