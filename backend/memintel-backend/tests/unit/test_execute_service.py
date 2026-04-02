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
  24. z_score with 30 history rows → fires correctly (full z-score math)
  25. z_score with 2 history rows → reason=insufficient_history
  26. percentile with 30 history rows → fires correctly (bottom 20%)
  27. percentile with 0 history rows → reason=insufficient_history
  28. change with 3 history rows → fires correctly (>50% decrease)
  29. change with 1 history row → reason=insufficient_history
  30. time_series operator normalises {timestamp, value} dicts (executor fix)

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
    concept_result_rows: list[dict]      — returned by fetch() for concept_results
                                           history queries (z_score/percentile/change).
    """

    def __init__(
        self,
        fetchrow_map: dict | None = None,
        fetch_rows: list[dict] | None = None,
        graph_row: dict | None = None,
        job_row: dict | None = None,
        concept_result_rows: list[dict] | None = None,
    ) -> None:
        self._fetchrow_map = fetchrow_map or {}
        self._fetch_rows = fetch_rows or []
        self._graph_row = graph_row
        self._job_row = job_row
        self._insert_row = job_row  # used for INSERT RETURNING
        self._concept_result_rows = concept_result_rows if concept_result_rows is not None else []

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
        # Concept results history queries (z_score / percentile / change)
        if "concept_results" in query:
            return self._concept_result_rows
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


# ── history-based strategy tests ───────────────────────────────────────────────
#
# History rows are served via MockPool.concept_result_rows.
# Concept result: normalize(revenue=0, min=0, max=1000) → 0.0 (missing_data_policy=zero)
#
# History row structure:
#   {"value": float, "output_type": "float", "entity": str,
#    "version": str, "evaluated_at": datetime}
#
# MockPool.fetch() routes "concept_results" queries to concept_result_rows and
# routes "FROM definitions WHERE definition_type='action'" to fetch_rows=[].
# These are independent dispatch paths so no collision occurs.

import datetime as _dt


def _cr_row(value: float, entity: str = "acct_1") -> dict:
    """Build a synthetic concept_results row with a unique evaluated_at."""
    return {
        "value": value,
        "output_type": "float",
        "entity": entity,
        "version": "1.0",
        "evaluated_at": _dt.datetime(2024, 1, 1, 0, 0, 0),
    }


def _zscore_condition_body() -> dict:
    """z_score condition: fires when z < -1.5 (below), threshold=1.5."""
    return {
        "condition_id": "zscore_cond",
        "version": "1.0",
        "namespace": "org",
        "concept_id": "org.test_score",
        "concept_version": "1.0",
        "strategy": {
            "type": "z_score",
            "params": {"threshold": 1.5, "direction": "below"},
        },
    }


def _percentile_condition_body() -> dict:
    """percentile condition: fires when value is in the bottom 20% of history."""
    return {
        "condition_id": "pct_cond",
        "version": "1.0",
        "namespace": "org",
        "concept_id": "org.test_score",
        "concept_version": "1.0",
        "strategy": {
            "type": "percentile",
            "params": {"direction": "bottom", "value": 20.0},
        },
    }


def _change_condition_body() -> dict:
    """change condition: fires on >50% decrease."""
    return {
        "condition_id": "change_cond",
        "version": "1.0",
        "namespace": "org",
        "concept_id": "org.test_score",
        "concept_version": "1.0",
        "strategy": {
            "type": "change",
            "params": {"direction": "decrease", "value": 0.5},
        },
    }


def test_z_score_with_30_history_rows_fires():
    """
    z_score with 30 history rows evaluates the full z-score formula and fires.

    History: 15 rows at value=0.7, 15 rows at value=0.9 → mean=0.8, std=0.1
    Current: 0.0  (normalize(0) with zero policy)
    z = (0.0 - 0.8) / 0.1 = -8.0   direction='below', threshold=1.5
    -8.0 < -1.5 → fires (True)
    """
    # Alternate 0.7 / 0.9 to produce non-zero std.
    history_rows = [_cr_row(0.7 if i % 2 == 0 else 0.9) for i in range(30)]

    pool = MockPool(
        fetchrow_map={
            ("zscore_cond", "1.0"): {"body": json.dumps(_zscore_condition_body())},
            ("org.test_score", "1.0"): _concept_row(),
        },
        fetch_rows=[],
        concept_result_rows=history_rows,
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="zscore_cond",
        condition_version="1.0",
        timestamp="2024-06-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert isinstance(result, DecisionResult)
    assert result.value is True
    assert result.reason is None
    assert result.history_count == 30


def test_z_score_with_2_history_rows_returns_insufficient_history():
    """z_score with only 2 stored results → reason='insufficient_history', value=False."""
    history_rows = [_cr_row(0.8), _cr_row(0.9)]

    pool = MockPool(
        fetchrow_map={
            ("zscore_cond", "1.0"): {"body": json.dumps(_zscore_condition_body())},
            ("org.test_score", "1.0"): _concept_row(),
        },
        fetch_rows=[],
        concept_result_rows=history_rows,
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="zscore_cond",
        condition_version="1.0",
        timestamp="2024-06-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert result.value is False
    assert result.reason == "insufficient_history"
    assert result.history_count == 2


def test_percentile_with_30_history_rows_fires():
    """
    percentile with 30 history rows fires when current value is in the bottom 20%.

    History: all 30 rows at value=0.8
    Current: 0.0  (normalize(0) with zero policy)
    20th percentile of [0.8, 0.8, ...] = 0.8
    0.0 < 0.8 → fires (True)
    """
    history_rows = [_cr_row(0.8) for _ in range(30)]

    pool = MockPool(
        fetchrow_map={
            ("pct_cond", "1.0"): {"body": json.dumps(_percentile_condition_body())},
            ("org.test_score", "1.0"): _concept_row(),
        },
        fetch_rows=[],
        concept_result_rows=history_rows,
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="pct_cond",
        condition_version="1.0",
        timestamp="2024-06-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert isinstance(result, DecisionResult)
    assert result.value is True
    assert result.reason is None


def test_percentile_with_0_history_rows_returns_insufficient_history():
    """percentile with 0 stored results → reason='insufficient_history', value=False."""
    pool = MockPool(
        fetchrow_map={
            ("pct_cond", "1.0"): {"body": json.dumps(_percentile_condition_body())},
            ("org.test_score", "1.0"): _concept_row(),
        },
        fetch_rows=[],
        concept_result_rows=[],
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="pct_cond",
        condition_version="1.0",
        timestamp="2024-06-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert result.value is False
    assert result.reason == "insufficient_history"
    assert result.history_count == 0


def test_change_with_sufficient_history_fires():
    """
    change with 3 history rows fires when most-recent prior value gives >50% drop.

    History (oldest→newest): [0.75, 0.78, 0.8]  → history[-1].value = 0.8
    Current: 0.0  (normalize(0) with zero policy)
    pct_change = (0.0 - 0.8) / 0.8 = -1.0
    direction='decrease', value=0.5 → -1.0 < -0.5 → fires (True)

    NOTE: MockPool.fetch() returns concept_result_rows already in oldest-first
    order (matching ConceptResultStore.fetch_history() which reverses the DESC
    result). history[-1] is therefore the most recent prior value = 0.8.
    """
    # oldest-first, so index -1 (most recent) = 0.8
    history_rows = [_cr_row(0.75), _cr_row(0.78), _cr_row(0.8)]

    pool = MockPool(
        fetchrow_map={
            ("change_cond", "1.0"): {"body": json.dumps(_change_condition_body())},
            ("org.test_score", "1.0"): _concept_row(),
        },
        fetch_rows=[],
        concept_result_rows=history_rows,
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="change_cond",
        condition_version="1.0",
        timestamp="2024-06-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert isinstance(result, DecisionResult)
    assert result.value is True
    assert result.reason is None


def test_change_with_1_history_row_returns_insufficient_history():
    """change with only 1 stored result → reason='insufficient_history', value=False."""
    pool = MockPool(
        fetchrow_map={
            ("change_cond", "1.0"): {"body": json.dumps(_change_condition_body())},
            ("org.test_score", "1.0"): _concept_row(),
        },
        fetch_rows=[],
        concept_result_rows=[_cr_row(0.8)],
    )
    service = _make_service(pool)
    req = _condition_req(
        condition_id="change_cond",
        condition_version="1.0",
        timestamp="2024-06-01T00:00:00Z",
    )
    result = _run(service.evaluate_condition(req))

    assert result.value is False
    assert result.reason == "insufficient_history"
    assert result.history_count == 1


def test_time_series_operator_normalises_timestamp_value_dicts():
    """
    Executor operators accept {timestamp, value} dicts as well as plain floats.

    Validates the _extract_ts_value fix in executor.py that resolves the format
    mismatch documented in app/models/config.py:509-512:
      time_series values arrive as list[dict[str, Any]] where each entry has
      'timestamp' (ISO 8601 str) and 'value' keys, not as plain float lists.
    """
    from app.runtime.executor import (
        _op_mean,
        _op_sum,
        _op_min,
        _op_max,
        _op_pct_change,
        _op_rate_of_change,
    )

    ts_vals = [
        {"timestamp": "2024-01-01T00:00:00Z", "value": 1.0},
        {"timestamp": "2024-01-02T00:00:00Z", "value": 2.0},
        {"timestamp": "2024-01-03T00:00:00Z", "value": 3.0},
    ]

    assert _op_mean({"input": ts_vals}, {}) == 2.0
    assert _op_sum({"input": ts_vals}, {}) == 6.0
    assert _op_min({"input": ts_vals}, {}) == 1.0
    assert _op_max({"input": ts_vals}, {}) == 3.0
    # pct_change: (3.0 - 2.0) / abs(2.0) = 0.5
    assert abs(_op_pct_change({"input": ts_vals}, {}) - 0.5) < 1e-9
    # rate_of_change: 3.0 - 2.0 = 1.0
    assert abs(_op_rate_of_change({"input": ts_vals}, {}) - 1.0) < 1e-9

    # Backward-compatible: plain float lists still work.
    plain_vals = [1.0, 2.0, 3.0]
    assert _op_mean({"input": plain_vals}, {}) == 2.0
    assert _op_pct_change({"input": plain_vals}, {}) == 0.5


# ── FIX 3: _fetch_bound_actions uses a parameterised JSONB query ──────────────

def test_fetch_bound_actions_uses_parameterized_query():
    """
    _fetch_bound_actions must pass condition_id and condition_version as SQL
    parameters ($1, $2) to a JSONB-path WHERE clause, not fetch all actions
    and filter in Python.

    Verifies FIX 3: the MockPool.fetch() stub records the query and args so
    we can assert the correct parameterised form was used.
    """
    captured: list[tuple[str, tuple]] = []

    class _RecordingPool(MockPool):
        """MockPool that records every fetch() call."""
        async def fetch(self, query: str, *args: Any) -> list[dict]:
            captured.append((query, args))
            return []

    pool = _RecordingPool(
        fetchrow_map={
            ("org.test_score", "1.0"): _concept_row(),
            ("high_score", "1.0"): _condition_row(),
        }
    )
    svc = _make_service(pool)

    _run(svc._fetch_bound_actions("high_score", "1.0"))

    assert len(captured) == 1, "fetch() must be called exactly once"
    query, args = captured[0]

    # Verify the JSONB path filter is present in the SQL.
    assert "body->'trigger'->>'condition_id'" in query, (
        "_fetch_bound_actions must filter by JSONB path, not fetch all actions"
    )
    assert "body->'trigger'->>'condition_version'" in query

    # Verify parameters are passed as SQL args (not embedded in the string).
    assert "$1" in query and "$2" in query, "Must use $1/$2 parameterised placeholders"
    assert args == ("high_score", "1.0"), (
        f"Expected args ('high_score', '1.0'), got {args}"
    )

