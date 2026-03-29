"""
tests/unit/test_compile_service.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for CompileService.

Coverage:
  1.  compile() — valid concept → ExecutionGraph with populated ir_hash
  2.  compile() — invalid concept (unknown operator) → MemintelError raised
  3.  compile() — idempotent recompile (same ir_hash) → store called, no error
  4.  compile() — ir_hash mismatch on recompile → CompilerInvariantError raised
  5.  compile() — graph_id is deterministic (same concept → same graph_id)
  6.  compile_semantic() — valid concept → SemanticGraph with semantic_hash
  7.  compile_semantic() — semantic_hash is stable across equivalent concepts
  8.  compile_semantic() — different concepts → different semantic_hash
  9.  compile_semantic() — invalid concept → MemintelError raised
  10. explain_plan() — valid concept → ExecutionPlan with correct fields
  11. explain_plan() — invalid concept → MemintelError raised
  12. explain_plan() — node_count matches concept features + primitives
  13. explain_plan() — primitive_fetches lists all declared primitive names
  14. _compute_semantic_hash() — same graph always produces same hash

Test isolation: each test builds its own MockPool. No real DB calls.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.compiler.dag_builder import DAGBuilder
from app.compiler.ir_generator import IRGenerator
from app.models.concept import ConceptDefinition, ExecutionGraph
from app.models.errors import CompilerInvariantError, ErrorType, MemintelError
from app.services.compile import CompileService, _compute_semantic_hash


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Minimal valid float concept with a single normalize node.
_CONCEPT_BODY = {
    "concept_id": "org.test_score",
    "version": "1.0",
    "namespace": "org",
    "output_type": "float",
    "primitives": {
        "revenue": {"type": "float", "missing_data_policy": "zero"},
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

# A concept with a different operator sequence but semantically identical
# computation to _CONCEPT_BODY (same operator, same topology, same types).
_CONCEPT_BODY_RENAMED = {
    "concept_id": "team.revenue_normalised",
    "version": "2.0",
    "namespace": "team",
    "output_type": "float",
    "primitives": {
        "revenue": {"type": "float", "missing_data_policy": "zero"},
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

# A semantically different concept (z_score_op instead of normalize).
_CONCEPT_BODY_DIFFERENT = {
    "concept_id": "org.zscore_score",
    "version": "1.0",
    "namespace": "org",
    "output_type": "float",
    "primitives": {
        "revenue": {"type": "float", "missing_data_policy": "zero"},
    },
    "features": {
        "score": {
            "op": "z_score_op",
            "inputs": {"input": "revenue"},
            "params": {},
        }
    },
    "output_feature": "score",
}

# An invalid concept: operator 'nonexistent_op' is not in OPERATOR_REGISTRY.
_INVALID_CONCEPT_BODY = {
    "concept_id": "org.bad_concept",
    "version": "1.0",
    "namespace": "org",
    "output_type": "float",
    "primitives": {
        "revenue": {"type": "float", "missing_data_policy": "zero"},
    },
    "features": {
        "score": {
            "op": "nonexistent_op",
            "inputs": {"input": "revenue"},
            "params": {},
        }
    },
    "output_feature": "score",
}


# ── MockPool ──────────────────────────────────────────────────────────────────

class MockPool:
    """
    Minimal asyncpg pool stub for CompileService tests.

    Tracks calls to execute() so tests can assert that GraphStore.store()
    issued the expected INSERT.  fetchrow() returns stored_graph_row when set,
    simulating an existing row in execution_graphs.
    """

    def __init__(self, stored_graph_row: dict | None = None) -> None:
        self._stored = stored_graph_row
        self.execute_calls: list[tuple] = []   # (query, *args)
        self.fetchrow_calls: list[tuple] = []

    async def fetchrow(self, query: str, *args: Any) -> dict | None:
        self.fetchrow_calls.append((query, *args))
        if "execution_graphs" in query:
            return self._stored
        return None

    async def execute(self, query: str, *args: Any) -> None:
        self.execute_calls.append((query, *args))


def _make_service(stored_graph_row: dict | None = None) -> CompileService:
    return CompileService(pool=MockPool(stored_graph_row=stored_graph_row))


def _run(coro):
    return asyncio.run(coro)


def _concept(body: dict) -> ConceptDefinition:
    return ConceptDefinition(**body)


def _compile_graph(concept: ConceptDefinition) -> ExecutionGraph:
    """Build and hash a graph the same way CompileService.compile() does."""
    g = DAGBuilder().build_dag(concept)
    IRGenerator().hash_graph(g)
    return g


# ── compile() ─────────────────────────────────────────────────────────────────

def test_compile_returns_execution_graph():
    """compile() → ExecutionGraph with non-empty graph_id and ir_hash."""
    service = _make_service()
    concept = _concept(_CONCEPT_BODY)
    result = _run(service.compile(concept))

    assert isinstance(result, ExecutionGraph)
    assert result.graph_id != ""
    assert result.ir_hash != "" and result.ir_hash != "sentinel"
    assert result.concept_id == "org.test_score"
    assert result.version == "1.0"


def test_compile_invalid_concept_raises():
    """compile() with an invalid concept raises MemintelError."""
    service = _make_service()
    concept = _concept(_INVALID_CONCEPT_BODY)

    with pytest.raises(MemintelError) as exc_info:
        _run(service.compile(concept))
    assert exc_info.value.error_type == ErrorType.REFERENCE_ERROR


def test_compile_idempotent_same_ir_hash():
    """
    Recompiling the same concept when the stored graph has the matching
    ir_hash succeeds without raising CompilerInvariantError.
    """
    concept = _concept(_CONCEPT_BODY)
    graph = _compile_graph(concept)

    # Simulate: graph already stored with the same ir_hash
    stored_row = {
        "graph_body": json.dumps(graph.model_dump(mode="json")),
        "created_at": None,
    }
    service = _make_service(stored_graph_row=stored_row)
    # Should not raise
    result = _run(service.compile(concept))
    assert result.ir_hash == graph.ir_hash


def test_compile_ir_hash_mismatch_raises():
    """
    When the stored graph has a DIFFERENT ir_hash, CompilerInvariantError
    is raised and the graph is not overwritten.
    """
    concept = _concept(_CONCEPT_BODY)
    graph = _compile_graph(concept)

    # Tamper with the stored hash to force a mismatch
    tampered = graph.model_dump(mode="json")
    tampered["ir_hash"] = "0" * 64
    stored_row = {
        "graph_body": json.dumps(tampered),
        "created_at": None,
    }
    service = _make_service(stored_graph_row=stored_row)

    with pytest.raises(CompilerInvariantError):
        _run(service.compile(concept))

    # execute() must NOT have been called (graph not overwritten)
    assert not service._pool.execute_calls


def test_compile_graph_id_is_deterministic():
    """Same concept always produces the same graph_id (SHA-256 of id:version)."""
    concept = _concept(_CONCEPT_BODY)

    r1 = _run(_make_service().compile(concept))
    r2 = _run(_make_service().compile(concept))

    assert r1.graph_id == r2.graph_id
    assert r1.ir_hash == r2.ir_hash


# ── compile_semantic() ────────────────────────────────────────────────────────

def test_compile_semantic_returns_semantic_graph():
    """compile_semantic() → SemanticGraph with concept_id, version, and hash."""
    service = _make_service()
    concept = _concept(_CONCEPT_BODY)
    result = _run(service.compile_semantic(concept))

    assert result.concept_id == "org.test_score"
    assert result.version == "1.0"
    assert len(result.semantic_hash) == 64   # SHA-256 hex digest
    assert isinstance(result.features, list)
    assert isinstance(result.input_primitives, list)
    assert "revenue" in result.input_primitives


def test_compile_semantic_stable_across_renames():
    """
    Two concepts that compute the same function have the same semantic_hash
    even when concept_id, version, and namespace differ.
    """
    c1 = _concept(_CONCEPT_BODY)
    c2 = _concept(_CONCEPT_BODY_RENAMED)

    r1 = _run(_make_service().compile_semantic(c1))
    r2 = _run(_make_service().compile_semantic(c2))

    assert r1.semantic_hash == r2.semantic_hash


def test_compile_semantic_different_for_different_computation():
    """Concepts that compute different functions have different semantic_hashes."""
    c1 = _concept(_CONCEPT_BODY)
    c2 = _concept(_CONCEPT_BODY_DIFFERENT)

    r1 = _run(_make_service().compile_semantic(c1))
    r2 = _run(_make_service().compile_semantic(c2))

    assert r1.semantic_hash != r2.semantic_hash


def test_compile_semantic_invalid_concept_raises():
    """compile_semantic() with an invalid concept raises MemintelError."""
    service = _make_service()
    concept = _concept(_INVALID_CONCEPT_BODY)

    with pytest.raises(MemintelError) as exc_info:
        _run(service.compile_semantic(concept))
    assert exc_info.value.error_type == ErrorType.REFERENCE_ERROR


def test_compile_semantic_does_not_store():
    """compile_semantic() must NOT call GraphStore.store() (no persistence)."""
    service = _make_service()
    concept = _concept(_CONCEPT_BODY)
    _run(service.compile_semantic(concept))

    # No INSERT should have been issued
    assert not service._pool.execute_calls


# ── explain_plan() ────────────────────────────────────────────────────────────

def test_explain_plan_returns_execution_plan():
    """explain_plan() → ExecutionPlan with correct structural fields."""
    service = _make_service()
    concept = _concept(_CONCEPT_BODY)
    plan = _run(service.explain_plan(concept))

    assert plan.concept_id == "org.test_score"
    assert plan.version == "1.0"
    assert plan.node_count >= 2           # at least 1 prim + 1 feature node
    assert len(plan.execution_order) == plan.node_count
    assert plan.critical_path_length >= 1
    assert isinstance(plan.parallelizable_groups, list)


def test_explain_plan_invalid_concept_raises():
    """explain_plan() with an invalid concept raises MemintelError."""
    service = _make_service()
    concept = _concept(_INVALID_CONCEPT_BODY)

    with pytest.raises(MemintelError) as exc_info:
        _run(service.explain_plan(concept))
    assert exc_info.value.error_type == ErrorType.REFERENCE_ERROR


def test_explain_plan_node_count_correct():
    """node_count equals number of nodes in the compiled graph."""
    concept = _concept(_CONCEPT_BODY)
    graph = DAGBuilder().build_dag(concept)

    service = _make_service()
    plan = _run(service.explain_plan(concept))

    assert plan.node_count == len(graph.nodes)


def test_explain_plan_primitive_fetches():
    """primitive_fetches lists all declared primitive source names."""
    service = _make_service()
    concept = _concept(_CONCEPT_BODY)
    plan = _run(service.explain_plan(concept))

    assert plan.primitive_fetches == ["revenue"]


def test_explain_plan_does_not_store():
    """explain_plan() must NOT call GraphStore.store() (no persistence)."""
    service = _make_service()
    concept = _concept(_CONCEPT_BODY)
    _run(service.explain_plan(concept))

    assert not service._pool.execute_calls


# ── _compute_semantic_hash() ──────────────────────────────────────────────────

def test_semantic_hash_determinism():
    """Same graph always produces the same semantic_hash."""
    concept = _concept(_CONCEPT_BODY)
    g1 = DAGBuilder().build_dag(concept)
    g2 = DAGBuilder().build_dag(concept)

    h1 = _compute_semantic_hash(g1)
    h2 = _compute_semantic_hash(g2)

    assert h1 == h2
    assert len(h1) == 64
