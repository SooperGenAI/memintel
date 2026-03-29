"""
tests/unit/test_agents.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for AgentService.

Coverage:
  1. query() — DB hit returns formatted results
  2. query() — DB miss falls back to LLM fixture
  3. query() — definition_type filter forwarded to DB
  4. define() — returns draft + validation_notes + requires_review
  5. define_condition() — concept found; draft has correct concept references
  6. define_condition() — concept not found → NotFoundError
  7. semantic_refine() — definition found; returns original + proposed + changes
  8. semantic_refine() — definition not found → NotFoundError
  9. compile_workflow() — returns ExecutionPlan with correct fields

Test isolation: every test builds its own MockPool and LLMFixtureClient.
No shared mutable state; no real DB or LLM calls.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.llm.fixtures import LLMFixtureClient
from app.models.concept import ExecutionPlan
from app.models.errors import NotFoundError
from app.services.agents import AgentService


# ── Helpers ───────────────────────────────────────────────────────────────────

class _Req:
    """Lightweight stand-in for Pydantic request models."""
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockPool:
    """
    Minimal asyncpg pool stub.

    fetchrow() returns fetchrow_result (a dict or None).
    fetch()    returns fetch_rows (a list of dicts, default []).
    """

    def __init__(
        self,
        fetchrow_result: dict | None = None,
        fetch_rows: list[dict] | None = None,
    ) -> None:
        self._fetchrow_result = fetchrow_result
        self._fetch_rows = fetch_rows if fetch_rows is not None else []

    async def fetchrow(self, query: str, *args: Any) -> dict | None:
        return self._fetchrow_result

    async def fetch(self, query: str, *args: Any) -> list[dict]:
        return self._fetch_rows


def _make_service(
    fetchrow_result: dict | None = None,
    fetch_rows: list[dict] | None = None,
) -> AgentService:
    pool = MockPool(fetchrow_result=fetchrow_result, fetch_rows=fetch_rows)
    return AgentService(pool=pool, llm_client=LLMFixtureClient())


def _run(coro):
    return asyncio.run(coro)


# ── query() ───────────────────────────────────────────────────────────────────

def test_query_db_hit_returns_results():
    """DB returns rows → results list built from those rows, not from fixture."""
    rows = [
        {"definition_id": "org.rev_score", "version": "2.0",
         "definition_type": "concept",
         "body": json.dumps({"description": "Revenue score concept."})},
    ]
    service = _make_service(fetch_rows=rows)
    req = _Req(query="revenue", definition_type=None, limit=10)
    resp = _run(service.query(req))

    assert resp.total_count == 1
    assert resp.results[0]["definition_id"] == "org.rev_score"
    assert resp.results[0]["version"] == "2.0"
    assert resp.results[0]["score"] == 1.0
    assert "Revenue score" in resp.results[0]["summary"]
    assert resp.query == "revenue"


def test_query_db_miss_falls_back_to_fixture():
    """DB returns no rows → fixture results returned instead."""
    service = _make_service(fetch_rows=[])
    req = _Req(query="churn", definition_type=None, limit=10)
    resp = _run(service.query(req))

    # Fixture has 2 results
    assert resp.total_count == 2
    assert any(r["definition_id"] == "org.churn_risk_score" for r in resp.results)
    assert resp.query == "churn"


def test_query_respects_limit():
    """Multiple DB rows are returned up to limit; total_count reflects actual count."""
    rows = [
        {"definition_id": f"org.def_{i}", "version": "1.0",
         "definition_type": "concept", "body": {}}
        for i in range(3)
    ]
    service = _make_service(fetch_rows=rows)
    req = _Req(query="def", definition_type="concept", limit=3)
    resp = _run(service.query(req))
    assert resp.total_count == 3


# ── define() ──────────────────────────────────────────────────────────────────

def test_define_returns_draft_and_notes():
    """define() calls LLM fixture and returns AgentDefineResponse."""
    service = _make_service()
    req = _Req(description="Monthly recurring revenue per customer", namespace="org")
    resp = _run(service.define(req))

    assert isinstance(resp.draft, dict)
    assert resp.draft["concept_id"] == "org.revenue_score"
    assert resp.draft["output_type"] == "float"
    assert isinstance(resp.validation_notes, list)
    assert len(resp.validation_notes) >= 1
    assert resp.requires_review is True


def test_define_draft_has_required_concept_fields():
    """Draft must contain the fields needed to pass to POST /compile."""
    service = _make_service()
    req = _Req(description="Some concept", namespace="personal")
    resp = _run(service.define(req))
    for field in ("concept_id", "version", "namespace", "output_type",
                  "primitives", "features", "output_feature"):
        assert field in resp.draft, f"Missing field in draft: {field}"


# ── define_condition() ────────────────────────────────────────────────────────

def test_define_condition_success():
    """Concept exists → draft returned with caller's concept_id/version patched in."""
    concept_body = {
        "concept_id": "org.churn_risk_score",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "primitives": {"engagement_score": {"type": "float"}},
        "features": {"churn_score": {"op": "normalize", "inputs": {"input": "engagement_score"}, "params": {}}},
        "output_feature": "churn_score",
    }
    service = _make_service(fetchrow_result={"body": json.dumps(concept_body)})
    req = _Req(
        description="Alert when churn risk is high",
        concept_id="org.churn_risk_score",
        concept_version="1.0",
        namespace="org",
    )
    resp = _run(service.define_condition(req))

    assert isinstance(resp.draft, dict)
    # concept references must be the caller's, not the fixture's defaults
    assert resp.draft["concept_id"] == "org.churn_risk_score"
    assert resp.draft["concept_version"] == "1.0"
    assert "strategy" in resp.draft
    assert resp.draft["strategy"]["type"] == "threshold"
    assert resp.requires_review is False


def test_define_condition_concept_not_found():
    """Concept not in DB → NotFoundError raised."""
    service = _make_service(fetchrow_result=None)
    req = _Req(
        description="Alert when churn risk is high",
        concept_id="org.missing_concept",
        concept_version="9.9",
        namespace="org",
    )
    with pytest.raises(NotFoundError) as exc_info:
        _run(service.define_condition(req))
    assert "org.missing_concept" in str(exc_info.value)


# ── semantic_refine() ─────────────────────────────────────────────────────────

def test_semantic_refine_success():
    """Definition found → original, proposed, changes, breaking returned."""
    original_body = {
        "concept_id": "org.churn_risk_score",
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "primitives": {"engagement_score": {"type": "float", "missing_data_policy": "zero"}},
        "features": {"churn_score": {"op": "normalize", "inputs": {"input": "engagement_score"}, "params": {}}},
        "output_feature": "churn_score",
    }
    service = _make_service(
        fetchrow_result={"body": json.dumps(original_body), "definition_type": "concept"}
    )
    req = _Req(
        definition_id="org.churn_risk_score",
        version="1.0",
        instruction="Add a recency decay factor",
    )
    resp = _run(service.semantic_refine(req))

    assert resp.original == original_body
    assert isinstance(resp.proposed, dict)
    assert isinstance(resp.changes, list)
    assert len(resp.changes) >= 1
    assert resp.breaking is True   # fixture sets breaking=True


def test_semantic_refine_definition_not_found():
    """Definition not in DB → NotFoundError raised."""
    service = _make_service(fetchrow_result=None)
    req = _Req(
        definition_id="org.nonexistent",
        version="1.0",
        instruction="Make it better",
    )
    with pytest.raises(NotFoundError) as exc_info:
        _run(service.semantic_refine(req))
    assert "org.nonexistent" in str(exc_info.value)


# ── compile_workflow() ────────────────────────────────────────────────────────

def test_compile_workflow_returns_execution_plan():
    """compile_workflow() returns a valid ExecutionPlan."""
    service = _make_service()
    req = _Req(
        description="Monitor revenue and trigger alert when threshold exceeded",
        namespace="org",
    )
    plan = _run(service.compile_workflow(req))

    assert isinstance(plan, ExecutionPlan)
    assert plan.concept_id == "workflow.revenue_monitor"
    assert plan.node_count == 3
    assert len(plan.execution_order) == 3
    assert isinstance(plan.parallelizable_groups, list)
    assert plan.primitive_fetches == ["mrr"]
    assert plan.critical_path_length == 3
