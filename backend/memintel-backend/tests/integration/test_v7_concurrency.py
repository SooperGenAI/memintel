"""
tests/integration/test_v7_concurrency.py
──────────────────────────────────────────────────────────────────────────────
T-7 Part 1 — Concurrent task creation and concept registration.

Verifies that concurrent operations do not produce duplicate DB rows,
constraint violations, or silently wrong outcomes.

All tests use:
  - asyncio.gather() to fire concurrent requests
  - Real asyncpg pool (no DB mocks)
  - httpx.AsyncClient with ASGITransport
  - Direct DB queries to verify final state

Concurrency notes
─────────────────
The db_pool fixture provides one event loop per test (see conftest.py).
All concurrent work is wrapped in a single async function and passed to
run() via loop.run_until_complete(asyncio.gather(...)).
httpx.AsyncClient is safe to use concurrently under asyncio — multiple
awaits on the same client share its internal connection pool.

Spec note on decision rows
──────────────────────────
POST /tasks creates task rows in the tasks table but does NOT create
decision rows. Decisions are written by the execute service
(POST /evaluate/full). The spec's assertion "5 decision rows (one per task)"
is incorrect for this endpoint; we assert task rows instead.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import concepts as concepts_route
from app.api.routes import tasks as tasks_route
from app.api.routes.concepts import (
    get_concept_compiler_service,
    get_concept_registration_service,
)
from app.api.routes.tasks import get_task_authoring_service
from app.models.errors import MemintelError, memintel_error_handler
from app.registry.definitions import DefinitionRegistry
from app.services.concept_compiler import ConceptCompilerService
from app.services.concept_registration import ConceptRegistrationService
from app.services.task_authoring import TaskAuthoringService
from app.stores import DefinitionStore, TaskStore
from app.stores.compile_token import CompileTokenStore

from tests.integration.conftest_v7 import compile_and_register


# ── Test-app factory ───────────────────────────────────────────────────────────

def _make_test_app(db_pool, llm_client: Any) -> FastAPI:
    """Minimal FastAPI app wired to the test DB pool and a mock LLM client."""
    app = FastAPI()
    app.state.db = db_pool
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.include_router(concepts_route.router)
    app.include_router(tasks_route.router)

    async def _compiler_svc() -> ConceptCompilerService:
        return ConceptCompilerService(
            llm_client=llm_client,
            token_store=CompileTokenStore(db_pool),
        )

    async def _registration_svc() -> ConceptRegistrationService:
        return ConceptRegistrationService()

    async def _task_svc() -> TaskAuthoringService:
        return TaskAuthoringService(
            task_store=TaskStore(db_pool),
            definition_registry=DefinitionRegistry(store=DefinitionStore(db_pool)),
            llm_client=llm_client,
        )

    app.dependency_overrides[get_concept_compiler_service]    = _compiler_svc
    app.dependency_overrides[get_concept_registration_service] = _registration_svc
    app.dependency_overrides[get_task_authoring_service]      = _task_svc
    return app


# ── Shared request bodies ──────────────────────────────────────────────────────

_TASK_BODY = {
    "intent":       "alert when loan repayment ratio is below 0.80",
    "entity_scope": "loan",
    "delivery": {
        "type":     "webhook",
        "endpoint": "https://concurrency-test.example.com/hook",
    },
    "stream":          False,
    "return_reasoning": False,
}

_COMPILE_BODY = {
    "identifier":       "loan.repayment_ratio",
    "description":      "Ratio of on-time payments to total payments due over 90 days",
    "output_type":      "float",
    "signal_names":     ["payments_on_time", "payments_due"],
    "stream":           False,
    "return_reasoning": False,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Five simultaneous task creations with the same entity
# ═══════════════════════════════════════════════════════════════════════════════

def test_concurrent_task_creation_same_entity(db_pool, run, llm_mock):
    """
    Five simultaneous POST /tasks with same intent + entity_scope.

    Verifies: no duplicate-key violations, no lost writes, 5 unique task_ids.

    Decision rows are NOT asserted here — POST /tasks does not create decision
    records. Only the execute pipeline writes to the decisions table.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = await asyncio.gather(*[
                client.post("/tasks", json=_TASK_BODY)
                for _ in range(5)
            ])
        return responses

    responses = run(_go())

    # All requests must succeed — no 5xx, no constraint errors
    for i, resp in enumerate(responses):
        assert resp.status_code == 200, (
            f"Request {i} failed with {resp.status_code}: {resp.text}"
        )

    # DB: exactly 5 task rows
    rows = run(db_pool.fetch("SELECT task_id FROM tasks"))
    assert len(rows) == 5, (
        f"Expected 5 task rows in DB; got {len(rows)}"
    )

    # All task_ids must be distinct — no shared rows
    task_ids = [str(r["task_id"]) for r in rows]
    assert len(set(task_ids)) == 5, (
        f"task_ids not unique: {task_ids}"
    )

    # Response task_ids must also match DB
    resp_ids = [r.json()["task_id"] for r in responses]
    assert set(resp_ids) == set(task_ids), (
        "Response task_ids do not match DB rows"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Five simultaneous registers with the SAME compile token
# ═══════════════════════════════════════════════════════════════════════════════

def test_concurrent_concept_registration_same_token(
    db_pool, run, llm_mock, loan_compile_request
):
    """
    Five simultaneous POST /concepts/register with the same compile_token.

    Verifies: exactly 1 succeeds (201), exactly 4 are rejected as consumed (409).
    The atomic UPDATE...RETURNING in consume() guarantees exactly-once redemption.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Sequential compile — get one token
            compile_resp = await client.post("/concepts/compile", json={
                "identifier":       loan_compile_request.identifier,
                "description":      loan_compile_request.description,
                "output_type":      loan_compile_request.output_type,
                "signal_names":     loan_compile_request.signal_names,
                "stream":           False,
                "return_reasoning": False,
            })
            assert compile_resp.status_code == 201, (
                f"compile failed: {compile_resp.text}"
            )
            token = compile_resp.json()["compile_token"]

            # 5 concurrent registers with the SAME token
            register_body = {
                "compile_token": token,
                "identifier":    loan_compile_request.identifier,
            }
            responses = await asyncio.gather(*[
                client.post("/concepts/register", json=register_body)
                for _ in range(5)
            ])
        return responses

    responses = run(_go())

    statuses = [r.status_code for r in responses]

    # Exactly 1 success
    assert statuses.count(201) == 1, (
        f"Expected exactly 1 × 201; got: {statuses}"
    )

    # Remaining 4 must all be 409 compile_token_consumed
    assert statuses.count(409) == 4, (
        f"Expected exactly 4 × 409; got: {statuses}"
    )

    # No 500s or unexpected codes
    assert all(s in (201, 409) for s in statuses), (
        f"Unexpected status codes: {statuses}"
    )

    # DB: exactly 1 concept row for this identifier
    rows = run(db_pool.fetch(
        "SELECT definition_id FROM definitions WHERE definition_id = $1 AND definition_type = 'concept'",
        loan_compile_request.identifier,
    ))
    assert len(rows) == 1, (
        f"Expected 1 concept row in DB; got {len(rows)}"
    )

    # All 409 responses must have the correct error type
    for resp in responses:
        if resp.status_code == 409:
            body = resp.json()
            err_type = body.get("error", {}).get("type", "")
            assert err_type == "compile_token_consumed", (
                f"Expected compile_token_consumed error; got: {body}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Five simultaneous compiles for the same identifier
# ═══════════════════════════════════════════════════════════════════════════════

def test_concurrent_compile_same_identifier(db_pool, run, llm_mock, loan_compile_request):
    """
    Five simultaneous POST /concepts/compile with the same identifier.

    Verifies: all 5 succeed (201), produce distinct token strings, and create
    5 rows in the compile_tokens table. Concurrent compilation of the same
    identifier must be safe — no locking at compile time.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = await asyncio.gather(*[
                client.post("/concepts/compile", json={
                    "identifier":       loan_compile_request.identifier,
                    "description":      loan_compile_request.description,
                    "output_type":      loan_compile_request.output_type,
                    "signal_names":     loan_compile_request.signal_names,
                    "stream":           False,
                    "return_reasoning": False,
                })
                for _ in range(5)
            ])
        return responses

    responses = run(_go())

    # All must succeed
    for i, resp in enumerate(responses):
        assert resp.status_code == 201, (
            f"Compile {i} failed with {resp.status_code}: {resp.text}"
        )

    # All compile_tokens must be distinct strings
    tokens = [r.json()["compile_token"] for r in responses]
    assert len(set(tokens)) == 5, (
        f"compile_tokens are not all distinct: {tokens}"
    )

    # DB: exactly 5 rows in compile_tokens for this identifier
    rows = run(db_pool.fetch(
        "SELECT token_string FROM compile_tokens WHERE identifier = $1",
        loan_compile_request.identifier,
    ))
    assert len(rows) == 5, (
        f"Expected 5 compile_token rows in DB; got {len(rows)}"
    )

    # Confirm all token strings in DB match the responses
    db_token_strings = {str(r["token_string"]) for r in rows}
    assert set(tokens) == db_token_strings, (
        "Token strings in DB do not match response tokens"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — Ten simultaneous task creations with different entity_scope values
# ═══════════════════════════════════════════════════════════════════════════════

def test_concurrent_task_creation_different_entities(db_pool, run, llm_mock):
    """
    Ten simultaneous POST /tasks with distinct entity_scope values.

    Verifies:
      - All 10 succeed (200).
      - 10 task rows in DB.
      - Each row has the correct entity_scope (no cross-entity contamination).
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    entities = [f"entity_{i:03d}" for i in range(1, 11)]

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = await asyncio.gather(*[
                client.post("/tasks", json={
                    **_TASK_BODY,
                    "entity_scope": entity,
                })
                for entity in entities
            ])
        return responses

    responses = run(_go())

    # All must succeed
    for i, resp in enumerate(responses):
        assert resp.status_code == 200, (
            f"Entity {entities[i]} failed: {resp.status_code}: {resp.text}"
        )

    # DB: exactly 10 rows
    rows = run(db_pool.fetch("SELECT task_id, entity_scope FROM tasks"))
    assert len(rows) == 10, (
        f"Expected 10 task rows; got {len(rows)}"
    )

    # All entity_scopes present, no cross-entity contamination
    db_entity_scopes = {r["entity_scope"] for r in rows}
    assert db_entity_scopes == set(entities), (
        f"entity_scope mismatch: DB has {db_entity_scopes}, expected {set(entities)}"
    )

    # Each entity_scope appears exactly once
    scope_counts = {}
    for r in rows:
        scope_counts[r["entity_scope"]] = scope_counts.get(r["entity_scope"], 0) + 1
    duplicates = {k: v for k, v in scope_counts.items() if v > 1}
    assert not duplicates, (
        f"Cross-entity contamination detected — duplicate entity_scopes: {duplicates}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — Concurrent compile + register race
# ═══════════════════════════════════════════════════════════════════════════════

def test_concurrent_compile_and_register_race(
    db_pool, run, llm_mock, loan_compile_request
):
    """
    Compile and register operations run concurrently without interference.

    Sequence:
      1. Pre-compile one token sequentially (to get a known valid token).
      2. Fire 3 new compiles + 1 register of the pre-compiled token, all concurrently.

    Verifies:
      - The register succeeds (201).
      - All 3 concurrent compiles succeed (201).
      - The pre-compiled token is consumed exactly once.
      - The concurrent compiles did not consume or invalidate the pre-compiled token.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Step 1: Sequential pre-compile
            compile_resp = await client.post("/concepts/compile", json={
                "identifier":       loan_compile_request.identifier,
                "description":      loan_compile_request.description,
                "output_type":      loan_compile_request.output_type,
                "signal_names":     loan_compile_request.signal_names,
                "stream":           False,
                "return_reasoning": False,
            })
            assert compile_resp.status_code == 201, compile_resp.text
            pre_token = compile_resp.json()["compile_token"]

            # Step 2: 3 concurrent compiles + 1 register, all in parallel
            compile_coros = [
                client.post("/concepts/compile", json={
                    "identifier":       loan_compile_request.identifier,
                    "description":      loan_compile_request.description,
                    "output_type":      loan_compile_request.output_type,
                    "signal_names":     loan_compile_request.signal_names,
                    "stream":           False,
                    "return_reasoning": False,
                })
                for _ in range(3)
            ]
            register_coro = client.post("/concepts/register", json={
                "compile_token": pre_token,
                "identifier":    loan_compile_request.identifier,
            })

            results = await asyncio.gather(*compile_coros, register_coro)
        return pre_token, results

    pre_token, results = run(_go())

    compile_responses = results[:3]
    register_response = results[3]

    # Register must succeed
    assert register_response.status_code == 201, (
        f"Register failed: {register_response.status_code}: {register_response.text}"
    )
    data = register_response.json()
    assert data["concept_id"] == loan_compile_request.identifier

    # All concurrent compiles must succeed
    for i, resp in enumerate(compile_responses):
        assert resp.status_code == 201, (
            f"Concurrent compile {i} failed: {resp.status_code}: {resp.text}"
        )

    # All 4 tokens (1 pre-compiled + 3 new) are distinct
    new_tokens = [r.json()["compile_token"] for r in compile_responses]
    all_tokens = [pre_token] + new_tokens
    assert len(set(all_tokens)) == 4, (
        f"Token strings not all distinct: {all_tokens}"
    )

    # Pre-compiled token is consumed in DB (used=True)
    row = run(db_pool.fetchrow(
        "SELECT used FROM compile_tokens WHERE token_string = $1",
        pre_token,
    ))
    assert row is not None, "Pre-compiled token not found in DB"
    assert row["used"] is True, (
        "Pre-compiled token should be marked used=True after register"
    )
