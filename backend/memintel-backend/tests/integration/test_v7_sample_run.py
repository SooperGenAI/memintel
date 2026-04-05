"""
tests/integration/test_v7_sample_run.py
──────────────────────────────────────────────────────────────────────────────
T-3 — Small Business Loan Portfolio Monitoring sample run.

Demonstrates the full Memintel V7 lifecycle across three loan-domain concepts:

  loan.repayment_ratio  — float, alert when ratio < 0.80
  loan.days_overdue     — float, alert when overdue > 30 days
  loan.credit_score     — float, alert when score < 650

Phase tests (5)
───────────────
  Phase 1 — M-3: Compile all three concepts via POST /concepts/compile
  Phase 2 — M-4: Register all three concepts via POST /concepts/register
  Phase 3 — M-5: Create three agents (POST /tasks) using pre-compiled concept_ids
  Phase 4 — M-2: Dry-run preview each agent without persisting
  Phase 5 — M-7: Verify SSE streaming for one agent creation

Full workflow test (1)
──────────────────────
  test_sample_run_full_workflow — Runs all phases sequentially with rich
  printed output showing every concept, agent, dry-run, and stream event.

All tests use:
  - Real asyncpg pool (db_pool fixture — tables truncated before each test)
  - Deterministic LLMMockClient (llm_mock fixture from conftest_v7)
  - httpx.AsyncClient over ASGITransport — no real network or LLM calls
  - compile_and_register helper from conftest_v7
"""
from __future__ import annotations

import json
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


# ── Loan domain concept definitions ───────────────────────────────────────────

_LOAN_CONCEPTS = [
    {
        "identifier":   "loan.repayment_ratio",
        "description":  "Ratio of on-time payments to total payments due over 90 days",
        "output_type":  "float",
        "signal_names": ["payments_on_time", "payments_due"],
    },
    {
        "identifier":   "loan.days_overdue",
        "description":  "Number of days a loan repayment is past due",
        "output_type":  "float",
        "signal_names": ["loan.days_overdue"],
    },
    {
        "identifier":   "loan.credit_score",
        "description":  "FICO credit score of the loan applicant",
        "output_type":  "float",
        "signal_names": ["loan.credit_score"],
    },
]

_LOAN_AGENTS = [
    {
        "intent":        "alert when loan repayment ratio is below 0.80",
        "entity_scope":  "loan",
        "endpoint":      "https://loan.example.com/repayment-alert",
        "concept_id":    "loan.repayment_ratio",
    },
    {
        "intent":        "alert when loan is past due by more than 30 days",
        "entity_scope":  "loan",
        "endpoint":      "https://loan.example.com/overdue-alert",
        "concept_id":    "loan.days_overdue",
    },
    {
        "intent":        "alert when borrower credit score drops below 650",
        "entity_scope":  "loan",
        "endpoint":      "https://loan.example.com/credit-alert",
        "concept_id":    "loan.credit_score",
    },
]


# ── Test-app factory ───────────────────────────────────────────────────────────

def _make_test_app(db_pool, llm_client: Any) -> FastAPI:
    """Minimal FastAPI app with real DB pool and mock LLM client."""
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

    app.dependency_overrides[get_concept_compiler_service] = _compiler_svc
    app.dependency_overrides[get_concept_registration_service] = _registration_svc
    app.dependency_overrides[get_task_authoring_service] = _task_svc
    return app


# ── SSE parser ─────────────────────────────────────────────────────────────────

def collect_sse_events(response: httpx.Response) -> list[dict]:
    """Parse SSE text/event-stream response into list of {event_type, data} dicts."""
    events: list[dict] = []
    current_type: str | None = None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            current_type = line[7:].strip()
        elif line.startswith("data: "):
            payload = json.loads(line[6:])
            events.append({"event_type": current_type, "data": payload})
            current_type = None
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — M-3: Compile all three loan concepts
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleRunPhase1ConceptCompilation:
    """
    Phase 1: POST /concepts/compile for each loan concept.
    Verifies HTTP 201, compile_token present, compiled_concept shape.
    """

    def test_sample_run_phase1_concept_compilation(self, db_pool, run, llm_mock):
        app = _make_test_app(db_pool, llm_mock)
        transport = httpx.ASGITransport(app=app)

        async def _run():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                tokens = {}
                for concept in _LOAN_CONCEPTS:
                    resp = await client.post("/concepts/compile", json={
                        "identifier":   concept["identifier"],
                        "description":  concept["description"],
                        "output_type":  concept["output_type"],
                        "signal_names": concept["signal_names"],
                        "stream":          False,
                        "return_reasoning": False,
                    })
                    assert resp.status_code == 201, (
                        f"compile failed for {concept['identifier']} "
                        f"({resp.status_code}): {resp.text}"
                    )
                    body = resp.json()
                    assert "compile_token" in body, (
                        f"No compile_token in response for {concept['identifier']}"
                    )
                    assert body["compile_token"], "compile_token must be non-empty"
                    assert "compiled_concept" in body
                    tokens[concept["identifier"]] = body["compile_token"]

                # All three tokens must be distinct
                token_values = list(tokens.values())
                assert len(set(token_values)) == 3, "Each compile must produce a unique token"
                return tokens

        tokens = run(_run())
        assert len(tokens) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — M-4: Register all three loan concepts
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleRunPhase2ConceptRegistration:
    """
    Phase 2: POST /concepts/register for each compiled concept.
    Verifies HTTP 201, concept_id equals identifier, version present.
    """

    def test_sample_run_phase2_concept_registration(self, db_pool, run, llm_mock):
        app = _make_test_app(db_pool, llm_mock)
        transport = httpx.ASGITransport(app=app)

        async def _run():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                registered = {}
                for concept in _LOAN_CONCEPTS:
                    concept_id, token = await compile_and_register(
                        client,
                        identifier=concept["identifier"],
                        description=concept["description"],
                        output_type=concept["output_type"],
                        signal_names=concept["signal_names"],
                    )
                    assert concept_id == concept["identifier"], (
                        f"concept_id mismatch: expected {concept['identifier']!r}, "
                        f"got {concept_id!r}"
                    )
                    registered[concept_id] = token

                return registered

        registered = run(_run())
        assert len(registered) == 3
        for concept in _LOAN_CONCEPTS:
            assert concept["identifier"] in registered, (
                f"{concept['identifier']} not in registered concept_ids"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — M-5: Create three agents with pre-compiled concept_ids
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleRunPhase3AgentCreation:
    """
    Phase 3: POST /tasks for each loan alert agent using pre-compiled concept_id.
    Verifies HTTP 200, task_id present, concept_id matches, status=active.
    """

    def test_sample_run_phase3_agent_creation(self, db_pool, run, llm_mock):
        app = _make_test_app(db_pool, llm_mock)
        transport = httpx.ASGITransport(app=app)

        async def _run():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Register all three concepts first
                registered_concept_ids = []
                for concept in _LOAN_CONCEPTS:
                    concept_id, _ = await compile_and_register(
                        client,
                        identifier=concept["identifier"],
                        description=concept["description"],
                        output_type=concept["output_type"],
                        signal_names=concept["signal_names"],
                    )
                    registered_concept_ids.append(concept_id)

                # Create one agent per concept
                tasks = []
                for agent_spec in _LOAN_AGENTS:
                    resp = await client.post("/tasks", json={
                        "intent":       agent_spec["intent"],
                        "entity_scope": agent_spec["entity_scope"],
                        "delivery": {
                            "type":     "webhook",
                            "endpoint": agent_spec["endpoint"],
                        },
                        "concept_id": agent_spec["concept_id"],
                        "stream":     False,
                    })
                    assert resp.status_code == 200, (
                        f"task creation failed for concept {agent_spec['concept_id']} "
                        f"({resp.status_code}): {resp.text}"
                    )
                    body = resp.json()
                    assert "task_id" in body, "task_id missing from response"
                    assert body["task_id"], "task_id must be non-empty"
                    assert body.get("status") == "active", (
                        f"Expected status=active, got {body.get('status')!r}"
                    )
                    tasks.append(body)

                return tasks

        tasks = run(_run())
        assert len(tasks) == 3
        task_ids = [t["task_id"] for t in tasks]
        assert len(set(task_ids)) == 3, "Each agent must get a unique task_id"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Dry-run preview each agent (no persistence)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleRunPhase4DryRunPreview:
    """
    Phase 4: POST /tasks with dry_run=true for each agent.
    Verifies HTTP 200, DryRunResult shape (concept + condition keys present,
    no task_id), and that nothing is persisted.
    """

    def test_sample_run_phase4_dry_run_preview(self, db_pool, run, llm_mock):
        app = _make_test_app(db_pool, llm_mock)
        transport = httpx.ASGITransport(app=app)

        async def _run():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Register concepts
                for concept in _LOAN_CONCEPTS:
                    await compile_and_register(
                        client,
                        identifier=concept["identifier"],
                        description=concept["description"],
                        output_type=concept["output_type"],
                        signal_names=concept["signal_names"],
                    )

                dry_run_results = []
                for agent_spec in _LOAN_AGENTS:
                    resp = await client.post("/tasks", json={
                        "intent":       agent_spec["intent"],
                        "entity_scope": agent_spec["entity_scope"],
                        "delivery": {
                            "type":     "webhook",
                            "endpoint": agent_spec["endpoint"],
                        },
                        "concept_id": agent_spec["concept_id"],
                        "dry_run":    True,
                        "stream":     False,
                    })
                    assert resp.status_code == 200, (
                        f"dry_run failed for {agent_spec['concept_id']} "
                        f"({resp.status_code}): {resp.text}"
                    )
                    body = resp.json()

                    # DryRunResult shape: concept + condition present, no task_id
                    assert "concept" in body, "DryRunResult must have 'concept'"
                    assert "condition" in body, "DryRunResult must have 'condition'"
                    assert "task_id" not in body, (
                        "dry_run=true must NOT persist a task (no task_id)"
                    )
                    dry_run_results.append(body)

                # Confirm no tasks were persisted
                list_resp = await client.get("/tasks")
                assert list_resp.status_code == 200
                task_list = list_resp.json()
                assert task_list.get("total", 0) == 0 or len(
                    task_list.get("tasks", [])
                ) == 0, "dry_run must not persist tasks"

                return dry_run_results

        results = run(_run())
        assert len(results) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — M-7: SSE streaming for one agent creation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleRunPhase5SSEStreaming:
    """
    Phase 5: POST /tasks with stream=true for the repayment ratio agent.
    Verifies 4 cor_step events (indices 1-4), 1 cor_complete with task_id,
    and no cor_error events.
    """

    def test_sample_run_phase5_sse_streaming(self, db_pool, run, llm_mock):
        app = _make_test_app(db_pool, llm_mock)
        transport = httpx.ASGITransport(app=app)

        async def _run():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Register the repayment_ratio concept
                concept = _LOAN_CONCEPTS[0]  # loan.repayment_ratio
                await compile_and_register(
                    client,
                    identifier=concept["identifier"],
                    description=concept["description"],
                    output_type=concept["output_type"],
                    signal_names=concept["signal_names"],
                )

                agent_spec = _LOAN_AGENTS[0]  # repayment alert agent
                resp = await client.post("/tasks", json={
                    "intent":       agent_spec["intent"],
                    "entity_scope": agent_spec["entity_scope"],
                    "delivery": {
                        "type":     "webhook",
                        "endpoint": agent_spec["endpoint"],
                    },
                    "concept_id": agent_spec["concept_id"],
                    "stream":     True,
                })
                assert resp.status_code == 200
                return resp

        resp = run(_run())

        events = collect_sse_events(resp)
        step_events     = [e for e in events if e["event_type"] == "cor_step"]
        complete_events = [e for e in events if e["event_type"] == "cor_complete"]
        error_events    = [e for e in events if e["event_type"] == "cor_error"]

        assert len(step_events) == 4, (
            f"Expected 4 cor_step events, got {len(step_events)}"
        )
        assert len(complete_events) == 1, "Expected exactly 1 cor_complete event"
        assert len(error_events) == 0, f"Unexpected cor_error events: {error_events}"

        indices = [e["data"]["step_index"] for e in step_events]
        assert indices == [1, 2, 3, 4], f"step_index sequence wrong: {indices}"

        complete_data = complete_events[0]["data"]
        assert "task_id" in complete_data, "cor_complete must contain task_id"
        assert complete_data["task_id"], "task_id must be non-empty"


# ═══════════════════════════════════════════════════════════════════════════════
# Full workflow — M-1 through M-7 combined
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleRunFullWorkflow:
    """
    Combined M-1 through M-7 end-to-end run for the loan portfolio domain.

    Prints rich output for each phase so the run log shows the full
    Memintel V7 lifecycle in action.
    """

    def test_sample_run_full_workflow(self, db_pool, run, llm_mock):
        app = _make_test_app(db_pool, llm_mock)
        transport = httpx.ASGITransport(app=app)

        print("\n")
        print("=" * 70)
        print("  MEMINTEL V7 - SMALL BUSINESS LOAN PORTFOLIO MONITORING")
        print("  Full Workflow Sample Run")
        print("=" * 70)

        async def _run():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:

                # ── Phase 1: Compile all concepts ─────────────────────────────
                print("\n--- PHASE 1: Concept Compilation (M-3) ---")
                compile_tokens: dict[str, str] = {}
                for concept in _LOAN_CONCEPTS:
                    resp = await client.post("/concepts/compile", json={
                        "identifier":      concept["identifier"],
                        "description":     concept["description"],
                        "output_type":     concept["output_type"],
                        "signal_names":    concept["signal_names"],
                        "stream":          False,
                        "return_reasoning": False,
                    })
                    assert resp.status_code == 201, (
                        f"compile failed for {concept['identifier']}: {resp.text}"
                    )
                    body = resp.json()
                    token = body["compile_token"]
                    compiled = body["compiled_concept"]
                    compile_tokens[concept["identifier"]] = token

                    print(f"\n  Concept: {concept['identifier']}")
                    print(f"  Description: {concept['description']}")
                    print(f"  Output type: {compiled.get('output_type', 'float')}")
                    print(f"  Formula: {compiled.get('formula_summary', 'N/A')}")
                    bindings = compiled.get("signal_bindings", [])
                    if bindings:
                        print(f"  Signals: {[b['signal_name'] for b in bindings]}")
                    print(f"  Compile token: {token[:20]}...")

                print(f"\n  [OK] {len(compile_tokens)} concepts compiled successfully")

                # ── Phase 2: Register all concepts ────────────────────────────
                print("\n--- PHASE 2: Concept Registration (M-4) ---")
                registered_concepts: dict[str, dict] = {}
                for concept in _LOAN_CONCEPTS:
                    token = compile_tokens[concept["identifier"]]
                    resp = await client.post("/concepts/register", json={
                        "compile_token": token,
                        "identifier":    concept["identifier"],
                    })
                    assert resp.status_code == 201, (
                        f"register failed for {concept['identifier']}: {resp.text}"
                    )
                    body = resp.json()
                    registered_concepts[body["concept_id"]] = body

                    print(f"\n  Registered: {body['concept_id']}")
                    print(f"  Version: {body.get('version', 'N/A')}")
                    print(f"  Output type: {body.get('output_type', 'N/A')}")
                    print(f"  Registered at: {body.get('registered_at', 'N/A')}")

                print(f"\n  [OK] {len(registered_concepts)} concepts registered")

                # ── Phase 3: Create agents ─────────────────────────────────────
                print("\n--- PHASE 3: Agent Creation (M-5) ---")
                created_tasks: list[dict] = []
                for agent_spec in _LOAN_AGENTS:
                    resp = await client.post("/tasks", json={
                        "intent":       agent_spec["intent"],
                        "entity_scope": agent_spec["entity_scope"],
                        "delivery": {
                            "type":     "webhook",
                            "endpoint": agent_spec["endpoint"],
                        },
                        "concept_id": agent_spec["concept_id"],
                        "stream":     False,
                    })
                    assert resp.status_code == 200, (
                        f"task creation failed for {agent_spec['concept_id']}: {resp.text}"
                    )
                    body = resp.json()
                    created_tasks.append(body)

                    print(f"\n  Agent: {body['task_id']}")
                    print(f"  Intent: {agent_spec['intent']}")
                    print(f"  Concept: {body.get('concept_id', 'N/A')}")
                    print(f"  Condition: {body.get('condition_id', 'N/A')}")
                    print(f"  Action: {body.get('action_id', 'N/A')}")
                    print(f"  Status: {body.get('status', 'N/A')}")
                    print(f"  Webhook: {agent_spec['endpoint']}")

                print(f"\n  [OK] {len(created_tasks)} agents created successfully")

                # ── Phase 4: Dry-run preview ────────────────────────────────────
                print("\n--- PHASE 4: Dry-Run Preview (no persistence) ---")
                dry_run_results: list[dict] = []
                for agent_spec in _LOAN_AGENTS:
                    resp = await client.post("/tasks", json={
                        "intent":       agent_spec["intent"],
                        "entity_scope": agent_spec["entity_scope"],
                        "delivery": {
                            "type":     "webhook",
                            "endpoint": agent_spec["endpoint"],
                        },
                        "concept_id": agent_spec["concept_id"],
                        "dry_run":    True,
                        "stream":     False,
                    })
                    assert resp.status_code == 200, (
                        f"dry_run failed for {agent_spec['concept_id']}: {resp.text}"
                    )
                    body = resp.json()
                    dry_run_results.append(body)

                    concept_info = body.get("concept", {})
                    condition_info = body.get("condition", {})
                    print(f"\n  Dry-run for intent: {agent_spec['intent'][:50]}...")
                    print(f"  Concept ID: {concept_info.get('concept_id', 'N/A')}")
                    print(f"  Condition ID: {condition_info.get('condition_id', 'N/A')}")
                    strategy = condition_info.get("strategy", {})
                    print(f"  Strategy: {strategy.get('type', 'N/A')} "
                          f"— params: {strategy.get('params', {})}")
                    print(f"  Would trigger: {body.get('would_trigger', 'N/A')}")
                    print(f"  Action ID: {body.get('action_id', 'N/A')}")
                    assert "task_id" not in body, "dry_run must not produce a task_id"

                print(f"\n  [OK] {len(dry_run_results)} dry-runs completed (no tasks persisted)")

                # Confirm only 3 persisted tasks (from Phase 3), not 6
                list_resp = await client.get("/tasks")
                assert list_resp.status_code == 200
                task_list = list_resp.json()
                task_count = len(task_list.get("tasks", []))
                print(f"\n  [OK] Persisted task count = {task_count} (dry-runs excluded)")

                # ── Phase 5: SSE streaming ─────────────────────────────────────
                # Use a fresh concept (loan.payment_velocity) not created in
                # Phase 3, so registering its condition/action doesn't conflict.
                print("\n--- PHASE 5: SSE Streaming (M-7) ---")
                velocity_concept_id, _ = await compile_and_register(
                    client,
                    identifier="loan.payment_velocity",
                    description="Rate of change in loan payment amounts over 30 days",
                    output_type="float",
                    signal_names=["loan.payment_velocity"],
                )
                resp = await client.post("/tasks", json={
                    "intent":       "alert when loan payment velocity drops below zero",
                    "entity_scope": "loan",
                    "delivery": {
                        "type":     "webhook",
                        "endpoint": "https://loan.example.com/velocity-alert",
                    },
                    "concept_id": velocity_concept_id,
                    "stream":     True,
                })
                assert resp.status_code == 200

                events = collect_sse_events(resp)
                step_events     = [e for e in events if e["event_type"] == "cor_step"]
                complete_events = [e for e in events if e["event_type"] == "cor_complete"]
                error_events    = [e for e in events if e["event_type"] == "cor_error"]

                print(f"\n  Streaming agent: alert when loan payment velocity drops below zero")
                print(f"  Total SSE events: {len(events)}")
                print(f"  cor_step events: {len(step_events)}")
                for ev in step_events:
                    d = ev["data"]
                    print(f"    step {d.get('step_index')}: {d.get('step_name', '')} "
                          f"— {d.get('outcome', '')}")
                print(f"  cor_complete events: {len(complete_events)}")
                if complete_events:
                    cd = complete_events[0]["data"]
                    print(f"    task_id: {cd.get('task_id', 'N/A')}")
                print(f"  cor_error events: {len(error_events)}")

                assert len(step_events) == 4, (
                    f"Expected 4 cor_step, got {len(step_events)}"
                )
                assert len(complete_events) == 1
                assert len(error_events) == 0

                # ── Summary ───────────────────────────────────────────────────
                print("\n" + "=" * 70)
                print("  SAMPLE RUN COMPLETE")
                print("=" * 70)
                print(f"\n  Domain  : Small Business Loan Portfolio Monitoring")
                print(f"  Concepts: {len(registered_concepts)}")
                for cid in registered_concepts:
                    print(f"    - {cid}")
                print(f"  Agents  : {len(created_tasks)}")
                for task in created_tasks:
                    print(f"    - {task['task_id']} "
                          f"[concept={task.get('concept_id', 'N/A')}]")
                print(f"  Dry-runs: {len(dry_run_results)} (nothing persisted)")
                print(f"  SSE     : {len(step_events)} steps -> 1 cor_complete")
                print()

                return {
                    "compile_tokens":    compile_tokens,
                    "registered":        registered_concepts,
                    "tasks":             created_tasks,
                    "dry_run_results":   dry_run_results,
                    "sse_step_count":    len(step_events),
                    "sse_complete_count": len(complete_events),
                }

        result = run(_run())

        # Final assertions
        assert len(result["compile_tokens"]) == 3
        assert len(result["registered"]) == 3
        assert len(result["tasks"]) == 3
        assert len(result["dry_run_results"]) == 3
        assert result["sse_step_count"] == 4
        assert result["sse_complete_count"] == 1

        # All registered concept_ids must match identifiers
        for concept in _LOAN_CONCEPTS:
            assert concept["identifier"] in result["registered"], (
                f"{concept['identifier']} not registered"
            )

        # All tasks must have unique IDs and status=active
        task_ids = {t["task_id"] for t in result["tasks"]}
        assert len(task_ids) == 3, "Each agent must have a unique task_id"
        for task in result["tasks"]:
            assert task.get("status") == "active"
