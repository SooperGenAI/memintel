"""
tests/integration/test_sse_integration.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for M-6 SSE streaming.

These tests require a running PostgreSQL database. They are automatically
skipped if the DB is unavailable (via the session-level _database_setup
fixture in conftest.py).

Test coverage
─────────────
 I-1  Full /tasks stream:      4 cor_step events + 1 cor_complete.
 I-2  Full /concepts/compile:  4 cor_step events + 1 cor_complete with compile_token.
 I-3  Simulated step timeout:  cor_error emitted, no events after it.

All tests use the service layer directly (no HTTP stack) with a real DB pool
for persistence.  HTTP header tests use a lightweight TestClient with mocked
services to avoid DB dependency.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from app.models.concept_compile import CompileConceptRequest
from app.models.task import CreateTaskRequest, DeliveryConfig, DeliveryType
from app.registry.definitions import DefinitionRegistry
from app.services.concept_compiler import ConceptCompilerService
from app.services.task_authoring import TaskAuthoringService
from app.stores.compile_token import CompileTokenStore
from app.stores.definition import DefinitionStore
from app.stores.task import TaskStore


# ── Mock LLM client for integration tests ─────────────────────────────────────

class _MockIntegrationLLM:
    """Routes LLM responses by context step (concept compiler) or always returns
    a valid task output (task authoring)."""

    def generate_task(self, intent: str, context: dict) -> dict:
        step = context.get("step")

        # ── Concept compiler steps ─────────────────────────────────────────
        if step == 1:
            return {"summary": "Intent: measure repayment ratio", "outcome": "accepted"}
        if step == 2:
            return {"summary": "Signals: payments_on_time, payments_due", "outcome": "accepted"}
        if step == 3:
            return {
                "summary": "Formula: payments_on_time / payments_due",
                "outcome": "accepted",
                "formula_summary": "payments_on_time / payments_due",
                "signal_bindings": [
                    {"signal_name": "payments_on_time", "role": "numerator"},
                    {"signal_name": "payments_due", "role": "denominator"},
                ],
            }
        if step == 4:
            return {"summary": "Type float is valid", "outcome": "accepted"}

        # ── Task authoring (no step key) ───────────────────────────────────
        return {
            "concept": {
                "concept_id": "integration.churn_risk",
                "version": "v1",
                "namespace": "org",
                "output_type": "float",
                "description": "Churn risk score",
                "primitives": {
                    "account.active_user_rate_30d": {
                        "type": "float",
                        "missing_data_policy": "zero",
                    }
                },
                "features": {
                    "output": {
                        "op": "z_score_op",
                        "inputs": {"input": "account.active_user_rate_30d"},
                        "params": {},
                    }
                },
                "output_feature": "output",
            },
            "condition": {
                "condition_id": "integration.churn_condition",
                "version": "v1",
                "concept_id": "integration.churn_risk",
                "concept_version": "v1",
                "namespace": "org",
                "strategy": {
                    "type": "threshold",
                    "params": {"direction": "below", "value": 0.35},
                },
            },
            "action": {
                "action_id": "integration.alert_action",
                "version": "v1",
                "namespace": "org",
                "config": {"type": "webhook", "endpoint": "https://example.com/hook"},
                "trigger": {
                    "fire_on": "true",
                    "condition_id": "integration.churn_condition",
                    "condition_version": "v1",
                },
            },
        }


class _TimeoutLLM:
    """Simulates a step timeout by raising asyncio.TimeoutError."""

    def generate_task(self, intent: str, context: dict) -> dict:
        raise asyncio.TimeoutError("integration_test_timeout")


# ── Helper: collect events from async generator ───────────────────────────────

def _collect_events(pool, coro_factory):
    """Run coro_factory(pool) in the current event loop and return all events."""
    loop = asyncio.get_event_loop()

    async def _go():
        events = []
        async for ev in coro_factory(pool):
            events.append(ev)
        return events

    return loop.run_until_complete(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# I-1  Full /tasks stream: 4 cor_step + 1 cor_complete
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskSSEIntegration:

    def test_tasks_stream_event_sequence(self, db_pool, run):
        """Full streaming pipeline: 4 steps then cor_complete with task_id."""
        task_store = TaskStore(db_pool)
        def_store = DefinitionStore(db_pool)
        registry = DefinitionRegistry(store=def_store)
        svc = TaskAuthoringService(
            task_store=task_store,
            definition_registry=registry,
            llm_client=_MockIntegrationLLM(),
        )
        req = CreateTaskRequest(
            intent="alert on high churn risk",
            entity_scope="account",
            delivery=DeliveryConfig(
                type=DeliveryType.WEBHOOK,
                endpoint="https://integration-test.example.com/hook",
            ),
            stream=True,
        )

        events = run(_collect_async(svc.create_task_stream(req)))

        step_events = [e for e in events if e["event_type"] == "cor_step"]
        complete_events = [e for e in events if e["event_type"] == "cor_complete"]
        error_events = [e for e in events if e["event_type"] == "cor_error"]

        assert len(step_events) == 4, f"Expected 4 cor_step, got {len(step_events)}"
        assert len(complete_events) == 1, "Expected 1 cor_complete"
        assert len(error_events) == 0, f"Unexpected cor_error: {error_events}"

        indices = [e["data"]["step_index"] for e in step_events]
        assert indices == [1, 2, 3, 4]

        complete_data = complete_events[0]["data"]
        assert "task_id" in complete_data

    def test_tasks_non_stream_unchanged(self, db_pool, run):
        """Non-streaming create_task still returns a Task."""
        task_store = TaskStore(db_pool)
        def_store = DefinitionStore(db_pool)
        registry = DefinitionRegistry(store=def_store)
        svc = TaskAuthoringService(
            task_store=task_store,
            definition_registry=registry,
            llm_client=_MockIntegrationLLM(),
        )
        req = CreateTaskRequest(
            intent="alert on high churn risk",
            entity_scope="account",
            delivery=DeliveryConfig(
                type=DeliveryType.WEBHOOK,
                endpoint="https://integration-test.example.com/hook",
            ),
            stream=False,
        )
        result = run(svc.create_task(req))
        assert hasattr(result, "task_id")
        assert result.task_id is not None


# ═══════════════════════════════════════════════════════════════════════════════
# I-2  Full /concepts/compile stream: 4 cor_step + 1 cor_complete with compile_token
# ═══════════════════════════════════════════════════════════════════════════════

class TestConceptCompileSSEIntegration:

    def test_compile_stream_event_sequence(self, db_pool, run):
        """Full concept compile streaming: 4 steps then cor_complete with compile_token."""
        token_store = CompileTokenStore(db_pool)
        svc = ConceptCompilerService(
            llm_client=_MockIntegrationLLM(),
            token_store=token_store,
        )
        req = CompileConceptRequest(
            identifier="integration.test_concept",
            description="Integration test concept for SSE streaming",
            output_type="float",
            signal_names=["payments_on_time", "payments_due"],
            stream=True,
        )

        events = run(_collect_async(svc.compile_stream(req, pool=db_pool)))

        step_events = [e for e in events if e["event_type"] == "cor_step"]
        complete_events = [e for e in events if e["event_type"] == "cor_complete"]
        error_events = [e for e in events if e["event_type"] == "cor_error"]

        assert len(step_events) == 4, f"Expected 4, got {len(step_events)}"
        assert len(complete_events) == 1, "Expected 1 cor_complete"
        assert len(error_events) == 0, f"Unexpected cor_error: {error_events}"

        indices = [e["data"]["step_index"] for e in step_events]
        assert indices == [1, 2, 3, 4]

        complete_data = complete_events[0]["data"]
        assert "compile_token" in complete_data
        assert complete_data["compile_token"], "compile_token must be non-empty"

    def test_compile_non_stream_unchanged(self, db_pool, run):
        """Non-streaming compile still returns CompileConceptResponse."""
        from app.models.concept_compile import CompileConceptResponse
        token_store = CompileTokenStore(db_pool)
        svc = ConceptCompilerService(
            llm_client=_MockIntegrationLLM(),
            token_store=token_store,
        )
        req = CompileConceptRequest(
            identifier="integration.non_stream_concept",
            description="Non-streaming integration test",
            output_type="float",
            signal_names=["signal_a"],
            stream=False,
        )
        result = run(svc.compile(req, pool=db_pool))
        assert isinstance(result, CompileConceptResponse)
        assert result.compile_token


# ═══════════════════════════════════════════════════════════════════════════════
# I-3  Simulated step timeout: cor_error emitted, no events after it
# ═══════════════════════════════════════════════════════════════════════════════

class TestSSETimeoutIntegration:

    def test_task_stream_timeout_cor_error(self, db_pool, run):
        """Timeout in task authoring LLM call → cor_error as last event."""
        task_store = TaskStore(db_pool)
        def_store = DefinitionStore(db_pool)
        registry = DefinitionRegistry(store=def_store)
        svc = TaskAuthoringService(
            task_store=task_store,
            definition_registry=registry,
            llm_client=_TimeoutLLM(),
        )
        req = CreateTaskRequest(
            intent="alert on churn",
            entity_scope="account",
            delivery=DeliveryConfig(
                type=DeliveryType.WEBHOOK,
                endpoint="https://example.com/hook",
            ),
            stream=True,
        )

        events = run(_collect_async(svc.create_task_stream(req)))

        error_events = [e for e in events if e["event_type"] == "cor_error"]
        assert len(error_events) == 1, f"Expected 1 cor_error, got {error_events}"
        last_idx = next(i for i, e in enumerate(events) if e["event_type"] == "cor_error")
        assert last_idx == len(events) - 1, "cor_error must be last"

        error_data = error_events[0]["data"]
        assert "failure_reason" in error_data
        assert "failed_at_step" in error_data

    def test_compile_stream_timeout_cor_error(self, db_pool, run):
        """Timeout in concept compiler → cor_error as last event."""
        token_store = CompileTokenStore(db_pool)
        svc = ConceptCompilerService(
            llm_client=_TimeoutLLM(),
            token_store=token_store,
        )
        req = CompileConceptRequest(
            identifier="integration.timeout_concept",
            description="Timeout test",
            output_type="float",
            signal_names=["s"],
            stream=True,
        )

        events = run(_collect_async(svc.compile_stream(req, pool=db_pool)))

        error_events = [e for e in events if e["event_type"] == "cor_error"]
        assert len(error_events) == 1
        last_idx = next(i for i, e in enumerate(events) if e["event_type"] == "cor_error")
        assert last_idx == len(events) - 1, "cor_error must be last"


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _collect_async(gen) -> list[dict]:
    """Collect all items from an async generator."""
    events = []
    async for item in gen:
        events.append(item)
    return events
