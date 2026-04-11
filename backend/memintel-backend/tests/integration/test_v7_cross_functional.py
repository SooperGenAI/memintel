"""
tests/integration/test_v7_cross_functional.py
──────────────────────────────────────────────────────────────────────────────
V7 cross-functional integration tests.

These tests exercise the full HTTP stack (routes → services → stores → DB)
using a lightweight FastAPI test app with:
  - Real asyncpg pool (from the `db_pool` fixture — fresh per test, tables
    truncated before each test).
  - Deterministic LLMMockClient (from the `llm_mock` fixture).
  - httpx.AsyncClient over ASGITransport for zero-network HTTP calls.

Test scenarios
──────────────
  Scenario 1  — Full M3→M4→M5 chain (1 test)
  Scenario 2  — Vocabulary-bounded agent creation (4 tests)
  Scenario 3  — M7 SSE streaming (5 tests)
  Scenario 4  — Reasoning trace correctness (4 tests)
  Scenario 5  — Backward compatibility (2 tests)

All tests are synchronous (using the `run` fixture from conftest.py) and share
`llm_mock`, `loan_compile_request`, and `loan_task_request` fixtures from
conftest_v7.py.
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


# ── Test-app factory ───────────────────────────────────────────────────────────

def _make_test_app(db_pool, llm_client: Any) -> FastAPI:
    """
    Build a minimal FastAPI app for route-level integration testing.

    Sets app.state.db so that get_db() works without a real lifespan.
    Overrides all service dependencies to inject the supplied db_pool and
    llm_client, bypassing the production LLM-client selection logic.
    """
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
    """
    Parse an SSE text/event-stream response body into a list of event dicts.

    Each item has the shape::

        {"event_type": "<type>", "data": <parsed_json>}

    Blank lines and comment lines are ignored.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Full M3→M4→M5 chain
# ─────────────────────────────────────────────────────────────────────────────

class TestScenario1FullChain:
    """
    Verifies the end-to-end M3→M4→M5 flow:
      M-3  POST /concepts/compile   → compile_token
      M-4  POST /concepts/register  → concept_id
      M-5  POST /tasks with concept_id → task with pre-compiled concept
    """

    def test_full_concept_to_agent_chain(
        self,
        db_pool,
        run,
        llm_mock,
        loan_compile_request,
    ) -> None:
        """Compile → register → create task; task.concept_id equals registered ID."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                concept_id, _ = await compile_and_register(
                    client,
                    identifier=loan_compile_request.identifier,
                    description=loan_compile_request.description,
                    output_type=loan_compile_request.output_type,
                    signal_names=loan_compile_request.signal_names,
                )

                resp = await client.post("/tasks", json={
                    "intent": "alert on repayment ratio below threshold",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                    "concept_id": concept_id,
                    "return_reasoning": True,
                    "stream": False,
                })
                assert resp.status_code == 200, resp.text
                data = resp.json()

                # The task must use the registered concept.
                assert data["concept_id"] == concept_id

                # Steps 1 and 2 are skipped when concept is pre-compiled.
                trace = data.get("reasoning_trace")
                assert trace is not None, "reasoning_trace absent"
                steps = {s["step_index"]: s for s in trace["steps"]}
                assert steps[1]["outcome"] == "skipped", steps[1]
                assert steps[2]["outcome"] == "skipped", steps[2]

        run(_go())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Vocabulary-bounded agent creation
# ─────────────────────────────────────────────────────────────────────────────

_REPAYMENT_TASK = {
    "intent": "alert on repayment ratio",
    "entity_scope": "loan",
    "delivery": {"type": "webhook", "endpoint": "https://test.example.com/hook"},
    "stream": False,
}


class TestScenario2VocabularyBounded:
    """
    vocabulary_context bounds the concept IDs the LLM may select from.

    LLMMockClient routes "repayment" intent → concept_id "loan.repayment_ratio".
    Tests verify accept / reject / absent / empty-list behaviour.
    """

    def test_vocabulary_bounded_agent_creation_accepts(
        self, db_pool, run, llm_mock
    ) -> None:
        """available_concept_ids contains the LLM-selected concept → 200 OK."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    **_REPAYMENT_TASK,
                    "vocabulary_context": {
                        "available_concept_ids": ["loan.repayment_ratio"],
                        "available_condition_ids": ["loan.repayment_below_threshold"],
                    },
                })
                assert resp.status_code == 200, resp.text

        run(_go())

    def test_vocabulary_bounded_agent_creation_rejects(
        self, db_pool, run, llm_mock
    ) -> None:
        """available_concept_ids does NOT contain the LLM-selected concept → 422."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    **_REPAYMENT_TASK,
                    "vocabulary_context": {
                        "available_concept_ids": ["some.other.concept"],
                        "available_condition_ids": ["some.other.condition"],
                    },
                })
                assert resp.status_code == 422, resp.text

        run(_go())

    def test_vocabulary_context_absent_uses_global_fallback(
        self, db_pool, run, llm_mock
    ) -> None:
        """No vocabulary_context → LLM selects freely → 200 OK."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json=_REPAYMENT_TASK)
                assert resp.status_code == 200, resp.text

        run(_go())

    def test_vocabulary_context_empty_lists_rejected(
        self, db_pool, run, llm_mock
    ) -> None:
        """Both lists empty → pre-LLM VocabularyMismatchError → 422; LLM never called."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    **_REPAYMENT_TASK,
                    "vocabulary_context": {
                        "available_concept_ids": [],
                        "available_condition_ids": [],
                    },
                })
                assert resp.status_code == 422, resp.text

        run(_go())
        # Pre-LLM guard fires before any generate_task call.
        assert llm_mock.call_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — M7 SSE streaming
# ─────────────────────────────────────────────────────────────────────────────

class _FailLLM:
    """Raises RuntimeError on every LLM call — triggers cor_error."""

    def generate_task(self, intent: str, context: dict) -> dict:
        raise RuntimeError("forced_failure_for_sse_error_test")

    def generate_compile_step(self, intent: str, context: dict) -> dict:
        raise RuntimeError("forced_failure_for_sse_error_test")


class TestScenario3SSEStreaming:
    """
    Verifies SSE stream shape for compile, task creation, dry-run, vocabulary,
    and error paths.
    """

    def test_concept_compile_sse_stream(self, db_pool, run, llm_mock) -> None:
        """POST /concepts/compile stream=True → 4 cor_step + 1 cor_complete with compile_token."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/concepts/compile", json={
                    "identifier": "sse.repayment_concept",
                    "description": "SSE repayment ratio stream test",
                    "output_type": "float",
                    "signal_names": ["payments_on_time", "payments_due"],
                    "stream": True,
                })
                assert resp.status_code == 200, resp.text

                events = collect_sse_events(resp)
                step_events    = [e for e in events if e["event_type"] == "cor_step"]
                complete_events = [e for e in events if e["event_type"] == "cor_complete"]
                error_events   = [e for e in events if e["event_type"] == "cor_error"]

                assert len(step_events) == 4, f"Expected 4 cor_step, got {len(step_events)}"
                assert len(complete_events) == 1, "Expected 1 cor_complete"
                assert len(error_events) == 0, f"Unexpected cor_error: {error_events}"
                assert [e["data"]["step_index"] for e in step_events] == [1, 2, 3, 4]
                assert "compile_token" in complete_events[0]["data"]
                assert complete_events[0]["data"]["compile_token"]

        run(_go())

    def test_task_create_sse_stream(self, db_pool, run, llm_mock) -> None:
        """POST /tasks stream=True → 4 cor_step + 1 cor_complete with task_id."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    "intent": "alert on overdue loans",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                    "stream": True,
                })
                assert resp.status_code == 200, resp.text

                events = collect_sse_events(resp)
                step_events     = [e for e in events if e["event_type"] == "cor_step"]
                complete_events  = [e for e in events if e["event_type"] == "cor_complete"]
                error_events    = [e for e in events if e["event_type"] == "cor_error"]

                assert len(step_events) == 4, f"Expected 4 cor_step, got {len(step_events)}"
                assert len(complete_events) == 1, "Expected 1 cor_complete"
                assert len(error_events) == 0, f"Unexpected cor_error: {error_events}"
                assert [e["data"]["step_index"] for e in step_events] == [1, 2, 3, 4]
                assert "task_id" in complete_events[0]["data"]
                assert complete_events[0]["data"]["task_id"]  # must be non-empty

        run(_go())

    def test_task_dry_run_sse_stream(self, db_pool, run, llm_mock) -> None:
        """POST /tasks dry_run=True stream=True → 4 cor_step + cor_complete (task_id=null)."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    "intent": "alert on overdue loans",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                    "stream": True,
                    "dry_run": True,
                })
                assert resp.status_code == 200, resp.text

                events = collect_sse_events(resp)
                step_events     = [e for e in events if e["event_type"] == "cor_step"]
                complete_events  = [e for e in events if e["event_type"] == "cor_complete"]

                assert len(step_events) == 4
                assert len(complete_events) == 1
                complete_data = complete_events[0]["data"]
                assert complete_data.get("task_id") is None
                assert complete_data.get("dry_run") is True

        run(_go())

    def test_sse_stream_with_vocabulary_context(self, db_pool, run, llm_mock) -> None:
        """stream=True with valid vocabulary_context → 4 cor_step + 1 cor_complete."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    "intent": "alert on repayment ratio",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                    "stream": True,
                    "vocabulary_context": {
                        "available_concept_ids": ["loan.repayment_ratio"],
                        "available_condition_ids": ["loan.repayment_below_threshold"],
                    },
                })
                assert resp.status_code == 200, resp.text

                events = collect_sse_events(resp)
                step_events     = [e for e in events if e["event_type"] == "cor_step"]
                complete_events  = [e for e in events if e["event_type"] == "cor_complete"]
                error_events    = [e for e in events if e["event_type"] == "cor_error"]

                assert len(step_events) == 4
                assert len(complete_events) == 1
                assert len(error_events) == 0, f"Unexpected cor_error: {error_events}"

        run(_go())

    def test_sse_stream_on_error_closes_cleanly(self, db_pool, run) -> None:
        """LLM failure in compile stream → cor_error emitted as last event."""
        fail_app = _make_test_app(db_pool, _FailLLM())

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=fail_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/concepts/compile", json={
                    "identifier": "fail.error_test_concept",
                    "description": "forced failure for SSE error test",
                    "output_type": "float",
                    "signal_names": ["s"],
                    "stream": True,
                })
                # HTTP 200 — status set before stream body is sent.
                assert resp.status_code == 200, resp.text

                events = collect_sse_events(resp)
                error_events = [e for e in events if e["event_type"] == "cor_error"]
                assert len(error_events) == 1, f"Expected 1 cor_error, got: {events}"
                # cor_error must be the last event in the stream.
                assert events[-1]["event_type"] == "cor_error", (
                    f"cor_error was not the last event: {[e['event_type'] for e in events]}"
                )
                error_data = error_events[0]["data"]
                assert "failure_reason" in error_data

        run(_go())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — Reasoning trace correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestScenario4ReasoningTrace:
    """
    Verifies reasoning_trace presence, structure, and step-outcome values.
    """

    def test_reasoning_trace_structure_live(self, db_pool, run, llm_mock) -> None:
        """return_reasoning=True, non-dry-run → trace has 4 steps with correct shape."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    "intent": "alert on overdue loans",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                    "return_reasoning": True,
                    "stream": False,
                })
                assert resp.status_code == 200, resp.text
                data = resp.json()

                assert "reasoning_trace" in data, "reasoning_trace absent"
                trace = data["reasoning_trace"]
                steps = trace["steps"]
                assert len(steps) == 4, f"Expected 4 steps, got {len(steps)}"
                assert [s["step_index"] for s in steps] == [1, 2, 3, 4]
                for step in steps:
                    assert step["outcome"] in ("accepted", "skipped", "failed"), step
                    assert "label" in step
                    assert "summary" in step
                # compilation_duration_ms is not set by the impl — may be absent or None.
                dur = trace.get("compilation_duration_ms")
                assert dur is None or dur > 0

        run(_go())

    def test_reasoning_trace_structure_dry_run(self, db_pool, run, llm_mock) -> None:
        """return_reasoning=True, dry_run=True → DryRunResult has no reasoning_trace."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    "intent": "alert on overdue loans",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                    "return_reasoning": True,
                    "dry_run": True,
                    "stream": False,
                })
                assert resp.status_code == 200, resp.text
                data = resp.json()
                # DryRunResult has no reasoning_trace field; must be absent from JSON.
                assert "reasoning_trace" not in data, (
                    "DryRunResult must not include reasoning_trace"
                )

        run(_go())

    def test_reasoning_trace_absent_by_default(self, db_pool, run, llm_mock) -> None:
        """return_reasoning defaults to False → reasoning_trace absent from JSON (Hard Rule 3)."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    "intent": "alert on overdue loans",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                    "stream": False,
                })
                assert resp.status_code == 200, resp.text
                data = resp.json()
                assert "reasoning_trace" not in data, (
                    "reasoning_trace must be absent when return_reasoning=False"
                )

        run(_go())

    def test_reasoning_trace_with_concept_id(
        self,
        db_pool,
        run,
        llm_mock,
        loan_compile_request,
    ) -> None:
        """return_reasoning=True + pre-compiled concept_id → steps 1 and 2 outcome='skipped'."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                concept_id, _ = await compile_and_register(
                    client,
                    identifier=loan_compile_request.identifier,
                    description=loan_compile_request.description,
                    output_type=loan_compile_request.output_type,
                    signal_names=loan_compile_request.signal_names,
                )
                resp = await client.post("/tasks", json={
                    "intent": "alert on repayment ratio below threshold",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                    "concept_id": concept_id,
                    "return_reasoning": True,
                    "stream": False,
                })
                assert resp.status_code == 200, resp.text
                data = resp.json()

                trace = data.get("reasoning_trace")
                assert trace is not None, "reasoning_trace absent"
                steps = {s["step_index"]: s for s in trace["steps"]}
                assert steps[1]["outcome"] == "skipped", steps[1]
                assert steps[2]["outcome"] == "skipped", steps[2]
                assert steps[3]["outcome"] == "accepted", steps[3]
                assert steps[4]["outcome"] == "accepted", steps[4]

        run(_go())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — Backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestScenario5BackwardCompat:
    """
    Verifies that V7 additions do not break existing response shapes.

    Hard Rule 3: reasoning_trace MUST be absent when return_reasoning=False.
    """

    def test_backward_compat_no_new_fields(self, db_pool, run, llm_mock) -> None:
        """POST /tasks without return_reasoning → response has no reasoning_trace; core fields present."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    "intent": "alert on overdue loans",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                })
                assert resp.status_code == 200, resp.text
                data = resp.json()

                # V7 field must be absent.
                assert "reasoning_trace" not in data
                # Pre-V7 core fields must still be present.
                for field in (
                    "task_id",
                    "intent",
                    "concept_id",
                    "condition_id",
                    "action_id",
                    "status",
                ):
                    assert field in data, f"core field '{field}' missing from Task response"

        run(_go())

    def test_backward_compat_dry_run_no_new_fields(self, db_pool, run, llm_mock) -> None:
        """POST /tasks dry_run=True → DryRunResult shape preserved; no reasoning_trace."""
        test_app = _make_test_app(db_pool, llm_mock)

        async def _go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=test_app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/tasks", json={
                    "intent": "alert on overdue loans",
                    "entity_scope": "loan",
                    "delivery": {
                        "type": "webhook",
                        "endpoint": "https://test.example.com/hook",
                    },
                    "dry_run": True,
                })
                assert resp.status_code == 200, resp.text
                data = resp.json()

                # V7 field must be absent.
                assert "reasoning_trace" not in data
                # DryRunResult must contain concept and condition previews.
                assert "concept" in data, "DryRunResult missing 'concept'"
                assert "condition" in data, "DryRunResult missing 'condition'"

        run(_go())
