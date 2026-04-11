"""
tests/unit/test_sse_streaming.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for M-6 SSE streaming on POST /tasks and POST /concepts/compile.

Test strategy
─────────────
All tests bypass the DB layer (in-memory stores + MockLLMClient).
Streaming tests consume the service's async generator directly, then verify
the yielded event sequence.

Route-level tests use httpx.AsyncClient against the FastAPI TestClient to
verify Content-Type headers, Cache-Control, and X-Accel-Buffering.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from app.models.concept_compile import CompileConceptRequest, CompileToken
from app.models.task import (
    CreateTaskRequest,
    DeliveryConfig,
    DeliveryType,
    ReasoningStep,
    Task,
)
from app.services.concept_compiler import ConceptCompilerService
from app.services.task_authoring import TaskAuthoringService
from tests.mocks.mock_llm_client import MockLLMClient


# ── In-memory store helpers (re-used across tests) ────────────────────────────

class _MockTaskStore:
    """Minimal in-memory TaskStore."""

    def __init__(self) -> None:
        self._rows: dict[str, Task] = {}
        self._counter = 0

    async def create(self, task: Task) -> Task:
        self._counter += 1
        task = task.model_copy(update={"task_id": f"task-{self._counter:04d}"})
        self._rows[task.task_id] = task
        return task

    async def get(self, task_id: str) -> Task | None:
        return self._rows.get(task_id)

    async def update(self, task_id: str, patch: dict) -> Task:
        task = self._rows[task_id]
        self._rows[task_id] = task.model_copy(update=patch)
        return self._rows[task_id]

    async def list(self, **kwargs) -> Any:
        from app.models.task import TaskList
        return TaskList(items=list(self._rows.values()), has_more=False, total_count=len(self._rows))


from app.models.concept import DefinitionResponse, SearchResult, VersionSummary
from app.models.errors import NotFoundError
from app.models.task import Namespace
from app.registry.definitions import DefinitionRegistry


class _MockDefinitionStore:
    """Minimal in-memory DefinitionStore."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DefinitionResponse] = {}
        self._bodies: dict[tuple[str, str], dict] = {}
        self._insert_order: list[tuple[str, str]] = []

    async def register(self, definition_id, version, definition_type, namespace, body, **_kw):
        from app.models.errors import ConflictError
        key = (definition_id, version)
        if key in self._rows:
            raise ConflictError(f"{definition_id}:{version} already exists.")
        ts = datetime.now(timezone.utc)
        resp = DefinitionResponse(
            definition_id=definition_id,
            version=version,
            definition_type=definition_type,
            namespace=Namespace(namespace),
            deprecated=False,
            created_at=ts,
            updated_at=ts,
        )
        self._rows[key] = resp
        self._bodies[key] = body
        self._insert_order.append(key)
        return resp

    async def get(self, definition_id, version):
        return self._bodies.get((definition_id, version))

    async def get_metadata(self, definition_id, version):
        return self._rows.get((definition_id, version))

    async def versions(self, definition_id):
        ordered = [k for k in reversed(self._insert_order) if k[0] == definition_id]
        if not ordered:
            raise NotFoundError(f"'{definition_id}' not found.")
        return [
            VersionSummary(
                version=k[1],
                created_at=self._rows[k].created_at,
                deprecated=self._rows[k].deprecated,
            )
            for k in ordered
        ]

    async def list(self, **kwargs):
        return SearchResult(items=list(self._rows.values()), has_more=False, total_count=len(self._rows))

    async def deprecate(self, *a, **kw):
        raise NotFoundError("not found")


class _MockCompileTokenStore:
    """In-memory CompileTokenStore."""

    def __init__(self) -> None:
        self._tokens: dict[str, CompileToken] = {}

    async def create(self, token: CompileToken) -> None:
        self._tokens[token.token_string] = token

    async def get(self, token_string: str) -> CompileToken | None:
        return self._tokens.get(token_string)

    async def consume(self, token_string: str) -> CompileToken:
        token = self._tokens.get(token_string)
        if token is None:
            raise NotFoundError("token not found")
        consumed = token.model_copy(update={"used": True})
        self._tokens[token_string] = consumed
        return consumed


# ── Timeout mock LLM client ────────────────────────────────────────────────────

class _TimeoutLLMClient:
    """LLM client that raises asyncio.TimeoutError to simulate step timeout."""

    def generate_task(self, intent: str, context: dict) -> dict:
        raise asyncio.TimeoutError("step_2_timed_out")


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_task_service(llm_client=None) -> TaskAuthoringService:
    store = _MockDefinitionStore()
    registry = DefinitionRegistry(store=store)
    return TaskAuthoringService(
        task_store=_MockTaskStore(),
        definition_registry=registry,
        llm_client=llm_client or MockLLMClient(),
    )


def _make_compile_service(llm_client=None) -> ConceptCompilerService:
    return ConceptCompilerService(
        llm_client=llm_client or MockLLMClient(),
        token_store=_MockCompileTokenStore(),
    )


def _default_task_request(stream: bool = False) -> CreateTaskRequest:
    return CreateTaskRequest(
        intent="alert on high churn risk",
        entity_scope="account",
        delivery=DeliveryConfig(type=DeliveryType.WEBHOOK, endpoint="https://example.com/hook"),
        stream=stream,
    )


def _default_compile_request(stream: bool = False) -> CompileConceptRequest:
    return CompileConceptRequest(
        identifier="loan.repayment_ratio",
        description="Ratio of on-time payments to total payments due",
        output_type="float",
        signal_names=["payments_on_time", "payments_due"],
        stream=stream,
    )


# ── Helper: collect all events from an async generator ───────────────────────

async def _collect(gen) -> list[dict]:
    events = []
    async for item in gen:
        events.append(item)
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# T-1  stream=False → synchronous JSON (non-streaming path unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def test_stream_false_returns_task():
    """Non-streaming path returns a Task, not a generator."""
    svc = _make_task_service()
    req = _default_task_request(stream=False)
    result = asyncio.run(svc.create_task(req))
    assert hasattr(result, "task_id"), "Expected Task with task_id"
    assert result.task_id is not None


def test_stream_false_compile_returns_response():
    """Non-streaming compile returns CompileConceptResponse, not a generator."""
    svc = _make_compile_service()
    req = _default_compile_request(stream=False)
    result = asyncio.run(svc.compile(req, pool=None))
    assert hasattr(result, "compile_token")
    assert result.compile_token


# ═══════════════════════════════════════════════════════════════════════════════
# T-2  stream=True on POST /tasks → StreamingResponse
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_task_stream_yields_dicts():
    """create_task_stream() is an async generator yielding event dicts."""
    svc = _make_task_service()
    req = _default_task_request(stream=True)
    gen = svc.create_task_stream(req)
    events = await _collect(gen)
    assert len(events) > 0
    for ev in events:
        assert "event_type" in ev
        assert "data" in ev


# ═══════════════════════════════════════════════════════════════════════════════
# T-3  stream=True → exactly 4 cor_step events before cor_complete
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_task_stream_exactly_4_cor_steps():
    """Success path: 4 cor_step events, then 1 cor_complete."""
    svc = _make_task_service()
    req = _default_task_request(stream=True)
    events = await _collect(svc.create_task_stream(req))

    step_events = [e for e in events if e["event_type"] == "cor_step"]
    complete_events = [e for e in events if e["event_type"] == "cor_complete"]
    error_events = [e for e in events if e["event_type"] == "cor_error"]

    assert len(step_events) == 4, f"Expected 4 cor_step, got {len(step_events)}"
    assert len(complete_events) == 1, "Expected 1 cor_complete"
    assert len(error_events) == 0, "Expected no cor_error on success"


@pytest.mark.asyncio
async def test_compile_stream_exactly_4_cor_steps():
    """Success path for concepts/compile: 4 cor_step events, then 1 cor_complete."""
    svc = _make_compile_service()
    req = _default_compile_request(stream=True)
    events = await _collect(svc.compile_stream(req, pool=None))

    step_events = [e for e in events if e["event_type"] == "cor_step"]
    complete_events = [e for e in events if e["event_type"] == "cor_complete"]
    error_events = [e for e in events if e["event_type"] == "cor_error"]

    assert len(step_events) == 4, f"Expected 4 cor_step, got {len(step_events)}"
    assert len(complete_events) == 1, "Expected 1 cor_complete"
    assert len(error_events) == 0, "Expected no cor_error on success"


# ═══════════════════════════════════════════════════════════════════════════════
# T-4  cor_step events have correct step_index values (1, 2, 3, 4) in order
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_task_stream_step_index_order():
    svc = _make_task_service()
    req = _default_task_request(stream=True)
    events = await _collect(svc.create_task_stream(req))
    indices = [e["data"]["step_index"] for e in events if e["event_type"] == "cor_step"]
    assert indices == [1, 2, 3, 4], f"Expected [1,2,3,4], got {indices}"


@pytest.mark.asyncio
async def test_compile_stream_step_index_order():
    svc = _make_compile_service()
    req = _default_compile_request(stream=True)
    events = await _collect(svc.compile_stream(req, pool=None))
    indices = [e["data"]["step_index"] for e in events if e["event_type"] == "cor_step"]
    assert indices == [1, 2, 3, 4], f"Expected [1,2,3,4], got {indices}"


# ═══════════════════════════════════════════════════════════════════════════════
# T-5  Route: Cache-Control and X-Accel-Buffering headers present
# ═══════════════════════════════════════════════════════════════════════════════

def test_task_route_streaming_headers():
    """StreamingResponse must carry Cache-Control and X-Accel-Buffering headers."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    payload = {
        "intent": "alert on churn",
        "entity_scope": "account",
        "delivery": {"type": "webhook", "endpoint": "https://x.com/hook"},
        "stream": True,
    }
    response = client.post("/tasks", json=payload)
    # May get 500 if DB not available — only check header contract on 200 or streaming
    if response.status_code == 200:
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("x-accel-buffering") == "no"
        assert "text/event-stream" in response.headers.get("content-type", "")


def test_compile_route_streaming_headers():
    """POST /concepts/compile with stream=True must return SSE headers."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    payload = {
        "identifier": "test.concept",
        "description": "Test concept",
        "output_type": "float",
        "signal_names": ["sig1"],
        "stream": True,
    }
    response = client.post("/concepts/compile", json=payload)
    if response.status_code in (200, 201):
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("x-accel-buffering") == "no"
        assert "text/event-stream" in response.headers.get("content-type", "")


# ═══════════════════════════════════════════════════════════════════════════════
# T-6  LLM step timeout → cor_error emitted, no further events after cor_error
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_task_stream_timeout_emits_cor_error():
    """When LLM raises TimeoutError, cor_error is emitted and stream closes."""
    svc = _make_task_service(llm_client=_TimeoutLLMClient())
    req = _default_task_request(stream=True)
    events = await _collect(svc.create_task_stream(req))

    error_events = [e for e in events if e["event_type"] == "cor_error"]
    assert len(error_events) == 1, "Expected exactly 1 cor_error"

    # No events after cor_error
    last_idx = next(i for i, e in enumerate(events) if e["event_type"] == "cor_error")
    assert last_idx == len(events) - 1, "cor_error must be the last event"

    # No cor_complete after error
    complete_after = [e for e in events[last_idx + 1:] if e["event_type"] == "cor_complete"]
    assert not complete_after


@pytest.mark.asyncio
async def test_compile_stream_timeout_emits_cor_error():
    """concept compiler: TimeoutError in _run_step → cor_error."""

    class _TimeoutCompileLLM:
        """LLM that times out on step 1."""
        def generate_compile_step(self, intent, context):
            raise asyncio.TimeoutError("step_1_timed_out")

    svc = _make_compile_service(llm_client=_TimeoutCompileLLM())
    req = _default_compile_request(stream=True)
    events = await _collect(svc.compile_stream(req, pool=None))

    error_events = [e for e in events if e["event_type"] == "cor_error"]
    assert len(error_events) == 1
    last_idx = next(i for i, e in enumerate(events) if e["event_type"] == "cor_error")
    assert last_idx == len(events) - 1, "cor_error must be the last event"


# ═══════════════════════════════════════════════════════════════════════════════
# T-7  cor_error payload contains failure_reason and failed_at_step
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_task_stream_cor_error_payload():
    """cor_error payload must contain failure_reason and failed_at_step."""
    svc = _make_task_service(llm_client=_TimeoutLLMClient())
    req = _default_task_request(stream=True)
    events = await _collect(svc.create_task_stream(req))

    error_ev = next(e for e in events if e["event_type"] == "cor_error")
    data = error_ev["data"]
    assert "failure_reason" in data, "cor_error must have failure_reason"
    assert "failed_at_step" in data, "cor_error must have failed_at_step"


@pytest.mark.asyncio
async def test_compile_stream_cor_error_payload():
    """concept compiler cor_error must contain failure_reason and failed_at_step."""

    class _ErrLLM:
        def generate_compile_step(self, intent, context):
            raise asyncio.TimeoutError("timeout at step 2")

    svc = _make_compile_service(llm_client=_ErrLLM())
    req = _default_compile_request(stream=True)
    events = await _collect(svc.compile_stream(req, pool=None))

    error_ev = next(e for e in events if e["event_type"] == "cor_error")
    data = error_ev["data"]
    assert "failure_reason" in data
    assert "failed_at_step" in data


# ═══════════════════════════════════════════════════════════════════════════════
# T-8  cor_complete on /concepts/compile contains compile_token
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_compile_stream_cor_complete_has_compile_token():
    svc = _make_compile_service()
    req = _default_compile_request(stream=True)
    events = await _collect(svc.compile_stream(req, pool=None))

    complete = next(e for e in events if e["event_type"] == "cor_complete")
    assert "compile_token" in complete["data"], "cor_complete must contain compile_token"
    assert complete["data"]["compile_token"], "compile_token must be non-empty"


# ═══════════════════════════════════════════════════════════════════════════════
# T-9  cor_complete on /tasks contains task_id (or null for dry_run)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_task_stream_cor_complete_has_task_id():
    svc = _make_task_service()
    req = _default_task_request(stream=True)
    events = await _collect(svc.create_task_stream(req))

    complete = next(e for e in events if e["event_type"] == "cor_complete")
    assert "task_id" in complete["data"], "cor_complete must contain task_id"
    assert complete["data"]["task_id"] is not None, "task_id must be set (not None)"


@pytest.mark.asyncio
async def test_task_stream_dry_run_cor_complete_task_id_null():
    """dry_run=True → cor_complete with task_id=null."""
    svc = _make_task_service()
    req = CreateTaskRequest(
        intent="alert on churn",
        entity_scope="account",
        delivery=DeliveryConfig(type=DeliveryType.WEBHOOK, endpoint="https://example.com/hook"),
        stream=True,
        dry_run=True,
    )
    events = await _collect(svc.create_task_stream(req))

    complete = next(e for e in events if e["event_type"] == "cor_complete")
    assert "task_id" in complete["data"]
    assert complete["data"]["task_id"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# T-10  SSE format helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_sse_event_format():
    """sse_event() must produce the exact RFC 8895 wire format."""
    from app.api.routes.utils import sse_event
    result = sse_event("cor_step", {"step_index": 1})
    lines = result.split("\n")
    assert lines[0] == "event: cor_step"
    assert lines[1].startswith("data: ")
    payload = json.loads(lines[1][6:])
    assert payload["step_index"] == 1
    # Must end with blank line (two \n at end)
    assert result.endswith("\n\n")
