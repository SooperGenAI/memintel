"""
tests/integration/test_v7_spec_conformance.py
──────────────────────────────────────────────────────────────────────────────
T-4 Part 3 — OpenAPI spec conformance tests.

Validates that actual API responses match the schema shapes defined in
developer_api.yaml. Canvas integration imports this spec as the source of
truth for type shapes and required fields; any mismatch breaks Canvas.

Approach
────────
  1. Load developer_api.yaml at module level.
  2. Walk required fields from each schema.
  3. For each response, check required fields are present and correctly typed.
  4. $ref references are resolved against components/schemas.
  5. Extra fields in the response are ALLOWED (additive responses are fine).

Tests
─────
  test_compile_concept_response_matches_spec    POST /concepts/compile 201
  test_register_concept_response_matches_spec   POST /concepts/register 201
  test_task_response_matches_spec               POST /tasks 200
  test_reasoning_step_schema                    reasoning_trace.steps fields
  test_sse_event_schema                         SSE cor_step / cor_complete payloads
  test_error_response_matches_spec              error envelope for 4xx errors
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest
import yaml

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

from tests.integration.conftest_v7 import LLMMockClient, compile_and_register


# ── Load spec ─────────────────────────────────────────────────────────────────

_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "developer_api.yaml"
)

with open(_YAML_PATH, encoding="utf-8") as _f:
    SPEC: dict = yaml.safe_load(_f)

_SCHEMAS: dict = SPEC.get("components", {}).get("schemas", {})


# ── Schema helpers ─────────────────────────────────────────────────────────────

def _resolve_ref(schema: dict) -> dict:
    """Recursively resolve $ref pointers against SPEC components/schemas."""
    if "$ref" in schema:
        ref: str = schema["$ref"]
        # Format: '#/components/schemas/SchemaName'
        parts = ref.lstrip("#/").split("/")
        node: Any = SPEC
        for part in parts:
            node = node[part]
        return _resolve_ref(node)
    return schema


def get_schema(path: str, method: str, response_code: str) -> dict:
    """Return the resolved JSON schema for a given path/method/status."""
    schema = (
        SPEC["paths"][path][method]["responses"][response_code]
        ["content"]["application/json"]["schema"]
    )
    return _resolve_ref(schema)


def assert_response_matches_schema(
    data: dict,
    schema: dict,
    *,
    context: str = "",
) -> None:
    """
    Assert that `data` satisfies the required fields and types in `schema`.

    - All `required` fields must be present in data.
    - Present fields must have types matching the schema type.
    - Extra fields in data are ALLOWED (additive responses are fine).
    - Nested objects and arrays are NOT recursively validated (too brittle).
    - $ref fields are resolved before type-checking.
    """
    required: list[str] = schema.get("required", [])
    properties: dict = schema.get("properties", {})

    for field in required:
        assert field in data, (
            f"[{context}] Required field '{field}' missing from response. "
            f"Got keys: {list(data.keys())}"
        )

    _TYPE_MAP = {
        "string":  str,
        "integer": int,
        "boolean": bool,
        "object":  dict,
        "array":   list,
        "number":  (int, float),
    }

    for field, field_schema in properties.items():
        if field not in data or data[field] is None:
            continue
        resolved = _resolve_ref(field_schema)
        expected_type = resolved.get("type")
        if expected_type is None:
            continue  # oneOf / anyOf — skip type check
        python_type = _TYPE_MAP.get(expected_type)
        if python_type is None:
            continue
        value = data[field]
        # booleans are ints in Python, so exclude bool when checking int
        if expected_type == "integer":
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"[{context}] Field '{field}' expected integer, "
                f"got {type(value).__name__}: {value!r}"
            )
        else:
            assert isinstance(value, python_type), (
                f"[{context}] Field '{field}' expected {expected_type}, "
                f"got {type(value).__name__}: {value!r}"
            )


# ── Test-app factory ───────────────────────────────────────────────────────────

def _make_test_app(db_pool, llm_client: Any) -> FastAPI:
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


def collect_sse_events(response: httpx.Response) -> list[dict]:
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


# ── Shared compile request body ────────────────────────────────────────────────

_COMPILE_BODY = {
    "identifier":       "spec.test_concept",
    "description":      "Spec conformance test concept",
    "output_type":      "float",
    "signal_names":     ["payments_on_time", "payments_due"],
    "stream":           False,
    "return_reasoning": False,
}

_TASK_BODY = {
    "intent":       "alert when loan repayment ratio is below 0.80",
    "entity_scope": "loan",
    "delivery": {
        "type":     "webhook",
        "endpoint": "https://spec-test.example.com/hook",
    },
    "stream": False,
}


# ═══════════════════════════════════════════════════════════════════════════════
# test_compile_concept_response_matches_spec
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompileConceptResponseMatchesSpec:

    def test_compile_response_matches_spec(self, db_pool, run):
        """POST /concepts/compile 201 — verify CompileConceptResult schema."""
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/concepts/compile",
                    json={**_COMPILE_BODY, "return_reasoning": False},
                )
                assert resp.status_code == 201, resp.text
                data = resp.json()

                # Validate against CompileConceptResult schema from spec
                schema = _resolve_ref({"$ref": "#/components/schemas/CompileConceptResult"})
                assert_response_matches_schema(data, schema, context="compile/201")

                # compile_token must be a non-empty string
                assert isinstance(data["compile_token"], str)
                assert data["compile_token"]

                # compiled_concept shape
                cc = data["compiled_concept"]
                assert isinstance(cc, dict)
                cc_schema = _resolve_ref({"$ref": "#/components/schemas/CompiledConcept"})
                assert_response_matches_schema(cc, cc_schema, context="compiled_concept")

                # expires_at must be a string (ISO 8601)
                assert isinstance(data["expires_at"], str)
                assert "T" in data["expires_at"] or "Z" in data["expires_at"]

                # reasoning_trace absent when return_reasoning=False
                assert "reasoning_trace" not in data or data["reasoning_trace"] is None, (
                    "reasoning_trace must be absent when return_reasoning=False"
                )

        run(_go())

    def test_compile_response_with_reasoning_matches_spec(self, db_pool, run):
        """compile with return_reasoning=True — reasoning_trace shape validated."""
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/concepts/compile",
                    json={**_COMPILE_BODY, "identifier": "spec.test_reason", "return_reasoning": True},
                )
                assert resp.status_code == 201, resp.text
                data = resp.json()

                # reasoning_trace must be present
                assert "reasoning_trace" in data and data["reasoning_trace"] is not None, (
                    "reasoning_trace must be present when return_reasoning=True"
                )
                rt = data["reasoning_trace"]
                rt_schema = _resolve_ref({"$ref": "#/components/schemas/ReasoningTrace"})
                assert_response_matches_schema(rt, rt_schema, context="reasoning_trace")

                # steps must be an array
                assert isinstance(rt["steps"], list)
                assert len(rt["steps"]) > 0

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_register_concept_response_matches_spec
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegisterConceptResponseMatchesSpec:

    def test_register_response_matches_spec(self, db_pool, run):
        """POST /concepts/register 201 — verify RegisteredConcept schema."""
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Compile first
                compile_resp = await client.post(
                    "/concepts/compile",
                    json=_COMPILE_BODY,
                )
                assert compile_resp.status_code == 201, compile_resp.text
                token = compile_resp.json()["compile_token"]

                # Register
                resp = await client.post(
                    "/concepts/register",
                    json={
                        "compile_token": token,
                        "identifier":    _COMPILE_BODY["identifier"],
                    },
                )
                assert resp.status_code == 201, resp.text
                data = resp.json()

                schema = _resolve_ref({"$ref": "#/components/schemas/RegisteredConcept"})
                assert_response_matches_schema(data, schema, context="register/201")

                # concept_id == identifier (documented behavior)
                assert data["concept_id"] == _COMPILE_BODY["identifier"]

                # registered_at is ISO 8601
                assert isinstance(data["registered_at"], str)
                assert "T" in data["registered_at"] or "Z" in data["registered_at"]

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_task_response_matches_spec
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskResponseMatchesSpec:

    def test_task_response_matches_spec(self, db_pool, run):
        """POST /tasks 200 — verify Task schema fields and types."""
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/tasks",
                    json={**_TASK_BODY, "return_reasoning": False},
                )
                assert resp.status_code == 200, resp.text
                data = resp.json()

                task_schema = _resolve_ref({"$ref": "#/components/schemas/Task"})
                assert_response_matches_schema(data, task_schema, context="task/200")

                # reasoning_trace absent when return_reasoning=False
                assert "reasoning_trace" not in data or data["reasoning_trace"] is None, (
                    "reasoning_trace must be absent when return_reasoning=False"
                )

        run(_go())

    def test_task_response_with_reasoning_matches_spec(self, db_pool, run):
        """POST /tasks with return_reasoning=True — reasoning_trace validated."""
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/tasks",
                    json={**_TASK_BODY, "return_reasoning": True},
                )
                assert resp.status_code == 200, resp.text
                data = resp.json()

                # reasoning_trace present and matches schema
                assert "reasoning_trace" in data and data["reasoning_trace"] is not None
                rt = data["reasoning_trace"]
                rt_schema = _resolve_ref({"$ref": "#/components/schemas/ReasoningTrace"})
                assert_response_matches_schema(rt, rt_schema, context="reasoning_trace")
                assert isinstance(rt["steps"], list)

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_reasoning_step_schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestReasoningStepSchema:

    def test_reasoning_step_schema(self, db_pool, run):
        """
        Each step in reasoning_trace.steps must match ReasoningStep schema:
          step_index: integer 1-4
          label: non-empty string
          summary: non-empty string
          outcome: one of [accepted, skipped, failed]
          candidates: list of strings or absent (never null)
        """
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/tasks",
                    json={**_TASK_BODY, "return_reasoning": True},
                )
                assert resp.status_code == 200, resp.text
                data = resp.json()
                steps = data["reasoning_trace"]["steps"]

                step_schema = _resolve_ref({"$ref": "#/components/schemas/ReasoningStep"})
                valid_outcomes = {"accepted", "skipped", "failed"}

                assert len(steps) > 0, "reasoning_trace.steps must be non-empty"
                for i, step in enumerate(steps):
                    ctx = f"step[{i}]"
                    assert_response_matches_schema(step, step_schema, context=ctx)

                    # step_index: integer 1-4
                    assert isinstance(step["step_index"], int), f"{ctx} step_index must be int"
                    assert 1 <= step["step_index"] <= 4, f"{ctx} step_index out of range"

                    # label: non-empty string
                    assert isinstance(step["label"], str) and step["label"], (
                        f"{ctx} label must be non-empty string"
                    )

                    # summary: non-empty string
                    assert isinstance(step["summary"], str) and step["summary"], (
                        f"{ctx} summary must be non-empty string"
                    )

                    # outcome: one of valid set
                    assert step["outcome"] in valid_outcomes, (
                        f"{ctx} outcome '{step['outcome']}' not in {valid_outcomes}"
                    )

                    # candidates: if present, must be a list (never null at step level)
                    if "candidates" in step and step["candidates"] is not None:
                        assert isinstance(step["candidates"], list), (
                            f"{ctx} candidates must be a list when present"
                        )
                        for c in step["candidates"]:
                            assert isinstance(c, str), (
                                f"{ctx} candidates items must be strings"
                            )

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_sse_event_schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestSSEEventSchema:

    def test_sse_event_schema(self, db_pool, run):
        """
        POST /tasks with stream=True: verify SSE event payloads match spec.
          cor_step: step_index (int), label (str), summary (str), outcome (str)
          cor_complete: task_id (str or absent for dry_run), status (str)
        """
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        cor_step_schema   = _resolve_ref({"$ref": "#/components/schemas/CorStepEvent"})
        cor_complete_schema = _resolve_ref({"$ref": "#/components/schemas/CorCompleteEvent"})

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/tasks",
                    json={**_TASK_BODY, "stream": True},
                )
                assert resp.status_code == 200, resp.text
                events = collect_sse_events(resp)

                step_events     = [e for e in events if e["event_type"] == "cor_step"]
                complete_events = [e for e in events if e["event_type"] == "cor_complete"]
                error_events    = [e for e in events if e["event_type"] == "cor_error"]

                assert len(step_events) == 4, f"Expected 4 cor_step, got {len(step_events)}"
                assert len(complete_events) == 1
                assert len(error_events) == 0

                for ev in step_events:
                    d = ev["data"]
                    assert_response_matches_schema(d, cor_step_schema, context="cor_step")
                    assert isinstance(d["step_index"], int)
                    assert isinstance(d["label"], str)
                    assert isinstance(d["summary"], str)
                    assert isinstance(d["outcome"], str)

                cd = complete_events[0]["data"]
                assert_response_matches_schema(cd, cor_complete_schema, context="cor_complete")
                if "task_id" in cd and cd["task_id"] is not None:
                    assert isinstance(cd["task_id"], str)

        run(_go())

    def test_sse_compile_event_schema(self, db_pool, run):
        """
        POST /concepts/compile with stream=True: cor_complete must contain compile_token.
        """
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/concepts/compile",
                    json={**_COMPILE_BODY, "identifier": "spec.stream_test", "stream": True},
                )
                assert resp.status_code == 200, resp.text  # SSE → 200 not 201
                events = collect_sse_events(resp)

                complete_events = [e for e in events if e["event_type"] == "cor_complete"]
                assert len(complete_events) == 1

                cd = complete_events[0]["data"]
                # compile stream complete must have compile_token
                assert "compile_token" in cd, f"cor_complete missing compile_token: {cd}"
                assert isinstance(cd["compile_token"], str) and cd["compile_token"]

        run(_go())


# ═══════════════════════════════════════════════════════════════════════════════
# test_error_response_matches_spec
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorResponseMatchesSpec:
    """
    Verify the error envelope shape for each V7 error type.

    Spec ErrorResponse:
      required: [error]
      error: {required: [type, message], properties: {type, message, location, suggestion}}
    """

    def _assert_error_envelope(self, resp_json: dict, expected_type: str | None = None) -> None:
        """Assert the error response matches the ErrorResponse spec schema."""
        error_schema = _resolve_ref({"$ref": "#/components/schemas/ErrorResponse"})
        assert_response_matches_schema(resp_json, error_schema, context="ErrorResponse")

        err = resp_json["error"]
        assert isinstance(err["type"], str) and err["type"], "error.type must be non-empty string"
        assert isinstance(err["message"], str) and err["message"], "error.message must be non-empty string"

        if expected_type is not None:
            assert err["type"] == expected_type, (
                f"Expected error type '{expected_type}', got '{err['type']}'"
            )

    def test_compile_token_not_found(self, db_pool, run):
        """POST /concepts/register with unknown token → 404, error type=compile_token_not_found."""
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/concepts/register",
                    json={
                        "compile_token": "nonexistent-token-aaaabbbbcccc",
                        "identifier":    "spec.does_not_exist",
                    },
                )
                assert resp.status_code == 404, (
                    f"Expected 404 for missing token, got {resp.status_code}: {resp.text}"
                )
                self._assert_error_envelope(resp.json(), "compile_token_not_found")

        run(_go())

    def test_compile_token_consumed(self, db_pool, run):
        """Using a token twice → 409, error type=compile_token_consumed."""
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Compile
                c_resp = await client.post(
                    "/concepts/compile",
                    json={**_COMPILE_BODY, "identifier": "spec.consumed_test"},
                )
                assert c_resp.status_code == 201, c_resp.text
                token = c_resp.json()["compile_token"]

                # Register once — consumes the token
                r1 = await client.post(
                    "/concepts/register",
                    json={"compile_token": token, "identifier": "spec.consumed_test"},
                )
                assert r1.status_code == 201, r1.text

                # Register again with same token → 409
                r2 = await client.post(
                    "/concepts/register",
                    json={"compile_token": token, "identifier": "spec.consumed_test"},
                )
                assert r2.status_code == 409, (
                    f"Expected 409 for consumed token, got {r2.status_code}: {r2.text}"
                )
                self._assert_error_envelope(r2.json(), "compile_token_consumed")

        run(_go())

    def test_compile_token_expired(self, db_pool, run):
        """Expired token → 400, error type=compile_token_expired."""
        import asyncpg

        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Compile to get a token
                c_resp = await client.post(
                    "/concepts/compile",
                    json={**_COMPILE_BODY, "identifier": "spec.expired_test"},
                )
                assert c_resp.status_code == 201, c_resp.text
                token = c_resp.json()["compile_token"]

                # Expire the token by backdating it in the DB
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE compile_tokens
                        SET    expires_at = NOW() - INTERVAL '1 hour'
                        WHERE  token_string = $1
                        """,
                        token,
                    )

                # Register with expired token → 400
                r = await client.post(
                    "/concepts/register",
                    json={"compile_token": token, "identifier": "spec.expired_test"},
                )
                assert r.status_code == 400, (
                    f"Expected 400 for expired token, got {r.status_code}: {r.text}"
                )
                self._assert_error_envelope(r.json(), "compile_token_expired")

        run(_go())

    def test_vocabulary_mismatch_error(self, db_pool, run):
        """
        POST /tasks with vocabulary_context whose available_concept_ids does
        not include the LLM-returned concept_id → 422 vocabulary_mismatch.
        """
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/tasks",
                    json={
                        **_TASK_BODY,
                        "vocabulary_context": {
                            "available_concept_ids":   ["nonexistent.concept.xyz"],
                            "available_condition_ids": ["nonexistent.condition.xyz"],
                        },
                    },
                )
                assert resp.status_code == 422, (
                    f"Expected 422 for vocabulary mismatch, got {resp.status_code}: {resp.text}"
                )
                self._assert_error_envelope(resp.json(), "vocabulary_mismatch")

        run(_go())

    def test_vocabulary_context_too_large(self, db_pool, run):
        """
        POST /tasks with >500 concept IDs → 422 vocabulary_context_too_large.
        """
        llm = LLMMockClient()
        app = _make_test_app(db_pool, llm)
        transport = httpx.ASGITransport(app=app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/tasks",
                    json={
                        **_TASK_BODY,
                        "vocabulary_context": {
                            "available_concept_ids": [
                                f"concept.id_{i}" for i in range(501)
                            ],
                            "available_condition_ids": [],
                        },
                    },
                )
                assert resp.status_code == 422, (
                    f"Expected 422 for too-large vocab, got {resp.status_code}: {resp.text}"
                )
                # VocabularyContext validates list size via a Pydantic @field_validator,
                # which FastAPI converts to a standard 422 with {"detail": [...]}.
                # This is NOT a MemintelError envelope — the rejection happens before the
                # service layer is reached.
                body = resp.json()
                assert "detail" in body, f"Expected FastAPI 422 detail, got: {body}"
                errs = body["detail"]
                assert any(
                    "available_concept_ids" in str(e.get("loc", ""))
                    for e in errs
                ), f"Expected available_concept_ids error in detail, got: {errs}"

        run(_go())
