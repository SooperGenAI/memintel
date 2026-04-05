"""
tests/unit/test_concept_compiler.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for ConceptCompilerService and POST /concepts/compile.

Service contract tests use MockConceptCompilerLLM + MockCompileTokenStore and
exercise the service directly — no database, no HTTP stack.

HTTP layer tests use a minimal FastAPI TestClient with dependency overrides.

Coverage
────────
  Service contract:
  1.  Valid request returns compile_token (non-empty string)
  2.  Valid request returns compiled_concept with formula_summary + signal_bindings
  3.  Valid request returns expires_at approximately now + TTL (within 2 seconds)
  4.  return_reasoning=True → reasoning_trace present, len(steps) == 4
  5.  return_reasoning=False → reasoning_trace ABSENT from serialised response dict
  6.  Unrecognised signal_names do NOT raise; Step 2 outcome is "accepted"
         (identity boundary — signal_names are never validated)
  7.  Step 4 outcome "failed" → TypeMismatchError with type "type_mismatch"
  8.  compile_token persisted in CompileTokenStore on success
  9.  ir_hash is deterministic: same request → same ir_hash on two calls
  10. Two calls with the same identifier return distinct compile_tokens
         (identifier is NOT locked at compile time)

  HTTP layer:
  11. POST /concepts/compile returns 201 on success
  12. POST /concepts/compile returns 422 on compilation_error
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

# Stub heavy optional deps before any app import
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import concepts as concepts_route
from app.api.routes.concepts import get_concept_compiler_service
from app.models.concept_compile import CompileConceptRequest, CompileConceptResponse
from app.models.errors import (
    CompilationError,
    MemintelError,
    TypeMismatchError,
    memintel_error_handler,
)
from app.persistence.db import get_db
from app.services.concept_compiler import ConceptCompilerService


# ── Mock: LLM client ──────────────────────────────────────────────────────────

class MockConceptCompilerLLM:
    """
    Mock LLM for concept compiler tests.

    Returns step-appropriate responses based on context["step"].
    step4_outcome controls whether Step 4 succeeds ("accepted") or triggers
    a type mismatch ("failed").
    """

    def __init__(self, step4_outcome: str = "accepted") -> None:
        self._step4_outcome = step4_outcome

    def generate_task(self, prompt: str, context: dict) -> dict:
        step = context.get("step", 0)
        signal_names: list[str] = context.get("signal_names") or ["signal_a", "signal_b"]
        identifier: str = context.get("identifier", "test.concept")
        output_type: str = context.get("output_type", "float")

        if step == 1:
            return {
                "summary": f"Identified intent: measure {identifier}",
                "outcome": "accepted",
                "entity_type": "entity",
                "desired_signal": identifier.split(".")[-1],
                "direction": "above",
                "urgency": "normal",
            }
        elif step == 2:
            return {
                "summary": "Signals imply ratio-based calculation",
                "outcome": "accepted",
                "domain_context": "payment domain",
            }
        elif step == 3:
            bindings = []
            if len(signal_names) >= 2:
                bindings = [
                    {"signal_name": signal_names[0], "role": "numerator"},
                    {"signal_name": signal_names[1], "role": "denominator"},
                ]
            else:
                bindings = [
                    {"signal_name": sig, "role": "input"} for sig in signal_names
                ]
            formula_parts = " / ".join(signal_names[:2]) if len(signal_names) >= 2 else identifier
            return {
                "summary": "Selected ratio formula strategy",
                "outcome": "accepted",
                "formula_summary": f"{formula_parts}, 90-day rolling window",
                "signal_bindings": bindings,
            }
        else:  # step 4
            return {
                "summary": f"Output type '{output_type}' validated",
                "outcome": self._step4_outcome,
            }


# ── Mock: CompileTokenStore ───────────────────────────────────────────────────

class MockCompileTokenStore:
    """In-memory CompileTokenStore."""

    def __init__(self) -> None:
        self._tokens: dict[str, Any] = {}

    async def create(self, token: Any) -> Any:
        self._tokens[token.token_string] = token
        return token

    async def get(self, token_string: str) -> Any:
        return self._tokens.get(token_string)

    async def consume(self, token_string: str) -> Any:
        raise NotImplementedError("not used in compile tests")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_request(**kwargs) -> CompileConceptRequest:
    """Build a minimal valid CompileConceptRequest."""
    defaults: dict = {
        "identifier": "loan.repayment_ratio",
        "description": "Ratio of payments made on time over total payments due",
        "output_type": "float",
        "signal_names": ["payments_on_time", "payments_due"],
    }
    defaults.update(kwargs)
    return CompileConceptRequest(**defaults)


def _make_service(
    *,
    step4_outcome: str = "accepted",
) -> tuple[ConceptCompilerService, MockCompileTokenStore]:
    """Build an isolated ConceptCompilerService with in-memory mock dependencies."""
    store = MockCompileTokenStore()
    service = ConceptCompilerService(
        llm_client=MockConceptCompilerLLM(step4_outcome=step4_outcome),
        token_store=store,
    )
    return service, store


# ── Service contract tests ─────────────────────────────────────────────────────

class TestCompileContract:

    @pytest.mark.asyncio
    async def test_compile_returns_non_empty_compile_token(self):
        service, _ = _make_service()
        response = await service.compile(_make_request(), pool=None)
        assert isinstance(response.compile_token, str)
        assert len(response.compile_token) > 0

    @pytest.mark.asyncio
    async def test_compile_returns_compiled_concept_with_formula_and_bindings(self):
        service, _ = _make_service()
        response = await service.compile(_make_request(), pool=None)

        cc = response.compiled_concept
        assert cc is not None
        assert isinstance(cc.formula_summary, str)
        assert len(cc.formula_summary) > 0
        assert isinstance(cc.signal_bindings, list)
        assert len(cc.signal_bindings) > 0
        assert all(hasattr(b, "signal_name") and hasattr(b, "role") for b in cc.signal_bindings)

    @pytest.mark.asyncio
    async def test_compile_returns_expires_at_approximately_correct(self):
        service, _ = _make_service()
        before = datetime.now(tz=timezone.utc)
        response = await service.compile(_make_request(), pool=None)
        after = datetime.now(tz=timezone.utc)

        # Default TTL is 1800 s; allow ±2 s for execution time.
        expected_min = before + timedelta(seconds=1798)
        expected_max = after  + timedelta(seconds=1802)

        expires_at = response.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        assert expected_min <= expires_at <= expected_max

    @pytest.mark.asyncio
    async def test_return_reasoning_true_includes_trace_with_four_steps(self):
        service, _ = _make_service()
        response = await service.compile(
            _make_request(return_reasoning=True), pool=None
        )
        assert response.reasoning_trace is not None
        assert len(response.reasoning_trace.steps) == 4
        step_indices = [s.step_index for s in response.reasoning_trace.steps]
        assert step_indices == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_return_reasoning_false_reasoning_trace_absent_from_dict(self):
        """
        reasoning_trace must be ABSENT from the serialised response,
        not null. The spec requires checking key absence, not None.
        """
        service, _ = _make_service()
        response = await service.compile(
            _make_request(return_reasoning=False), pool=None
        )
        # Serialise with exclude_none=True to match the API wire format.
        response_dict = response.model_dump(exclude_none=True)
        assert "reasoning_trace" not in response_dict

    @pytest.mark.asyncio
    async def test_unrecognised_signal_names_not_rejected(self):
        """
        Identity boundary test: signal_names are semantic hints from the caller's
        domain. Memintel MUST NOT validate them against any internal registry.
        An unrecognised signal_name is NOT an error. Step 2 outcome must be
        "accepted" regardless of whether the signal names are known.
        """
        service, _ = _make_service()
        request = _make_request(
            signal_names=["xyzzy_unknown_signal_123", "foobar_totally_made_up_456"],
            return_reasoning=True,
        )
        # Must not raise — unrecognised signal names are semantic hints only.
        response = await service.compile(request, pool=None)

        assert response.compile_token
        # Step 2 (Signal Identification) must succeed with outcome "accepted".
        signal_step = response.reasoning_trace.steps[1]  # steps are 0-indexed; step_index=2
        assert signal_step.step_index == 2
        assert signal_step.label == "Signal Identification"
        assert signal_step.outcome == "accepted"

    @pytest.mark.asyncio
    async def test_step4_type_mismatch_raises_type_mismatch_error(self):
        """
        When Step 4 returns outcome='failed', compile() must raise
        TypeMismatchError with error_type == "type_mismatch".
        """
        service, _ = _make_service(step4_outcome="failed")
        with pytest.raises(TypeMismatchError) as exc_info:
            await service.compile(_make_request(), pool=None)
        assert exc_info.value.error_type.value == "type_mismatch"
        assert exc_info.value.http_status == 422

    @pytest.mark.asyncio
    async def test_compile_token_persisted_in_store(self):
        """After compile(), the token is retrievable from CompileTokenStore."""
        service, store = _make_service()
        response = await service.compile(_make_request(), pool=None)

        persisted = await store.get(response.compile_token)
        assert persisted is not None
        assert persisted.token_string == response.compile_token
        assert persisted.identifier == "loan.repayment_ratio"
        assert persisted.used is False

    @pytest.mark.asyncio
    async def test_ir_hash_is_deterministic(self):
        """
        Same request with the same signal_names and identifier must produce
        the same ir_hash on two separate compile() calls.
        """
        service1, store1 = _make_service()
        service2, store2 = _make_service()

        request = _make_request()
        response1 = await service1.compile(request, pool=None)
        response2 = await service2.compile(request, pool=None)

        token1 = await store1.get(response1.compile_token)
        token2 = await store2.get(response2.compile_token)

        assert token1 is not None and token2 is not None
        assert token1.ir_hash == token2.ir_hash

    @pytest.mark.asyncio
    async def test_two_calls_same_identifier_return_distinct_tokens(self):
        """
        The identifier is NOT locked at compile time — only at registration.
        Two compile calls for the same identifier must return distinct tokens.
        """
        service1, _ = _make_service()
        service2, _ = _make_service()

        request = _make_request(identifier="loan.repayment_ratio")
        response1 = await service1.compile(request, pool=None)
        response2 = await service2.compile(request, pool=None)

        assert response1.compile_token != response2.compile_token
        assert (
            response1.compiled_concept.identifier
            == response2.compiled_concept.identifier
            == "loan.repayment_ratio"
        )


# ── HTTP layer tests ──────────────────────────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    yield


_VALID_BODY = {
    "identifier": "loan.repayment_ratio",
    "description": "Ratio of on-time payments to total payments due",
    "output_type": "float",
    "signal_names": ["payments_on_time", "payments_due"],
}


def _make_test_app(
    service: ConceptCompilerService,
) -> tuple[FastAPI, TestClient]:
    """
    Build a minimal FastAPI test app with the concepts router,
    the given service injected, and the DB pool overridden to None.
    """
    app = FastAPI(lifespan=_null_lifespan)
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.include_router(concepts_route.router)

    app.dependency_overrides[get_concept_compiler_service] = lambda: service
    app.dependency_overrides[get_db] = lambda: None

    return app, TestClient(app, raise_server_exceptions=True)


class TestCompileRoute:

    def test_route_returns_201_on_success(self):
        """POST /concepts/compile returns HTTP 201 with compile_token + compiled_concept."""
        service, _ = _make_service()
        _, client = _make_test_app(service)

        response = client.post("/concepts/compile", json=_VALID_BODY)

        assert response.status_code == 201
        data = response.json()
        assert "compile_token" in data
        assert isinstance(data["compile_token"], str)
        assert len(data["compile_token"]) > 0
        assert "compiled_concept" in data
        assert "expires_at" in data
        # reasoning_trace must be absent (not null) when return_reasoning=False
        assert "reasoning_trace" not in data

    def test_route_returns_422_on_compilation_error(self):
        """POST /concepts/compile returns HTTP 422 when CoR Step 1 fails."""

        class _FailStep1LLM:
            def generate_task(self, prompt: str, context: dict) -> dict:
                # Step 1 returns outcome "failed" — triggers CompilationError
                if context.get("step") == 1:
                    return {"summary": "could not parse intent", "outcome": "failed"}
                return {"summary": "ok", "outcome": "accepted"}

        store = MockCompileTokenStore()
        service = ConceptCompilerService(
            llm_client=_FailStep1LLM(),
            token_store=store,
        )
        _, client = _make_test_app(service)

        response = client.post("/concepts/compile", json=_VALID_BODY)

        assert response.status_code == 422
        data = response.json()
        assert data["error"]["type"] == "compilation_error"
