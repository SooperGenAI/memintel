"""
tests/unit/test_concepts_register.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for ConceptRegistrationService and POST /concepts/register.

Service contract tests use MockCompileTokenStore + MockDefinitionStore and
exercise the service directly — no database, no HTTP stack.

HTTP layer tests use a minimal FastAPI TestClient with dependency overrides.

Coverage
────────
  Service contract:
  1.  Valid request returns RegisterConceptResponse with concept_id + registered_at
  2.  consume() is called with the token string from the request
  3.  Identifier mismatch → IdentifierMismatchError (HTTP 422)
  4.  Expired token → CompileTokenExpiredError (HTTP 400)
  5.  Consumed token → CompileTokenConsumedError (HTTP 409)
  6.  Token not found → CompileTokenNotFoundError (HTTP 404)
  7.  Idempotent registration: same (identifier, ir_hash) → same concept_id (HTTP 201)
  8.  Identifier conflict: same identifier + different ir_hash → IdentifierConflictError (HTTP 409)

  HTTP layer:
  9.  POST /concepts/register returns 201 on success
  10. POST /concepts/register returns 422 on identifier_mismatch
"""
from __future__ import annotations

import sys
import uuid
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
from app.api.routes.concepts import get_concept_registration_service
from app.models.concept_compile import CompileToken, RegisterConceptRequest
from app.models.concept import DefinitionResponse
from app.models.errors import (
    CompileTokenConsumedError,
    CompileTokenExpiredError,
    CompileTokenNotFoundError,
    ConflictError,
    IdentifierConflictError,
    IdentifierMismatchError,
    MemintelError,
    memintel_error_handler,
)
from app.models.task import Namespace
from app.persistence.db import get_db
from app.services.concept_registration import ConceptRegistrationService


# ── Mock: CompileTokenStore ───────────────────────────────────────────────────

class MockCompileTokenStore:
    """
    In-memory CompileTokenStore for registration tests.

    Supports create(), get(), and consume() with realistic error behaviour.
    Tokens with expires_at in the past raise CompileTokenExpiredError.
    Tokens with used=True raise CompileTokenConsumedError.
    Missing tokens raise CompileTokenNotFoundError.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, CompileToken] = {}
        self._consume_calls: list[str] = []   # record consume() call args

    def seed(self, token: CompileToken) -> None:
        """Pre-load a token so tests can control state precisely."""
        self._tokens[token.token_string] = token

    async def create(self, token: CompileToken) -> CompileToken:
        self._tokens[token.token_string] = token
        return token

    async def get(self, token_string: str) -> CompileToken | None:
        return self._tokens.get(token_string)

    async def consume(self, token_string: str) -> CompileToken:
        self._consume_calls.append(token_string)
        token = self._tokens.get(token_string)
        if token is None:
            raise CompileTokenNotFoundError("compile_token not found.")
        if token.used:
            raise CompileTokenConsumedError("compile_token has already been consumed.")
        now = datetime.now(tz=timezone.utc)
        exp = token.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp <= now:
            raise CompileTokenExpiredError("compile_token has expired.")
        # Mark used (mutate a copy)
        self._tokens[token_string] = token.model_copy(update={"used": True})
        return self._tokens[token_string]


# ── Mock: DefinitionStore ─────────────────────────────────────────────────────

class MockDefinitionStore:
    """
    In-memory DefinitionStore for registration tests.

    Tracks registered definitions by (definition_id, version).
    Raises ConflictError on duplicate (definition_id, version).
    get_metadata() returns stored DefinitionResponse.
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DefinitionResponse] = {}

    async def register(
        self,
        definition_id: str,
        version: str,
        definition_type: str,
        namespace: str,
        body: dict,
        meaning_hash: str | None = None,
        ir_hash: str | None = None,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key in self._rows:
            raise ConflictError(
                f"Definition '{definition_id}' version '{version}' is already registered."
            )
        defn = DefinitionResponse(
            definition_id=definition_id,
            version=version,
            definition_type=definition_type,
            namespace=Namespace(namespace),
            ir_hash=ir_hash,
            meaning_hash=meaning_hash,
            created_at=datetime.now(tz=timezone.utc),
        )
        self._rows[key] = defn
        return defn

    async def get_metadata(
        self, definition_id: str, version: str
    ) -> DefinitionResponse | None:
        return self._rows.get((definition_id, version))


# ── Helpers ───────────────────────────────────────────────────────────────────

_TOKEN_STRING = "test-token-string-abc123"
_IDENTIFIER   = "loan.repayment_ratio"
_IR_HASH      = "sha256_abc123deadbeef"
_OUTPUT_TYPE  = "float"


def _make_token(
    *,
    identifier: str = _IDENTIFIER,
    ir_hash: str = _IR_HASH,
    output_type: str = _OUTPUT_TYPE,
    used: bool = False,
    expired: bool = False,
) -> CompileToken:
    """Build a CompileToken with predictable defaults."""
    now = datetime.now(tz=timezone.utc)
    expires_at = (
        now - timedelta(seconds=60)   # already expired
        if expired
        else now + timedelta(seconds=1800)
    )
    return CompileToken(
        token_id=str(uuid.uuid4()),
        token_string=_TOKEN_STRING,
        identifier=identifier,
        ir_hash=ir_hash,
        output_type=output_type,
        expires_at=expires_at,
        used=used,
        created_at=now,
    )


def _make_request(
    *,
    compile_token: str = _TOKEN_STRING,
    identifier: str = _IDENTIFIER,
) -> RegisterConceptRequest:
    return RegisterConceptRequest(
        compile_token=compile_token,
        identifier=identifier,
    )


def _make_service(
    token_store: MockCompileTokenStore | None = None,
    definition_store: MockDefinitionStore | None = None,
) -> ConceptRegistrationService:
    return ConceptRegistrationService(
        token_store=token_store or MockCompileTokenStore(),
        definition_store=definition_store or MockDefinitionStore(),
    )


# ── Service contract tests ─────────────────────────────────────────────────────

class TestRegisterContract:

    @pytest.mark.asyncio
    async def test_valid_request_returns_response_with_concept_id_and_registered_at(self):
        """Happy path: returns RegisterConceptResponse with concept_id + registered_at."""
        token_store = MockCompileTokenStore()
        token_store.seed(_make_token())
        service = _make_service(token_store=token_store)

        response = await service.register(_make_request(), pool=None)

        assert response.concept_id == _IDENTIFIER
        assert response.identifier == _IDENTIFIER
        assert response.version == "1.0.0"
        assert response.output_type == _OUTPUT_TYPE
        assert isinstance(response.registered_at, datetime)

    @pytest.mark.asyncio
    async def test_consume_called_with_correct_token_string(self):
        """consume() must be invoked with the token string from the request."""
        token_store = MockCompileTokenStore()
        token_store.seed(_make_token())
        service = _make_service(token_store=token_store)

        await service.register(_make_request(compile_token=_TOKEN_STRING), pool=None)

        assert token_store._consume_calls == [_TOKEN_STRING]

    @pytest.mark.asyncio
    async def test_identifier_mismatch_raises_identifier_mismatch_error(self):
        """
        Token identifier ≠ request identifier → IdentifierMismatchError (HTTP 422).
        No write to DefinitionStore must occur.
        """
        token_store = MockCompileTokenStore()
        token_store.seed(_make_token(identifier="original.concept"))
        definition_store = MockDefinitionStore()
        service = _make_service(token_store=token_store, definition_store=definition_store)

        with pytest.raises(IdentifierMismatchError) as exc_info:
            await service.register(
                _make_request(identifier="different.concept"),
                pool=None,
            )

        assert exc_info.value.http_status == 422
        assert exc_info.value.error_type.value == "identifier_mismatch"
        # No definition was registered
        assert len(definition_store._rows) == 0

    @pytest.mark.asyncio
    async def test_expired_token_raises_compile_token_expired_error(self):
        """Expired token → CompileTokenExpiredError (HTTP 400)."""
        token_store = MockCompileTokenStore()
        token_store.seed(_make_token(expired=True))
        service = _make_service(token_store=token_store)

        with pytest.raises(CompileTokenExpiredError) as exc_info:
            await service.register(_make_request(), pool=None)

        assert exc_info.value.http_status == 400

    @pytest.mark.asyncio
    async def test_consumed_token_raises_compile_token_consumed_error(self):
        """Already-used token → CompileTokenConsumedError (HTTP 409)."""
        token_store = MockCompileTokenStore()
        token_store.seed(_make_token(used=True))
        service = _make_service(token_store=token_store)

        with pytest.raises(CompileTokenConsumedError) as exc_info:
            await service.register(_make_request(), pool=None)

        assert exc_info.value.http_status == 409

    @pytest.mark.asyncio
    async def test_token_not_found_raises_compile_token_not_found_error(self):
        """Unknown token string → CompileTokenNotFoundError (HTTP 404)."""
        token_store = MockCompileTokenStore()  # empty — no token seeded
        service = _make_service(token_store=token_store)

        with pytest.raises(CompileTokenNotFoundError) as exc_info:
            await service.register(
                _make_request(compile_token="nonexistent-token"),
                pool=None,
            )

        assert exc_info.value.http_status == 404

    @pytest.mark.asyncio
    async def test_idempotent_registration_returns_same_concept_id(self):
        """
        Two register calls for the same (identifier, ir_hash) both return HTTP 201
        with the SAME concept_id. The second call must NOT raise IdentifierConflictError.

        Each call uses a separate token (two separate compile→register flows
        both for the same identifier + formula).
        """
        _TOKEN_A = "token-a-aaaaaa"
        _TOKEN_B = "token-b-bbbbbb"

        token_a = _make_token(ir_hash=_IR_HASH)
        token_b = _make_token(ir_hash=_IR_HASH)
        # Give them distinct token_strings
        token_a = token_a.model_copy(update={"token_string": _TOKEN_A})
        token_b = token_b.model_copy(update={"token_string": _TOKEN_B})

        # Shared definition store — simulates one DB for both calls
        shared_def_store = MockDefinitionStore()
        token_store = MockCompileTokenStore()
        token_store.seed(token_a)
        token_store.seed(token_b)

        svc_a = ConceptRegistrationService(
            token_store=token_store, definition_store=shared_def_store
        )
        svc_b = ConceptRegistrationService(
            token_store=token_store, definition_store=shared_def_store
        )

        resp_a = await svc_a.register(_make_request(compile_token=_TOKEN_A), pool=None)
        resp_b = await svc_b.register(_make_request(compile_token=_TOKEN_B), pool=None)

        assert resp_a.concept_id == resp_b.concept_id == _IDENTIFIER
        assert resp_a.version == resp_b.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_identifier_conflict_different_ir_hash_raises(self):
        """
        Same identifier already registered with a different ir_hash
        → IdentifierConflictError (HTTP 409).
        """
        _TOKEN_FIRST  = "token-first-111"
        _TOKEN_SECOND = "token-second-222"

        # First token: ir_hash A
        token_first = _make_token(ir_hash="sha256_aaaa").model_copy(
            update={"token_string": _TOKEN_FIRST}
        )
        # Second token: same identifier, different ir_hash
        token_second = _make_token(ir_hash="sha256_bbbb").model_copy(
            update={"token_string": _TOKEN_SECOND}
        )

        shared_def_store = MockDefinitionStore()
        token_store = MockCompileTokenStore()
        token_store.seed(token_first)
        token_store.seed(token_second)

        # First registration succeeds
        svc = ConceptRegistrationService(
            token_store=token_store, definition_store=shared_def_store
        )
        await svc.register(_make_request(compile_token=_TOKEN_FIRST), pool=None)

        # Second registration with different ir_hash → conflict
        with pytest.raises(IdentifierConflictError) as exc_info:
            await svc.register(_make_request(compile_token=_TOKEN_SECOND), pool=None)

        assert exc_info.value.http_status == 409
        assert exc_info.value.error_type.value == "identifier_conflict"


# ── HTTP layer tests ──────────────────────────────────────────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    yield


_VALID_REGISTER_BODY = {
    "compile_token": _TOKEN_STRING,
    "identifier": _IDENTIFIER,
}


def _make_test_app(service: ConceptRegistrationService) -> TestClient:
    """Build a minimal FastAPI test app with the concepts router injected."""
    app = FastAPI(lifespan=_null_lifespan)
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.include_router(concepts_route.router)

    app.dependency_overrides[get_concept_registration_service] = lambda: service
    app.dependency_overrides[get_db] = lambda: None

    return TestClient(app, raise_server_exceptions=True)


class TestRegisterRoute:

    def test_route_returns_201_on_success(self):
        """POST /concepts/register returns HTTP 201 with concept_id + registered_at."""
        token_store = MockCompileTokenStore()
        token_store.seed(_make_token())
        service = _make_service(token_store=token_store)
        client = _make_test_app(service)

        response = client.post("/concepts/register", json=_VALID_REGISTER_BODY)

        assert response.status_code == 201
        data = response.json()
        assert data["concept_id"] == _IDENTIFIER
        assert data["identifier"] == _IDENTIFIER
        assert data["version"] == "1.0.0"
        assert data["output_type"] == _OUTPUT_TYPE
        assert "registered_at" in data

    def test_route_returns_422_on_identifier_mismatch(self):
        """POST /concepts/register returns HTTP 422 when identifier does not match token."""
        token_store = MockCompileTokenStore()
        token_store.seed(_make_token(identifier="original.concept"))
        service = _make_service(token_store=token_store)
        client = _make_test_app(service)

        response = client.post(
            "/concepts/register",
            json={
                "compile_token": _TOKEN_STRING,
                "identifier": "different.concept",  # mismatch
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert data["error"]["type"] == "identifier_mismatch"
