"""
tests/unit/test_compile_token_store.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for CompileTokenStore contract and get_compile_token_ttl().

Store tests use an in-memory MockCompileTokenStore that implements the
identical logical contract as the real CompileTokenStore but requires no
database connection.  This follows the established unit-test pattern in
this codebase (see test_calibration.py).

TTL tests exercise get_compile_token_ttl() directly with monkeypatched env vars.

Concurrent safety test uses asyncio.gather to verify that exactly one of two
simultaneous consume() calls succeeds when both race for the same token.
"""
from __future__ import annotations

import asyncio
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.config.compile_token_ttl import get_compile_token_ttl
from app.models.concept_compile import CompileToken
from app.models.errors import (
    CompileTokenConsumedError,
    CompileTokenExpiredError,
    CompileTokenNotFoundError,
    ConflictError,
)


# ── In-memory MockCompileTokenStore ──────────────────────────────────────────
#
# Mirrors the exact logical contract of CompileTokenStore without touching
# the database.  The consume() method uses asyncio.Lock to simulate the
# atomic UPDATE...WHERE used=FALSE that the real store relies on in Postgres.

class MockCompileTokenStore:
    """
    In-memory implementation of CompileTokenStore for unit testing.

    Implements the same public interface as the real store:
      create(token) → CompileToken
      get(token_string) → CompileToken | None
      consume(token_string) → CompileToken  (raises on any failure)

    Thread-/coroutine-safety: consume() acquires a per-token asyncio.Lock
    before inspecting and mutating state, mirroring the row-level locking
    that Postgres provides in the real implementation.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, CompileToken] = {}
        # Per-token locks to simulate atomic UPDATE...WHERE used=FALSE
        self._locks: dict[str, asyncio.Lock] = {}

    # ── create ────────────────────────────────────────────────────────────────

    async def create(self, token: CompileToken) -> CompileToken:
        if token.token_string in self._tokens:
            raise ConflictError(
                "compile_token with token_string already exists.",
                location="token_string",
            )
        self._tokens[token.token_string] = token
        self._locks[token.token_string] = asyncio.Lock()
        return token

    # ── get ───────────────────────────────────────────────────────────────────

    async def get(self, token_string: str) -> CompileToken | None:
        return self._tokens.get(token_string)

    # ── consume ───────────────────────────────────────────────────────────────

    async def consume(self, token_string: str) -> CompileToken:
        """
        Atomically marks the token as used.

        Uses a per-token asyncio.Lock to mirror the Postgres row-level locking
        that the real implementation relies on.  Raises the same error types
        as CompileTokenStore.consume() for the same failure conditions.
        """
        lock = self._locks.get(token_string)

        if lock is None:
            # Token never existed
            raise CompileTokenNotFoundError("compile_token not found.")

        async with lock:
            token = self._tokens.get(token_string)
            if token is None:
                raise CompileTokenNotFoundError("compile_token not found.")

            if token.used:
                raise CompileTokenConsumedError(
                    "compile_token has already been consumed.",
                    suggestion="Call POST /concepts/compile to obtain a new token.",
                )

            now = datetime.now(tz=timezone.utc)
            # Ensure both sides are timezone-aware for comparison
            expires_at = token.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

            if expires_at <= now:
                # HTTP 400, not 410 — cross-team contract with Canvas consumer.
                raise CompileTokenExpiredError(
                    "compile_token has expired. Call POST /concepts/compile again.",
                )

            # Valid — mark used and return updated token
            updated = token.model_copy(update={"used": True})
            self._tokens[token_string] = updated
            return updated


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_token(
    *,
    identifier: str = "loan.repayment_ratio",
    expired: bool = False,
    used: bool = False,
) -> CompileToken:
    """Build a minimal CompileToken for testing."""
    now = datetime.now(tz=timezone.utc)
    if expired:
        expires_at = now - timedelta(minutes=5)
    else:
        expires_at = now + timedelta(minutes=30)

    return CompileToken(
        token_id=str(uuid.uuid4()),
        token_string=secrets.token_urlsafe(32),
        identifier=identifier,
        ir_hash="sha256_" + secrets.token_hex(16),
        expires_at=expires_at,
        used=used,
        created_at=now,
    )


# ── create() ─────────────────────────────────────────────────────────────────

class TestCreate:

    @pytest.mark.asyncio
    async def test_create_stores_token_successfully(self):
        store = MockCompileTokenStore()
        token = _make_token()
        result = await store.create(token)
        assert result.token_string == token.token_string
        assert result.identifier == token.identifier
        assert result.used is False

    @pytest.mark.asyncio
    async def test_create_duplicate_token_string_raises_conflict(self):
        store = MockCompileTokenStore()
        token = _make_token()
        await store.create(token)
        with pytest.raises(ConflictError):
            await store.create(token)

    @pytest.mark.asyncio
    async def test_create_two_tokens_same_identifier_both_succeed(self):
        """Two pending tokens for the same identifier can coexist simultaneously.
        The identifier is NOT locked at compile time — only at registration."""
        store = MockCompileTokenStore()
        t1 = _make_token(identifier="loan.repayment_ratio")
        t2 = _make_token(identifier="loan.repayment_ratio")
        r1 = await store.create(t1)
        r2 = await store.create(t2)
        assert r1.token_string != r2.token_string
        assert r1.identifier == r2.identifier == "loan.repayment_ratio"


# ── get() ─────────────────────────────────────────────────────────────────────

class TestGet:

    @pytest.mark.asyncio
    async def test_get_returns_token_by_token_string(self):
        store = MockCompileTokenStore()
        token = _make_token()
        await store.create(token)
        result = await store.get(token.token_string)
        assert result is not None
        assert result.token_string == token.token_string

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown_token_string(self):
        store = MockCompileTokenStore()
        result = await store.get("does_not_exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_does_not_mutate_token(self):
        """get() must not change used status or any other field."""
        store = MockCompileTokenStore()
        token = _make_token()
        await store.create(token)
        r1 = await store.get(token.token_string)
        r2 = await store.get(token.token_string)
        assert r1.used is False
        assert r2.used is False


# ── consume() ────────────────────────────────────────────────────────────────

class TestConsume:

    @pytest.mark.asyncio
    async def test_consume_succeeds_for_valid_unused_non_expired_token(self):
        store = MockCompileTokenStore()
        token = _make_token()
        await store.create(token)
        result = await store.consume(token.token_string)
        assert result.token_string == token.token_string
        assert result.used is True

    @pytest.mark.asyncio
    async def test_consume_twice_raises_consumed_on_second_attempt(self):
        """Calling consume() twice on the same token: first succeeds, second raises."""
        store = MockCompileTokenStore()
        token = _make_token()
        await store.create(token)
        await store.consume(token.token_string)  # first — succeeds
        with pytest.raises(CompileTokenConsumedError):
            await store.consume(token.token_string)  # second — must raise

    @pytest.mark.asyncio
    async def test_consume_raises_expired_for_expired_token(self):
        """Token with expires_at in the past → CompileTokenExpiredError."""
        store = MockCompileTokenStore()
        token = _make_token(expired=True)
        await store.create(token)
        with pytest.raises(CompileTokenExpiredError):
            await store.consume(token.token_string)

    @pytest.mark.asyncio
    async def test_consume_raises_not_found_for_unknown_token(self):
        store = MockCompileTokenStore()
        with pytest.raises(CompileTokenNotFoundError):
            await store.consume("totally_unknown_token_string")

    @pytest.mark.asyncio
    async def test_expired_error_is_http_400_not_410(self):
        """
        Cross-team contract: CompileTokenExpiredError.http_status must be 400.
        Do not change to 410.
        """
        store = MockCompileTokenStore()
        token = _make_token(expired=True)
        await store.create(token)
        with pytest.raises(CompileTokenExpiredError) as exc_info:
            await store.consume(token.token_string)
        assert exc_info.value.http_status == 400
        assert exc_info.value.http_status != 410

    @pytest.mark.asyncio
    async def test_consumed_error_is_http_409(self):
        store = MockCompileTokenStore()
        token = _make_token()
        await store.create(token)
        await store.consume(token.token_string)
        with pytest.raises(CompileTokenConsumedError) as exc_info:
            await store.consume(token.token_string)
        assert exc_info.value.http_status == 409

    @pytest.mark.asyncio
    async def test_not_found_error_is_http_404(self):
        store = MockCompileTokenStore()
        with pytest.raises(CompileTokenNotFoundError) as exc_info:
            await store.consume("no_such_token")
        assert exc_info.value.http_status == 404

    @pytest.mark.asyncio
    async def test_consume_marks_token_as_used_in_store(self):
        """After consume(), get() must reflect used=True."""
        store = MockCompileTokenStore()
        token = _make_token()
        await store.create(token)
        await store.consume(token.token_string)
        fetched = await store.get(token.token_string)
        assert fetched is not None
        assert fetched.used is True


# ── Concurrent safety ─────────────────────────────────────────────────────────

class TestConcurrentSafety:

    @pytest.mark.asyncio
    async def test_concurrent_consume_exactly_one_succeeds(self):
        """
        Two coroutines calling consume() simultaneously on the same token:
        exactly one must succeed and one must raise CompileTokenConsumedError.

        The MockCompileTokenStore uses asyncio.Lock per token to simulate the
        atomic UPDATE...WHERE used=FALSE that Postgres provides in the real
        implementation.
        """
        store = MockCompileTokenStore()
        token = _make_token()
        await store.create(token)

        successes: list[CompileToken] = []
        failures: list[Exception] = []

        async def try_consume() -> None:
            try:
                result = await store.consume(token.token_string)
                successes.append(result)
            except (CompileTokenConsumedError, CompileTokenNotFoundError) as exc:
                failures.append(exc)

        # Fire both coroutines concurrently
        await asyncio.gather(try_consume(), try_consume())

        assert len(successes) == 1, (
            f"Expected exactly 1 success, got {len(successes)}"
        )
        assert len(failures) == 1, (
            f"Expected exactly 1 failure, got {len(failures)}"
        )
        assert isinstance(failures[0], CompileTokenConsumedError)

    @pytest.mark.asyncio
    async def test_concurrent_consume_multiple_races(self):
        """Five simultaneous consume() calls: exactly one succeeds."""
        store = MockCompileTokenStore()
        token = _make_token()
        await store.create(token)

        successes: list[CompileToken] = []
        failures: list[Exception] = []

        async def try_consume() -> None:
            try:
                result = await store.consume(token.token_string)
                successes.append(result)
            except (CompileTokenConsumedError, CompileTokenNotFoundError) as exc:
                failures.append(exc)

        await asyncio.gather(*[try_consume() for _ in range(5)])

        assert len(successes) == 1
        assert len(failures) == 4


# ── TTL configuration ─────────────────────────────────────────────────────────

class TestGetCompileTokenTtl:

    def test_default_is_1800_when_env_var_absent(self, monkeypatch):
        monkeypatch.delenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", raising=False)
        assert get_compile_token_ttl() == 1800

    def test_env_var_value_is_used_when_set(self, monkeypatch):
        monkeypatch.setenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "900")
        assert get_compile_token_ttl() == 900

    def test_ttl_below_300_is_clamped_to_300(self, monkeypatch):
        monkeypatch.setenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "60")
        assert get_compile_token_ttl() == 300

    def test_ttl_of_exactly_300_is_accepted_as_is(self, monkeypatch):
        monkeypatch.setenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "300")
        assert get_compile_token_ttl() == 300

    def test_ttl_of_1_is_clamped_to_300(self, monkeypatch):
        monkeypatch.setenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "1")
        assert get_compile_token_ttl() == 300

    def test_ttl_of_0_is_clamped_to_300(self, monkeypatch):
        monkeypatch.setenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "0")
        assert get_compile_token_ttl() == 300

    def test_ttl_of_299_is_clamped_to_300(self, monkeypatch):
        monkeypatch.setenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "299")
        assert get_compile_token_ttl() == 300

    def test_ttl_of_301_is_accepted_as_is(self, monkeypatch):
        monkeypatch.setenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "301")
        assert get_compile_token_ttl() == 301

    def test_invalid_non_integer_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "not_a_number")
        assert get_compile_token_ttl() == 1800

    def test_empty_string_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", "")
        assert get_compile_token_ttl() == 1800


# ── Import smoke test ─────────────────────────────────────────────────────────

class TestImports:

    def test_compile_token_store_importable_from_stores(self):
        from app.persistence.stores import get_compile_token_store
        from app.stores import CompileTokenStore
        from app.stores.compile_token import CompileTokenStore as CT
        assert CT is CompileTokenStore

    def test_ttl_function_importable_from_config(self):
        from app.config import get_compile_token_ttl as fn
        assert callable(fn)
