"""
tests/integration/test_v7_ttl_boundary.py
──────────────────────────────────────────────────────────────────────────────
T-7 Part 2 — Compile token TTL boundary behaviour.

Purpose: verify that the boundary condition (TTL-1s = valid, TTL+1s = expired)
is correctly enforced, and that degenerate edge cases (exactly NOW(), concurrent
requests at boundary) are handled safely.

Technique: Direct DB manipulation of expires_at after token creation.
No sleep is used except where explicitly testing timing behaviour.

Locked HTTP status codes (compile token errors):
  COMPILE_TOKEN_EXPIRED    → 400  (not 410 — cross-team contract)
  COMPILE_TOKEN_CONSUMED   → 409
  COMPILE_TOKEN_NOT_FOUND  → 404

Boundary semantics: the SQL check is `expires_at > NOW()` (strict greater-than).
  - expires_at = NOW()        → EXPIRED (400)
  - expires_at > NOW()        → VALID
  - expires_at < NOW()        → EXPIRED (400)
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import concepts as concepts_route
from app.api.routes.concepts import (
    get_concept_compiler_service,
    get_concept_registration_service,
)
from app.config.compile_token_ttl import get_compile_token_ttl
from app.models.errors import MemintelError, memintel_error_handler
from app.services.concept_compiler import ConceptCompilerService
from app.services.concept_registration import ConceptRegistrationService
from app.stores.compile_token import CompileTokenStore


# ── Test-app factory ───────────────────────────────────────────────────────────

def _make_compile_app(db_pool, llm_client: Any) -> FastAPI:
    """Minimal FastAPI app with compile + register routes only."""
    app = FastAPI()
    app.state.db = db_pool
    app.add_exception_handler(MemintelError, memintel_error_handler)
    app.include_router(concepts_route.router)

    async def _compiler_svc() -> ConceptCompilerService:
        return ConceptCompilerService(
            llm_client=llm_client,
            token_store=CompileTokenStore(db_pool),
        )

    async def _registration_svc() -> ConceptRegistrationService:
        return ConceptRegistrationService()

    app.dependency_overrides[get_concept_compiler_service]    = _compiler_svc
    app.dependency_overrides[get_concept_registration_service] = _registration_svc
    return app


_COMPILE_BODY = {
    "identifier":       "ttl.test_concept",
    "description":      "Ratio of on-time payments to total payments due over 90 days",
    "output_type":      "float",
    "signal_names":     ["signal_a"],
    "stream":           False,
    "return_reasoning": False,
}

_REGISTER_IDENTIFIER = "ttl.test_concept"


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def _set_expiry_sql(db_pool, token_string: str, sql_expr: str) -> None:
    """
    Set expires_at on a compile_token using a raw SQL timestamp expression.

    sql_expr is injected directly — use only with trusted literals such as
    'NOW() + INTERVAL ...' or 'NOW() - INTERVAL ...'.
    """
    await db_pool.execute(
        f"UPDATE compile_tokens SET expires_at = {sql_expr} WHERE token_string = $1",
        token_string,
    )


async def _compile_and_get_token(client: httpx.AsyncClient) -> str:
    """Compile a concept and return the compile_token string."""
    resp = await client.post("/concepts/compile", json=_COMPILE_BODY)
    assert resp.status_code == 201, f"compile failed: {resp.text}"
    return resp.json()["compile_token"]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Token valid one second before expiry
# ═══════════════════════════════════════════════════════════════════════════════

def test_token_valid_one_second_before_expiry(db_pool, run, llm_mock):
    """
    A token with expires_at = NOW() + 2s should succeed after 1 second.
    At the time of registration (T+1s), the token has ~1 second remaining.
    The SQL check 'expires_at > NOW()' evaluates to True → 201.
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            token = await _compile_and_get_token(client)

            # Set expiry to NOW() + 2 seconds
            await _set_expiry_sql(db_pool, token, "NOW() + INTERVAL '2 seconds'")

            # Wait 1 second — token now has ~1 second remaining
            await asyncio.sleep(1)

            # Register: should succeed (1 second still remaining)
            resp = await client.post("/concepts/register", json={
                "compile_token": token,
                "identifier":    _REGISTER_IDENTIFIER,
            })
        return resp

    resp = run(_go())
    assert resp.status_code == 201, (
        f"Expected 201 (token still valid); got {resp.status_code}: {resp.text}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Token expired one second in the past
# ═══════════════════════════════════════════════════════════════════════════════

def test_token_expired_one_second_after_expiry(db_pool, run, llm_mock):
    """
    A token with expires_at = NOW() - 1s is already expired.
    Register must return 400 with error.type == 'compile_token_expired'.
    400, not 410 — this is a locked cross-team contract.
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            token = await _compile_and_get_token(client)

            # Set expiry 1 second in the past
            await _set_expiry_sql(db_pool, token, "NOW() - INTERVAL '1 second'")

            resp = await client.post("/concepts/register", json={
                "compile_token": token,
                "identifier":    _REGISTER_IDENTIFIER,
            })
        return resp

    resp = run(_go())
    assert resp.status_code == 400, (
        f"Expected 400 (expired); got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    err_type = body.get("error", {}).get("type", "")
    assert err_type == "compile_token_expired", (
        f"Expected compile_token_expired; got: {body}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Token at exact boundary (expires_at = NOW())
# ═══════════════════════════════════════════════════════════════════════════════

def test_token_expires_at_exact_boundary(db_pool, run, llm_mock):
    """
    Boundary condition: expires_at set to database NOW().

    The SQL check is `expires_at > NOW()` (strict greater-than, not >=).
    A token expiring exactly at the current DB time is expired.

    In practice, the UPDATE sets expires_at = NOW() and then the following
    SELECT in consume() runs a moment later, so expires_at <= NOW() → expired.
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            token = await _compile_and_get_token(client)

            # Set expiry to exactly NOW() — boundary case
            await _set_expiry_sql(db_pool, token, "NOW()")

            # Register immediately — expires_at is at or before NOW() → expired
            resp = await client.post("/concepts/register", json={
                "compile_token": token,
                "identifier":    _REGISTER_IDENTIFIER,
            })
        return resp

    resp = run(_go())
    assert resp.status_code == 400, (
        f"Expected 400 (boundary expired — strict '>'); got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    err_type = body.get("error", {}).get("type", "")
    assert err_type == "compile_token_expired", (
        f"Expected compile_token_expired at exact boundary; got: {body}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — TTL from config applied correctly at token creation
# ═══════════════════════════════════════════════════════════════════════════════

def test_token_ttl_from_config(db_pool, run, llm_mock):
    """
    The token's expires_at must be approximately created_at + TTL.
    Reads the TTL from get_compile_token_ttl() and compares to the DB row.
    Tolerance: 2 seconds (account for test overhead).
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)
    ttl = get_compile_token_ttl()

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            token_string = await _compile_and_get_token(client)

        row = await db_pool.fetchrow(
            "SELECT created_at, expires_at FROM compile_tokens WHERE token_string = $1",
            token_string,
        )
        return token_string, row

    token_string, row = run(_go())

    assert row is not None, f"Token {token_string!r} not found in DB"

    created_at: datetime = row["created_at"]
    expires_at: datetime = row["expires_at"]

    # Both should be timezone-aware from asyncpg TIMESTAMPTZ
    expected_expiry = created_at + timedelta(seconds=ttl)
    delta = abs((expires_at - expected_expiry).total_seconds())

    assert delta <= 2, (
        f"expires_at {expires_at} is not within 2s of created_at + TTL "
        f"({expected_expiry}). Delta: {delta}s. TTL: {ttl}s."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — TTL minimum clamp (no DB required)
# ═══════════════════════════════════════════════════════════════════════════════

def test_token_ttl_minimum_clamp():
    """
    MEMINTEL_COMPILE_TOKEN_TTL_SECONDS=60 (below 300 minimum).
    get_compile_token_ttl() must return 300 (clamped, not rejected).

    No DB or HTTP needed — tests the config module directly.
    """
    original = os.environ.get("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS")
    try:
        os.environ["MEMINTEL_COMPILE_TOKEN_TTL_SECONDS"] = "60"
        result = get_compile_token_ttl()
    finally:
        if original is None:
            os.environ.pop("MEMINTEL_COMPILE_TOKEN_TTL_SECONDS", None)
        else:
            os.environ["MEMINTEL_COMPILE_TOKEN_TTL_SECONDS"] = original

    assert result == 300, (
        f"Expected clamped TTL of 300s; got {result}. "
        "Values below minimum must be clamped, not rejected."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — Expired token is NOT marked as consumed
# ═══════════════════════════════════════════════════════════════════════════════

def test_expired_token_not_consumed(db_pool, run, llm_mock):
    """
    Attempting to register an expired token returns 400.
    The token must NOT be marked as used=True — it was not consumed.

    Rationale: an expired token that is attempted but rejected must remain
    unconsumed so an admin can extend it (re-set expires_at) if needed.
    The atomic UPDATE in consume() guards: `used = FALSE AND expires_at > NOW()`.
    An expired token never matches → used stays False.
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            token = await _compile_and_get_token(client)

            # Expire the token
            await _set_expiry_sql(db_pool, token, "NOW() - INTERVAL '5 seconds'")

            # Attempt registration (should fail with 400)
            resp = await client.post("/concepts/register", json={
                "compile_token": token,
                "identifier":    _REGISTER_IDENTIFIER,
            })

        # Check DB: token must NOT be consumed
        row = await db_pool.fetchrow(
            "SELECT used FROM compile_tokens WHERE token_string = $1",
            token,
        )
        return resp, row

    resp, row = run(_go())

    assert resp.status_code == 400, (
        f"Expected 400 (expired token); got {resp.status_code}: {resp.text}"
    )
    assert row is not None, "Token row not found in DB after expired register attempt"
    assert row["used"] is False, (
        "Expired token must NOT be marked used=True after a failed register attempt. "
        f"DB row shows used={row['used']}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — Simultaneous requests at TTL boundary
# ═══════════════════════════════════════════════════════════════════════════════

def test_simultaneous_requests_at_ttl_boundary(db_pool, run, llm_mock):
    """
    Three simultaneous register requests for a token expiring in ~1 second.

    Verifies:
      - At most 1 returns 201 (token consumed before expiry).
      - Remaining return 400 (expired) or 409 (consumed) — not 500.
      - The server handles the boundary race without crashing.

    Note: the assertion is "at most 1" success because in rare cases under
    high system load, all 3 requests may arrive after the 1-second window,
    resulting in 0 successes (all 400s). This is correct boundary behaviour.
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            token = await _compile_and_get_token(client)

            # Set expiry to 1 second from now
            await _set_expiry_sql(db_pool, token, "NOW() + INTERVAL '1 second'")

            # Fire 3 concurrent registers immediately
            register_body = {
                "compile_token": token,
                "identifier":    _REGISTER_IDENTIFIER,
            }
            responses = await asyncio.gather(*[
                client.post("/concepts/register", json=register_body)
                for _ in range(3)
            ])
        return responses

    responses = run(_go())

    statuses = [r.status_code for r in responses]

    # No 500s — boundary race must not crash the server
    assert 500 not in statuses, (
        f"Server crashed at TTL boundary: {statuses}"
    )

    # At most 1 success
    assert statuses.count(201) <= 1, (
        f"Expected at most 1 × 201 at TTL boundary; got: {statuses}"
    )

    # All must be in expected range
    assert all(s in (200, 201, 400, 409) for s in statuses), (
        f"Unexpected status codes at TTL boundary: {statuses}"
    )
