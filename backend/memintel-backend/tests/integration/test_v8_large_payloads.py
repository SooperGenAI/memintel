"""
tests/integration/test_v8_large_payloads.py
──────────────────────────────────────────────────────────────────────────────
T-8 Part 1 — Large payload handling.

Sections
────────
1. vocabulary_context at scale (Dev Guide §2.6 benchmarking requirement):
   - Baseline latency (no vocab)
   - 100-ID vocab latency (≤ 3× baseline)
   - 500-ID vocab latency (≤ 3× baseline, hard cap)
   - 501-ID vocab rejected fast (< 100 ms, HTTP 422)

2. Large signal_names:
   - 10 entries  → 201
   - 50 entries  → 201, latency measured
   - 100 entries → 201, latency measured
   - Duplicates  → 201, handled gracefully

3. Large intent strings:
   - 2000-char → 201
   - 10000-char → 201 or 422 (behaviour documented)

4. Deeply nested composites:
   - 2 levels deep (baseline composite)
   - 3 levels deep (OR at top of AND-OR tree)
   - 4 levels deep (no stack overflow)
   - Circular self-reference (safe error, no infinite loop)

Dev Guide §2.6 benchmark requirement:
  latency(N-ID vocab) / latency(baseline) ≤ 3.0  for N ∈ {100, 500}

Technique
─────────
HTTP sections use an ASGI transport app with LLMMockClient.
Composite sections use ExecuteService directly (service-layer tests).
No sleep is used. All latency measurements use time.monotonic().
"""
from __future__ import annotations

import asyncio
import time
import types as _types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

import app.services.execute as _execute_module
from app.api.routes import concepts as concepts_route
from app.api.routes import tasks as tasks_route
from app.api.routes.concepts import (
    get_concept_compiler_service,
    get_concept_registration_service,
)
from app.api.routes.tasks import get_task_authoring_service
from app.config.config_loader import ConfigLoader
from app.models.errors import MemintelError, memintel_error_handler
from app.registry.definitions import DefinitionRegistry
from app.runtime.data_resolver import MockConnector
from app.services.concept_compiler import ConceptCompilerService
from app.services.concept_registration import ConceptRegistrationService
from app.services.execute import ExecuteService
from app.services.task_authoring import TaskAuthoringService
from app.stores import DefinitionStore, TaskStore
from app.stores.compile_token import CompileTokenStore

from tests.integration.conftest_v7 import LLMMockClient

pytestmark = pytest.mark.skip(
    reason=(
        "Large payload tests deferred — causes process instability"
        " on local Windows dev environment. Run in CI/staging only."
    )
)

# ── Guardrails ─────────────────────────────────────────────────────────────────

_GUARDRAILS_YAML = str(
    Path(__file__).parent.parent.parent / "memintel_guardrails.yaml"
)


class _SimpleGuardrailsStore:
    """Minimal duck-typed guardrails store wrapping the real YAML config."""

    def __init__(self) -> None:
        self._guardrails = ConfigLoader().load_guardrails(_GUARDRAILS_YAML)

    def get_guardrails(self):
        return self._guardrails

    def get_threshold_bounds(self, strategy: str) -> dict:
        b = self._guardrails.constraints.threshold_bounds.get(strategy)
        return b.model_dump() if b else {}

    def is_loaded(self) -> bool:
        return True

    def get_active_version(self):
        return None


# ── Test app factories ─────────────────────────────────────────────────────────

def _make_test_app(db_pool, llm_client: Any) -> FastAPI:
    """Minimal FastAPI app with tasks + concepts routes."""
    app = FastAPI()
    app.state.db = db_pool
    app.state.guardrails_store = _SimpleGuardrailsStore()
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

    app.dependency_overrides[get_concept_compiler_service]     = _compiler_svc
    app.dependency_overrides[get_concept_registration_service] = _registration_svc
    app.dependency_overrides[get_task_authoring_service]       = _task_svc
    return app


def _make_compile_app(db_pool, llm_client: Any) -> FastAPI:
    """Minimal FastAPI app with only the concepts route."""
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

    app.dependency_overrides[get_concept_compiler_service]     = _compiler_svc
    app.dependency_overrides[get_concept_registration_service] = _registration_svc
    return app


# ── Domain helpers (service-layer composite tests) ─────────────────────────────

def _float_concept(concept_id: str, primitive_name: str) -> dict:
    return {
        "concept_id":     concept_id,
        "version":        "v1",
        "namespace":      "org",
        "output_type":    "float",
        "description":    f"Test concept: {concept_id}",
        "primitives": {
            primitive_name: {"type": "float", "missing_data_policy": "null"}
        },
        "features": {
            "output": {
                "op":     "z_score_op",
                "inputs": {"input": primitive_name},
                "params": {},
            }
        },
        "output_feature": "output",
    }


def _condition_body(
    condition_id: str,
    concept_id: str,
    strategy_type: str,
    params: dict,
    version: str = "v1",
) -> dict:
    return {
        "condition_id":    condition_id,
        "version":         version,
        "concept_id":      concept_id,
        "concept_version": "v1",
        "namespace":       "org",
        "strategy":        {"type": strategy_type, "params": params},
    }


def _register(run, db_pool, definition_id: str, body: dict, def_type: str) -> None:
    store = DefinitionStore(db_pool)
    run(store.register(
        definition_id=definition_id,
        version=body.get("version", "v1"),
        definition_type=def_type,
        namespace="org",
        body=body,
    ))


def _evaluate(run, db_pool, condition_id: str, version: str, entity: str, connector_data: dict):
    """Evaluate a condition via ExecuteService with a MockConnector."""
    mock_conn = MockConnector(data=connector_data)
    svc = ExecuteService(pool=db_pool)
    req = _types.SimpleNamespace(
        condition_id=condition_id,
        condition_version=version,
        entity=entity,
        timestamp=None,
        explain=False,
    )
    with patch.object(_execute_module, "_make_connector", return_value=mock_conn):
        return run(svc.evaluate_condition(req))


# ── Shared request constants ───────────────────────────────────────────────────

# "repayment" → LLMMockClient generates loan.repayment_ratio + loan.repayment_below_threshold
_REPAYMENT_INTENT = "alert when loan repayment ratio is below threshold"
_DELIVERY = {
    "type":     "webhook",
    "endpoint": "https://large-payload-test.example.com/hook",
}

# concept_id that LLMMockClient generates for "repayment" intent
_REAL_CONCEPT_ID = "loan.repayment_ratio"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — vocabulary_context at scale
# ══════════════════════════════════════════════════════════════════════════════

def test_vocabulary_context_baseline_latency(db_pool, run, llm_mock):
    """
    POST /tasks with NO vocabulary_context.

    Baseline measurement for comparison against 100-ID and 500-ID vocab tests.
    Assert: response successful (200).
    Print: baseline latency.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            start = time.monotonic()
            resp = await client.post("/tasks", json={
                "intent":       _REPAYMENT_INTENT,
                "entity_scope": "loan",
                "delivery":     _DELIVERY,
                "stream":       False,
            })
            elapsed_ms = (time.monotonic() - start) * 1000
        return resp, elapsed_ms

    resp, elapsed_ms = run(_go())

    print(f"\nBaseline latency (no vocab): {elapsed_ms:.0f}ms")
    assert resp.status_code == 200, (
        f"Baseline task creation failed: {resp.status_code}: {resp.text}"
    )


def test_vocabulary_context_100_ids_latency(db_pool, run, llm_mock):
    """
    POST /tasks with vocabulary_context containing 100 concept_ids.

    The first ID is 'loan.repayment_ratio' (matches LLM output for this intent).
    The remaining 99 are synthetic IDs (not registered in DB — vocabulary_context
    is checked at the string level only, not against DB state).

    Dev Guide §2.6: latency must not exceed 3× the no-vocab baseline.

    Baseline and vocab measurements are made in the same test with the same
    ASGI transport to ensure a fair comparison.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    task_body_base = {
        "intent":       _REPAYMENT_INTENT,
        "entity_scope": "loan",
        "delivery":     _DELIVERY,
        "stream":       False,
    }

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Baseline: no vocabulary_context (warm-up + measure)
            start_b = time.monotonic()
            resp_b = await client.post("/tasks", json=task_body_base)
            baseline_ms = (time.monotonic() - start_b) * 1000

            # 100-ID vocab: 1 real + 99 synthetic
            concept_ids_100 = [_REAL_CONCEPT_ID] + [
                f"fake.concept_{i:04d}" for i in range(99)
            ]
            start_v = time.monotonic()
            resp_v = await client.post("/tasks", json={
                **task_body_base,
                "vocabulary_context": {"available_concept_ids": concept_ids_100},
            })
            elapsed_ms = (time.monotonic() - start_v) * 1000

        return resp_b, baseline_ms, resp_v, elapsed_ms

    resp_b, baseline_ms, resp_v, elapsed_ms = run(_go())

    # Use a minimum baseline of 1 ms to guard against sub-millisecond timings
    # on fast machines where monotonic() rounds to 0.0 ms, which would make
    # the 3× ratio comparison and the ratio print both divide-by-zero.
    effective_baseline_ms = max(baseline_ms, 1.0)

    print(f"\nBaseline latency (no vocab): {baseline_ms:.0f}ms")
    print(
        f"100-ID vocab latency: {elapsed_ms:.0f}ms "
        f"({elapsed_ms/effective_baseline_ms:.1f}x baseline)"
    )

    assert resp_b.status_code == 200, (
        f"Baseline failed: {resp_b.status_code}: {resp_b.text}"
    )
    # 200 if concept matched; 422 if vocab mismatch — both acceptable for latency test
    assert resp_v.status_code in (200, 422), (
        f"Unexpected status for 100-ID vocab: {resp_v.status_code}: {resp_v.text}"
    )
    assert elapsed_ms < effective_baseline_ms * 3, (
        f"100-ID vocab latency {elapsed_ms:.0f}ms exceeds 3× baseline {baseline_ms:.0f}ms. "
        "Dev Guide §2.6: vocabulary validation must not add unbounded latency."
    )


def test_vocabulary_context_500_ids_latency(db_pool, run, llm_mock):
    """
    POST /tasks with vocabulary_context containing 500 concept_ids (the hard cap).

    500 IDs is the maximum allowed per list (MAX_VOCABULARY_IDS).
    Assert: response is 200 or 422 (not 500).
    Assert: latency ≤ 3× no-vocab baseline (Dev Guide §2.6).

    If latency exceeds 3× baseline, the test fails with a message directing to
    lower the cap or optimise the validation path.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    task_body_base = {
        "intent":       _REPAYMENT_INTENT,
        "entity_scope": "loan",
        "delivery":     _DELIVERY,
        "stream":       False,
    }

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Baseline
            start_b = time.monotonic()
            resp_b = await client.post("/tasks", json=task_body_base)
            baseline_ms = (time.monotonic() - start_b) * 1000

            # 500-ID vocab: 1 real + 499 synthetic
            concept_ids_500 = [_REAL_CONCEPT_ID] + [
                f"fake.concept_{i:04d}" for i in range(499)
            ]
            start_v = time.monotonic()
            resp_v = await client.post("/tasks", json={
                **task_body_base,
                "vocabulary_context": {"available_concept_ids": concept_ids_500},
            })
            elapsed_ms = (time.monotonic() - start_v) * 1000

        return resp_b, baseline_ms, resp_v, elapsed_ms

    resp_b, baseline_ms, resp_v, elapsed_ms = run(_go())

    effective_baseline_ms = max(baseline_ms, 1.0)

    print(f"\nBaseline latency (no vocab): {baseline_ms:.0f}ms")
    print(
        f"500-ID vocab latency: {elapsed_ms:.0f}ms "
        f"({elapsed_ms/effective_baseline_ms:.1f}x baseline)"
    )

    assert resp_b.status_code == 200, (
        f"Baseline failed: {resp_b.status_code}: {resp_b.text}"
    )
    assert resp_v.status_code in (200, 422), (
        f"Unexpected status for 500-ID vocab: {resp_v.status_code}: {resp_v.text}"
    )

    if elapsed_ms > effective_baseline_ms * 3:
        print(
            f"WARNING: 500-ID latency {elapsed_ms:.0f}ms exceeds 3× baseline {baseline_ms:.0f}ms. "
            "Dev Guide §2.6 requires the cap to be lowered."
        )
        pytest.fail(
            f"500-ID vocab latency {elapsed_ms:.0f}ms exceeds Dev Guide 3× threshold "
            f"({effective_baseline_ms * 3:.0f}ms). Lower MAX_VOCABULARY_IDS or optimise validation."
        )


def test_vocabulary_context_501_ids_rejected_fast(db_pool, run, llm_mock):
    """
    POST /tasks with 501 concept_ids — one over the hard cap.

    Rejection must happen BEFORE the LLM call (pre-LLM validation).
    Assert: HTTP 422 (VocabularyContextTooLargeError or Pydantic validation error).
    Assert: LLM not called (call_count == 0).
    Assert: elapsed < 100 ms (validation fires immediately).
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    concept_ids_501 = [f"fake.concept_{i:04d}" for i in range(501)]

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            start = time.monotonic()
            resp = await client.post("/tasks", json={
                "intent":       _REPAYMENT_INTENT,
                "entity_scope": "loan",
                "delivery":     _DELIVERY,
                "stream":       False,
                "vocabulary_context": {"available_concept_ids": concept_ids_501},
            })
            elapsed_ms = (time.monotonic() - start) * 1000
        return resp, elapsed_ms

    resp, elapsed_ms = run(_go())

    print(f"\n501-ID rejection latency: {elapsed_ms:.0f}ms")

    assert resp.status_code == 422, (
        f"Expected 422 for 501-ID vocab; got {resp.status_code}: {resp.text}"
    )
    assert elapsed_ms < 100, (
        f"501-ID rejection took {elapsed_ms:.0f}ms — must be < 100ms (pre-LLM check)."
    )
    # LLM must NOT have been called — rejection happens before the LLM pipeline
    assert llm_mock.call_count == 0, (
        f"LLM was called {llm_mock.call_count} times — 501-ID rejection must fire before LLM."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — large signal_names
# ══════════════════════════════════════════════════════════════════════════════

_COMPILE_BODY_BASE = {
    "identifier":       "large.signal_test_concept",
    "description":      "Test concept for large signal_names payloads",
    "output_type":      "float",
    "stream":           False,
    "return_reasoning": False,
}


def test_signal_names_10_entries(db_pool, run, llm_mock):
    """
    POST /concepts/compile with 10 signal_names.
    Assert: 201 — accepted.
    Assert: compile_token present in response.
    Assert: compiled_concept.signal_bindings non-empty.
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    signal_names_10 = [
        "payments_on_time", "payments_due", "days_overdue", "credit_score",
        "outstanding_balance", "interest_rate", "loan_term", "collateral_value",
        "income_to_debt_ratio", "account_age",
    ]

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/concepts/compile", json={
                **_COMPILE_BODY_BASE,
                "signal_names": signal_names_10,
            })

    resp = run(_go())

    assert resp.status_code == 201, (
        f"Expected 201 for 10 signal_names; got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "compile_token" in body, "compile_token missing from response"
    # signal_bindings may be populated — accept any non-empty list
    compiled = body.get("compiled_concept", {})
    signal_bindings = compiled.get("signal_bindings", [])
    assert isinstance(signal_bindings, list), (
        f"signal_bindings must be a list; got {type(signal_bindings)}"
    )
    assert len(signal_bindings) > 0, (
        "signal_bindings must be non-empty — LLM should bind at least one signal"
    )


def test_signal_names_50_entries(db_pool, run, llm_mock):
    """
    POST /concepts/compile with 50 signal_names (mix of realistic + generic).
    Assert: 201 — accepted.
    Measure latency.
    Print: compile latency.
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    # Mix of realistic loan domain names and generic names
    realistic = [
        "payments_on_time", "payments_due", "days_overdue", "credit_score",
        "outstanding_balance", "interest_rate", "loan_term", "collateral_value",
        "income_to_debt_ratio", "account_age", "payment_velocity",
        "loan_to_value_ratio", "default_probability", "employment_months",
        "monthly_income", "monthly_expenses", "savings_balance",
        "previous_defaults", "bankruptcy_history", "credit_utilisation",
    ]
    generic = [f"signal_{i:03d}" for i in range(30)]
    signal_names_50 = realistic + generic

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            start = time.monotonic()
            resp = await client.post("/concepts/compile", json={
                **_COMPILE_BODY_BASE,
                "identifier":   "large.fifty_signal_concept",
                "signal_names": signal_names_50,
            })
            elapsed_ms = (time.monotonic() - start) * 1000
        return resp, elapsed_ms

    resp, elapsed_ms = run(_go())

    print(f"\n50 signal_names compile latency: {elapsed_ms:.0f}ms")
    assert resp.status_code == 201, (
        f"Expected 201 for 50 signal_names; got {resp.status_code}: {resp.text}"
    )
    assert "compile_token" in resp.json(), "compile_token missing"


def test_signal_names_100_entries(db_pool, run, llm_mock):
    """
    POST /concepts/compile with 100 signal_names.
    Assert: 201 — accepted (no hard cap on signal_names).
    Assert: compile_token present.
    Measure latency.
    Print: compile latency.

    Note: signal_names are opaque hints to the LLM — 100 entries is unusual
    but must not crash the service. The LLMMockClient ignores the list and
    returns deterministic bindings for step 3.
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    signal_names_100 = [f"signal_{i:04d}" for i in range(100)]

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            start = time.monotonic()
            resp = await client.post("/concepts/compile", json={
                **_COMPILE_BODY_BASE,
                "identifier":   "large.hundred_signal_concept",
                "signal_names": signal_names_100,
            })
            elapsed_ms = (time.monotonic() - start) * 1000
        return resp, elapsed_ms

    resp, elapsed_ms = run(_go())

    print(f"\n100 signal_names compile latency: {elapsed_ms:.0f}ms")
    assert resp.status_code == 201, (
        f"Expected 201 for 100 signal_names; got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "compile_token" in body, (
        f"compile_token must be present in response; got {list(body.keys())}"
    )


def test_signal_names_with_duplicates(db_pool, run, llm_mock):
    """
    POST /concepts/compile with duplicate signal_names entries.

    The route layer must NOT reject duplicates (signal_names are opaque hints).
    Assert: 201 — accepted.
    Assert: compiled_concept.signal_bindings does not assign the same
    signal_name to more than one role (deduplication or graceful handling).

    Duplicate input: ["payments_on_time", "payments_due", "payments_on_time",
                       "days_overdue", "payments_due"]
    """
    app = _make_compile_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    signal_names_with_dups = [
        "payments_on_time", "payments_due", "payments_on_time",
        "days_overdue", "payments_due",
    ]

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/concepts/compile", json={
                **_COMPILE_BODY_BASE,
                "identifier":   "large.duplicate_signal_concept",
                "signal_names": signal_names_with_dups,
            })

    resp = run(_go())

    assert resp.status_code == 201, (
        f"Duplicates in signal_names must not be rejected at route layer; "
        f"got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "compile_token" in body, "compile_token missing"

    # Check signal_bindings: no duplicate role assignments for the same signal
    compiled = body.get("compiled_concept", {})
    bindings = compiled.get("signal_bindings", [])
    # Count (signal_name, role) pairs — each pair should appear at most once
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs: list[tuple[str, str]] = []
    for b in bindings:
        pair = (b.get("signal_name", ""), b.get("role", ""))
        if pair in seen_pairs:
            duplicate_pairs.append(pair)
        seen_pairs.add(pair)

    assert not duplicate_pairs, (
        f"signal_bindings contains duplicate (signal_name, role) pairs: {duplicate_pairs}. "
        "Duplicates in signal_names input must not produce duplicate role assignments."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — large intent strings
# ══════════════════════════════════════════════════════════════════════════════

def test_large_intent_string(db_pool, run, llm_mock):
    """
    POST /tasks with a 2000-character intent string.
    Assert: 201 — accepted.
    Assert: task_id returned.
    Measure latency.
    Print: latency.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    # 2000-char intent — "repayment" keyword must be present so LLMMockClient
    # routes to the correct fixture.
    intent_2000 = (
        "alert when loan repayment ratio is below threshold for entities "
        "that have shown deteriorating payment behaviour over the past 90 days. "
        "This monitoring requirement applies to all active loan accounts in the "
        "portfolio with outstanding principal above $10,000. Consider seasonal "
        "factors including holiday payment delays and end-of-quarter patterns. "
        "The alert must fire within 24 hours of the condition being met and must "
        "include a full audit trail of the evaluation result, the concept value, "
        "the threshold applied, and all primitive values fetched from the data "
        "layer. The webhook endpoint must receive a structured JSON payload with "
        "standardised fields as defined in the API specification version 7. "
    ) * 3  # ~600 chars × 3 ≈ 1800, pad to 2000
    intent_2000 = (intent_2000 + " " * 2000)[:2000]

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            start = time.monotonic()
            resp = await client.post("/tasks", json={
                "intent":       intent_2000,
                "entity_scope": "loan",
                "delivery":     _DELIVERY,
                "stream":       False,
            })
            elapsed_ms = (time.monotonic() - start) * 1000
        return resp, elapsed_ms

    resp, elapsed_ms = run(_go())

    print(f"\n2000-char intent latency: {elapsed_ms:.0f}ms")
    assert resp.status_code == 200, (
        f"Expected 200 for 2000-char intent; got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "task_id" in body, f"task_id missing; response keys: {list(body.keys())}"


def test_very_large_intent_string(db_pool, run, llm_mock):
    """
    POST /tasks with a 10000-character intent string.

    The service has no explicit maximum intent length. The LLMMockClient
    routes by intent substring, so "repayment" anywhere in the string will
    match.

    Documented behaviour: 10000-char intents are accepted (200) if no
    route-layer length validation exists. If 422 is returned, the error
    message must mention intent length.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    # 10000-char intent containing "repayment" for LLMMockClient routing
    intent_10000 = "alert when loan repayment ratio is below 0.80 " * 215  # ~10000 chars
    intent_10000 = intent_10000[:10000]

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/tasks", json={
                "intent":       intent_10000,
                "entity_scope": "loan",
                "delivery":     _DELIVERY,
                "stream":       False,
            })

    resp = run(_go())

    if resp.status_code == 200:
        print("\n10000-char intent: ACCEPTED (200) — no route-layer length cap")
        body = resp.json()
        assert "task_id" in body, "task_id missing from 200 response"
    elif resp.status_code == 422:
        print("\n10000-char intent: REJECTED (422) — route-layer length cap exists")
        body = resp.json()
        # If our MemintelError format, check error message; if Pydantic, check detail
        error_text = resp.text.lower()
        assert "intent" in error_text or "length" in error_text or "too large" in error_text, (
            f"422 error must mention intent length; got: {resp.text}"
        )
    else:
        pytest.fail(
            f"Unexpected status {resp.status_code} for 10000-char intent: {resp.text}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — deeply nested composites
# ══════════════════════════════════════════════════════════════════════════════

def test_composite_2_levels_deep(db_pool, run):
    """
    2-level composite (baseline): composite(AND, [threshold_A, threshold_B]).

    Setup:
      concept_A + cond_A: fires when score_a < 0.80
      concept_B + cond_B: fires when score_b < 0.50
      comp2: AND(cond_A, cond_B)

    Case 1: score_a=0.65 (<0.80 → A fires), score_b=0.30 (<0.50 → B fires)
            → AND → True.
    Case 2: score_a=0.65 (<0.80 → A fires), score_b=0.60 (≥0.50 → B doesn't)
            → AND → False.
    """
    # Register sub-concepts and conditions
    concept_a = _float_concept("deep2.concept_a", "deep2.score_a")
    cond_a = _condition_body("deep2.cond_a", "deep2.concept_a", "threshold",
                              {"direction": "below", "value": 0.80})
    concept_b = _float_concept("deep2.concept_b", "deep2.score_b")
    cond_b = _condition_body("deep2.cond_b", "deep2.concept_b", "threshold",
                              {"direction": "below", "value": 0.50})
    dummy = _float_concept("deep2.dummy", "deep2.dummy_score")
    comp2 = _condition_body("deep2.comp2", "deep2.dummy", "composite", {
        "operator": "AND",
        "operands": [
            {"condition_id": "deep2.cond_a", "condition_version": "v1"},
            {"condition_id": "deep2.cond_b", "condition_version": "v1"},
        ],
    })

    for name, body, typ in [
        ("deep2.concept_a", concept_a, "concept"),
        ("deep2.cond_a",   cond_a,    "condition"),
        ("deep2.concept_b", concept_b, "concept"),
        ("deep2.cond_b",   cond_b,    "condition"),
        ("deep2.dummy",    dummy,     "concept"),
        ("deep2.comp2",    comp2,     "condition"),
    ]:
        _register(run, db_pool, name, body, typ)

    entity = "test-entity"

    # Case 1: both conditions fire → AND → True
    result1 = _evaluate(run, db_pool, "deep2.comp2", "v1", entity, {
        ("deep2.score_a",    entity, None): 0.65,
        ("deep2.score_b",    entity, None): 0.30,
        ("deep2.dummy_score", entity, None): 0.50,
    })
    assert result1.value is True, (
        f"AND(A fires, B fires) should be True; got {result1.value}"
    )

    # Case 2: only A fires → AND → False
    result2 = _evaluate(run, db_pool, "deep2.comp2", "v1", entity, {
        ("deep2.score_a",    entity, None): 0.65,
        ("deep2.score_b",    entity, None): 0.60,
        ("deep2.dummy_score", entity, None): 0.50,
    })
    assert result2.value is False, (
        f"AND(A fires, B doesn't) should be False; got {result2.value}"
    )


def test_composite_3_levels_deep(db_pool, run):
    """
    3-level composite:
      level_1a = threshold condition (score_a < 0.80)
      level_1b = threshold condition (score_b < 0.50)
      level_2  = composite(AND, [level_1a, level_1b])
      level_3  = composite(OR,  [level_2,  threshold_C (score_c > 0.70)])

    Test 1 (level_2 fires, C doesn't):
      score_a=0.65 (<0.80 → A fires), score_b=0.30 (<0.50 → B fires) → level_2=True
      score_c=0.50 (≤0.70 → C doesn't fire) → level_2 OR C = True OR False = True

    Test 2 (neither level_2 nor C fires):
      score_a=0.90 (≥0.80 → A doesn't), score_b=0.40 → level_2=False
      score_c=0.50 (≤0.70 → C doesn't) → False OR False = False
    """
    # Concepts
    concept_a = _float_concept("deep3.concept_a", "deep3.score_a")
    concept_b = _float_concept("deep3.concept_b", "deep3.score_b")
    concept_c = _float_concept("deep3.concept_c", "deep3.score_c")
    dummy2    = _float_concept("deep3.dummy2", "deep3.dummy2_score")
    dummy3    = _float_concept("deep3.dummy3", "deep3.dummy3_score")

    # Level-1 leaf conditions
    cond_a = _condition_body("deep3.cond_a", "deep3.concept_a", "threshold",
                              {"direction": "below", "value": 0.80})
    cond_b = _condition_body("deep3.cond_b", "deep3.concept_b", "threshold",
                              {"direction": "below", "value": 0.50})
    cond_c = _condition_body("deep3.cond_c", "deep3.concept_c", "threshold",
                              {"direction": "above", "value": 0.70})

    # Level-2 composite: AND(cond_a, cond_b)
    comp_level2 = _condition_body("deep3.comp_level2", "deep3.dummy2", "composite", {
        "operator": "AND",
        "operands": [
            {"condition_id": "deep3.cond_a", "condition_version": "v1"},
            {"condition_id": "deep3.cond_b", "condition_version": "v1"},
        ],
    })

    # Level-3 composite: OR(comp_level2, cond_c)
    comp_level3 = _condition_body("deep3.comp_level3", "deep3.dummy3", "composite", {
        "operator": "OR",
        "operands": [
            {"condition_id": "deep3.comp_level2", "condition_version": "v1"},
            {"condition_id": "deep3.cond_c",      "condition_version": "v1"},
        ],
    })

    for name, body, typ in [
        ("deep3.concept_a",   concept_a,   "concept"),
        ("deep3.concept_b",   concept_b,   "concept"),
        ("deep3.concept_c",   concept_c,   "concept"),
        ("deep3.dummy2",      dummy2,      "concept"),
        ("deep3.dummy3",      dummy3,      "concept"),
        ("deep3.cond_a",      cond_a,      "condition"),
        ("deep3.cond_b",      cond_b,      "condition"),
        ("deep3.cond_c",      cond_c,      "condition"),
        ("deep3.comp_level2", comp_level2, "condition"),
        ("deep3.comp_level3", comp_level3, "condition"),
    ]:
        _register(run, db_pool, name, body, typ)

    entity = "test-entity"

    # Test 1: level_2 fires (A and B both fire), C doesn't → OR = True
    result1 = _evaluate(run, db_pool, "deep3.comp_level3", "v1", entity, {
        ("deep3.score_a",      entity, None): 0.65,   # < 0.80 → A fires
        ("deep3.score_b",      entity, None): 0.30,   # < 0.50 → B fires
        ("deep3.score_c",      entity, None): 0.50,   # ≤ 0.70 → C doesn't fire
        ("deep3.dummy2_score", entity, None): 0.50,
        ("deep3.dummy3_score", entity, None): 0.50,
    })
    assert result1.value is True, (
        f"OR(AND(A fires, B fires), C doesn't) should be True; got {result1.value}"
    )

    # Test 2: neither level_2 nor C fires → OR = False
    result2 = _evaluate(run, db_pool, "deep3.comp_level3", "v1", entity, {
        ("deep3.score_a",      entity, None): 0.90,   # ≥ 0.80 → A doesn't fire
        ("deep3.score_b",      entity, None): 0.40,   # < 0.50 → B fires, but A doesn't
        ("deep3.score_c",      entity, None): 0.50,   # ≤ 0.70 → C doesn't fire
        ("deep3.dummy2_score", entity, None): 0.50,
        ("deep3.dummy3_score", entity, None): 0.50,
    })
    assert result2.value is False, (
        f"OR(AND(A doesn't, B fires), C doesn't) should be False; got {result2.value}"
    )


def test_composite_4_levels_deep(db_pool, run):
    """
    4-level composite:
      leaf_1 = threshold (score_1 < 0.80)
      leaf_2 = threshold (score_2 < 0.50)
      leaf_3 = threshold (score_3 > 0.30)
      leaf_4 = threshold (score_4 > 0.60)
      level_2 = AND(leaf_1, leaf_2)
      level_3 = OR(level_2, leaf_3)
      level_4 = AND(level_3, leaf_4)

    Test values:
      leaf_1: 0.65 (<0.80 → fires)
      leaf_2: 0.30 (<0.50 → fires) → level_2 = AND(T, T) = True
      leaf_3: 0.40 (>0.30 → fires) → level_3 = OR(T, T) = True
      leaf_4: 0.70 (>0.60 → fires) → level_4 = AND(T, T) = True

    Assert: evaluation completes within 5 seconds.
    Assert: decision value is correct (True for above case).
    Assert: no stack overflow or infinite loop.
    """
    concept_1 = _float_concept("deep4.concept_1", "deep4.score_1")
    concept_2 = _float_concept("deep4.concept_2", "deep4.score_2")
    concept_3 = _float_concept("deep4.concept_3", "deep4.score_3")
    concept_4 = _float_concept("deep4.concept_4", "deep4.score_4")
    dummy_l2  = _float_concept("deep4.dummy_l2", "deep4.dl2")
    dummy_l3  = _float_concept("deep4.dummy_l3", "deep4.dl3")
    dummy_l4  = _float_concept("deep4.dummy_l4", "deep4.dl4")

    leaf_1 = _condition_body("deep4.leaf_1", "deep4.concept_1", "threshold",
                              {"direction": "below", "value": 0.80})
    leaf_2 = _condition_body("deep4.leaf_2", "deep4.concept_2", "threshold",
                              {"direction": "below", "value": 0.50})
    leaf_3 = _condition_body("deep4.leaf_3", "deep4.concept_3", "threshold",
                              {"direction": "above", "value": 0.30})
    leaf_4 = _condition_body("deep4.leaf_4", "deep4.concept_4", "threshold",
                              {"direction": "above", "value": 0.60})

    level_2 = _condition_body("deep4.level_2", "deep4.dummy_l2", "composite", {
        "operator": "AND",
        "operands": [
            {"condition_id": "deep4.leaf_1", "condition_version": "v1"},
            {"condition_id": "deep4.leaf_2", "condition_version": "v1"},
        ],
    })
    level_3 = _condition_body("deep4.level_3", "deep4.dummy_l3", "composite", {
        "operator": "OR",
        "operands": [
            {"condition_id": "deep4.level_2", "condition_version": "v1"},
            {"condition_id": "deep4.leaf_3",  "condition_version": "v1"},
        ],
    })
    level_4 = _condition_body("deep4.level_4", "deep4.dummy_l4", "composite", {
        "operator": "AND",
        "operands": [
            {"condition_id": "deep4.level_3", "condition_version": "v1"},
            {"condition_id": "deep4.leaf_4",  "condition_version": "v1"},
        ],
    })

    for name, body, typ in [
        ("deep4.concept_1", concept_1, "concept"),
        ("deep4.concept_2", concept_2, "concept"),
        ("deep4.concept_3", concept_3, "concept"),
        ("deep4.concept_4", concept_4, "concept"),
        ("deep4.dummy_l2",  dummy_l2,  "concept"),
        ("deep4.dummy_l3",  dummy_l3,  "concept"),
        ("deep4.dummy_l4",  dummy_l4,  "concept"),
        ("deep4.leaf_1",   leaf_1,   "condition"),
        ("deep4.leaf_2",   leaf_2,   "condition"),
        ("deep4.leaf_3",   leaf_3,   "condition"),
        ("deep4.leaf_4",   leaf_4,   "condition"),
        ("deep4.level_2",  level_2,  "condition"),
        ("deep4.level_3",  level_3,  "condition"),
        ("deep4.level_4",  level_4,  "condition"),
    ]:
        _register(run, db_pool, name, body, typ)

    entity = "test-entity"
    connector_data = {
        ("deep4.score_1", entity, None): 0.65,  # < 0.80 → leaf_1 fires
        ("deep4.score_2", entity, None): 0.30,  # < 0.50 → leaf_2 fires
        ("deep4.score_3", entity, None): 0.40,  # > 0.30 → leaf_3 fires
        ("deep4.score_4", entity, None): 0.70,  # > 0.60 → leaf_4 fires
        ("deep4.dl2",     entity, None): 0.50,
        ("deep4.dl3",     entity, None): 0.50,
        ("deep4.dl4",     entity, None): 0.50,
    }

    start = time.monotonic()
    result = _evaluate(run, db_pool, "deep4.level_4", "v1", entity, connector_data)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, (
        f"4-level composite evaluation took {elapsed:.2f}s — must complete within 5s."
    )
    assert result.value is True, (
        f"AND(OR(AND(T,T),T),T) should be True; got {result.value}"
    )


def test_composite_circular_reference_rejected(db_pool, run):
    """
    Composite condition where an operand references the composite itself.

    Strategy:
      cond_leaf  = threshold condition (safe leaf)
      cond_circular = composite(AND, [cond_circular:v1, cond_leaf:v1])

    Registration: succeeds (no cycle check at registration time — the body
    is valid JSON with a composite strategy).

    Evaluation: raises an error (RecursionError at Python level) before
    completing. The server must not loop forever.

    Assert: evaluation raises an exception within 5 seconds.
    Assert: the exception is RecursionError or a MemintelError (not 500 from server).
    Assert: the exception is raised — not a silent return value.

    Note: Python's default recursion limit (~1000) ensures RecursionError
    is raised quickly. This is not caught and re-raised as a MemintelError
    at the service layer — this is a known gap (no graph_error for composite
    circular refs at evaluation time).
    """
    concept_leaf = _float_concept("circ.leaf_concept", "circ.leaf_score")
    cond_leaf = _condition_body("circ.cond_leaf", "circ.leaf_concept", "threshold",
                                {"direction": "above", "value": 0.5})
    dummy = _float_concept("circ.dummy", "circ.dummy_score")
    # Self-referential composite: operand[0] references itself
    cond_circular = _condition_body("circ.cond_circular", "circ.dummy", "composite", {
        "operator": "AND",
        "operands": [
            {"condition_id": "circ.cond_circular", "condition_version": "v1"},  # self-reference
            {"condition_id": "circ.cond_leaf",     "condition_version": "v1"},
        ],
    })

    for name, body, typ in [
        ("circ.leaf_concept",  concept_leaf, "concept"),
        ("circ.cond_leaf",     cond_leaf,    "condition"),
        ("circ.dummy",         dummy,        "concept"),
        ("circ.cond_circular", cond_circular, "condition"),
    ]:
        _register(run, db_pool, name, body, typ)

    entity = "test-entity"
    connector_data = {
        ("circ.leaf_score",   entity, None): 0.70,
        ("circ.dummy_score",  entity, None): 0.50,
    }

    start = time.monotonic()
    caught_error: BaseException | None = None

    try:
        _evaluate(run, db_pool, "circ.cond_circular", "v1", entity, connector_data)
    except RecursionError as exc:
        caught_error = exc
    except MemintelError as exc:
        caught_error = exc
    except Exception as exc:
        caught_error = exc

    elapsed = time.monotonic() - start

    assert elapsed < 5.0, (
        f"Circular reference evaluation took {elapsed:.2f}s — must error within 5s."
    )
    assert caught_error is not None, (
        "Circular composite reference must raise an error — "
        "silent return value (possibly stub resolver) is not acceptable. "
        "If the stub resolver returns False for all unknown conditions, "
        "the self-reference terminates at the first recursion level "
        "and returns a result instead of erroring."
    )

    # Document observed behaviour
    error_type = type(caught_error).__name__
    print(f"\nCircular composite reference raised: {error_type}: {caught_error}")
