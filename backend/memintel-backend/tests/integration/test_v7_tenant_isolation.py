"""
tests/integration/test_v7_tenant_isolation.py
──────────────────────────────────────────────────────────────────────────────
T-7 Part 4 — Multi-tenant isolation via vocabulary_context.

Purpose: verify that vocabulary_context prevents cross-tenant concept bleed.
Org A's concept_ids must not be selectable when Org B's vocabulary is active.

Background
──────────
Memintel has no built-in org/tenant model. All concept_ids live in one global
registry. vocabulary_context is the ONLY mechanism preventing Org A from using
Org B's concepts. If it fails, tenants can silently access each other's
decision logic.

How vocabulary enforcement works
─────────────────────────────────
POST /tasks triggers the LLM pipeline. In Step 2 (Concept Selection), the LLM
returns a concept body with a concept_id field. The service then checks:
  - Is concept_id in vocabulary_context.available_concept_ids? (if vocab present)
  - Is condition_id in vocabulary_context.available_condition_ids? (if vocab present)

If either check fails → 422 vocabulary_mismatch.
If vocabulary_context is absent → no check (global fallback).
If both lists are empty → 422 before LLM is called (call_count stays 0).

LLM routing in this test suite
────────────────────────────────
The LLMMockClient from conftest_v7 routes by intent substring:
  'repayment' → concept_id='loan.repayment_ratio',
                condition_id='loan.repayment_below_threshold'
  'overdue'   → concept_id='loan.days_overdue',
                condition_id='loan.overdue_condition'

Tenant assignments (for test purposes):
  Org A concepts:    loan.repayment_ratio, loan.credit_score
  Org A conditions:  loan.repayment_below_threshold, loan.credit_below_threshold
  Org B concepts:    loan.days_overdue, loan.payment_velocity
  Org B conditions:  loan.overdue_condition, loan.velocity_below_threshold
"""
from __future__ import annotations

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

from tests.integration.conftest_v7 import LLMMockClient


# ── Tenant vocabulary constants ────────────────────────────────────────────────
#
# Concept IDs match what LLMMockClient returns for each intent keyword.
# Org A uses "repayment" and "credit" intents; Org B uses "overdue" and "velocity".

_ORG_A_CONCEPT_IDS    = ["loan.repayment_ratio", "loan.credit_score"]
_ORG_A_CONDITION_IDS  = ["loan.repayment_below_threshold", "loan.credit_below_threshold"]

_ORG_B_CONCEPT_IDS    = ["loan.days_overdue", "loan.payment_velocity"]
_ORG_B_CONDITION_IDS  = ["loan.overdue_condition", "loan.velocity_below_threshold"]

_ORG_A_VOCAB = {
    "available_concept_ids":   _ORG_A_CONCEPT_IDS,
    "available_condition_ids": _ORG_A_CONDITION_IDS,
}
_ORG_B_VOCAB = {
    "available_concept_ids":   _ORG_B_CONCEPT_IDS,
    "available_condition_ids": _ORG_B_CONDITION_IDS,
}

# Shared concept in both org A and org B vocabularies
_SHARED_CONCEPT_ID    = "loan.repayment_ratio"
_SHARED_CONDITION_ID  = "loan.repayment_below_threshold"

_DELIVERY = {
    "type":     "webhook",
    "endpoint": "https://tenant-test.example.com/hook",
}


# ── Test-app factory ───────────────────────────────────────────────────────────

def _make_test_app(db_pool, llm_client: Any) -> FastAPI:
    """Minimal FastAPI app for vocabulary enforcement tests."""
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

    app.dependency_overrides[get_concept_compiler_service]    = _compiler_svc
    app.dependency_overrides[get_concept_registration_service] = _registration_svc
    app.dependency_overrides[get_task_authoring_service]      = _task_svc
    return app


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Org A vocabulary excludes Org B concepts
# ═══════════════════════════════════════════════════════════════════════════════

def test_org_a_vocabulary_excludes_org_b_concepts(db_pool, run, llm_mock):
    """
    POST /tasks with Org A's vocabulary_context and an "overdue" intent.
    LLMMockClient returns loan.days_overdue (an Org B concept).
    loan.days_overdue is NOT in Org A's available_concept_ids → 422.

    Verifies: vocabulary_context hard-blocks cross-tenant concept access.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/tasks", json={
                "intent":             "alert when loan is overdue by more than 30 days",
                "entity_scope":       "loan",
                "delivery":           _DELIVERY,
                "stream":             False,
                "vocabulary_context": _ORG_A_VOCAB,
            })

    resp = run(_go())
    assert resp.status_code == 422, (
        f"Expected 422 (vocabulary_mismatch); got {resp.status_code}: {resp.text}. "
        "Org A vocabulary must exclude Org B's loan.days_overdue concept."
    )
    body = resp.json()
    err_type = body.get("error", {}).get("type", "")
    assert "vocabulary" in err_type.lower() or "semantic" in err_type.lower(), (
        f"Expected vocabulary_mismatch error type; got: {err_type}. Full: {body}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Org B vocabulary excludes Org A concepts
# ═══════════════════════════════════════════════════════════════════════════════

def test_org_b_vocabulary_excludes_org_a_concepts(db_pool, run, llm_mock):
    """
    POST /tasks with Org B's vocabulary_context and a "repayment" intent.
    LLMMockClient returns loan.repayment_ratio (an Org A concept).
    loan.repayment_ratio is NOT in Org B's available_concept_ids → 422.

    Mirrors test_org_a_vocabulary_excludes_org_b_concepts for the opposite direction.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/tasks", json={
                "intent":             "alert when loan repayment ratio is below threshold",
                "entity_scope":       "loan",
                "delivery":           _DELIVERY,
                "stream":             False,
                "vocabulary_context": _ORG_B_VOCAB,
            })

    resp = run(_go())
    assert resp.status_code == 422, (
        f"Expected 422 (vocabulary_mismatch); got {resp.status_code}: {resp.text}. "
        "Org B vocabulary must exclude Org A's loan.repayment_ratio concept."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Absent vocabulary_context sees all concepts (global fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def test_absent_vocabulary_context_sees_all_concepts(db_pool, run, llm_mock):
    """
    POST /tasks with NO vocabulary_context.
    LLMMockClient returns loan.days_overdue (an Org B concept).
    Without restriction, this should succeed.

    Verifies: absent vocabulary = global fallback. Canvas must never omit
    vocabulary_context for authenticated org requests.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/tasks", json={
                "intent":       "alert when loan is overdue by more than 30 days",
                "entity_scope": "loan",
                "delivery":     _DELIVERY,
                "stream":       False,
                # vocabulary_context intentionally absent
            })

    resp = run(_go())
    assert resp.status_code == 200, (
        f"Expected 200 (no vocabulary restriction); got {resp.status_code}: {resp.text}"
    )
    data = resp.json()
    # Task uses the Org B concept — no cross-tenant restriction without vocab
    assert data.get("concept_id") == "loan.days_overdue", (
        f"Expected concept_id=loan.days_overdue (no vocab filter); got: {data.get('concept_id')}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — Single-concept vocabulary is fully enforced
# ═══════════════════════════════════════════════════════════════════════════════

def test_single_concept_vocabulary_still_validated(db_pool, run, llm_mock):
    """
    A vocabulary with exactly 1 concept_id is fully enforced — not bypassed
    as a 'trivial' case.

    Part A: LLM returns the correct concept (loan.repayment_ratio) → 200.
    Part B: same vocabulary, LLM returns a different concept (loan.days_overdue) → 422.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    single_vocab = {
        "available_concept_ids":   ["loan.repayment_ratio"],
        "available_condition_ids": ["loan.repayment_below_threshold"],
    }

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Part A: intent that produces the single allowed concept → 200
            resp_a = await client.post("/tasks", json={
                "intent":             "alert when loan repayment ratio is below threshold",
                "entity_scope":       "loan",
                "delivery":           _DELIVERY,
                "stream":             False,
                "vocabulary_context": single_vocab,
            })

            # Part B: intent that produces a different concept → 422
            resp_b = await client.post("/tasks", json={
                "intent":             "alert when loan is overdue",
                "entity_scope":       "loan",
                "delivery":           _DELIVERY,
                "stream":             False,
                "vocabulary_context": single_vocab,
            })
        return resp_a, resp_b

    resp_a, resp_b = run(_go())

    assert resp_a.status_code == 200, (
        f"Part A: expected 200 (concept in single-vocab); got {resp_a.status_code}: {resp_a.text}"
    )
    assert resp_b.status_code == 422, (
        f"Part B: expected 422 (concept NOT in single-vocab); got {resp_b.status_code}: {resp_b.text}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — Overlapping concepts available to multiple orgs
# ═══════════════════════════════════════════════════════════════════════════════

def test_vocabulary_context_with_overlapping_concepts(db_pool, run, llm_mock):
    """
    concept_x = loan.repayment_ratio is included in BOTH Org A and Org B vocabularies.

    POST /tasks with Org A vocabulary → concept_x → 200.
    POST /tasks with Org B vocabulary → concept_x → 200.

    Verifies: a concept available to multiple orgs is not exclusively locked
    to one org. Shared/platform concepts must work across all tenants.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    # Both vocabs include the shared concept
    vocab_a_with_shared = {
        "available_concept_ids":   [_SHARED_CONCEPT_ID] + _ORG_A_CONCEPT_IDS,
        "available_condition_ids": [_SHARED_CONDITION_ID] + _ORG_A_CONDITION_IDS,
    }
    vocab_b_with_shared = {
        "available_concept_ids":   [_SHARED_CONCEPT_ID] + _ORG_B_CONCEPT_IDS,
        "available_condition_ids": [_SHARED_CONDITION_ID] + _ORG_B_CONDITION_IDS,
    }

    shared_intent = "alert when loan repayment ratio is below threshold"

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp_a = await client.post("/tasks", json={
                "intent":             shared_intent,
                "entity_scope":       "org_a",
                "delivery":           _DELIVERY,
                "stream":             False,
                "vocabulary_context": vocab_a_with_shared,
            })
            resp_b = await client.post("/tasks", json={
                "intent":             shared_intent,
                "entity_scope":       "org_b",
                "delivery":           _DELIVERY,
                "stream":             False,
                "vocabulary_context": vocab_b_with_shared,
            })
        return resp_a, resp_b

    resp_a, resp_b = run(_go())

    assert resp_a.status_code == 200, (
        f"Org A with shared concept: expected 200; got {resp_a.status_code}: {resp_a.text}"
    )
    assert resp_b.status_code == 200, (
        f"Org B with shared concept: expected 200; got {resp_b.status_code}: {resp_b.text}"
    )

    # Both used the shared concept
    assert resp_a.json()["concept_id"] == _SHARED_CONCEPT_ID
    assert resp_b.json()["concept_id"] == _SHARED_CONCEPT_ID


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — Empty vocabulary rejected before LLM
# ═══════════════════════════════════════════════════════════════════════════════

def test_empty_vocabulary_rejected_before_llm(db_pool, run):
    """
    vocabulary_context with both lists empty → 422 before LLM is called.
    llm_mock.call_count must remain 0.

    Verifies: the guard fires at the service layer before the LLM pipeline,
    so Canvas misconfigurations are caught immediately.
    """
    fresh_llm = LLMMockClient()   # fresh instance so call_count starts at 0
    app = _make_test_app(db_pool, fresh_llm)
    transport = httpx.ASGITransport(app=app)

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/tasks", json={
                "intent":       "alert when loan repayment ratio is below threshold",
                "entity_scope": "loan",
                "delivery":     _DELIVERY,
                "stream":       False,
                "vocabulary_context": {
                    "available_concept_ids":   [],
                    "available_condition_ids": [],
                },
            })

    resp = run(_go())
    assert resp.status_code == 422, (
        f"Expected 422 for empty vocabulary; got {resp.status_code}: {resp.text}"
    )
    assert fresh_llm.call_count == 0, (
        f"LLM must NOT be called when vocabulary is empty; call_count = {fresh_llm.call_count}. "
        "The empty-vocabulary guard must fire before the LLM pipeline."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — Condition ID isolation works independently of concept ID
# ═══════════════════════════════════════════════════════════════════════════════

def test_cross_tenant_bleed_via_condition_ids(db_pool, run, llm_mock):
    """
    Build a vocabulary where the concept_id IS allowed but the condition_id
    returned by the LLM is NOT in available_condition_ids.

    Org A vocab: concept_ids = [loan.repayment_ratio] (correct for repayment intent)
                 condition_ids = [loan.overdue_condition] (WRONG — this is an Org B ID)

    LLM for "repayment" intent returns:
      concept_id=loan.repayment_ratio   → IN vocab (passes concept check)
      condition_id=loan.repayment_below_threshold → NOT in vocab → 422

    Verifies: condition_id isolation is independent of concept_id isolation.
    A matching concept_id is not sufficient if the condition_id bleeds.
    """
    app = _make_test_app(db_pool, llm_mock)
    transport = httpx.ASGITransport(app=app)

    # Concept allowed, but condition_ids contain ONLY an org-B condition
    mismatched_vocab = {
        "available_concept_ids":   ["loan.repayment_ratio"],
        "available_condition_ids": ["loan.overdue_condition"],   # Org B condition
    }

    async def _go():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/tasks", json={
                "intent":             "alert when loan repayment ratio is below threshold",
                "entity_scope":       "loan",
                "delivery":           _DELIVERY,
                "stream":             False,
                "vocabulary_context": mismatched_vocab,
            })

    resp = run(_go())

    assert resp.status_code == 422, (
        f"Expected 422 (condition_id NOT in vocab); got {resp.status_code}: {resp.text}. "
        "loan.repayment_below_threshold is not in available_condition_ids — should be rejected."
    )
