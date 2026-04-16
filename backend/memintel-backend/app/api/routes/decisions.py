"""
app/api/routes/decisions.py
──────────────────────────────────────────────────────────────────────────────
Decision explanation endpoint.

Endpoints
─────────
  POST /decisions/explain    explainDecision — full explanation for a decision

Ownership rules
───────────────
This endpoint is deterministic — no LLM involvement.

POST /decisions/explain:
  Returns a full explanation of a decision result for a specific entity at a
  specific timestamp:
    - Why the condition fired or did not fire
    - The concept value that was evaluated
    - The threshold or label applied (strategy-aware)
    - Driver contributions from each input signal (sum to 1.0)

  timestamp is required (ISO 8601 UTC). The explanation re-executes the same
  concept execution path as POST /evaluate/full — the result is deterministic
  for any (condition_id, condition_version, entity, timestamp) tuple.

  HTTP 404 — decision record not found for the given (condition_id,
             condition_version, entity, timestamp).

  Decision output field naming (per developer_api.yaml x-ts-note):
    DecisionExplanation.decision — the decision output in explain responses.
    DecisionResult.value         — the same concept in evaluation results.
    These use different field names by spec design — do not rename either.

Error handling
──────────────
MemintelError subclasses (NotFoundError) are caught globally by the exception
handler in main.py. Routes do not catch them here.
"""
from __future__ import annotations

import structlog

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import require_api_key
from app.compiler.dag_builder import DAGBuilder
from app.compiler.ir_generator import IRGenerator
from app.models.condition import DecisionExplanation
from app.models.concept import ConceptDefinition
from app.persistence.db import get_db
from app.registry.definitions import DefinitionRegistry
from app.runtime.cache import ResultCache
from app.runtime.condition_evaluator import ConditionEvaluator
from app.runtime.data_resolver import DataResolver, MockConnector
from app.runtime.executor import ConceptExecutor
from app.services.explanation import ExplanationService
from app.stores.decision import DecisionStore
from app.stores.definition import DefinitionStore
from app.stores.graph import GraphStore

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/decisions", tags=["Decisions"])


# ── Request body ───────────────────────────────────────────────────────────────
# Inline schema from developer_api.yaml (not a named component).

class ExplainDecisionRequest(BaseModel):
    condition_id: str = Field(..., max_length=255)
    condition_version: str = Field(..., max_length=50)
    entity: str = Field(..., max_length=512)
    timestamp: str    # ISO 8601 UTC — required; deterministic for given tuple


# ── Service dependency ─────────────────────────────────────────────────────────

async def get_explanation_service(
    request: Request,
    pool: asyncpg.Pool = Depends(get_db),
) -> ExplanationService:
    """
    FastAPI dependency — returns an ExplanationService backed by the shared pool.

    ExplanationService.explain_decision() re-executes the concept and condition
    evaluation paths deterministically and ranks driver contributions.
    No LLM involvement.

    Wires the real connector registry and primitive_sources from app.state so
    that primitive fetches use production data connectors (Postgres, REST) when
    available, falling back to MockConnector for primitives with no configured
    source.
    """
    definition_store = DefinitionStore(pool)
    definition_registry = DefinitionRegistry(store=definition_store)
    result_cache = ResultCache()
    graph_store = GraphStore(pool)
    executor = ConceptExecutor(result_cache=result_cache, graph_store=graph_store)
    evaluator = ConditionEvaluator(executor=executor, result_cache=result_cache)

    connector_registry = getattr(request.app.state, "connector_registry", None)
    config = getattr(request.app.state, "config", None)
    primitive_sources = {}
    if config is not None and config.primitive_sources:
        primitive_sources = dict(config.primitive_sources)
    dynamic = getattr(request.app.state, "dynamic_primitive_sources", None)
    if dynamic:
        primitive_sources.update(dynamic)
    async_registry = {}
    if connector_registry is not None:
        async_registry = connector_registry._registry

    data_resolver = DataResolver(
        connector=MockConnector(data={}),
        backoff_base=0.0,
        primitive_sources=primitive_sources,
        async_connector_registry=async_registry,
    )
    decision_store = DecisionStore(pool)
    return ExplanationService(
        definition_registry=definition_registry,
        concept_executor=executor,
        condition_evaluator=evaluator,
        data_resolver=data_resolver,
        decision_store=decision_store,
    )


# ── POST /decisions/explain ────────────────────────────────────────────────────

@router.post(
    "/explain",
    summary="Explain a specific decision result for an entity",
    response_model=DecisionExplanation,
    status_code=200,
)
async def explain_decision(
    req: ExplainDecisionRequest,
    service: ExplanationService = Depends(get_explanation_service),
    _: None = Depends(require_api_key),
) -> DecisionExplanation:
    """
    Return a full explanation of a decision result.

    Explains why the condition fired or did not fire for the given entity at
    the given timestamp — the concept value evaluated, the threshold or label
    applied, and the ranked contribution of each input signal.

    Supports both boolean and categorical decision outputs:
      Boolean strategies  — threshold_applied populated; label_matched is None.
      Equals strategy     — label_matched populated; threshold_applied is None.
      Composite strategy  — both threshold_applied and label_matched are None.

    Invariants:
      drivers[].contribution values sum to 1.0 across all drivers.
      threshold_applied is None for equals and composite strategies.
      label_matched is None for all non-equals strategies.

    HTTP 404 — no decision record found for (condition_id, condition_version,
               entity, timestamp).
    """
    log.info(
        "explain_decision_request",
        extra={
            "condition_id": req.condition_id,
            "condition_version": req.condition_version,
            "entity": req.entity,
        },
    )
    return await service.explain_decision(
        condition_id=req.condition_id,
        condition_version=req.condition_version,
        entity=req.entity,
        timestamp=req.timestamp,
    )


# ── GET /decisions/{decision_id}/verify ────────────────────────────────────────

class VerifyDecisionResponse(BaseModel):
    """
    Tamper-evidence response for a stored decision record.

    verified=True  — the ir_hash stored in the decision matches the hash
                     computed from the current concept definition. The
                     execution graph has not been altered since the decision
                     was recorded.

    verified=False — hash mismatch: the concept definition was modified after
                     the decision was recorded, or the decision record was
                     tampered with directly.

    stored_hash    — the ir_hash value written to the decisions table at
                     evaluation time. None if no ir_hash was stored.

    computed_hash  — the ir_hash freshly computed from the concept definition
                     referenced by the decision record.
    """
    decision_id: str
    verified: bool
    stored_hash: str | None
    computed_hash: str


async def _get_verify_deps(
    pool: asyncpg.Pool = Depends(get_db),
) -> tuple[DecisionStore, DefinitionStore]:
    return DecisionStore(pool), DefinitionStore(pool)


@router.get(
    "/{decision_id}/verify",
    summary="Verify tamper evidence for a stored decision",
    response_model=VerifyDecisionResponse,
    status_code=200,
)
async def verify_decision(
    decision_id: str,
    deps: tuple[DecisionStore, DefinitionStore] = Depends(_get_verify_deps),
    _: None = Depends(require_api_key),
) -> VerifyDecisionResponse:
    """
    Verify that the ir_hash stored in a decision record matches the hash
    freshly computed from the concept definition it references.

    verified=True  — the concept graph is unchanged since this decision.
    verified=False — the concept definition was modified after this decision
                     was recorded (or the record was tampered with).

    HTTP 404 — decision not found.
    HTTP 422 — concept definition referenced by the decision not found.
    """
    decision_store, definition_store = deps

    record = await decision_store.get(decision_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Decision '{decision_id}' not found.")

    concept_body = await definition_store.get(record.concept_id, record.concept_version)
    if concept_body is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Concept '{record.concept_id}' v'{record.concept_version}' "
                f"referenced by decision '{decision_id}' not found in registry."
            ),
        )

    concept = ConceptDefinition.model_validate(concept_body)
    graph = DAGBuilder().build_dag(concept)
    computed_hash = IRGenerator().hash_graph(graph)

    verified = record.ir_hash == computed_hash

    log.info(
        "verify_decision",
        extra={
            "decision_id": decision_id,
            "verified": verified,
            "stored_hash": record.ir_hash,
            "computed_hash": computed_hash,
        },
    )

    return VerifyDecisionResponse(
        decision_id=decision_id,
        verified=verified,
        stored_hash=record.ir_hash,
        computed_hash=computed_hash,
    )
