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
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.models.condition import DecisionExplanation
from app.persistence.db import get_db
from app.registry.definitions import DefinitionRegistry
from app.runtime.cache import ResultCache
from app.runtime.condition_evaluator import ConditionEvaluator
from app.runtime.data_resolver import DataResolver, MockConnector
from app.runtime.executor import ConceptExecutor
from app.services.explanation import ExplanationService
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
        primitive_sources = config.primitive_sources
    async_registry = {}
    if connector_registry is not None:
        async_registry = connector_registry._registry

    data_resolver = DataResolver(
        connector=MockConnector(data={}),
        backoff_base=0.0,
        primitive_sources=primitive_sources,
        async_connector_registry=async_registry,
    )
    return ExplanationService(
        definition_registry=definition_registry,
        concept_executor=executor,
        condition_evaluator=evaluator,
        data_resolver=data_resolver,
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
