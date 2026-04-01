"""
app/api/routes/execute.py
──────────────────────────────────────────────────────────────────────────────
Execution pipeline endpoints — concept evaluation, batch, range, async, and
full ψ → φ → α pipeline.

Two routers are exported:
  evaluate_router — registered with prefix="/evaluate" in main.py
    POST /evaluate/full              evaluateFull           — full ψ → φ → α pipeline
    POST /evaluate/condition         evaluateCondition      — φ layer for one entity
    POST /evaluate/condition/batch   evaluateConditionBatch — φ for N entities

  router — registered with prefix="/execute" in main.py
    POST /execute                    execute       — ψ layer (concept only)
    POST /execute/batch              executeBatch  — ψ for N entities
    POST /execute/range              executeRange  — ψ over a time range
    POST /execute/async              executeAsync  — enqueue async ψ job (→ 202)
    POST /execute/graph              executeGraph  — execute a pre-compiled graph (elevated)

Ownership rules
───────────────
All endpoints are deterministic when timestamp is provided. No LLM involvement
at runtime.

POST /execute/evaluate/full:
  Runs the complete ψ → φ → α pipeline. Fetches primitives, evaluates the
  concept graph, evaluates the condition, and dispatches bound actions.

  timestamp — optional; when present the result is deterministic and cached.
  explain=True — populates result.explanation with per-node trace.
  dry_run=True — simulates pipeline without firing actions.

  HTTP 404 — concept, condition, or entity not found.
  HTTP 408 — execution timeout.
  HTTP 422 — execution error (missing data, null propagation failure).

POST /execute:
  ψ layer only — concept execution without condition or action evaluation.
  Returns ConceptResult. timestamp controls deterministic vs snapshot mode.

POST /execute/batch:
  Runs ψ for N entities against the same (concept_id, version).
  Returns HTTP 200 always; per-entity errors in BatchExecuteItem.error.

POST /execute/range:
  Runs ψ for one entity over a closed time range [from, to] at an interval.
  Returns list[ConceptResult] in chronological order.

POST /execute/async:
  Enqueues a ψ execution job and returns a Job immediately.
  Poll GET /jobs/{job_id} for status and result.

POST /execute/graph:
  Executes a pre-compiled graph by graph_id, bypassing compilation.
  ir_hash, if provided, is verified before execution — mismatch → HTTP 409.

Route registration order
────────────────────────
Literal-path sub-routes (/evaluate/full, /batch, /range, /async, /graph) are
registered BEFORE the root POST "" to prevent FastAPI routing ambiguity.

Error handling
──────────────
MemintelError subclasses are caught globally by the exception handler in
main.py — routes do not catch them here.
"""
from __future__ import annotations

import json
import structlog

import asyncpg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from typing import Any

from app.api.deps import require_elevated_key
from app.compiler.dag_builder import DAGBuilder
from app.models.concept import ConceptDefinition
from app.models.condition import ConditionDefinition, DecisionValue
from app.models.errors import ErrorType, MemintelError
from app.models.result import (
    BatchExecuteResult,
    ConceptResult,
    DecisionResult,
    ExecuteRequest,
    ExecuteGraphRequest,
    FullPipelineResult,
    Job,
)
from app.persistence.db import get_db
from app.runtime.cache import ResultCache
from app.runtime.data_resolver import DataResolver, StaticDataConnector
from app.runtime.executor import ConceptExecutor
from app.services.execute import ExecuteService
from app.strategies.change import ChangeStrategy as ChangeStrategyImpl
from app.strategies.composite import CompositeStrategy as CompositeStrategyImpl
from app.strategies.equals import EqualsStrategy as EqualsStrategyImpl
from app.strategies.percentile import PercentileStrategy as PercentileStrategyImpl
from app.strategies.threshold import ThresholdStrategy as ThresholdStrategyImpl
from app.strategies.z_score import ZScoreStrategy as ZScoreStrategyImpl

_STRATEGY_IMPLS = {
    "threshold":  ThresholdStrategyImpl,
    "percentile": PercentileStrategyImpl,
    "z_score":    ZScoreStrategyImpl,
    "change":     ChangeStrategyImpl,
    "equals":     EqualsStrategyImpl,
    "composite":  CompositeStrategyImpl,
}

log = structlog.get_logger(__name__)

# ── Two routers — separate URL prefixes ───────────────────────────────────────
evaluate_router = APIRouter(tags=["Execution"])   # registered at /evaluate
router = APIRouter(tags=["Execution"])             # registered at /execute


# ── Request body models ───────────────────────────────────────────────────────

class EvaluateFullRequest(BaseModel):
    concept_id: str
    concept_version: str
    condition_id: str
    condition_version: str
    entity: str
    timestamp: str | None = None    # ISO 8601 UTC — when present, deterministic
    explain: bool = False
    dry_run: bool = False


class EvaluateConditionRequest(BaseModel):
    condition_id: str
    condition_version: str
    entity: str
    timestamp: str | None = None    # ISO 8601 UTC — when present, deterministic
    explain: bool = False


class EvaluateConditionBatchRequest(BaseModel):
    condition_id: str
    condition_version: str
    entities: list[str]
    timestamp: str | None = None    # applied to all entities in the batch


class ExecuteBatchRequest(BaseModel):
    id: str
    version: str
    entities: list[str]
    timestamp: str | None = None    # ISO 8601 UTC — when present, deterministic
    explain: bool = False
    dry_run: bool = False           # simulate without firing actions


class ExecuteRangeRequest(BaseModel):
    id: str
    version: str
    entity: str
    from_timestamp: str    # ISO 8601 UTC — range start (inclusive)
    to_timestamp: str      # ISO 8601 UTC — range end (inclusive)
    interval: str          # ISO 8601 duration, e.g. 'PT1H', 'P1D'
    explain: bool = False


# ── Service dependency ─────────────────────────────────────────────────────────

async def get_execute_service(
    request: Request,
    pool: asyncpg.Pool = Depends(get_db),
) -> ExecuteService:
    """
    FastAPI dependency — returns an ExecuteService backed by the shared pool.

    ExecuteService drives the ψ → φ → α pipeline, batch execution, range
    execution, and async job enqueuing. No LLM involvement.

    Injects connector_registry and primitive_sources from app.state when
    available so real data connectors (Postgres, REST) are used on the
    execution path.
    """
    connector_registry = getattr(request.app.state, "connector_registry", None)
    config = getattr(request.app.state, "config", None)
    primitive_sources = {}
    if config is not None and config.primitive_sources:
        primitive_sources = config.primitive_sources
    return ExecuteService(
        pool=pool,
        connector_registry=connector_registry,
        primitive_sources=primitive_sources,
    )


# ── evaluate_router: POST /evaluate/full ──────────────────────────────────────

@evaluate_router.post(
    "/full",
    summary="Execute concept + condition + action pipeline",
    response_model=FullPipelineResult,
    status_code=200,
)
async def evaluate_full(
    req: EvaluateFullRequest,
    service: ExecuteService = Depends(get_execute_service),
) -> FullPipelineResult:
    """
    Run the full ψ → φ → α pipeline for a given entity.

    Executes the concept (ψ), evaluates the condition (φ), and dispatches
    bound actions (α). Fully deterministic when timestamp is provided.

    actions_triggered[] is nested inside decision (DecisionResult), not at
    the top level of FullPipelineResult — this is a hard API contract.

    dry_run=True simulates the pipeline without firing actions.
    explain=True populates result.explanation with per-node trace.

    HTTP 404 — concept, condition, or entity not found.
    HTTP 408 — execution timeout.
    HTTP 422 — execution error (missing data, null propagation).
    """
    log.info(
        "evaluate_full_request",
        concept_id=req.concept_id,
        concept_version=req.concept_version,
        condition_id=req.condition_id,
        entity=req.entity,
        dry_run=req.dry_run,
    )
    return await service.evaluate_full(req)


# ── evaluate_router: POST /evaluate/condition ─────────────────────────────────

@evaluate_router.post(
    "/condition",
    summary="Evaluate a condition for one entity",
    response_model=DecisionResult,
    status_code=200,
)
async def evaluate_condition(
    req: EvaluateConditionRequest,
    service: ExecuteService = Depends(get_execute_service),
) -> DecisionResult:
    """
    Run the φ layer (condition evaluation) for a single entity.

    Returns a DecisionResult with decision_value, strategy details, and any
    actions_triggered. The concept must be cached or the service will fetch
    primitives and execute the ψ layer first.

    HTTP 404 — condition or entity not found.
    HTTP 408 — execution timeout.
    """
    log.info(
        "evaluate_condition_request",
        condition_id=req.condition_id,
        condition_version=req.condition_version,
        entity=req.entity,
    )
    return await service.evaluate_condition(req)


# ── evaluate_router: POST /evaluate/condition/batch ───────────────────────────

@evaluate_router.post(
    "/condition/batch",
    summary="Evaluate a condition for multiple entities",
    response_model=list[DecisionResult],
    status_code=200,
)
async def evaluate_condition_batch(
    req: EvaluateConditionBatchRequest,
    service: ExecuteService = Depends(get_execute_service),
) -> list[DecisionResult]:
    """
    Run condition evaluation (φ layer) for N entities in a single call.

    Always returns HTTP 200. Per-entity failures are captured inside each
    DecisionResult rather than failing the entire batch.

    Returns results in the same order as the input entities list.
    """
    log.info(
        "evaluate_condition_batch_request",
        condition_id=req.condition_id,
        condition_version=req.condition_version,
        entity_count=len(req.entities),
    )
    return await service.evaluate_condition_batch(req)


# ── POST /execute/batch ────────────────────────────────────────────────────────

@router.post(
    "/batch",
    summary="Execute concept for multiple entities",
    response_model=BatchExecuteResult,
    status_code=200,
)
async def execute_batch(
    req: ExecuteBatchRequest,
    service: ExecuteService = Depends(get_execute_service),
) -> BatchExecuteResult:
    """
    Run concept execution for N entities against the same (id, version).

    Always returns HTTP 200. Per-entity failures are reported in
    BatchExecuteItem.error — they do not fail the whole batch.

    Returns results in input order. total and failed counters allow callers
    to detect partial failure without inspecting each item.
    """
    log.info(
        "execute_batch_request",
        id=req.id,
        version=req.version,
        entity_count=len(req.entities),
    )
    return await service.execute_batch(req)


# ── POST /execute/range ────────────────────────────────────────────────────────

@router.post(
    "/range",
    summary="Execute concept over a time range",
    response_model=list[ConceptResult],
    status_code=200,
)
async def execute_range(
    req: ExecuteRangeRequest,
    service: ExecuteService = Depends(get_execute_service),
) -> list[ConceptResult]:
    """
    Run concept execution for one entity across a closed time range.

    Returns one ConceptResult per interval step, in chronological order.
    Each result has deterministic=True because all timestamps are explicit.

    Useful for backtesting, historical analysis, and chart data generation.
    """
    log.info(
        "execute_range_request",
        id=req.id,
        version=req.version,
        entity=req.entity,
    )
    return await service.execute_range(req)


# ── POST /execute/async ────────────────────────────────────────────────────────

@router.post(
    "/async",
    summary="Enqueue an async concept execution job",
    response_model=Job,
    status_code=202,
)
async def execute_async(
    req: ExecuteRequest,
    service: ExecuteService = Depends(get_execute_service),
) -> Job:
    """
    Enqueue a concept execution job and return immediately.

    The response contains a job_id — poll GET /jobs/{job_id} at the
    returned poll_interval_seconds to retrieve the result once the job
    reaches status='completed'.

    Use for long-running executions or when synchronous latency is
    unacceptable. The job executes the same path as POST /execute.

    Returns HTTP 202 Accepted — the job is queued, not yet complete.
    """
    log.info(
        "execute_async_request",
        id=req.id,
        version=req.version,
        entity=req.entity,
    )
    return await service.execute_async(req)


# ── POST /execute/graph ────────────────────────────────────────────────────────

@router.post(
    "/graph",
    summary="Execute a pre-compiled graph",
    response_model=ConceptResult,
    status_code=200,
)
async def execute_graph(
    req: ExecuteGraphRequest,
    service: ExecuteService = Depends(get_execute_service),
    _: None = Depends(require_elevated_key),
) -> ConceptResult:
    """
    Execute a pre-compiled execution graph by graph_id.

    Bypasses the compilation step — use on the hot path: compile once at
    startup, cache graph_id, then call this endpoint on every evaluation.

    ir_hash — if provided, verified against the stored graph before execution.
    Mismatch → HTTP 409 (stale graph_id in caller; recompile and update cache).

    Requires elevated key (X-Elevated-Key header) → HTTP 403 if absent.

    HTTP 403 — elevated key missing or invalid.
    HTTP 404 — graph_id not found.
    HTTP 409 — ir_hash mismatch.
    """
    log.info(
        "execute_graph_request",
        graph_id=req.graph_id,
        entity=req.entity,
    )
    return await service.execute_graph(req)


# ── POST /execute/static ───────────────────────────────────────────────────────

class StaticExecuteRequest(BaseModel):
    condition_id: str
    condition_version: str
    entity: str
    data: dict[str, Any]      # {primitive_name: value}


@router.post(
    "/static",
    summary="Evaluate a condition with inline data (test-only)",
    response_model=DecisionValue,
    status_code=200,
)
async def execute_static(
    req: StaticExecuteRequest,
    pool: asyncpg.Pool = Depends(get_db),
) -> dict:
    """
    Evaluate a registered condition using caller-supplied primitive values.

    No real data connectors or memintel_config.yaml entries are needed.
    Compiles the linked concept on the fly, evaluates it with the provided
    data dict, then runs the condition strategy.

    This endpoint is for local testing only — it does not require auth and
    bypasses the production data pipeline entirely.

    Returns the DecisionValue as a JSON object.
    """
    # 1. Fetch condition definition body from DB.
    row = await pool.fetchrow(
        """
        SELECT body FROM definitions
        WHERE definition_id = $1 AND version = $2
          AND definition_type = 'condition'
        LIMIT 1
        """,
        req.condition_id,
        req.condition_version,
    )
    if row is None:
        raise MemintelError(
            ErrorType.NOT_FOUND,
            f"Condition '{req.condition_id}' version '{req.condition_version}' not found.",
            location=f"{req.condition_id}:{req.condition_version}",
        )
    raw_condition = row["body"]
    condition_dict = json.loads(raw_condition) if isinstance(raw_condition, str) else (raw_condition or {})
    condition = ConditionDefinition(**condition_dict)

    # 2. Fetch concept definition body from DB.
    concept_row = await pool.fetchrow(
        """
        SELECT body FROM definitions
        WHERE definition_id = $1 AND version = $2
          AND definition_type = 'concept'
        LIMIT 1
        """,
        condition.concept_id,
        condition.concept_version,
    )
    if concept_row is None:
        raise MemintelError(
            ErrorType.NOT_FOUND,
            f"Concept '{condition.concept_id}' version '{condition.concept_version}' not found.",
            location=f"{condition.concept_id}:{condition.concept_version}",
        )
    raw_concept = concept_row["body"]
    concept_dict = json.loads(raw_concept) if isinstance(raw_concept, str) else (raw_concept or {})
    concept = ConceptDefinition(**concept_dict)

    # 3. Compile concept → ExecutionGraph.
    graph = DAGBuilder().build_dag(concept)

    # 4. Wire inline data through StaticDataConnector.
    connector = StaticDataConnector(req.data)
    resolver = DataResolver(connector, backoff_base=0.0)

    # 5. Execute concept to get ConceptResult.
    executor = ConceptExecutor(ResultCache())
    concept_result = executor.execute_graph(graph, req.entity, resolver)

    # 6. Evaluate condition strategy → DecisionValue.
    strategy_cls = _STRATEGY_IMPLS.get(condition.strategy.type.value)
    if strategy_cls is None:
        raise MemintelError(
            ErrorType.EXECUTION_ERROR,
            f"Unknown strategy type '{condition.strategy.type.value}'.",
        )
    strategy_instance = strategy_cls()
    decision = strategy_instance.evaluate(
        concept_result,
        [],
        condition.strategy.params.model_dump(),
        condition_id=req.condition_id,
        condition_version=req.condition_version,
    )

    log.info(
        "execute_static",
        condition_id=req.condition_id,
        condition_version=req.condition_version,
        entity=req.entity,
        decision=decision.value,
    )
    return decision.model_dump()


# ── POST /execute ──────────────────────────────────────────────────────────────
# Registered last — after all sub-path routes.

@router.post(
    "",
    summary="Execute a concept for an entity",
    response_model=ConceptResult,
    status_code=200,
)
async def execute(
    req: ExecuteRequest,
    service: ExecuteService = Depends(get_execute_service),
) -> ConceptResult:
    """
    Run ψ layer execution for a single entity.

    Returns a ConceptResult. timestamp controls execution mode:
      present → deterministic; result.deterministic=True; cached.
      absent  → snapshot mode; result.deterministic=False; not cached.

    explain=True populates result.explanation (per-node trace).

    HTTP 404 — concept or entity not found.
    HTTP 408 — execution timeout.
    HTTP 422 — execution error (missing data, null propagation).
    """
    log.info(
        "execute_request",
        id=req.id,
        version=req.version,
        entity=req.entity,
    )
    return await service.execute(req)
