"""
app/api/routes/compile.py
──────────────────────────────────────────────────────────────────────────────
Compiler endpoints — concept compilation to execution graphs.

Endpoints (paths relative to the /compile prefix added by main.py)
─────────
  POST /compile              compile           — compile concept to ExecutionGraph
  POST /compile/semantic     compileSemantic   — produce SemanticGraph
  POST /compile/explain-plan compileExplainPlan — explain execution plan (no run)
  GET  /compile/graphs/{id}  getGraph          — retrieve a stored graph by id

Note on GET /graphs/{id}
────────────────────────
Per core-spec.md, the graph retrieval endpoint is GET /graphs/{graphId} (root
level). Because main.py registers this router with prefix="/compile", the
graph endpoint is reachable at GET /compile/graphs/{id}. Correcting the path
requires a dedicated graphs.router registered with prefix="/graphs" in main.py.

Ownership rules
───────────────
All endpoints are deterministic — no LLM involvement.

POST /compile:
  Validates, compiles, and stores a concept definition as an ExecutionGraph.
  Returns the stored graph. The ir_hash is a SHA-256 of the canonical graph JSON.

  Graph replacement invariant:
    Recompiling an unchanged definition MUST produce the same ir_hash.
    If the existing graph has a different ir_hash → HTTP 409 (invariant violation).

  HTTP 400 — validation errors in the concept definition.
  HTTP 404 — referenced primitives or features not found.
  HTTP 409 — ir_hash mismatch on recompile (CompilerInvariantError).
  HTTP 422 — type errors detected by the type checker.

POST /compile/semantic:
  Produces a SemanticGraph — the canonical semantic view of the concept.
  Returns a stable semantic_hash for equivalence detection and deduplication.
  Does NOT compile to an ExecutionGraph and does NOT persist anything.

POST /compile/explain-plan:
  Returns an ExecutionPlan describing execution order and parallelizable groups.
  The SQL EXPLAIN equivalent for a concept. Does NOT compile or execute.

GET /compile/graphs/{graph_id}:
  Retrieves a stored ExecutionGraph by graph_id.
  Use to inspect the compiled IR or pass graph_id to POST /execute/graph.

Route registration order
────────────────────────
Literal-path routes (/semantic, /explain-plan, /graphs/{id}) are registered
BEFORE the root POST "" to prevent FastAPI routing ambiguity.

Error handling
──────────────
MemintelError subclasses are caught globally by the exception handler in
main.py — routes do not catch them here.
"""
from __future__ import annotations

import structlog

import asyncpg
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import require_elevated_key
from app.models.concept import (
    ConceptDefinition,
    ExecutionGraph,
    ExecutionPlan,
    SemanticGraph,
)
from app.models.errors import NotFoundError
from app.persistence.db import get_db
from app.persistence.stores import get_graph_store
from app.services.compile import CompileService
from app.stores import GraphStore

log = structlog.get_logger(__name__)

router = APIRouter(tags=["Compiler"])


# ── Request bodies ─────────────────────────────────────────────────────────────

class CompileRequest(BaseModel):
    concept: ConceptDefinition    # full concept definition body to compile


class CompileSemanticRequest(BaseModel):
    concept: ConceptDefinition


class ExplainPlanRequest(BaseModel):
    concept: ConceptDefinition


# ── Service dependency ─────────────────────────────────────────────────────────

async def get_compile_service(
    pool: asyncpg.Pool = Depends(get_db),
) -> CompileService:
    """
    FastAPI dependency — returns a CompileService backed by the shared pool.

    CompileService validates concept definitions against the type system,
    produces execution graphs and semantic views. Fully deterministic — no LLM.
    """
    return CompileService(pool=pool)


# ── POST /compile/semantic ─────────────────────────────────────────────────────
# Registered before the root POST "" to avoid path-matching ambiguity.

@router.post(
    "/semantic",
    summary="Produce the semantic view of a concept",
    response_model=SemanticGraph,
    status_code=200,
)
async def compile_semantic(
    req: CompileSemanticRequest,
    service: CompileService = Depends(get_compile_service),
    _: None = Depends(require_elevated_key),
) -> SemanticGraph:
    """
    Produce a SemanticGraph from a concept definition without compiling.

    Returns a stable semantic_hash suitable for deduplication and equivalence
    detection. Two concepts with the same semantic_hash are semantically
    equivalent — they compute the same function.

    Does NOT persist anything.

    HTTP 400 — validation errors.
    HTTP 422 — type errors.
    """
    log.info(
        "compile_semantic_request",
        concept_id=req.concept.concept_id,
        version=req.concept.version,
    )
    return await service.compile_semantic(req.concept)


# ── POST /compile/explain-plan ─────────────────────────────────────────────────

@router.post(
    "/explain-plan",
    summary="Explain the execution plan for a concept",
    response_model=ExecutionPlan,
    status_code=200,
)
async def compile_explain_plan(
    req: ExplainPlanRequest,
    service: CompileService = Depends(get_compile_service),
) -> ExecutionPlan:
    """
    Return the execution plan for a concept without compiling or executing.

    Equivalent to SQL EXPLAIN — returns execution_order, parallelizable_groups,
    primitive_fetches, and critical_path_length. Does NOT persist anything.

    Use to understand execution structure before committing a compile.

    HTTP 400 — validation errors.
    HTTP 422 — type errors.
    """
    log.info(
        "compile_explain_plan_request",
        concept_id=req.concept.concept_id,
        version=req.concept.version,
    )
    return await service.explain_plan(req.concept)


# ── GET /compile/graphs/{graph_id} ────────────────────────────────────────────

@router.get(
    "/graphs/{graph_id}",
    summary="Get a stored execution graph",
    response_model=ExecutionGraph,
    status_code=200,
)
async def get_graph(
    graph_id: str,
    store: GraphStore = Depends(get_graph_store),
) -> ExecutionGraph:
    """
    Retrieve a stored execution graph by graph_id.

    Use to inspect the compiled IR or to obtain the ir_hash for audit
    verification. Pass graph_id to POST /execute/graph for execution.

    HTTP 404 — graph not found.
    """
    log.info("get_graph_request", graph_id=graph_id)
    graph = await store.get(graph_id)
    if graph is None:
        raise NotFoundError(f"Graph '{graph_id}' not found.", location="graph_id")
    return graph


# ── POST /compile ──────────────────────────────────────────────────────────────
# Registered last — after all sub-path routes.

@router.post(
    "",
    summary="Compile a concept to an execution graph",
    response_model=ExecutionGraph,
    status_code=200,
)
async def compile_concept(
    req: CompileRequest,
    service: CompileService = Depends(get_compile_service),
    _: None = Depends(require_elevated_key),
) -> ExecutionGraph:
    """
    Validate, compile, and store a concept as an ExecutionGraph.

    Returns the stored ExecutionGraph with a stable graph_id and ir_hash.
    The graph is stored in the execution_graphs table and can be retrieved
    by graph_id via GET /compile/graphs/{graph_id} or executed via
    POST /execute/graph.

    Graph replacement invariant:
      Recompiling an unchanged definition produces the same ir_hash.
      If the existing graph has a different ir_hash → HTTP 409.

    HTTP 400 — validation errors in concept definition.
    HTTP 404 — referenced primitive or feature not found.
    HTTP 409 — ir_hash mismatch on recompile (CompilerInvariantError).
    HTTP 422 — type errors detected by type checker.
    """
    log.info(
        "compile_request",
        concept_id=req.concept.concept_id,
        version=req.concept.version,
    )
    return await service.compile(req.concept)
