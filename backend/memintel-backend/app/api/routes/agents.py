"""
app/api/routes/agents.py
──────────────────────────────────────────────────────────────────────────────
Agent endpoints — LLM-assisted definition authoring and semantic operations.

Endpoints (paths relative to the /agents prefix added by main.py)
─────────
  POST /agents/query             query           — NL query over registry
  POST /agents/define            define          — LLM-assisted concept definition
  POST /agents/define-condition  defineCondition — LLM-assisted condition definition
  POST /agents/semantic-refine   semanticRefine  — refine an existing definition
  POST /agents/workflows/compile compileWorkflow — compile a multi-step workflow

Ownership rules
───────────────
All endpoints invoke the LLM — they are NOT deterministic. Each endpoint
involves at least one LLM call for intent resolution or generation.

POST /agents/query:
  Natural language query over the definition registry. The LLM interprets
  the query and returns matching definitions with relevance scores.

POST /agents/define:
  Given a natural language description, produces a ConceptDefinition draft.
  The result is a proposal — it must be validated via POST /compile before
  registration. Returns AgentDefineResponse with the draft + validation notes.

POST /agents/define-condition:
  Given a natural language description, proposes a condition definition with
  strategy selection. Returns AgentDefineResponse.
  HTTP 404 — referenced concept (concept_id, concept_version) not found.

POST /agents/semantic-refine:
  Given an existing definition and a refinement instruction, produces an
  updated definition. Returns SemanticRefineResponse with original, proposed,
  and diff. breaking=True signals the change affects the semantic_hash.
  HTTP 404 — definition not found.

POST /agents/workflows/compile:
  Compiles a multi-step workflow description into an ExecutionPlan.

Error handling
──────────────
MemintelError subclasses are caught globally by the exception handler in
main.py — routes do not catch them here.
"""
from __future__ import annotations

import structlog
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import require_elevated_key
from app.models.concept import ExecutionPlan
from app.persistence.db import get_db
from app.services.agents import AgentService

log = structlog.get_logger(__name__)

router = APIRouter(tags=["Agents"])


# ── Inline request/response models ────────────────────────────────────────────
# These models correspond to agent endpoints not yet defined in developer_api.yaml.

class AgentQueryRequest(BaseModel):
    query: str
    definition_type: str | None = None   # optional filter
    limit: int = 10


class AgentQueryResponse(BaseModel):
    results: list[dict[str, Any]]   # [{definition_id, version, score, summary}]
    query: str
    total_count: int


class AgentDefineRequest(BaseModel):
    description: str
    namespace: str = "private"


class AgentDefineConditionRequest(BaseModel):
    description: str
    concept_id: str
    concept_version: str
    namespace: str = "private"


class AgentDefineResponse(BaseModel):
    draft: dict[str, Any]          # proposed definition body
    validation_notes: list[str]    # advisory notes from the LLM
    requires_review: bool          # True when LLM confidence is low


class SemanticRefineRequest(BaseModel):
    definition_id: str
    version: str
    instruction: str    # natural language refinement instruction


class SemanticRefineResponse(BaseModel):
    original: dict[str, Any]
    proposed: dict[str, Any]
    changes: list[dict[str, Any]]   # [{field, from, to, reason}]
    breaking: bool                  # True if the change is semantically breaking


class WorkflowCompileRequest(BaseModel):
    description: str    # natural language workflow description
    namespace: str = "private"


# ── Service dependency ─────────────────────────────────────────────────────────

async def get_agent_service(
    pool: asyncpg.Pool = Depends(get_db),
) -> AgentService:
    """
    FastAPI dependency — returns an AgentService backed by the shared pool.

    AgentService invokes the LLM for all operations in this router.
    """
    return AgentService(pool=pool)


# ── POST /agents/query ────────────────────────────────────────────────────────

@router.post(
    "/query",
    summary="Query the definition registry using natural language",
    response_model=AgentQueryResponse,
    status_code=200,
)
async def query(
    req: AgentQueryRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentQueryResponse:
    """
    Run a natural language query over the definition registry.

    The LLM interprets the query and returns matching definitions ranked by
    relevance. Use to discover existing concepts and conditions before
    authoring new definitions.
    """
    log.info("agent_query_request", query=req.query)
    return await service.query(req)


# ── POST /agents/define ────────────────────────────────────────────────────────

@router.post(
    "/define",
    summary="Define a concept from a natural language description",
    response_model=AgentDefineResponse,
    status_code=200,
)
async def define(
    req: AgentDefineRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentDefineResponse:
    """
    Produce a ConceptDefinition draft from a natural language description.

    The result is a proposal — validate it via POST /compile/semantic before
    registering. requires_review=True indicates the LLM confidence is low
    and human review is recommended.
    """
    log.info("agent_define_request")
    return await service.define(req)


# ── POST /agents/define-condition ─────────────────────────────────────────────

@router.post(
    "/define-condition",
    summary="Define a condition from a natural language description",
    response_model=AgentDefineResponse,
    status_code=200,
)
async def define_condition(
    req: AgentDefineConditionRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentDefineResponse:
    """
    Produce a ConditionDefinition draft from a natural language description.

    Strategy selection (threshold, percentile, z_score, equals, etc.) is
    performed by the LLM based on the description and the concept output type.

    HTTP 404 — referenced concept (concept_id, concept_version) not found.
    """
    log.info(
        "agent_define_condition_request",
        concept_id=req.concept_id,
        concept_version=req.concept_version,
    )
    return await service.define_condition(req)


# ── POST /agents/semantic-refine ──────────────────────────────────────────────

@router.post(
    "/semantic-refine",
    summary="Refine an existing definition using a natural language instruction",
    response_model=SemanticRefineResponse,
    status_code=200,
)
async def semantic_refine(
    req: SemanticRefineRequest,
    service: AgentService = Depends(get_agent_service),
    _: None = Depends(require_elevated_key),
) -> SemanticRefineResponse:
    """
    Apply a natural language refinement instruction to an existing definition.

    Returns the original, proposed, and a list of changes with reasons.
    breaking=True indicates the change affects the semantic_hash — a new
    version should be created rather than updating in place.

    HTTP 404 — definition not found.
    """
    log.info(
        "semantic_refine_request",
        definition_id=req.definition_id,
        version=req.version,
    )
    return await service.semantic_refine(req)


# ── POST /agents/workflows/compile ────────────────────────────────────────────

@router.post(
    "/workflows/compile",
    summary="Compile a multi-step workflow from a natural language description",
    response_model=ExecutionPlan,
    status_code=200,
)
async def compile_workflow(
    req: WorkflowCompileRequest,
    service: AgentService = Depends(get_agent_service),
) -> ExecutionPlan:
    """
    Compile a multi-step workflow description into an ExecutionPlan.

    The workflow describes a sequence of concept evaluations and condition
    checks. The LLM resolves referenced concepts and conditions from the
    registry and produces an ExecutionPlan with execution_order and
    parallelizable_groups.
    """
    log.info("compile_workflow_request")
    return await service.compile_workflow(req)
