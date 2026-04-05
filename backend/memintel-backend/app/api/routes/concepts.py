"""
app/api/routes/concepts.py
──────────────────────────────────────────────────────────────────────────────
Concept lifecycle endpoints — V7 two-phase compile + register flow.

Endpoints (paths relative to the /concepts prefix in main.py)
─────────
  POST /concepts/compile    compileConcept   — run 4-step CoR pipeline (V7 M-3)
  POST /concepts/register   registerConcept  — consume token + register (V7 M-4)

POST /concepts/compile
──────────────────────
Runs the 4-step Chain of Reasoning (Intent Parsing, Signal Identification,
DAG Construction, Type Validation) to produce a compile_token and
compiled_concept summary.

signal_names are opaque semantic hints from the caller's domain. Memintel
MUST NOT validate them against any internal registry. An unrecognised
signal_name is never an error.

HTTP 201 — compilation succeeded.
HTTP 422 — compilation_error (CoR step failed) or type_mismatch.
HTTP 500 — internal_error.

Error handling
──────────────
MemintelError subclasses (CompilationError, TypeMismatchError) are caught
globally by the exception handler in main.py — this route does not catch them.
"""
from __future__ import annotations

import structlog
import asyncpg

from fastapi import APIRouter, Depends

from app.api.deps import require_api_key
from app.models.concept_compile import CompileConceptRequest, CompileConceptResponse
from app.persistence.db import get_db
from app.services.concept_compiler import ConceptCompilerService

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/concepts", tags=["Concepts"])


# ── Service dependency ─────────────────────────────────────────────────────────

async def get_concept_compiler_service() -> ConceptCompilerService:
    """
    FastAPI dependency — returns a ConceptCompilerService.

    The service auto-selects the LLM client from USE_LLM_FIXTURES env var.
    The pool is passed directly to compile() rather than stored on the service.
    """
    return ConceptCompilerService()


# ── POST /concepts/compile ────────────────────────────────────────────────────

@router.post(
    "/compile",
    summary="Compile a concept via 4-step CoR pipeline",
    response_model=CompileConceptResponse,
    response_model_exclude_none=True,
    status_code=201,
)
async def compile_concept(
    req: CompileConceptRequest,
    service: ConceptCompilerService = Depends(get_concept_compiler_service),
    pool: asyncpg.Pool = Depends(get_db),
    _: None = Depends(require_api_key),
) -> CompileConceptResponse:
    """
    Run the 4-step Chain of Reasoning to compile a concept description
    into a CompiledConcept and a single-use compile_token.

    Phase 1 of two-phase concept registration:
      1. POST /concepts/compile   → returns compile_token (this endpoint)
      2. POST /concepts/register  → consumes compile_token, returns concept_id (M-4)

    The compile_token is single-use with a TTL (default 30 min, configurable
    via MEMINTEL_COMPILE_TOKEN_TTL_SECONDS). expires_at is included in the
    response so Canvas can display a countdown timer.

    signal_names are opaque semantic hints — never validated against any
    internal registry. Any signal_name is accepted.

    return_reasoning=true — includes reasoning_trace in the response body.
    return_reasoning=false — reasoning_trace is ABSENT from the response.

    HTTP 201 — success.
    HTTP 422 — compilation_error or type_mismatch.
    HTTP 500 — internal_error.
    """
    log.info(
        "compile_concept_request",
        identifier=req.identifier,
        signal_count=len(req.signal_names),
        return_reasoning=req.return_reasoning,
    )
    return await service.compile(req, pool)
