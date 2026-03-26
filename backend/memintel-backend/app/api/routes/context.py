"""
app/api/routes/context.py
────────────────────────────────────────────────────────────────────────────────
Application Context endpoints.

Endpoints
─────────
  POST /context                       createContext   — create a new context version
  GET  /context                       getActiveContext — return the active context
  GET  /context/versions              listVersions    — list all context versions
  GET  /context/versions/{version}    getContextVersion — fetch a specific version
  GET  /context/impact                getImpact       — task distribution across versions

Invariants
──────────
  - POST /context always supersedes the previous active context atomically.
  - GET /context returns HTTP 404 when no context has been created yet.
  - GET /context/versions/{version} returns HTTP 404 for unknown versions.
  - GET /context/impact never raises — returns zeros when no context exists.

Error handling
──────────────
NotFoundError → HTTP 404 (raised by ContextService).
MemintelError subclasses are caught globally by the exception handler in main.py.
"""
from __future__ import annotations

import structlog

import asyncpg
from fastapi import APIRouter, Depends

from app.models.context import (
    ApplicationContext,
    ContextImpactResult,
    CreateContextRequest,
)
from app.persistence.db import get_db
from app.services.context import ContextService
from app.stores.context import ContextStore

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/context", tags=["Context"])


# ── Service dependency ──────────────────────────────────────────────────────────

async def get_context_service(
    pool: asyncpg.Pool = Depends(get_db),
) -> ContextService:
    """FastAPI dependency — returns a ContextService backed by the shared pool."""
    return ContextService(store=ContextStore(pool))


# ── POST /context ───────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Create a new application context",
    response_model=ApplicationContext,
    status_code=201,
)
async def create_context(
    req: CreateContextRequest,
    service: ContextService = Depends(get_context_service),
) -> ApplicationContext:
    """
    Create a new application context version.

    The new context becomes the active context immediately. The previous active
    context is deactivated atomically. Context versions are never deleted.

    bias_direction inside calibration_bias (if supplied) is always auto-derived
    from false_negative_cost vs false_positive_cost — any caller-supplied value
    is silently overwritten.
    """
    ctx = await service.create_context(req)
    log.info("context_created", context_id=ctx.context_id, version=ctx.version)
    return ctx


# ── GET /context ────────────────────────────────────────────────────────────────

@router.get(
    "",
    summary="Get the active application context",
    response_model=ApplicationContext,
    status_code=200,
)
async def get_active_context(
    service: ContextService = Depends(get_context_service),
) -> ApplicationContext:
    """
    Return the currently active application context.

    HTTP 404 — no context has been created yet.
    """
    return await service.get_active_context()


# ── GET /context/versions ───────────────────────────────────────────────────────

@router.get(
    "/versions",
    summary="List all context versions",
    response_model=list[ApplicationContext],
    status_code=200,
)
async def list_versions(
    service: ContextService = Depends(get_context_service),
) -> list[ApplicationContext]:
    """Return all context versions ordered by created_at descending (newest first)."""
    return await service.list_versions()


# ── GET /context/versions/{version} ────────────────────────────────────────────

@router.get(
    "/versions/{version}",
    summary="Get a specific context version",
    response_model=ApplicationContext,
    status_code=200,
)
async def get_context_version(
    version: str,
    service: ContextService = Depends(get_context_service),
) -> ApplicationContext:
    """
    Return the context for the given version string (e.g. "v1", "v2").

    HTTP 404 — version not found.
    """
    return await service.get_context_version(version)


# ── GET /context/impact ─────────────────────────────────────────────────────────

@router.get(
    "/impact",
    summary="Get context impact on tasks",
    response_model=ContextImpactResult,
    status_code=200,
)
async def get_impact(
    service: ContextService = Depends(get_context_service),
) -> ContextImpactResult:
    """
    Return task distribution across context versions.

    older_version_task_ids lists tasks compiled under a previous context version
    (or no context at all). These tasks may benefit from re-creation under the
    current context to pick up updated domain knowledge.

    Never raises — returns all-zero counts when no context exists.
    """
    return await service.get_impact()
