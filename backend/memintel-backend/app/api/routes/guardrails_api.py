"""
app/api/routes/guardrails_api.py
────────────────────────────────────────────────────────────────────────────────
Guardrails API endpoints — manage guardrails versions without server restart.

Endpoints
─────────
  POST /guardrails                        createGuardrails    — create a new version
  GET  /guardrails                        getActiveGuardrails — return active version
  GET  /guardrails/versions               listVersions        — list all versions
  GET  /guardrails/versions/{version}     getGuardrailsVersion — fetch a specific version
  GET  /guardrails/impact                 getImpact           — task distribution

Invariants
──────────
  - POST /guardrails requires MEMINTEL_ELEVATED_KEY (X-Elevated-Key header).
  - POST /guardrails reloads in memory synchronously before returning 201.
  - GET /guardrails returns HTTP 404 when no API version has been posted yet.
  - GET /guardrails/versions returns an empty list when no versions exist.
  - GET /guardrails/versions/{version} returns HTTP 404 for unknown versions.
  - Only one guardrails version is active at any time.
  - Guardrails versions are never deleted — only superseded.

Error handling
──────────────
NotFoundError → HTTP 404 (raised by GuardrailsApiService).
MemintelError subclasses are caught globally by the exception handler in main.py.
HTTPException 403 is raised by require_elevated_key when the key is absent/invalid.
"""
from __future__ import annotations

import structlog

import asyncpg
from fastapi import APIRouter, Depends, Request

from app.api.deps import require_elevated_key
from app.models.errors import NotFoundError
from app.models.guardrails_api import (
    CreateGuardrailsRequest,
    GuardrailsImpactResult,
    GuardrailsVersion,
)
from app.persistence.db import get_db
from app.services.guardrails_api import GuardrailsApiService
from app.stores.guardrails import GuardrailsStore as GuardrailsVersionStore

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/guardrails", tags=["Guardrails"])

_NO_API_VERSION_MESSAGE = (
    "No guardrails defined via API. Guardrails loaded from "
    "memintel_guardrails.yaml at startup."
)


# ── Service dependency ──────────────────────────────────────────────────────────

async def get_guardrails_api_service(
    request: Request,
    pool: asyncpg.Pool = Depends(get_db),
) -> GuardrailsApiService:
    """FastAPI dependency — returns a GuardrailsApiService backed by the shared pool."""
    config_store = getattr(request.app.state, "guardrails_store", None)
    store = GuardrailsVersionStore(pool)
    return GuardrailsApiService(store=store, config_store=config_store)


# ── POST /guardrails ────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Create a new guardrails version",
    response_model=GuardrailsVersion,
    status_code=201,
    dependencies=[Depends(require_elevated_key)],
)
async def create_guardrails(
    req: CreateGuardrailsRequest,
    service: GuardrailsApiService = Depends(get_guardrails_api_service),
) -> GuardrailsVersion:
    """
    Create a new guardrails version.

    The new version becomes active immediately — the previous version is
    deactivated atomically. The in-memory GuardrailsStore is reloaded
    synchronously before this response is returned.

    Requires X-Elevated-Key header matching MEMINTEL_ELEVATED_KEY.

    HTTP 400 — validation error (unknown strategy, invalid severity level, etc.).
    HTTP 403 — elevated key missing or incorrect.
    """
    created = await service.create_guardrails(req)
    log.info(
        "guardrails_version_created",
        guardrails_id=created.guardrails_id,
        version=created.version,
    )
    return created


# ── GET /guardrails ─────────────────────────────────────────────────────────────

@router.get(
    "",
    summary="Get the active guardrails version",
    response_model=GuardrailsVersion,
    status_code=200,
)
async def get_active_guardrails(
    service: GuardrailsApiService = Depends(get_guardrails_api_service),
) -> GuardrailsVersion:
    """
    Return the currently active API guardrails version.

    HTTP 404 — no guardrails version has been posted via the API yet.
    The message indicates that file-based guardrails are in use.
    """
    version = await service.get_active_guardrails()
    if version is None:
        raise NotFoundError(_NO_API_VERSION_MESSAGE)
    return version


# ── GET /guardrails/versions ────────────────────────────────────────────────────

@router.get(
    "/versions",
    summary="List all guardrails versions",
    response_model=list[GuardrailsVersion],
    status_code=200,
)
async def list_versions(
    service: GuardrailsApiService = Depends(get_guardrails_api_service),
) -> list[GuardrailsVersion]:
    """Return all guardrails versions ordered by created_at descending (newest first)."""
    return await service.list_versions()


# ── GET /guardrails/versions/{version} ─────────────────────────────────────────

@router.get(
    "/versions/{version}",
    summary="Get a specific guardrails version",
    response_model=GuardrailsVersion,
    status_code=200,
)
async def get_guardrails_version(
    version: str,
    service: GuardrailsApiService = Depends(get_guardrails_api_service),
) -> GuardrailsVersion:
    """
    Return the guardrails version for the given version string (e.g. "v1", "v2").

    HTTP 404 — version not found.
    """
    return await service.get_guardrails_version(version)


# ── GET /guardrails/impact ──────────────────────────────────────────────────────

@router.get(
    "/impact",
    summary="Get guardrails impact on tasks",
    response_model=GuardrailsImpactResult,
    status_code=200,
)
async def get_impact(
    service: GuardrailsApiService = Depends(get_guardrails_api_service),
) -> GuardrailsImpactResult:
    """
    Return task distribution across guardrails versions.

    older_version_task_ids lists tasks compiled under a previous guardrails
    version (or no guardrails version at all). These tasks may benefit from
    re-creation under the current guardrails configuration.

    Never raises — returns all-zero counts when no guardrails version exists.
    """
    return await service.get_impact()
