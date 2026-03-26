"""
app/services/context.py
────────────────────────────────────────────────────────────────────────────────
ContextService — application context lifecycle management.

Responsibilities
────────────────
  - Translate CreateContextRequest into ApplicationContext and delegate to
    ContextStore.create(), which handles version assignment and atomic
    deactivation of the previous context.
  - Proxy read operations (get_active, get_version, list_versions, get_impact)
    to ContextStore, raising NotFoundError where appropriate.

Error mapping
─────────────
  get_active_context()   → raises NotFoundError if no context exists.
  get_context_version()  → raises NotFoundError if version not found.
"""
from __future__ import annotations

import structlog

from app.models.context import (
    ApplicationContext,
    ContextImpactResult,
    CreateContextRequest,
)
from app.models.errors import NotFoundError
from app.stores.context import ContextStore

log = structlog.get_logger(__name__)


class ContextService:
    """
    Business logic layer for ApplicationContext.

    Public API
    ----------
    create_context(req)         → ApplicationContext
    get_active_context()        → ApplicationContext   (raises NotFoundError)
    get_context_version(ver)    → ApplicationContext   (raises NotFoundError)
    list_versions()             → list[ApplicationContext]
    get_impact()                → ContextImpactResult
    """

    def __init__(self, store: ContextStore) -> None:
        self._store = store

    async def create_context(self, req: CreateContextRequest) -> ApplicationContext:
        """
        Build an ApplicationContext from the request and persist it.

        version is assigned by ContextStore.create().
        """
        context = ApplicationContext(
            domain=req.domain,
            behavioural=req.behavioural,
            semantic_hints=req.semantic_hints,
            calibration_bias=req.calibration_bias,
        )
        created = await self._store.create(context)
        log.info("context_created", context_id=created.context_id, version=created.version)
        return created

    async def get_active_context(self) -> ApplicationContext:
        """
        Return the active context.

        Raises NotFoundError if no context has been created yet.
        """
        ctx = await self._store.get_active()
        if ctx is None:
            raise NotFoundError("No active application context exists.")
        return ctx

    async def get_context_version(self, version: str) -> ApplicationContext:
        """
        Return the context for the given version string.

        Raises NotFoundError if the version does not exist.
        """
        ctx = await self._store.get_version(version)
        if ctx is None:
            raise NotFoundError(f"Context version '{version}' not found.")
        return ctx

    async def list_versions(self) -> list[ApplicationContext]:
        """Return all context versions, newest first."""
        return await self._store.list_versions()

    async def get_impact(self) -> ContextImpactResult:
        """Return task distribution across context versions."""
        return await self._store.get_impact()
