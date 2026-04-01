"""
app/api/routes/actions.py
──────────────────────────────────────────────────────────────────────────────
Action inspection and direct-trigger endpoints.

Endpoints
─────────
  GET  /actions/{id}          getAction     — fetch action definition by id + version
  POST /actions/{id}/trigger  triggerAction — fire action directly, bypassing pipeline

Ownership rules
───────────────
Both endpoints are deterministic — no LLM involvement.

GET /actions/{id}:
  version is REQUIRED as a query parameter. Returns the ActionDefinition stored
  in the definitions table (definition_type='action'). HTTP 404 if not found.

POST /actions/{id}/trigger:
  Triggers a registered action directly for a given entity, bypassing the full
  ψ → φ → α pipeline evaluation. Intended for:
    - Verifying action config before go-live
    - Staging environment testing
    - Manual trigger for debugging / support workflows

  version is in the request body (ActionTriggerRequest) — not in the path.
  Actions are always addressed by explicit (action_id, version) with no 'latest'
  resolution.

  dry_run=True simulates the trigger without making any HTTP calls or external
  side effects. Returns status='would_trigger' in the response.

  Actions are best-effort — the route returns HTTP 200 even if delivery fails.
  A failed delivery is recorded in ActionResult.status='failed' with
  ActionResult.error populated. It does NOT return HTTP 5xx.

  fire_on evaluation:
    When an explicit decision value is not provided (direct trigger bypasses the
    condition evaluation path), the action is fired unconditionally unless the
    action executor determines a skip condition from the trigger config.

Note on GET /actions (list endpoint)
──────────────────────────────────────
A GET /actions list endpoint is implied by the ActionList model but is not
defined in developer_api.yaml v2.1. The current DefinitionStore.list() omits
the body column (JSONB), making it impossible to populate ActionDefinition.config
and ActionDefinition.trigger without N+1 queries. This endpoint should be
implemented in the store layer before being exposed here.

Error handling
──────────────
MemintelError subclasses are caught globally by the exception handler in
main.py — routes do not catch them here.
"""
from __future__ import annotations

import structlog
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.deps import require_elevated_key
from app.models.action import ActionDefinition, ActionResult, ActionTriggerRequest
from app.models.concept import DefinitionResponse
from app.models.errors import NotFoundError
from app.persistence.db import get_db
from app.persistence.stores import get_definition_store
from app.services.action import ActionService
from app.stores import DefinitionStore

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/actions", tags=["Actions"])


class ActionListResponse(BaseModel):
    """Response body for GET /actions."""
    actions: list[ActionDefinition]
    total: int
    limit: int
    offset: int


# ── Service dependency ─────────────────────────────────────────────────────────

async def get_action_service(
    pool: asyncpg.Pool = Depends(get_db),
) -> ActionService:
    """
    FastAPI dependency — returns an ActionService backed by the shared pool.

    ActionService dispatches webhook, notification, workflow, and register
    actions. It evaluates fire_on, handles dry_run, and wraps failures in
    ActionResult rather than raising exceptions — actions are best-effort.
    """
    return ActionService(pool=pool)


# ── GET /actions ──────────────────────────────────────────────────────────────

@router.get(
    "",
    summary="List action definitions",
    response_model=ActionListResponse,
    status_code=200,
)
async def list_actions(
    namespace: str = Query(..., description="Namespace to list actions from"),
    limit: int = Query(default=100, ge=1, le=100, description="Maximum number of actions to return"),
    offset: int = Query(default=0, ge=0, description="Number of actions to skip"),
    store: DefinitionStore = Depends(get_definition_store),
) -> ActionListResponse:
    """
    Return a list of non-deprecated ActionDefinitions for the given namespace.

    Results are ordered newest-first (created_at DESC). Deprecated actions
    are excluded. Returns an empty list when no actions match — never 404.

    HTTP 422 — namespace query parameter missing.
    """
    actions, total = await _list_actions_with_total(
        store, namespace=namespace, limit=limit, offset=offset
    )
    return ActionListResponse(
        actions=actions,
        total=total,
        limit=limit,
        offset=offset,
    )


async def _list_actions_with_total(
    store: Any,
    namespace: str,
    limit: int,
    offset: int,
) -> tuple[list, int]:
    """Fetch page and total count concurrently."""
    import asyncio
    page_task = asyncio.create_task(
        store.list_actions(namespace=namespace, limit=limit, offset=offset)
    )
    count_task = asyncio.create_task(store.count_actions(namespace=namespace))
    actions = await page_task
    total = await count_task
    return actions, total


# ── POST /actions ─────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Register an action definition",
    response_model=DefinitionResponse,
    status_code=201,
)
async def register_action(
    action: ActionDefinition,
    store: DefinitionStore = Depends(get_definition_store),
    _: None = Depends(require_elevated_key),
) -> DefinitionResponse:
    """
    Register a new ActionDefinition in the definitions store.

    The (action_id, version) pair is immutable once registered — calling
    this endpoint again with the same action_id and version raises HTTP 409.

    Requires elevated key (X-Elevated-Key header) → HTTP 403 if absent.

    HTTP 201 — action registered successfully.
    HTTP 403 — elevated key missing or invalid.
    HTTP 409 — (action_id, version) already registered.
    """
    log.info(
        "register_action_request",
        action_id=action.action_id,
        version=action.version,
        action_type=action.config.type,
    )
    return await store.register(
        definition_id=action.action_id,
        version=action.version,
        definition_type="action",
        namespace=action.namespace.value,
        body=action.model_dump(mode="json"),
    )


# ── GET /actions/{action_id} ───────────────────────────────────────────────────

@router.get(
    "/{action_id}",
    summary="Get an action definition",
    response_model=ActionDefinition,
    status_code=200,
)
async def get_action(
    action_id: str,
    version: str = Query(
        ...,
        description="Action version to retrieve (required)",
    ),
    store: DefinitionStore = Depends(get_definition_store),
) -> ActionDefinition:
    """
    Return the full ActionDefinition for the given (action_id, version).

    Includes config (type-specific delivery parameters), trigger (fire_on rule
    and condition binding), namespace, deprecation status, and creation timestamp.

    HTTP 404 — action not found or version does not exist.
    """
    body = await store.get(action_id, version)
    if body is None:
        raise NotFoundError(
            f"Action '{action_id}' version '{version}' not found.",
            location="action_id",
        )
    return ActionDefinition.model_validate(body)


# ── POST /actions/{action_id}/trigger ─────────────────────────────────────────

@router.post(
    "/{action_id}/trigger",
    summary="Trigger an action directly",
    response_model=ActionResult,
    status_code=200,
)
async def trigger_action(
    action_id: str,
    req: ActionTriggerRequest,
    store: DefinitionStore = Depends(get_definition_store),
    service: ActionService = Depends(get_action_service),
) -> ActionResult:
    """
    Fire a registered action directly for a given entity, bypassing the
    full pipeline evaluation.

    version is in the request body — not in the path. Actions are always
    addressed by explicit (action_id, version); there is no 'latest' resolution.

    dry_run=True simulates the trigger without making HTTP calls or producing
    external side effects. Returns status='would_trigger'.

    Action execution is best-effort — this endpoint always returns HTTP 200.
    A failed delivery is captured in ActionResult.status='failed' with
    ActionResult.error populated. HTTP 5xx is never returned for action failures.

    HTTP 404 — action (action_id, version) not found.
    """
    body = await store.get(action_id, req.version)
    if body is None:
        raise NotFoundError(
            f"Action '{action_id}' version '{req.version}' not found.",
            location="action_id",
        )

    action = ActionDefinition.model_validate(body)

    log.info(
        "trigger_action_request",
        action_id=action_id,
        version=req.version,
        action_type=action.config.type,
        dry_run=req.dry_run,
    )

    return await service.trigger(action=action, req=req)
