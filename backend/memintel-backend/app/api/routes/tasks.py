"""
app/api/routes/tasks.py
──────────────────────────────────────────────────────────────────────────────
Task lifecycle endpoints — creation, listing, inspection, update, and deletion.

Endpoints
─────────
  POST   /tasks          createTask   — create from natural language intent
  GET    /tasks          listTasks    — paginated list with optional status filter
  GET    /tasks/{id}     getTask      — fetch a single task by id
  PATCH  /tasks/{id}     updateTask   — update status / condition_version / delivery / entity_scope
  DELETE /tasks/{id}     deleteTask   — soft delete (status → 'deleted')

Ownership rules
───────────────
POST /tasks is the ONLY endpoint that calls the LLM (via TaskAuthoringService).
All other endpoints are deterministic — no LLM involvement.

PATCH /tasks/{id} — condition_version rebinding validation:
  If condition_version is in the patch body, the route validates that the new
  version exists in the definition registry for the same condition_id before
  passing the update to the store. A deprecated but valid version is accepted
  (with a warning log); a missing version → HTTP 404.

DELETE /tasks/{id} — soft delete only:
  Sets status='deleted'. The row is retained for audit. Attempting to DELETE
  a task that is already deleted → HTTP 404 (treat as not found from the
  API perspective — no active task exists at that id).

PATCH on a deleted task → HTTP 409 (ConflictError raised by TaskStore.update()).

HTTP status codes mirror the API spec in developer_api.yaml.
MemintelError subclasses are caught globally by the exception handler in
main.py — routes do not catch them here.
"""
from __future__ import annotations

import structlog

import asyncpg
from fastapi import APIRouter, Depends, Query, Request

from app.models.errors import NotFoundError
from app.models.task import (
    CreateTaskRequest,
    Task,
    TaskList,
    TaskStatus,
    TaskUpdateRequest,
)
from app.models.result import DryRunResult
from app.persistence.db import get_db
from app.persistence.stores import get_task_store
from app.registry.definitions import DefinitionRegistry
from app.services.task_authoring import TaskAuthoringService
from app.stores import TaskStore, DefinitionStore, ContextStore

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["Tasks"])


# ── Service dependency ─────────────────────────────────────────────────────────

async def get_task_authoring_service(
    request: Request,
    pool: asyncpg.Pool = Depends(get_db),
) -> TaskAuthoringService:
    """
    FastAPI dependency — returns a TaskAuthoringService wired to the pool
    and the guardrails loaded at startup.

    TaskAuthoringService owns the LLM pipeline: guardrails loading, concept +
    condition + action generation, compiler validation, and task persistence.
    It is the only service that calls the LLM.
    """
    task_store = TaskStore(pool)
    definition_registry = DefinitionRegistry(store=DefinitionStore(pool))
    guardrails_store = getattr(request.app.state, "guardrails_store", None)
    guardrails = guardrails_store.get_guardrails() if guardrails_store else None
    return TaskAuthoringService(
        task_store=task_store,
        definition_registry=definition_registry,
        guardrails=guardrails,
        context_store=ContextStore(pool),
        guardrails_store=guardrails_store,
    )


# ── POST /tasks ────────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Create a task from natural language intent",
    response_model=None,   # oneOf Task | DryRunResult — FastAPI can't express this natively
    status_code=200,
)
async def create_task(
    req: CreateTaskRequest,
    service: TaskAuthoringService = Depends(get_task_authoring_service),
) -> Task | DryRunResult:
    """
    Submit a natural language intent to the LLM pipeline.

    The system classifies intent, resolves primitives, selects a strategy,
    resolves parameters, generates a concept + condition, binds an action,
    validates, compiles, and persists a version-pinned Task.

    Pass dry_run=true to preview the resolved definitions without persisting.
    Returns a DryRunResult when dry_run=true, a Task otherwise.

    HTTP 400 — validation error in the request body.
    HTTP 422 — intent could not be resolved (primitive not found, no valid
               strategy, or action binding failed).
    """
    log.info(
        "create_task_request",
        intent_length=len(req.intent),
        dry_run=req.dry_run,
    )
    result = await service.create_task(req)
    return result


# ── GET /tasks ─────────────────────────────────────────────────────────────────

@router.get(
    "",
    summary="List tasks",
    response_model=TaskList,
    status_code=200,
)
async def list_tasks(
    status: TaskStatus | None = Query(
        default=None,
        description="Filter by task status (active, paused, deleted)",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of tasks to return (1–100)",
    ),
    cursor: str | None = Query(
        default=None,
        description="Pagination cursor — task_id of the last item seen",
    ),
    store: TaskStore = Depends(get_task_store),
) -> TaskList:
    """
    Return a paginated list of tasks for the current workspace.

    Deleted tasks are excluded by default. Pass status='deleted' to list only
    deleted tasks. Cursor-based pagination: pass the returned next_cursor as
    the cursor query parameter on the next request.
    """
    return await store.list(status=status, limit=limit, cursor=cursor)


# ── GET /tasks/{id} ────────────────────────────────────────────────────────────

@router.get(
    "/{task_id}",
    summary="Get task details",
    response_model=Task,
    status_code=200,
)
async def get_task(
    task_id: str,
    store: TaskStore = Depends(get_task_store),
) -> Task:
    """
    Return the full task definition by task_id.

    Includes concept, condition, and action references (id + version for each)
    plus execution metadata (status, created_at, last_triggered_at).

    HTTP 404 — task not found.
    """
    task = await store.get(task_id)
    if task is None:
        raise NotFoundError(f"Task '{task_id}' not found.", location="task_id")
    return task


# ── PATCH /tasks/{id} ─────────────────────────────────────────────────────────

@router.patch(
    "/{task_id}",
    summary="Update a task",
    response_model=Task,
    status_code=200,
)
async def update_task(
    task_id: str,
    body: TaskUpdateRequest,
    service: TaskAuthoringService = Depends(get_task_authoring_service),
) -> Task:
    """
    Update operational task settings.

    Allowed fields: condition_version, delivery, entity_scope, status (active/paused).
    At least one must be provided — empty patch → HTTP 400 (enforced by model validator).

    Forbidden fields: concept_id, concept_version, condition_id, action_id,
    action_version. Provide any of these → HTTP 400.

    condition_version rebinding:
      The new version must exist in the definition registry for the same
      condition_id as the task's current condition. Missing version → HTTP 404.
      A deprecated version is accepted with a warning log.

    Deleted task → HTTP 409 (ConflictError from store).
    Not found → HTTP 404.
    """
    return await service.update_task(task_id, body)


# ── DELETE /tasks/{id} ────────────────────────────────────────────────────────

@router.delete(
    "/{task_id}",
    summary="Delete a task",
    response_model=Task,
    status_code=200,
)
async def delete_task(
    task_id: str,
    service: TaskAuthoringService = Depends(get_task_authoring_service),
) -> Task:
    """
    Soft-delete a task (status → 'deleted').

    The row is retained for audit. Deleted tasks are excluded from GET /tasks
    by default and are never evaluated. Deletion is irreversible via the API.

    HTTP 404 — task not found or already deleted (no active task at this id).
    """
    return await service.delete_task(task_id)
