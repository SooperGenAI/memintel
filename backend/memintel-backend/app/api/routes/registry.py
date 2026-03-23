"""
app/api/routes/registry.py
──────────────────────────────────────────────────────────────────────────────
Definition registry and feature registry endpoints.

Endpoints (paths relative to the /registry prefix added by main.py)
────────────────────────────────
  GET    /registry/definitions                     listDefinitions    — paginated list
  POST   /registry/definitions                     registerDefinition — create/update
  GET    /registry/definitions/{id}/versions       getVersions        — version history
  GET    /registry/definitions/{id}/lineage        getLineage         — version chain
  GET    /registry/definitions/{id}/semantic-diff  getSemanticDiff    — version diff
  POST   /registry/definitions/{id}/deprecate      deprecate          — mark deprecated
  POST   /registry/definitions/{id}/promote        promote            — namespace promote
  POST   /registry/definitions/similar             findSimilar        — semantic search

  GET    /registry/search                          searchDefinitions  — keyword search

  POST   /registry/features                        registerFeature    — register feature
  GET    /registry/features                        listFeatures       — search features
  GET    /registry/features/{id}                   getFeature         — get by id
  GET    /registry/features/{id}/usages            getFeatureUsages   — list usages

Ownership rules
───────────────
All endpoints are deterministic — no LLM involvement.

Definition registry semantics:
  Definitions are immutable once published. register() creates a new version
  if (definition_id, version) already exists and the body matches; it raises
  HTTP 409 if the body differs (immutability invariant).

  Deprecation is irreversible. deprecated=True prevents the definition from
  being bound in new tasks (type checker enforces).

  Promotion moves a definition to a higher namespace (private → org → public).
  Namespace transitions are unidirectional — demotion is not supported.

Semantic diff:
  version_from and version_to are required query params.
  equivalence_status values:
    equivalent — same meaning_hash; identical computation; safe to promote.
    compatible — meaning changed but backward-compatible.
    breaking   — downstream definitions may be invalidated.
    unknown    — treat as breaking.

Route registration order
────────────────────────
/definitions/similar and /search and /features* are registered BEFORE the
parameterised /definitions/{id}/* routes to prevent path capture.

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
from app.models.concept import (
    DefinitionResponse,
    LineageResult,
    SearchResult,
    SemanticDiffResult,
    VersionSummary,
)
from app.persistence.db import get_db
from app.services.registry import RegistryService

log = structlog.get_logger(__name__)

router = APIRouter(tags=["Registry"])


# ── Inline request/response models ────────────────────────────────────────────
# These models correspond to Internal Platform API endpoints not yet defined
# in developer_api.yaml. Inline until a dedicated models file is created.

class RegisterDefinitionRequest(BaseModel):
    definition_id: str
    version: str
    definition_type: str          # concept | condition | action | primitive
    namespace: str
    body: dict[str, Any]          # raw definition body (stored as JSONB)


class DeprecateRequest(BaseModel):
    replacement_version: str | None = None


class PromoteRequest(BaseModel):
    target_namespace: str         # namespace to promote into


class FindSimilarRequest(BaseModel):
    definition_id: str
    version: str
    limit: int = 10


class RegisterFeatureRequest(BaseModel):
    feature_id: str
    version: str
    body: dict[str, Any]


class RegisteredFeature(BaseModel):
    feature_id: str
    version: str
    semantic_hash: str
    body: dict[str, Any]
    created_at: str | None = None


class FeatureSearchResult(BaseModel):
    items: list[RegisteredFeature]
    has_more: bool
    next_cursor: str | None = None
    total_count: int


class FeatureRegistrationResult(BaseModel):
    feature_id: str
    version: str
    semantic_hash: str
    deduplicated: bool    # True if an existing equivalent feature was found


class UsageResult(BaseModel):
    feature_id: str
    usages: list[dict[str, Any]]   # list of {definition_id, version, definition_type}
    total_count: int


class DefinitionsBatchRequest(BaseModel):
    definitions: list[RegisterDefinitionRequest]   # batch of definitions to register


class DefinitionsBatchResult(BaseModel):
    registered: int       # count of successfully registered definitions
    skipped: int          # count skipped (already exists, body matches)
    failed: int           # count that raised errors (body mismatch, type error, etc.)
    errors: list[dict[str, Any]]   # [{definition_id, version, error}]


# ── Service dependency ─────────────────────────────────────────────────────────

async def get_registry_service(
    pool: asyncpg.Pool = Depends(get_db),
) -> RegistryService:
    """
    FastAPI dependency — returns a RegistryService backed by the shared pool.

    RegistryService manages definition registration, versioning, semantic diff,
    deprecation, namespace promotion, and feature registry. No LLM involvement.
    """
    return RegistryService(pool=pool)


# ── POST /registry/definitions/similar ────────────────────────────────────────
# Registered before /definitions/{id}/* routes to prevent path capture.

@router.post(
    "/definitions/similar",
    summary="Find semantically similar definitions",
    response_model=SearchResult,
    status_code=200,
)
async def find_similar(
    req: FindSimilarRequest,
    service: RegistryService = Depends(get_registry_service),
) -> SearchResult:
    """
    Return definitions that are semantically similar to the given (id, version).

    Similarity is computed from the semantic_hash distance. Useful for
    detecting near-duplicates before registering a new definition.

    HTTP 404 — source definition not found.
    """
    log.info(
        "find_similar_request",
        definition_id=req.definition_id,
        version=req.version,
    )
    return await service.find_similar(req)


# ── GET /registry/search ───────────────────────────────────────────────────────

@router.get(
    "/search",
    summary="Search definitions by keyword",
    response_model=SearchResult,
    status_code=200,
)
async def search_definitions(
    q: str = Query(..., description="Search query string"),
    definition_type: str | None = Query(
        default=None,
        description="Filter by definition type (concept, condition, action, primitive)",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None, description="Pagination cursor"),
    service: RegistryService = Depends(get_registry_service),
) -> SearchResult:
    """
    Search definitions by keyword across id, namespace, and metadata.

    Returns a paginated SearchResult with cursor-based pagination.
    """
    log.info("search_definitions_request", q=q)
    return await service.search(
        q=q,
        definition_type=definition_type,
        limit=limit,
        cursor=cursor,
    )


# ── POST /registry/features ────────────────────────────────────────────────────

@router.post(
    "/features",
    summary="Register a feature",
    response_model=FeatureRegistrationResult,
    status_code=200,
)
async def register_feature(
    req: RegisterFeatureRequest,
    service: RegistryService = Depends(get_registry_service),
    _: None = Depends(require_elevated_key),
) -> FeatureRegistrationResult:
    """
    Register a new feature in the feature registry.

    Semantic deduplication is applied: if an equivalent feature already
    exists (same semantic_hash), deduplicated=True is returned and the
    existing feature_id is preserved.

    HTTP 409 — feature_id + version already exists with different body.
    """
    log.info(
        "register_feature_request",
        feature_id=req.feature_id,
        version=req.version,
    )
    return await service.register_feature(req)


# ── GET /registry/features/{id}/usages ────────────────────────────────────────
# Registered before /features/{id} to prevent path capture.

@router.get(
    "/features/{feature_id}/usages",
    summary="Get usages of a feature",
    response_model=UsageResult,
    status_code=200,
)
async def get_feature_usages(
    feature_id: str,
    service: RegistryService = Depends(get_registry_service),
) -> UsageResult:
    """
    Return all definitions that depend on the given feature.

    Each usage entry contains definition_id, version, and definition_type
    of the consuming definition.

    HTTP 404 — feature not found.
    """
    log.info("get_feature_usages_request", feature_id=feature_id)
    return await service.get_feature_usages(feature_id)


# ── GET /registry/features/{id} ───────────────────────────────────────────────

@router.get(
    "/features/{feature_id}",
    summary="Get a registered feature",
    response_model=RegisteredFeature,
    status_code=200,
)
async def get_feature(
    feature_id: str,
    service: RegistryService = Depends(get_registry_service),
) -> RegisteredFeature:
    """
    Return a registered feature by feature_id.

    HTTP 404 — feature not found.
    """
    log.info("get_feature_request", feature_id=feature_id)
    return await service.get_feature(feature_id)


# ── GET /registry/features ─────────────────────────────────────────────────────

@router.get(
    "/features",
    summary="Search features",
    response_model=FeatureSearchResult,
    status_code=200,
)
async def list_features(
    q: str | None = Query(default=None, description="Search query string"),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None, description="Pagination cursor"),
    service: RegistryService = Depends(get_registry_service),
) -> FeatureSearchResult:
    """
    Search features in the feature registry.

    Returns a paginated FeatureSearchResult with cursor-based pagination.
    Pass q to filter by feature_id or metadata keyword.
    """
    log.info("list_features_request")
    return await service.list_features(q=q, limit=limit, cursor=cursor)


# ── GET /registry/definitions ─────────────────────────────────────────────────

@router.get(
    "/definitions",
    summary="List definitions",
    response_model=SearchResult,
    status_code=200,
)
async def list_definitions(
    definition_type: str | None = Query(
        default=None,
        description="Filter by type (concept, condition, action, primitive)",
    ),
    namespace: str | None = Query(default=None, description="Filter by namespace"),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None, description="Pagination cursor"),
    service: RegistryService = Depends(get_registry_service),
) -> SearchResult:
    """
    Return a paginated list of definitions.

    Filter by definition_type and/or namespace. Cursor-based pagination —
    pass next_cursor from the previous response as cursor on the next request.
    """
    log.info("list_definitions_request")
    return await service.list_definitions(
        definition_type=definition_type,
        namespace=namespace,
        limit=limit,
        cursor=cursor,
    )


# ── POST /registry/definitions ────────────────────────────────────────────────

@router.post(
    "/definitions",
    summary="Register a definition",
    response_model=DefinitionResponse,
    status_code=200,
)
async def register_definition(
    req: RegisterDefinitionRequest,
    service: RegistryService = Depends(get_registry_service),
    _: None = Depends(require_elevated_key),
) -> DefinitionResponse:
    """
    Register (create or update) a definition in the registry.

    Definitions are immutable once published. If (definition_id, version)
    already exists with identical body, the existing record is returned.
    If it exists with a different body → HTTP 409 (immutability violation).

    For concepts, meaning_hash and ir_hash are computed at registration time.

    Requires elevated key (X-Elevated-Key header) → HTTP 403 if absent.

    HTTP 403 — elevated key missing or invalid.
    HTTP 409 — definition already exists with different body.
    """
    log.info(
        "register_definition_request",
        definition_id=req.definition_id,
        version=req.version,
        definition_type=req.definition_type,
    )
    return await service.register(req)


# ── POST /registry/definitions/batch ─────────────────────────────────────────

@router.post(
    "/definitions/batch",
    summary="Register multiple definitions in a single request",
    response_model=DefinitionsBatchResult,
    status_code=200,
)
async def register_definitions_batch(
    req: DefinitionsBatchRequest,
    service: RegistryService = Depends(get_registry_service),
    _: None = Depends(require_elevated_key),
) -> DefinitionsBatchResult:
    """
    Register a batch of definitions atomically.

    Each definition is processed independently. Failures for individual
    definitions are captured in the errors list and do not block the rest
    of the batch. The response always returns HTTP 200.

    Requires elevated key (X-Elevated-Key header) → HTTP 403 if absent.

    HTTP 403 — elevated key missing or invalid.
    """
    log.info(
        "register_definitions_batch_request",
        count=len(req.definitions),
    )
    return await service.register_batch(req)


# ── GET /registry/definitions/{id}/versions ───────────────────────────────────
# Literal sub-path routes registered before the root /{id} route.

@router.get(
    "/definitions/{definition_id}/versions",
    summary="Get version history for a definition",
    response_model=list[VersionSummary],
    status_code=200,
)
async def get_versions(
    definition_id: str,
    service: RegistryService = Depends(get_registry_service),
) -> list[VersionSummary]:
    """
    Return all versions of a definition, newest-first (created_at DESC).

    Includes deprecation status, meaning_hash, and ir_hash per version.

    HTTP 404 — definition not found.
    """
    log.info("get_versions_request", definition_id=definition_id)
    return await service.get_versions(definition_id)


# ── GET /registry/definitions/{id}/lineage ────────────────────────────────────

@router.get(
    "/definitions/{definition_id}/lineage",
    summary="Get lineage for a definition",
    response_model=LineageResult,
    status_code=200,
)
async def get_lineage(
    definition_id: str,
    service: RegistryService = Depends(get_registry_service),
) -> LineageResult:
    """
    Return the version chain and namespace promotion history of a definition.

    promoted_to maps version strings to the namespace they were promoted to.

    HTTP 404 — definition not found.
    """
    log.info("get_lineage_request", definition_id=definition_id)
    return await service.get_lineage(definition_id)


# ── GET /registry/definitions/{id}/semantic-diff ──────────────────────────────

@router.get(
    "/definitions/{definition_id}/semantic-diff",
    summary="Get semantic diff between two versions",
    response_model=SemanticDiffResult,
    status_code=200,
)
async def get_semantic_diff(
    definition_id: str,
    version_from: str = Query(..., description="Base version (older)"),
    version_to: str = Query(..., description="Target version (newer)"),
    service: RegistryService = Depends(get_registry_service),
) -> SemanticDiffResult:
    """
    Compare two versions of a definition and return the semantic diff.

    equivalence_status:
      equivalent — same meaning_hash; identical computation; safe to promote.
      compatible — meaning changed but backward-compatible.
      breaking   — downstream definitions may be invalidated.
      unknown    — treat as breaking.

    HTTP 404 — definition or either version not found.
    """
    log.info(
        "get_semantic_diff_request",
        definition_id=definition_id,
        version_from=version_from,
        version_to=version_to,
    )
    return await service.get_semantic_diff(definition_id, version_from, version_to)


# ── POST /registry/definitions/{id}/deprecate ─────────────────────────────────

@router.post(
    "/definitions/{definition_id}/deprecate",
    summary="Deprecate a definition version",
    response_model=DefinitionResponse,
    status_code=200,
)
async def deprecate_definition(
    definition_id: str,
    version: str = Query(..., description="Version to deprecate"),
    req: DeprecateRequest = DeprecateRequest(),
    service: RegistryService = Depends(get_registry_service),
) -> DefinitionResponse:
    """
    Mark a definition version as deprecated.

    Deprecation is irreversible. Deprecated definitions cannot be bound in
    new tasks, but existing bindings continue to function.

    replacement_version (optional) — a newer version to reference in the
    deprecation notice.

    HTTP 404 — definition or version not found.
    """
    log.info(
        "deprecate_definition_request",
        definition_id=definition_id,
        version=version,
    )
    return await service.deprecate(
        definition_id=definition_id,
        version=version,
        replacement_version=req.replacement_version,
    )


# ── POST /registry/definitions/{id}/promote ───────────────────────────────────

@router.post(
    "/definitions/{definition_id}/promote",
    summary="Promote a definition to a higher namespace",
    response_model=DefinitionResponse,
    status_code=200,
)
async def promote_definition(
    definition_id: str,
    version: str = Query(..., description="Version to promote"),
    req: PromoteRequest = PromoteRequest(target_namespace="org"),
    service: RegistryService = Depends(get_registry_service),
) -> DefinitionResponse:
    """
    Promote a definition version to a higher namespace.

    Namespace transitions are unidirectional (private → org → public).
    Demotion is not supported. Review the semantic diff before promoting
    a breaking change.

    HTTP 404 — definition or version not found.
    HTTP 422 — namespace transition is invalid.
    """
    log.info(
        "promote_definition_request",
        definition_id=definition_id,
        version=version,
        target_namespace=req.target_namespace,
    )
    return await service.promote(
        definition_id=definition_id,
        version=version,
        target_namespace=req.target_namespace,
    )
