"""
app/api/routes/dynamic_registry.py
──────────────────────────────────────────────────────────────────────────────
Dynamic primitive and connector registration API.

Endpoints (paths relative to /v1 prefix added in main.py)
──────────────────────────────────────────────────────────
  POST   /v1/connectors            — register a new connector (elevated key)
  GET    /v1/connectors            — list all registered connectors (api key)
  DELETE /v1/connectors/{name}     — remove a connector (elevated key)
  POST   /v1/primitives            — register a new primitive (elevated key)
  GET    /v1/primitives            — list all registered primitives (api key)
  DELETE /v1/primitives/{name}     — remove a primitive (elevated key)

Connector params (host, port, database, user, password / token etc.) are
stored Fernet-encrypted in the DB.  Responses never include the raw params.

Runtime integration
───────────────────
On successful POST the in-memory registries on app.state are updated
immediately so the new connector/primitive is available to DataResolver
without a restart.

On DELETE the live registries are updated immediately.

Startup reload
──────────────
main.py loads all rows from registered_connectors / registered_primitives at
startup and calls _rebuild_live_connector() / _register_dynamic_primitive()
to reconstruct live objects, mirroring what POST does at runtime.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.deps import require_api_key, require_elevated_key
from app.models.config import VALID_PRIMITIVE_TYPES, PrimitiveSourceConfig
from app.persistence.db import get_db
from app.stores.dynamic_registry import DynamicRegistryStore
from app.utils.encryption import decrypt, encrypt

log = structlog.get_logger(__name__)

router = APIRouter(tags=["Dynamic Registry"])

# Supported connector types for dynamic registration
_VALID_CONNECTOR_TYPES = frozenset({"postgres", "rest"})


# ── Request / Response models ──────────────────────────────────────────────────

class RegisterConnectorRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    connector_type: str = Field(..., description="postgres or rest")
    params: dict[str, Any] = Field(
        ...,
        description=(
            "Connector-specific parameters. "
            "Postgres: host, port, database, user, password. "
            "REST: base_url, auth (optional), timeout_ms (optional)."
        ),
    )


class ConnectorResponse(BaseModel):
    name: str
    connector_type: str
    created_at: datetime


class RegisterPrimitiveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    primitive_type: str = Field(..., description="Any valid Memintel primitive type")
    connector_name: str | None = Field(
        default=None,
        description="Name of the connector to use. Must be registered before use.",
    )
    query: str | None = Field(
        default=None,
        description="SQL query or REST path for fetching this primitive.",
    )
    json_path: str | None = Field(
        default=None,
        description="Dot-notation JSON path for REST connectors (e.g. 'data.value').",
    )
    source_config: dict | None = Field(
        default=None,
        description=(
            "Nested source config object. SQL query is extracted from "
            "source_config.access.query if query is not provided directly."
        ),
    )

    def resolved_query(self) -> str | None:
        """
        Return the effective SQL query.
        Explicit `query` field wins; falls back to source_config.access.query.
        """
        if self.query:
            return self.query
        if self.source_config:
            access = self.source_config.get("access") or {}
            return access.get("query") or None
        return None

    def resolved_connector_name(self) -> str | None:
        """
        Return the effective connector name.
        Explicit `connector_name` wins; falls back to source_config.identifier.
        """
        if self.connector_name:
            return self.connector_name
        if self.source_config:
            return self.source_config.get("identifier") or None
        return None


class PrimitiveResponse(BaseModel):
    name: str
    primitive_type: str
    connector_name: str | None
    query: str | None
    json_path: str | None
    created_at: datetime | None = None  # absent for yaml-loaded primitives


# ── Internal helpers shared with main.py ──────────────────────────────────────

async def _rebuild_live_connector(
    name: str,
    connector_type: str,
    params: dict,
    primitive_rows: list[dict],
    connector_registry: Any,
) -> None:
    """
    Instantiate a connector from decrypted params and add it to the live registry.

    primitive_rows — rows from registered_primitives whose connector_name == name,
                     used to build the PrimitiveSourceConfig map for this connector.
    """
    from app.connectors.postgres import PostgresConnector
    from app.connectors.rest import RestConnector
    from app.models.config import ConnectorConfig

    # Build PrimitiveSourceConfig map for this connector
    prim_sources: dict[str, PrimitiveSourceConfig] = {
        row["name"]: PrimitiveSourceConfig(
            connector=name,
            query=row["query"] or "",
            json_path=row["json_path"],
        )
        for row in primitive_rows
    }

    # model_construct() bypasses all Pydantic validators — required here because
    # dynamic connectors carry plaintext credentials (already encrypted at rest),
    # while ConnectorConfig's _validate_password enforces ${ENV_VAR} references
    # for the static YAML config path.  Do not use model_construct() on the
    # static config loading path.
    config = ConnectorConfig.model_construct(type=connector_type, **params)

    if connector_type == "postgres":
        conn_obj = PostgresConnector(config, prim_sources)
    elif connector_type == "rest":
        conn_obj = RestConnector(config, prim_sources)
    else:
        log.warning("dynamic_connector_unknown_type", name=name, connector_type=connector_type)
        return

    connector_registry.register(name, conn_obj)
    log.info("dynamic_connector_registered", name=name, connector_type=connector_type)


def _register_dynamic_primitive(
    row: dict,
    primitive_registry: Any,
    dynamic_primitive_sources: dict,
) -> None:
    """
    Register a primitive row into the in-memory primitive_registry and
    dynamic_primitive_sources dict.
    """
    from app.models.config import PrimitiveConfig

    # model_construct() bypasses the 'source' required-field validator.
    # Dynamic primitives use dynamic_primitive_sources (PrimitiveSourceConfig)
    # for data access at runtime — PrimitiveConfig.source is not used on this path.
    primitive_registry.register(
        PrimitiveConfig.model_construct(
            name=row["name"],
            type=row["primitive_type"],
            missing_data_policy=row.get("missing_data_policy") or "null",
        )
    )

    if row.get("connector_name"):
        dynamic_primitive_sources[row["name"]] = PrimitiveSourceConfig(
            connector=row["connector_name"],
            query=row["query"] or "",
            json_path=row.get("json_path"),
        )
    log.info("dynamic_primitive_registered", name=row["name"])


# ── POST /v1/connectors ────────────────────────────────────────────────────────

@router.post(
    "/connectors",
    response_model=ConnectorResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_connector(
    body: RegisterConnectorRequest,
    request: Request,
    pool: asyncpg.Pool = Depends(get_db),
    _: None = Depends(require_elevated_key),
) -> ConnectorResponse:
    """
    Register a new connector and make it immediately available at runtime.

    Params are encrypted with Fernet before storage.
    The new connector is added to the live ConnectorRegistry without a restart.

    HTTP 400 — unsupported connector_type.
    HTTP 409 — a connector with this name already exists.
    """
    if body.connector_type not in _VALID_CONNECTOR_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported connector_type '{body.connector_type}'. "
                   f"Supported: {sorted(_VALID_CONNECTOR_TYPES)}",
        )

    try:
        params_encrypted = encrypt(json.dumps(body.params))
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    store = DynamicRegistryStore(pool)
    try:
        row = await store.create_connector(
            name=body.name,
            connector_type=body.connector_type,
            params_encrypted=params_encrypted,
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A connector named '{body.name}' already exists.",
        )

    # Update live registry — primitives for this connector will be added
    # as they are registered via POST /v1/primitives
    connector_registry = getattr(request.app.state, "connector_registry", None)
    if connector_registry is not None:
        await _rebuild_live_connector(
            name=body.name,
            connector_type=body.connector_type,
            params=body.params,
            primitive_rows=[],
            connector_registry=connector_registry,
        )

    log.info("register_connector_ok", name=body.name, connector_type=body.connector_type)
    return ConnectorResponse(**row)


# ── GET /v1/connectors ─────────────────────────────────────────────────────────

@router.get(
    "/connectors",
    response_model=list[ConnectorResponse],
    status_code=status.HTTP_200_OK,
)
async def list_connectors(
    pool: asyncpg.Pool = Depends(get_db),
    _: None = Depends(require_api_key),
) -> list[ConnectorResponse]:
    """
    List all dynamically registered connectors.  Params are never returned.
    """
    rows = await DynamicRegistryStore(pool).list_connectors()
    return [ConnectorResponse(**r) for r in rows]


# ── DELETE /v1/connectors/{name} ──────────────────────────────────────────────

@router.delete(
    "/connectors/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_connector(
    name: str,
    request: Request,
    pool: asyncpg.Pool = Depends(get_db),
    _: None = Depends(require_elevated_key),
) -> None:
    """
    Remove a connector from the DB and from the live ConnectorRegistry.

    HTTP 404 — connector not found.
    """
    store = DynamicRegistryStore(pool)
    deleted = await store.delete_connector(name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector '{name}' not found.",
        )

    connector_registry = getattr(request.app.state, "connector_registry", None)
    if connector_registry is not None and name in connector_registry._registry:
        conn_obj = connector_registry._registry.pop(name)
        try:
            if hasattr(conn_obj, "close"):
                await conn_obj.close()
        except Exception:
            pass

    log.info("delete_connector_ok", name=name)


# ── POST /v1/primitives ────────────────────────────────────────────────────────

@router.post(
    "/primitives",
    response_model=PrimitiveResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_primitive(
    body: RegisterPrimitiveRequest,
    request: Request,
    pool: asyncpg.Pool = Depends(get_db),
    _: None = Depends(require_elevated_key),
) -> PrimitiveResponse:
    """
    Register a new primitive and make it immediately available at runtime.

    If connector_name is provided it must already be registered (static config
    or dynamic).  The primitive is added to the live PrimitiveRegistry and
    dynamic_primitive_sources without a restart.

    HTTP 400 — invalid primitive_type or unresolvable connector_name.
    HTTP 409 — a primitive with this name already exists.
    """
    if body.primitive_type not in VALID_PRIMITIVE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid primitive_type '{body.primitive_type}'. "
                   f"Valid types: {sorted(VALID_PRIMITIVE_TYPES)}",
        )

    # Resolve effective connector_name and query from either flat fields or source_config
    effective_connector = body.resolved_connector_name()
    effective_query = body.resolved_query()

    if effective_connector is not None:
        connector_registry = getattr(request.app.state, "connector_registry", None)
        # Check both static config connectors and dynamically registered ones
        known = (
            set(connector_registry._registry.keys()) if connector_registry else set()
        )
        config = getattr(request.app.state, "config", None)
        if config is not None and config.connectors:
            known |= set(config.connectors.keys())
        if effective_connector not in known:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Connector '{effective_connector}' is not registered. "
                       f"Register it first via POST /v1/connectors.",
            )

    store = DynamicRegistryStore(pool)
    try:
        row = await store.create_primitive(
            name=body.name,
            primitive_type=body.primitive_type,
            connector_name=effective_connector,
            query=effective_query,
            json_path=body.json_path,
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A primitive named '{body.name}' already exists.",
        )

    # Update live registries
    primitive_registry = getattr(request.app.state, "primitive_registry", None)
    dynamic_primitive_sources = getattr(
        request.app.state, "dynamic_primitive_sources", None
    )
    if primitive_registry is not None:
        _register_dynamic_primitive(row, primitive_registry, dynamic_primitive_sources or {})
        # If connector already has a live instance, add this primitive's source to it
        if effective_connector and dynamic_primitive_sources is not None:
            connector_registry = getattr(request.app.state, "connector_registry", None)
            if connector_registry is not None:
                conn_obj = connector_registry._registry.get(effective_connector)
                if conn_obj is not None and hasattr(conn_obj, "_primitive_sources"):
                    conn_obj._primitive_sources[body.name] = PrimitiveSourceConfig(
                        connector=effective_connector,
                        query=effective_query or "",
                        json_path=body.json_path,
                    )

    log.info("register_primitive_ok", name=body.name, primitive_type=body.primitive_type)
    return PrimitiveResponse(**row)


# ── GET /v1/primitives ─────────────────────────────────────────────────────────

@router.get(
    "/primitives",
    response_model=list[PrimitiveResponse],
    status_code=status.HTTP_200_OK,
)
async def list_primitives(
    request: Request,
    _: None = Depends(require_api_key),
) -> list[PrimitiveResponse]:
    """
    List all primitives available at runtime — yaml-loaded (from
    memintel_config.yaml or CLIENT_CONFIG_DIR) and dynamically registered
    via POST /v1/primitives.

    Reads from the in-memory PrimitiveRegistry on app.state, which is
    populated at startup and is the authoritative source in demo mode
    (CLIENT_CONFIG_DIR set). Does not query the registered_primitives DB
    table, so it is fast and works correctly when no DB primitives exist.
    """
    registry = getattr(request.app.state, "primitive_registry", None)
    if registry is None:
        return []

    dynamic_sources = getattr(request.app.state, "dynamic_primitive_sources", None) or {}
    result = []
    for prim in registry.list_all():
        src = dynamic_sources.get(prim.name)
        result.append(PrimitiveResponse(
            name=prim.name,
            primitive_type=prim.type,
            connector_name=src.connector if src else None,
            query=src.query if src else None,
            json_path=src.json_path if src else None,
            created_at=None,
        ))
    return result


# ── DELETE /v1/primitives/{name} ──────────────────────────────────────────────

@router.delete(
    "/primitives/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_primitive(
    name: str,
    request: Request,
    pool: asyncpg.Pool = Depends(get_db),
    _: None = Depends(require_elevated_key),
) -> None:
    """
    Remove a primitive from the DB and from the live registries.

    HTTP 404 — primitive not found.
    """
    store = DynamicRegistryStore(pool)
    deleted = await store.delete_primitive(name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Primitive '{name}' not found.",
        )

    # Remove from live primitive_registry (if present)
    primitive_registry = getattr(request.app.state, "primitive_registry", None)
    if primitive_registry is not None and name in primitive_registry._primitives:
        del primitive_registry._primitives[name]

    # Remove from dynamic_primitive_sources
    dynamic_primitive_sources = getattr(
        request.app.state, "dynamic_primitive_sources", None
    )
    if dynamic_primitive_sources is not None:
        dynamic_primitive_sources.pop(name, None)

    log.info("delete_primitive_ok", name=name)
