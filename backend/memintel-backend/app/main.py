"""
Memintel Python Backend — application entry point.

Registers all route modules and wires the FastAPI lifespan (DB pool + Redis).
No logic lives here — only wiring.

Startup invariants (enforced in lifespan)
─────────────────────────────────────────
The system MUST NOT start if any of the following fail:
  - Config file missing or failing schema validation
  - Any ${ENV_VAR} reference unresolved
  - Guardrails file missing or failing schema validation
  - strategy_registry empty in the loaded guardrails
  - DB pool or Redis connection cannot be established

On failure: the specific error is logged and the process exits with code 1.
DO NOT start with partial configuration under any circumstances.
DO NOT silently fall back to defaults.

Config path resolution:
  The MEMINTEL_CONFIG_PATH environment variable must point to the
  memintel_config.yaml file. If it is not set the process exits immediately.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import (
    actions,
    agents,
    compile,
    concepts,
    conditions,
    context,
    decisions,
    dynamic_registry,
    execute,
    feedback,
    guardrails_api,
    jobs,
    registry,
    tasks,
)
from app.config import ConfigError, ConfigLoader, GuardrailsStore, PrimitiveRegistry
from app.models.errors import (
    ErrorDetail,
    ErrorResponse,
    ErrorType,
    MemintelError,
    memintel_error_handler,
)
from app.observability import configure_structlog
from app.persistence.db import create_db_pool, create_redis_client

configure_structlog()
log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan — startup and shutdown logic.

    Startup sequence (all-or-nothing):
      1. Resolve MEMINTEL_CONFIG_PATH
      2. Load and validate memintel_config.yaml
      3. Load and validate memintel_guardrails.yaml (path from config)
      4. Verify strategy_registry is non-empty
      5. Populate PrimitiveRegistry from config
      6. Create DB pool
      7. Create Redis client
      8. Store all singletons on app.state

    Any failure in steps 1–7 logs the error and calls sys.exit(1).
    """
    # ── Step 1: Resolve config path ────────────────────────────────────────────
    config_path = os.environ.get("MEMINTEL_CONFIG_PATH")
    if not config_path:
        log.error(
            "startup_failed",
            reason="MEMINTEL_CONFIG_PATH environment variable is not set. "
                   "Set it to the path of your memintel_config.yaml file.",
        )
        sys.exit(1)

    loader = ConfigLoader()

    # ── Step 2: Load config ────────────────────────────────────────────────────
    try:
        config = loader.load(config_path)
    except ConfigError as e:
        log.error("startup_failed", reason=f"Config load error: {e}")
        sys.exit(1)

    # ── Step 3 & 4: Load guardrails (ConfigLoader enforces non-empty registry) ─
    guardrails_path = str(
        Path(config_path).parent / config.guardrails_path
    )
    guardrails_store = GuardrailsStore()
    try:
        await guardrails_store.load(guardrails_path)
    except ConfigError as e:
        log.error("startup_failed", reason=f"Guardrails load error: {e}")
        sys.exit(1)

    # ── Step 5: Populate primitive registry ────────────────────────────────────
    primitive_registry = PrimitiveRegistry()
    primitive_registry.load_from_config(config)

    if not primitive_registry.list_all():
        log.error(
            "startup_failed",
            reason="No primitives registered — config.primitives is empty.",
        )
        sys.exit(1)

    # ── Step 5b: Build ConnectorRegistry ────────────────────────────────────────
    from app.connectors import ConnectorRegistry
    try:
        connector_registry = await ConnectorRegistry.from_config(config)
    except Exception as e:
        log.error("startup_failed", reason=f"Connector registry build failed: {e}")
        sys.exit(1)

    # Optional health checks (non-fatal — warns but does not block startup)
    if not config.environment.skip_connector_health_check:
        for name, conn in connector_registry._registry.items():
            try:
                healthy = await conn.health_check()
                if not healthy:
                    log.warning("connector_unhealthy", name=name)
            except Exception as exc:
                log.warning("connector_health_check_failed", name=name, error=str(exc))

    # ── Step 6: DB pool ────────────────────────────────────────────────────────
    try:
        db_pool = await create_db_pool(os.environ["DATABASE_URL"])
    except Exception as e:
        log.error("startup_failed", reason=f"DB pool creation failed: {e}")
        sys.exit(1)

    # ── Step 6a: Check for API guardrails override in DB ───────────────────────
    # If an admin has previously posted to POST /guardrails, the API version
    # takes precedence over the file-based guardrails from this point forward.
    # Failure here is non-fatal — file-based guardrails remain active.
    try:
        from app.stores.guardrails import GuardrailsStore as GuardrailsVersionStore
        guardrails_version_store = GuardrailsVersionStore(db_pool)
        db_reloaded = await guardrails_store.reload_from_db(guardrails_version_store)
        if db_reloaded:
            active_ver = guardrails_store.get_active_version()
            log.info(
                "guardrails_api_version_loaded",
                version=active_ver.version if active_ver else None,
            )
    except Exception as e:
        log.warning("guardrails_db_check_failed", reason=str(e))

    # ── Step 6b: Load dynamic registrations from DB ───────────────────────────
    # When CLIENT_CONFIG_DIR is set (demo mode) the yaml files in that folder
    # are the sole source of primitives — skip registered_primitives entirely to
    # prevent cross-demo contamination (e.g. AcmeBank rows appearing in XBRL).
    # When CLIENT_CONFIG_DIR is not set (production) reload dynamic connectors
    # and primitives registered via POST /v1/connectors and POST /v1/primitives
    # so they survive a restart without re-registration.
    dynamic_primitive_sources: dict = {}
    if os.environ.get("CLIENT_CONFIG_DIR"):
        log.info(
            "demo_mode_skip_db_primitives",
            client_config_dir=os.environ.get("CLIENT_CONFIG_DIR"),
        )
    else:
        try:
            from app.stores.dynamic_registry import DynamicRegistryStore
            from app.api.routes.dynamic_registry import (
                _rebuild_live_connector,
                _register_dynamic_primitive,
            )
            from app.utils.encryption import decrypt

            dyn_store = DynamicRegistryStore(db_pool)
            connector_rows = await dyn_store.list_connectors_with_params()
            primitive_rows = await dyn_store.list_primitives()

            # Index primitives by connector_name for efficient lookup
            by_connector: dict[str, list[dict]] = {}
            for pr in primitive_rows:
                cn = pr.get("connector_name")
                if cn:
                    by_connector.setdefault(cn, []).append(pr)

            import json as _json
            for cr in connector_rows:
                try:
                    params = _json.loads(decrypt(cr["params_encrypted"]))
                    await _rebuild_live_connector(
                        name=cr["name"],
                        connector_type=cr["connector_type"],
                        params=params,
                        primitive_rows=by_connector.get(cr["name"], []),
                        connector_registry=connector_registry,
                    )
                except Exception as exc:
                    log.warning(
                        "dynamic_connector_reload_failed",
                        name=cr["name"],
                        error=str(exc),
                    )

            for pr in primitive_rows:
                try:
                    _register_dynamic_primitive(pr, primitive_registry, dynamic_primitive_sources)
                except Exception as exc:
                    log.warning(
                        "dynamic_primitive_reload_failed",
                        name=pr["name"],
                        error=str(exc),
                    )

            log.info(
                "dynamic_registrations_loaded",
                connectors=len(connector_rows),
                primitives=len(primitive_rows),
            )
        except Exception as exc:
            log.warning("dynamic_registrations_load_failed", error=str(exc))

    # ── Step 7: Redis client ───────────────────────────────────────────────────
    try:
        redis = await create_redis_client(os.environ["REDIS_URL"])
    except Exception as e:
        log.error("startup_failed", reason=f"Redis client creation failed: {e}")
        sys.exit(1)

    # ── Store singletons on app.state ──────────────────────────────────────────
    app.state.config = config
    app.state.db = db_pool
    app.state.redis = redis
    app.state.guardrails_store = guardrails_store
    app.state.primitive_registry = primitive_registry
    app.state.connector_registry = connector_registry
    app.state.dynamic_primitive_sources = dynamic_primitive_sources
    # Elevated key for internal platform endpoints — read from environment.
    # Absent → all elevated endpoints return HTTP 403.
    app.state.elevated_key = os.environ.get("MEMINTEL_ELEVATED_KEY")
    # API key for developer-facing read endpoints — read from environment.
    # Absent → require_api_key is permissive (development mode).
    app.state.api_key = os.environ.get("MEMINTEL_API_KEY")

    log.info(
        "startup_complete",
        primitives=len(primitive_registry.list_all()),
        strategies=len(guardrails_store.get_strategy_registry()),
    )

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    await db_pool.close()
    await redis.close()
    await connector_registry.close_all()
    log.info("shutdown_complete")


app = FastAPI(
    title="Memintel Backend API",
    version="1.0.0",
    lifespan=lifespan,
    redoc_url=None,       # disable default ReDoc route; custom route below
)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/redoc", include_in_schema=False)
async def redoc() -> HTMLResponse:
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Memintel API Reference",
        redoc_js_url="/static/redoc.standalone.js",
    )

# ── CORS middleware ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception handlers ──────────────────────────────────────────────────────

app.add_exception_handler(MemintelError, memintel_error_handler)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Map FastAPI HTTPException to the canonical ErrorResponse wire shape.

    Covers elevated-key 403s (raised as HTTPException in deps.py) and any
    other HTTPExceptions FastAPI raises internally (e.g. 405 Method Not Allowed).
    """
    detail = exc.detail
    # If the detail is already our ErrorResponse dict, pass it through.
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                type=ErrorType.AUTH_ERROR if exc.status_code in (401, 403)
                else ErrorType.NOT_FOUND if exc.status_code == 404
                else ErrorType.EXECUTION_ERROR,
                message=str(detail) if detail else HTTPException(exc.status_code).detail,
            )
        ).model_dump(mode="json"),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Override FastAPI's default 422 handler to strip the 'input' field from
    validation errors, preventing user-supplied data from being echoed back.
    """
    errors = [
        {"loc": e["loc"], "msg": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(asyncpg.PostgresError)
async def postgres_error_handler(request: Request, exc: asyncpg.PostgresError) -> JSONResponse:
    """
    Map asyncpg database errors to safe HTTP responses.

    Prevents raw PostgreSQL error messages (which may include query fragments
    or data) from leaking to callers.
    """
    if isinstance(exc, asyncpg.CheckViolationError):
        return JSONResponse(status_code=422, content={"detail": "Invalid field value"})
    if isinstance(exc, asyncpg.DataError):
        return JSONResponse(status_code=422, content={"detail": "Invalid data format"})
    log.error(
        "postgres_error",
        exc_type=type(exc).__name__,
        exc_str=str(exc),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unexpected exceptions (unhandled Python errors).
    Always returns HTTP 500 ErrorResponse.

    Logs the exception before responding.
    """
    log.error(
        "unhandled_exception",
        exc_type=type(exc).__name__,
        exc_str=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                type=ErrorType.EXECUTION_ERROR,
                message="An unexpected error occurred. Please try again or contact support.",
            )
        ).model_dump(mode="json"),
    )


# ── Route registration ──────────────────────────────────────────────────────
# evaluate_router is registered at /evaluate (not /execute) per developer_api.yaml.
app.include_router(execute.evaluate_router,          prefix="/evaluate",   tags=["Execution"])
app.include_router(execute.router,                   prefix="/execute",    tags=["Execution"])
app.include_router(compile.router,                   prefix="/compile",    tags=["Compiler"])
app.include_router(registry.router,                  prefix="/registry",   tags=["Registry"])
app.include_router(dynamic_registry.router,          prefix="/v1",         tags=["Dynamic Registry"])
app.include_router(agents.router,           prefix="/agents",     tags=["Agents"])
app.include_router(concepts.router,                               tags=["Concepts"])
app.include_router(tasks.router,                                  tags=["Tasks"])
app.include_router(conditions.router,                             tags=["Conditions"])
app.include_router(decisions.router,                              tags=["Decisions"])
app.include_router(feedback.router,                               tags=["Feedback"])
app.include_router(actions.router,                                tags=["Actions"])
app.include_router(jobs.router,             prefix="/jobs",       tags=["Jobs"])
app.include_router(context.router,                                tags=["Context"])
app.include_router(guardrails_api.router,                        tags=["Guardrails"])
