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

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    actions,
    agents,
    compile,
    conditions,
    context,
    decisions,
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
    # Elevated key for internal platform endpoints — read from environment.
    # Absent → all elevated endpoints return HTTP 403.
    app.state.elevated_key = os.environ.get("MEMINTEL_ELEVATED_KEY")

    log.info(
        "startup_complete",
        primitives=len(primitive_registry.list_all()),
        strategies=len(guardrails_store.get_strategy_registry()),
    )

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    await db_pool.close()
    await redis.close()
    log.info("shutdown_complete")


app = FastAPI(
    title="Memintel Backend API",
    version="1.0.0",
    lifespan=lifespan,
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


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unexpected exceptions (unhandled Python errors,
    RequestValidationError, etc.). Always returns HTTP 500 ErrorResponse.

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
app.include_router(execute.evaluate_router, prefix="/evaluate",   tags=["Execution"])
app.include_router(execute.router,          prefix="/execute",    tags=["Execution"])
app.include_router(compile.router,          prefix="/compile",    tags=["Compiler"])
app.include_router(registry.router,         prefix="/registry",   tags=["Registry"])
app.include_router(agents.router,           prefix="/agents",     tags=["Agents"])
app.include_router(tasks.router,                                  tags=["Tasks"])
app.include_router(conditions.router,                             tags=["Conditions"])
app.include_router(decisions.router,                              tags=["Decisions"])
app.include_router(feedback.router,                               tags=["Feedback"])
app.include_router(actions.router,                                tags=["Actions"])
app.include_router(jobs.router,             prefix="/jobs",       tags=["Jobs"])
app.include_router(context.router,                                tags=["Context"])
app.include_router(guardrails_api.router,                        tags=["Guardrails"])
