"""
app/persistence/__init__.py
──────────────────────────────────────────────────────────────────────────────
Re-exports the public surface of the persistence layer.

Import from this package rather than individual modules:
    from app.persistence import lifespan, get_db, get_redis
    from app.persistence import get_task_store, get_cache
    from app.persistence import make_cache_key, ResultCache
"""

from app.persistence.db import (
    create_db_pool,
    create_redis_client,
    lifespan,
    get_db,
    get_redis,
)

from app.persistence.cache import (
    CACHE_KEY,
    TTL_DETERMINISTIC,
    make_cache_key,
    ResultCache,
    get_cache,
)

from app.persistence.stores import (
    get_task_store,
    get_definition_store,
    get_feedback_store,
    get_calibration_token_store,
    get_graph_store,
    get_job_store,
)

__all__ = [
    # db
    "create_db_pool",
    "create_redis_client",
    "lifespan",
    "get_db",
    "get_redis",
    # cache
    "CACHE_KEY",
    "TTL_DETERMINISTIC",
    "make_cache_key",
    "ResultCache",
    "get_cache",
    # stores
    "get_task_store",
    "get_definition_store",
    "get_feedback_store",
    "get_calibration_token_store",
    "get_graph_store",
    "get_job_store",
]
