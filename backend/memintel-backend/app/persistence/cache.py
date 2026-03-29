"""
app/persistence/cache.py
──────────────────────────────────────────────────────────────────────────────
Redis-backed result cache for concept execution outputs (ConceptResult / Rₜ).

This is a performance layer, not a store of record. The authoritative
source of truth for any concept result is always re-execution via the runtime.

Cache key contract
──────────────────
Key format: "result:{concept_id}:{version}:{safe_entity}:{safe_ts}"

  concept_id and version are namespace-qualified strings that contain only
  dots, hyphens, and alphanumerics — safe for raw interpolation.

  entity and timestamp may contain colons, slashes, plus signs, and other
  characters that would create ambiguous keys if interpolated directly.
  Both are URL-percent-encoded before embedding.

  timestamp=None → sentinel literal "SNAPSHOT" (never URL-encoded).

  Examples (after encoding):
    entity='user:123', ts='2024-01-15T09:00:00Z'
    → "result:org.churn_risk:1.2:user%3A123:2024-01-15T09%3A00%3A00Z"

    entity='user_abc', ts=None
    → "result:org.churn_risk:1.2:user_abc:SNAPSHOT"

SNAPSHOT rules (CRITICAL)
─────────────────────────
SNAPSHOT keys (timestamp=None) MUST NEVER be written to Redis.
  - set() is a silent no-op when the key contains "SNAPSHOT".
  - get() always returns None for SNAPSHOT keys without touching Redis.

Rationale: snapshot results reflect current-state at call time. Caching them
  would cause stale reads on the next call. The determinism contract requires
  that snapshot execution is always live.

TTL rules
─────────
  Deterministic (timestamp supplied): TTL_DETERMINISTIC = 1 year.
    Same inputs always produce the same output, so this entry is safe to hold
    indefinitely. A long TTL (rather than no expiry) allows Redis to apply
    maxmemory eviction policies without being blocked by non-expiring keys.

  Snapshot (timestamp=None): never written; TTL is irrelevant.

CACHE_KEY type
──────────────
CACHE_KEY = tuple[str, str, str, str | None]
  (concept_id, version, entity, timestamp)

  Pass a CACHE_KEY to make_cache_key() to get the Redis string key.
  The tuple form is used internally by the runtime and execute_graph path;
  the string form is what ResultCache stores and retrieves.

FastAPI dependency
──────────────────
get_cache() → yields a ResultCache bound to request.app.state.redis.

Usage in routes / services:

    from fastapi import Depends
    from app.persistence.cache import ResultCache, get_cache

    @router.post("/execute")
    async def execute(
        req: ExecuteRequest,
        cache: ResultCache = Depends(get_cache),
    ):
        key = make_cache_key(req.id, req.version, req.entity, req.timestamp)
        cached = await cache.get(key)
        if cached:
            return cached
        result = await runtime.execute(...)
        await cache.set(key, result)
        return result
"""
from __future__ import annotations

import logging
from urllib.parse import quote

import redis.asyncio as aioredis
from fastapi import Request

from app.models.result import ConceptResult

log = logging.getLogger(__name__)

# ── Type alias ────────────────────────────────────────────────────────────────

#: Cache key tuple: (concept_id, version, entity, timestamp | None).
#: Pass to make_cache_key() to get the Redis string key.
CACHE_KEY = tuple[str, str, str, str | None]

# ── TTL constant ──────────────────────────────────────────────────────────────

#: TTL for deterministic results (timestamp supplied). 1 year in seconds.
#: A long TTL allows Redis memory management without blocking on non-expiring keys.
TTL_DETERMINISTIC: int = 60 * 60 * 24 * 365

#: Sentinel used in the key when timestamp is None (snapshot mode).
#: Keys containing this string are never written to or read from Redis.
_SNAPSHOT_SENTINEL: str = "SNAPSHOT"


# ── Cache key builder ─────────────────────────────────────────────────────────

def make_cache_key(
    concept_id: str,
    version: str,
    entity: str,
    timestamp: str | None,
) -> str:
    """
    Build the Redis cache key for a concept execution result.

    entity and timestamp are URL-percent-encoded to prevent key collisions
    caused by special characters (colons, slashes, plus signs, etc.).

    timestamp=None produces the SNAPSHOT sentinel, which is never stored.

    Returns a string of the form:
        "result:{concept_id}:{version}:{safe_entity}:{safe_ts}"
    """
    safe_entity = quote(entity, safe="")
    safe_ts = (
        quote(timestamp, safe="")
        if timestamp is not None
        else _SNAPSHOT_SENTINEL
    )
    return f"result:{concept_id}:{version}:{safe_entity}:{safe_ts}"


# ── ResultCache ───────────────────────────────────────────────────────────────

class ResultCache:
    """
    Redis-backed cache for ConceptResult (Rₜ) objects.

    Constructed with an aioredis.Redis client. The client is shared across
    requests — it is created once at startup and stored on app.state.redis.

    SNAPSHOT keys are silently ignored in both get() and set(). Callers do
    not need to check for snapshot mode before calling — the cache handles it.
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    # ── get ───────────────────────────────────────────────────────────────────

    async def get(self, key: str) -> ConceptResult | None:
        """
        Return a cached ConceptResult, or None if not found.

        Returns None immediately (without touching Redis) for SNAPSHOT keys.
        Returns None on a cache miss.
        """
        if _SNAPSHOT_SENTINEL in key:
            return None

        raw = await self._redis.get(key)
        if raw is None:
            return None

        try:
            return ConceptResult.model_validate_json(raw)
        except Exception:
            log.warning(
                "cache_deserialise_error",
                extra={"key": key},
                exc_info=True,
            )
            return None

    # ── set ───────────────────────────────────────────────────────────────────

    async def set(self, key: str, result: ConceptResult) -> None:
        """
        Store a ConceptResult under the given key with a 1-year TTL.

        Silent no-op for SNAPSHOT keys — snapshot results must never be
        persisted to Redis (they are current-state and would go stale).
        """
        if _SNAPSHOT_SENTINEL in key:
            return

        serialised = result.model_dump_json()
        await self._redis.setex(key, TTL_DETERMINISTIC, serialised)
        log.debug("cache_set", extra={"key": key})

    # ── invalidate ────────────────────────────────────────────────────────────

    async def invalidate(self, key: str) -> None:
        """
        Delete a single cache entry.

        No-op for SNAPSHOT keys (they are never in Redis).
        No-op if the key does not exist.
        """
        if _SNAPSHOT_SENTINEL in key:
            return

        deleted = await self._redis.delete(key)
        if deleted:
            log.info("cache_invalidated", extra={"key": key})

    # ── invalidate_concept ────────────────────────────────────────────────────

    async def invalidate_concept(self, concept_id: str, version: str) -> int:
        """
        Delete all cached results for a (concept_id, version) pair.

        Uses SCAN + DELETE to avoid blocking Redis with a single large DEL.
        Returns the number of keys deleted.

        Use after a concept version is deprecated or its graph is recompiled
        to force fresh execution on the next request.

        Note: SCAN is non-atomic. In a clustered Redis setup this should be
        called on all cluster nodes. For single-node Redis (the typical
        Memintel deployment) this is safe.
        """
        pattern = f"result:{concept_id}:{version}:*"
        deleted_count = 0
        cursor = 0

        while True:
            cursor, keys = await self._redis.scan(
                cursor, match=pattern, count=100
            )
            if keys:
                await self._redis.delete(*keys)
                deleted_count += len(keys)
            if cursor == 0:
                break

        if deleted_count:
            log.info(
                "cache_invalidated_concept",
                extra={
                    "concept_id": concept_id,
                    "version": version,
                    "keys_deleted": deleted_count,
                },
            )
        return deleted_count


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_cache(request: Request) -> ResultCache:
    """
    FastAPI dependency — returns a ResultCache bound to app.state.redis.

    The Redis client is shared across all requests. The ResultCache wrapper
    is lightweight (no I/O on construction) so a new instance per-request
    is fine.

    Inject via Depends(get_cache):

        cache: ResultCache = Depends(get_cache)
    """
    return ResultCache(request.app.state.redis)
