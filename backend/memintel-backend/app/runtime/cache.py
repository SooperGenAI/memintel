"""
app/runtime/cache.py
──────────────────────────────────────────────────────────────────────────────
In-memory result cache for concept execution results.

Cache key contract (from py-instructions.md and core-spec.md §1C):
  Key MUST be: (concept_id, version, entity, timestamp)
  A None timestamp is a DIFFERENT key from any specific timestamp.
  Snapshot results (timestamp=None) MUST NOT be cached at all — not even
  transiently. The executor enforces this by calling set() only when timestamp
  is not None.

TTL semantics:
  Deterministic (timestamp set) → cached indefinitely.
    Same (concept_id, version, entity, timestamp) always produces the same
    result — caching forever is safe and correct.
  Snapshot (no timestamp)       → never persisted in this cache.
    set() is a no-op for None-timestamp keys; get() always returns None.

This implementation is in-memory (dict-backed). In production, back with Redis
or Memcached. The interface is the same; swap the _store backend.

No LLM in the cache path — this is a deterministic lookup/store only.
"""
from __future__ import annotations

from typing import NamedTuple

from app.models.result import ConceptResult

# Cache key: (concept_id, version, entity, timestamp)
# timestamp=None is a valid but never-cached key (snapshot mode).
CacheKey = tuple[str, str, str, str | None]


class ResultCache:
    """
    Thread-safe (GIL-protected) in-memory cache for ConceptResult objects.

    Key invariants enforced here:
      - set() is a no-op when key[3] (timestamp) is None.
      - get() always returns None for None-timestamp keys (snapshot mode).
      - Two keys (id, ver, entity, "2024-01-01T00:00:00Z") and
        (id, ver, entity, None) are completely independent entries.

    The cache is intentionally not LRU-bounded in this implementation.
    Production deployments should add eviction via TTL or capacity limits.
    """

    def __init__(self) -> None:
        self._store: dict[CacheKey, ConceptResult] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, key: CacheKey) -> ConceptResult | None:
        """
        Return the cached result for ``key``, or None on a miss.

        Always returns None for snapshot keys (timestamp=None) because snapshot
        results are never stored.
        """
        if key[3] is None:
            return None  # snapshot results are never cached
        return self._store.get(key)

    def set(self, key: CacheKey, result: ConceptResult) -> None:
        """
        Store ``result`` under ``key``.

        No-op when key[3] (timestamp) is None — snapshot results must not
        be persisted beyond the current request context.
        """
        if key[3] is None:
            return  # DETERMINISM CONTRACT: never cache snapshot results
        self._store[key] = result

    def invalidate(self, key: CacheKey) -> None:
        """Remove ``key`` from the cache if present; no-op otherwise."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Evict all entries. Used in tests and on graceful shutdown."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
