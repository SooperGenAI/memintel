"""
test_cache.py
─────────────────────────────────────────────────────────────────────────────
Redis / runtime cache behaviour test.

Sections
────────
  1. ResultCache (runtime/cache.py) — in-memory, synchronous
     1a. Basic get/set/miss for deterministic keys
     1b. Snapshot keys (timestamp=None) are never stored
     1c. Cache key uniqueness: same concept+version, different entity → different slots
     1d. invalidate() removes a single key
     1e. clear() evicts everything

  2. ConceptExecutor cache integration — double-execution prevention
     2a. First call executes; second call (same timestamp) hits ResultCache, skips
         the connector entirely (fetch_call_count unchanged)
     2b. Snapshot mode (timestamp=None): both calls reach the connector because
         snapshot results are never cached

  3. Redis integration (persistence/cache.py) — async, Redis-backed
     3a. Connectivity: write, read-back, delete
     3b. SNAPSHOT sentinel: set() is a no-op; get() returns None without touching Redis
     3c. make_cache_key() format and URL-encoding of special characters
     3d. invalidate_concept(): SCAN + DELETE cleans up all keys for a (concept, version)
     3e. TTL_DETERMINISTIC is 1 year (31 536 000 s)

Run:
    python test_cache.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── allow imports from app/ ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from app.compiler.dag_builder import DAGBuilder
from app.models.concept import ConceptDefinition
from app.models.result import ConceptOutputType, ConceptResult
from app.runtime.cache import CacheKey, ResultCache
from app.runtime.data_resolver import DataResolver, MockConnector
from app.runtime.executor import ConceptExecutor

# ── colour helpers ────────────────────────────────────────────────────────────

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

_results: list[tuple[str, bool, str]] = []   # (label, passed, detail)


def _report(label: str, passed: bool, detail: str = "") -> None:
    tag = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    line = f"  [{tag}] {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    _results.append((label, passed, detail))


def _section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("-" * 60)


# ── fixture concept (from threshold_task.json) ────────────────────────────────

_FIXTURES_DIR = Path(__file__).parent / "app" / "llm" / "fixtures"

def _load_concept() -> ConceptDefinition:
    raw = json.loads((_FIXTURES_DIR / "threshold_task.json").read_text())
    return ConceptDefinition(**raw["concept"])


def _make_concept_result(
    concept_id: str = "org.churn_risk_score",
    version: str = "1.0",
    entity: str = "acme_corp",
    value: float = 0.75,
    timestamp: str | None = "2024-01-01T00:00:00Z",
) -> ConceptResult:
    return ConceptResult(
        value=value,
        type=ConceptOutputType.FLOAT,
        entity=entity,
        version=version,
        deterministic=(timestamp is not None),
        timestamp=timestamp,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ResultCache (runtime/cache.py) — in-memory, synchronous
# ═════════════════════════════════════════════════════════════════════════════

def run_section_1() -> None:
    _section("SECTION 1 — ResultCache (in-memory, runtime/cache.py)")

    # 1a. Basic get / set / miss ───────────────────────────────────────────────
    cache = ResultCache()
    result = _make_concept_result(entity="acme_corp")
    key_acme: CacheKey = ("org.churn_risk_score", "1.0", "acme_corp", "2024-01-01T00:00:00Z")
    key_small: CacheKey = ("org.churn_risk_score", "1.0", "small_co",  "2024-01-01T00:00:00Z")

    cache.set(key_acme, result)
    hit = cache.get(key_acme)
    _report(
        "1a. set then get — returns stored result",
        hit is not None and hit.value == 0.75 and hit.entity == "acme_corp",
        f"value={getattr(hit, 'value', None)}",
    )

    miss = cache.get(key_small)
    _report(
        "1a. get different entity — returns None",
        miss is None,
        f"got={miss!r}",
    )

    # 1b. Snapshot keys (timestamp=None) are never stored ─────────────────────
    cache2 = ResultCache()
    snap_result = _make_concept_result(timestamp=None)
    snap_key: CacheKey = ("org.churn_risk_score", "1.0", "acme_corp", None)

    cache2.set(snap_key, snap_result)   # must be a no-op
    snap_hit = cache2.get(snap_key)
    _report(
        "1b. snapshot key (timestamp=None) — never stored; get returns None",
        snap_hit is None and len(cache2) == 0,
        f"cache size={len(cache2)}, hit={snap_hit!r}",
    )

    # 1c. Key uniqueness: same concept+version, different entities ─────────────
    cache3 = ResultCache()
    r_acme  = _make_concept_result(entity="acme_corp", value=0.90)
    r_small = _make_concept_result(entity="small_co",  value=0.20)
    k_acme:  CacheKey = ("org.churn_risk_score", "1.0", "acme_corp", "2024-01-01T00:00:00Z")
    k_small: CacheKey = ("org.churn_risk_score", "1.0", "small_co",  "2024-01-01T00:00:00Z")

    cache3.set(k_acme,  r_acme)
    cache3.set(k_small, r_small)

    got_acme  = cache3.get(k_acme)
    got_small = cache3.get(k_small)
    _report(
        "1c. separate entities stored in independent slots",
        got_acme is not None and got_acme.value == 0.90
        and got_small is not None and got_small.value == 0.20,
        f"acme={getattr(got_acme,'value',None)}, small={getattr(got_small,'value',None)}",
    )

    # 1d. invalidate() removes one key, leaves others ─────────────────────────
    cache3.invalidate(k_acme)
    _report(
        "1d. invalidate(acme) — acme gone, small still present",
        cache3.get(k_acme) is None and cache3.get(k_small) is not None,
        f"after_invalidate cache size={len(cache3)}",
    )

    # 1e. clear() evicts all ───────────────────────────────────────────────────
    cache3.clear()
    _report(
        "1e. clear() — cache is empty",
        len(cache3) == 0,
        f"cache size={len(cache3)}",
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ConceptExecutor cache integration
# ═════════════════════════════════════════════════════════════════════════════

def run_section_2() -> None:
    _section("SECTION 2 — ConceptExecutor cache integration")

    concept = _load_concept()
    graph   = DAGBuilder().build_dag(concept)

    TS          = "2024-01-01T00:00:00Z"
    ENTITY      = "acme_corp"
    PRIMITIVE   = "engagement_score"
    PRIM_VALUE  = 5.0   # normalize(5.0)=0.833 > 0.80 threshold

    # 2a. Deterministic: second call hits ResultCache; connector not called again ──
    shared_cache    = ResultCache()
    shared_connector = MockConnector(
        data={(PRIMITIVE, ENTITY, TS): PRIM_VALUE}
    )

    executor = ConceptExecutor(result_cache=shared_cache)

    # First execution — must call the connector.
    resolver1 = DataResolver(shared_connector, backoff_base=0.0)
    result1   = executor.execute_graph(graph, ENTITY, resolver1, timestamp=TS)
    count_after_first = shared_connector.fetch_call_count

    # Second execution — same key; ResultCache should return immediately.
    resolver2 = DataResolver(shared_connector, backoff_base=0.0)
    result2   = executor.execute_graph(graph, ENTITY, resolver2, timestamp=TS)
    count_after_second = shared_connector.fetch_call_count

    _report(
        "2a. first call — connector was called at least once",
        count_after_first >= 1,
        f"fetch_call_count={count_after_first}",
    )
    _report(
        "2a. second call (same timestamp) — connector NOT called again (cache hit)",
        count_after_second == count_after_first,
        f"count before={count_after_first}, after={count_after_second}",
    )
    _report(
        "2a. both calls return identical result values",
        result1.value == result2.value and result1.entity == result2.entity,
        f"result1.value={result1.value:.4f}, result2.value={result2.value:.4f}",
    )
    _report(
        "2a. result is deterministic=True (timestamp was provided)",
        result1.deterministic is True,
        f"deterministic={result1.deterministic}",
    )

    # 2b. Snapshot mode (timestamp=None): both calls must reach the connector ──
    snap_cache     = ResultCache()
    snap_connector = MockConnector(
        data={(PRIMITIVE, ENTITY, None): PRIM_VALUE}
    )
    snap_executor  = ConceptExecutor(result_cache=snap_cache)

    resolver_s1 = DataResolver(snap_connector, backoff_base=0.0)
    snap_r1     = snap_executor.execute_graph(graph, ENTITY, resolver_s1, timestamp=None)
    count_snap1 = snap_connector.fetch_call_count

    resolver_s2 = DataResolver(snap_connector, backoff_base=0.0)
    snap_r2     = snap_executor.execute_graph(graph, ENTITY, resolver_s2, timestamp=None)
    count_snap2 = snap_connector.fetch_call_count

    _report(
        "2b. snapshot mode — second call also reaches connector (no caching)",
        count_snap2 > count_snap1,
        f"count after call1={count_snap1}, after call2={count_snap2}",
    )
    _report(
        "2b. snapshot result is deterministic=False",
        snap_r1.deterministic is False,
        f"deterministic={snap_r1.deterministic}",
    )
    _report(
        "2b. snapshot cache size stays 0 (nothing stored)",
        len(snap_cache) == 0,
        f"snap_cache size={len(snap_cache)}",
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Redis integration (persistence/cache.py) — async
# ═════════════════════════════════════════════════════════════════════════════

async def run_section_3_async() -> None:
    from app.persistence.cache import (
        ResultCache as RedisResultCache,
        TTL_DETERMINISTIC,
        make_cache_key,
    )
    import redis.asyncio as aioredis

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

    # 3a. Connectivity ─────────────────────────────────────────────────────────
    try:
        client = aioredis.from_url(redis_url, decode_responses=True)
        await client.ping()
        redis_reachable = True
    except Exception as exc:
        _report(
            "3a. Redis connectivity — PING",
            False,
            f"Could not reach Redis at {redis_url}: {exc}",
        )
        print(f"\n  {YELLOW}Skipping remaining Section 3 tests (Redis unreachable).{RESET}")
        return

    TEST_KEY = "test_cache_py:connectivity"
    await client.set(TEST_KEY, "hello", ex=30)
    val = await client.get(TEST_KEY)
    await client.delete(TEST_KEY)
    confirm_gone = await client.get(TEST_KEY) is None

    _report(
        "3a. Redis write / read-back / delete",
        val == "hello" and confirm_gone,
        f"read_back={val!r}, gone_after_delete={confirm_gone}",
    )

    # 3b. SNAPSHOT sentinel — no Redis I/O ─────────────────────────────────────
    redis_cache = RedisResultCache(client)
    snap_key = make_cache_key("org.test", "1.0", "entity_x", None)
    assert "SNAPSHOT" in snap_key, "make_cache_key did not embed SNAPSHOT sentinel"

    snap_result = _make_concept_result(timestamp=None)

    # set() must be a silent no-op
    await redis_cache.set(snap_key, snap_result)
    raw_in_redis = await client.get(snap_key)

    # get() must return None without touching Redis
    got = await redis_cache.get(snap_key)

    _report(
        "3b. SNAPSHOT key — set() writes nothing to Redis",
        raw_in_redis is None,
        f"redis.get(snap_key)={raw_in_redis!r}",
    )
    _report(
        "3b. SNAPSHOT key — get() returns None",
        got is None,
        f"got={got!r}",
    )

    # 3c. make_cache_key() format and URL-encoding ─────────────────────────────
    k1 = make_cache_key("org.churn_risk", "1.2", "user:123", "2024-01-15T09:00:00Z")
    k2 = make_cache_key("org.churn_risk", "1.2", "user_abc",  None)
    k3 = make_cache_key("org.churn_risk", "1.2", "user/a+b",  "2024-01-01T00:00:00Z")

    _report(
        "3c. key with colons in entity — URL-encoded",
        "user%3A123" in k1 and k1.startswith("result:org.churn_risk:1.2:"),
        f"key={k1}",
    )
    _report(
        "3c. key with timestamp=None — SNAPSHOT sentinel embedded",
        k2.endswith(":SNAPSHOT"),
        f"key={k2}",
    )
    _report(
        "3c. key with slashes and plus in entity — URL-encoded",
        "user%2Fa%2Bb" in k3,
        f"key={k3}",
    )

    # 3d. Full round-trip: set deterministic result, get it back ───────────────
    det_key = make_cache_key("org.test_rt", "1.0", "entity_rt", "2024-06-01T00:00:00Z")
    det_result = _make_concept_result(
        concept_id="org.test_rt", entity="entity_rt",
        value=0.42, timestamp="2024-06-01T00:00:00Z",
    )
    await redis_cache.set(det_key, det_result)
    got_rt = await redis_cache.get(det_key)

    _report(
        "3d. deterministic result — round-trip set/get via Redis",
        got_rt is not None and abs(got_rt.value - 0.42) < 1e-9,
        f"stored=0.42, retrieved={getattr(got_rt,'value',None)}",
    )

    # Check TTL was set (within a generous window)
    ttl = await client.ttl(det_key)
    _report(
        "3d. deterministic result — TTL set to ~1 year",
        TTL_DETERMINISTIC - 5 <= ttl <= TTL_DETERMINISTIC,
        f"TTL_DETERMINISTIC={TTL_DETERMINISTIC}s, actual_ttl={ttl}s",
    )

    # 3e. invalidate_concept() — SCAN + DELETE ─────────────────────────────────
    # Seed three keys for the same (concept, version).
    concept_id, version = "org.sweep_test", "2.0"
    sweep_keys = [
        make_cache_key(concept_id, version, f"entity_{i}", f"2024-0{i+1}-01T00:00:00Z")
        for i in range(3)
    ]
    sweep_results = [_make_concept_result(entity=f"entity_{i}", value=float(i)) for i in range(3)]
    for k, r in zip(sweep_keys, sweep_results):
        await redis_cache.set(k, r)

    deleted = await redis_cache.invalidate_concept(concept_id, version)
    remaining = [await client.get(k) for k in sweep_keys]

    _report(
        "3e. invalidate_concept() — deleted all 3 keys",
        deleted == 3 and all(v is None for v in remaining),
        f"deleted={deleted}, remaining_in_redis={sum(v is not None for v in remaining)}",
    )

    # 3f. TTL_DETERMINISTIC constant value ─────────────────────────────────────
    _report(
        "3f. TTL_DETERMINISTIC is 1 year (31 536 000 s)",
        TTL_DETERMINISTIC == 60 * 60 * 24 * 365,
        f"TTL_DETERMINISTIC={TTL_DETERMINISTIC}",
    )

    # Cleanup
    await client.delete(det_key)
    await client.aclose()


def run_section_3() -> None:
    _section("SECTION 3 — Redis integration (persistence/cache.py)")
    asyncio.run(run_section_3_async())


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{BOLD}Memintel cache behaviour test{RESET}")
    print("=" * 60)

    run_section_1()
    run_section_2()
    run_section_3()

    # ── Summary ───────────────────────────────────────────────────────────────
    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'=' * 60}")
    print(f"{BOLD}Summary: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({RED}{failed} failed{RESET}{BOLD})", end="")
    print(RESET)

    if failed:
        print(f"\n{RED}Failed tests:{RESET}")
        for label, ok, detail in _results:
            if not ok:
                print(f"  ✗  {label}")
                if detail:
                    print(f"       {detail}")
        sys.exit(1)
    else:
        print(f"{GREEN}All tests passed.{RESET}")


if __name__ == "__main__":
    main()
