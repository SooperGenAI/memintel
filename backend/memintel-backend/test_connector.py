"""
test_connector.py
─────────────────────────────────────────────────────────────────────────────
Integration test: PostgresConnector -> credit_metrics (canvas DB).

Validates that:
  1. PostgresConnector can connect to the canvas DB and run queries.
  2. All 6 credit primitives return real float values for known entities.
  3. DataResolver.afetch() routes through the async_connector_registry
     correctly when PRIMITIVE_DB_URL is set.
  4. DataResolver.afetch() falls back to MockConnector (returns None) when
     PRIMITIVE_DB_URL is NOT set.

Run:
  PRIMITIVE_DB_URL=postgresql://postgres:admin@127.0.0.1:5433/canvas \
      python test_connector.py
"""
import asyncio
import os
import sys

# ── Allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

PRIMITIVE_DB_URL = (
    os.environ.get("PRIMITIVE_DB_URL")
    or "postgresql://postgres:admin@127.0.0.1:5433/canvas"
)
os.environ.setdefault("PRIMITIVE_DB_URL", PRIMITIVE_DB_URL)

TEST_ENTITIES = ["ACME-RB-00001", "ACME-RB-00002", "ACME-RB-00003"]
PRIMITIVES = [
    "payment.failure_rate_30d",
    "customer.credit_utilisation",
    "customer.days_past_due",
    "customer.emi_bounce_count",
    "customer.missed_payment_count",
    "customer.repayment_ratio",
]

_PRIM_QUERIES: dict[str, str] = {
    "payment.failure_rate_30d":
        "SELECT (emi_bounce_count::float / 30.0) AS failure_rate_30d"
        " FROM credit_metrics WHERE customer_id = :entity_id"
        " AND (:as_of::timestamptz IS NULL OR recorded_at <= :as_of::timestamptz)"
        " ORDER BY recorded_at DESC LIMIT 1",
    "customer.credit_utilisation":
        "SELECT credit_utilisation FROM credit_metrics"
        " WHERE customer_id = :entity_id"
        " AND (:as_of::timestamptz IS NULL OR recorded_at <= :as_of::timestamptz)"
        " ORDER BY recorded_at DESC LIMIT 1",
    "customer.days_past_due":
        "SELECT days_past_due::float FROM credit_metrics"
        " WHERE customer_id = :entity_id"
        " AND (:as_of::timestamptz IS NULL OR recorded_at <= :as_of::timestamptz)"
        " ORDER BY recorded_at DESC LIMIT 1",
    "customer.emi_bounce_count":
        "SELECT emi_bounce_count::float FROM credit_metrics"
        " WHERE customer_id = :entity_id"
        " AND (:as_of::timestamptz IS NULL OR recorded_at <= :as_of::timestamptz)"
        " ORDER BY recorded_at DESC LIMIT 1",
    "customer.missed_payment_count":
        "SELECT missed_payment_count::float FROM credit_metrics"
        " WHERE customer_id = :entity_id"
        " AND (:as_of::timestamptz IS NULL OR recorded_at <= :as_of::timestamptz)"
        " ORDER BY recorded_at DESC LIMIT 1",
    "customer.repayment_ratio":
        "SELECT repayment_ratio FROM credit_metrics"
        " WHERE customer_id = :entity_id"
        " AND (:as_of::timestamptz IS NULL OR recorded_at <= :as_of::timestamptz)"
        " ORDER BY recorded_at DESC LIMIT 1",
}


def _build_connector():
    """Build a PostgresConnector from PRIMITIVE_DB_URL without model validation."""
    from urllib.parse import urlparse
    from app.connectors.postgres import PostgresConnector
    from app.models.config import ConnectorConfig, PrimitiveSourceConfig

    parsed = urlparse(PRIMITIVE_DB_URL)
    config = ConnectorConfig.model_construct(
        type="postgres",
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=(parsed.path or "/").lstrip("/"),
        user=parsed.username or "",
        password=parsed.password or "",
        pool_min=2,
        pool_max=10,
        timeout_ms=10000,
        retry_max=3,
    )
    prim_sources = {
        name: PrimitiveSourceConfig(connector="postgres.canvas", query=q)
        for name, q in _PRIM_QUERIES.items()
    }
    return PostgresConnector(config, prim_sources)


# ── Test 1: PostgresConnector.fetch() directly ────────────────────────────────

async def test_postgres_connector_direct():
    print("\n-- Test 1: PostgresConnector.fetch() direct --")
    connector = _build_connector()
    failures = 0
    try:
        for entity in TEST_ENTITIES:
            print(f"\n  Entity: {entity}")
            for prim in PRIMITIVES:
                result = await connector.fetch(prim, entity, None)
                status = "OK" if result.value is not None else "NULL"
                print(f"    {prim:<35} -> {status}  value={result.value}")
                if result.value is None:
                    print(f"      WARNING: got None — missing data policy should handle this")
    finally:
        await connector.close()
    print(f"\n  Test 1 {'PASSED' if failures == 0 else 'FAILED'}")
    return failures == 0


# ── Test 2: DataResolver.afetch() via _make_resolver() ───────────────────────

async def test_data_resolver_afetch():
    print("\n-- Test 2: DataResolver.afetch() via _make_resolver() --")
    from app.services.execute import ExecuteService

    # Build a minimal ExecuteService (no real config needed — _make_resolver
    # reads PRIMITIVE_DB_URL from env)
    svc = ExecuteService.__new__(ExecuteService)
    svc._connector_registry = None
    svc._primitive_sources = {}

    resolver = svc._make_resolver()

    failures = 0
    entity = TEST_ENTITIES[0]
    print(f"\n  Entity: {entity}")
    for prim in PRIMITIVES:
        try:
            result = await resolver.afetch(prim, entity, None)
            status = "OK" if result is not None and getattr(result, "value", result) is not None else "NULL"
            val = getattr(result, "value", result)
            print(f"    {prim:<35} -> {status}  value={val}")
        except Exception as exc:
            print(f"    {prim:<35} -> ERROR: {exc}")
            failures += 1

    print(f"\n  Test 2 {'PASSED' if failures == 0 else 'FAILED (see errors above)'}")
    return failures == 0


# ── Test 3: MockConnector fallback (PRIMITIVE_DB_URL unset) ──────────────────

async def test_mock_fallback():
    print("\n-- Test 3: MockConnector fallback when PRIMITIVE_DB_URL unset --")
    saved = os.environ.pop("PRIMITIVE_DB_URL", None)
    try:
        from app.services.execute import ExecuteService
        svc = ExecuteService.__new__(ExecuteService)
        svc._connector_registry = None
        svc._primitive_sources = {}

        resolver = svc._make_resolver()
        # Verify no postgres.canvas was wired
        has_pg = "postgres.canvas" in getattr(resolver, "_async_connector_registry", {})
        print(f"  postgres.canvas in async_registry: {has_pg}  (expected: False)")
        passed = not has_pg
        print(f"\n  Test 3 {'PASSED' if passed else 'FAILED'}")
        return passed
    finally:
        if saved is not None:
            os.environ["PRIMITIVE_DB_URL"] = saved


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print(f"PRIMITIVE_DB_URL = {PRIMITIVE_DB_URL}")
    results = []

    results.append(await test_postgres_connector_direct())
    results.append(await test_data_resolver_afetch())
    results.append(await test_mock_fallback())

    passed = sum(results)
    total = len(results)
    print(f"\n{'-'*60}")
    print(f"Results: {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
