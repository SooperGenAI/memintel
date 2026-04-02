"""
tests/performance/test_indexes.py
──────────────────────────────────────────────────────────────────────────────
Index existence and EXPLAIN plan tests.

Index existence tests (14): verify each expected index is present in
pg_indexes using the indexdef column.

EXPLAIN plan tests (4): with 5 000+ seed rows, verify that key query
patterns use an index scan rather than a sequential scan.

All tests are marked @pytest.mark.performance and skip gracefully if the
database is unavailable.
"""
from __future__ import annotations

import pytest

# ── helpers ────────────────────────────────────────────────────────────────────

async def _index_exists(pool, table: str, index_name: str) -> bool:
    """Return True if an index with the given name exists on the given table."""
    row = await pool.fetchrow(
        """
        SELECT COUNT(*) AS cnt
        FROM pg_indexes
        WHERE tablename = $1 AND indexname = $2
        """,
        table,
        index_name,
    )
    return row["cnt"] > 0


async def _explain(pool, query: str, *args) -> str:
    """Run EXPLAIN on a query and return the full plan as a single string."""
    rows = await pool.fetch(f"EXPLAIN {query}", *args)
    return "\n".join(r[0] for r in rows)


# ── Index existence tests ──────────────────────────────────────────────────────

@pytest.mark.performance
class TestIndexExistence:
    """Verify every expected index is present after all migrations."""

    # decisions ----------------------------------------------------------------

    def test_decisions_entity_concept_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "decisions", "idx_decisions_entity_concept")
        ), "Missing index: idx_decisions_entity_concept"

    def test_decisions_condition_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "decisions", "idx_decisions_condition")
        ), "Missing index: idx_decisions_condition (added in 0008)"

    def test_decisions_evaluated_at_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "decisions", "idx_decisions_evaluated_at")
        ), "Missing index: idx_decisions_evaluated_at (added in 0008)"

    # concept_results ----------------------------------------------------------

    def test_concept_results_lookup_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "concept_results", "idx_concept_results_lookup")
        ), "Missing index: idx_concept_results_lookup"

    def test_concept_results_evaluated_at_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "concept_results", "idx_concept_results_evaluated_at")
        ), "Missing index: idx_concept_results_evaluated_at (added in 0008)"

    # feedback_records ---------------------------------------------------------

    def test_feedback_condition_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "feedback_records", "idx_feedback_condition")
        ), "Missing index: idx_feedback_condition"

    def test_feedback_entity_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "feedback_records", "idx_feedback_entity")
        ), "Missing index: idx_feedback_entity"

    def test_feedback_recorded_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "feedback_records", "idx_feedback_recorded")
        ), "Missing index: idx_feedback_recorded"

    def test_feedback_lookup_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "feedback_records", "idx_feedback_lookup")
        ), "Missing index: idx_feedback_lookup"

    # definitions --------------------------------------------------------------

    def test_definitions_type_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "definitions", "idx_definitions_type")
        ), "Missing index: idx_definitions_type"

    def test_definitions_namespace_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "definitions", "idx_definitions_namespace")
        ), "Missing index: idx_definitions_namespace"

    def test_definitions_namespace_type_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "definitions", "idx_definitions_namespace_type")
        ), "Missing index: idx_definitions_namespace_type (added in 0008)"

    def test_definitions_created_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "definitions", "idx_definitions_created")
        ), "Missing index: idx_definitions_created"

    # tasks --------------------------------------------------------------------

    def test_tasks_status_index(self, db_pool, _loop):
        assert _loop.run_until_complete(
            _index_exists(db_pool, "tasks", "idx_tasks_status")
        ), "Missing index: idx_tasks_status"


# ── EXPLAIN plan tests ─────────────────────────────────────────────────────────

@pytest.mark.performance
class TestExplainPlans:
    """
    Verify that high-volume query patterns use index scans, not sequential scans.

    Each test seeds 5 000+ rows via the session-scoped `seed_explain_data`
    fixture, then runs EXPLAIN and asserts the plan contains "Index Scan"
    (matches both "Index Scan" and "Bitmap Index Scan").
    """

    def test_decisions_by_entity_concept_uses_index(
        self, db_pool, _loop, seed_explain_data
    ):
        """
        DecisionStore.list_for_entity() filters on (entity_id, concept_id)
        ORDER BY evaluated_at DESC.  Should hit idx_decisions_entity_concept.
        """
        plan = _loop.run_until_complete(
            _explain(
                db_pool,
                "SELECT * FROM decisions "
                "WHERE entity_id = $1 AND concept_id = $2 "
                "ORDER BY evaluated_at DESC LIMIT 100",
                "entity-0",
                "concept-0",
            )
        )
        assert "Index Scan" in plan, (
            f"Expected index scan on decisions (entity_id, concept_id); got:\n{plan}"
        )
        assert "Seq Scan on decisions" not in plan, (
            f"Unexpected sequential scan on decisions:\n{plan}"
        )

    def test_decisions_by_condition_uses_index(
        self, db_pool, _loop, seed_explain_data
    ):
        """
        Explanation and audit queries filter decisions by (condition_id, condition_version).
        Should hit idx_decisions_condition (added in 0008).
        """
        plan = _loop.run_until_complete(
            _explain(
                db_pool,
                "SELECT * FROM decisions "
                "WHERE condition_id = $1 AND condition_version = $2 "
                "LIMIT 100",
                "condition-0",
                "v1",
            )
        )
        assert "Index Scan" in plan, (
            f"Expected index scan on decisions (condition_id, condition_version); got:\n{plan}"
        )
        assert "Seq Scan on decisions" not in plan, (
            f"Unexpected sequential scan on decisions:\n{plan}"
        )

    def test_concept_results_by_concept_entity_uses_index(
        self, db_pool, _loop, seed_explain_data
    ):
        """
        ConceptResultStore.fetch_history() filters on (concept_id, entity)
        ORDER BY evaluated_at DESC.  Should hit idx_concept_results_lookup.
        """
        plan = _loop.run_until_complete(
            _explain(
                db_pool,
                "SELECT * FROM concept_results "
                "WHERE concept_id = $1 AND entity = $2 "
                "ORDER BY evaluated_at DESC LIMIT 10",
                "concept-0",
                "entity-0",
            )
        )
        assert "Index Scan" in plan, (
            f"Expected index scan on concept_results (concept_id, entity); got:\n{plan}"
        )
        assert "Seq Scan on concept_results" not in plan, (
            f"Unexpected sequential scan on concept_results:\n{plan}"
        )

    def test_definitions_by_namespace_type_uses_index(
        self, db_pool, _loop, seed_explain_data
    ):
        """
        DefinitionStore.list() and list_actions() filter on both namespace AND
        definition_type.  Should hit idx_definitions_namespace_type (added in 0008).
        """
        plan = _loop.run_until_complete(
            _explain(
                db_pool,
                "SELECT * FROM definitions "
                "WHERE namespace = $1 AND definition_type = $2 "
                "ORDER BY created_at DESC LIMIT 50",
                "org",
                "concept",
            )
        )
        assert "Index Scan" in plan, (
            f"Expected index scan on definitions (namespace, definition_type); got:\n{plan}"
        )
        assert "Seq Scan on definitions" not in plan, (
            f"Unexpected sequential scan on definitions:\n{plan}"
        )
