"""Add missing indexes for production query patterns.

Cross-reference of store SQL queries against existing indexes revealed four
gaps where high-volume tables are queried on un-indexed columns.

Missing indexes added
─────────────────────
1. decisions (condition_id, condition_version)
   Rationale: explanation and audit queries filter decisions by condition.
   Currently no index exists on these columns; the only decisions index is
   (entity_id, concept_id, evaluated_at DESC).

2. decisions (evaluated_at DESC)
   Rationale: time-range queries on the decisions table (e.g. "all decisions
   in the last hour") require a standalone evaluated_at index. The existing
   composite covers evaluated_at only as its third column — unusable for
   ORDER BY or range scans without the leading (entity_id, concept_id) prefix.

3. concept_results (evaluated_at DESC)
   Rationale: same reasoning as decisions.evaluated_at. The composite
   (concept_id, entity, evaluated_at DESC) covers evaluated_at as third
   column only; standalone time scans would full-scan the table.

4. definitions (namespace, definition_type)
   Rationale: DefinitionStore.list(), list_actions(), and count_actions()
   all filter on BOTH namespace AND definition_type. Separate single-column
   indexes exist (idx_definitions_namespace and idx_definitions_type) but
   the planner must either pick one and filter, or bitmap-combine two index
   scans. A single composite covering both enables a single efficient scan.

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-02
"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── decisions: condition lookup ────────────────────────────────────────────
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_decisions_condition "
        "ON decisions (condition_id, condition_version)"
    )

    # ── decisions: time-range queries ─────────────────────────────────────────
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_decisions_evaluated_at "
        "ON decisions (evaluated_at DESC)"
    )

    # ── concept_results: time-range queries ───────────────────────────────────
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_concept_results_evaluated_at "
        "ON concept_results (evaluated_at DESC)"
    )

    # ── definitions: combined namespace + type filter ─────────────────────────
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_definitions_namespace_type "
        "ON definitions (namespace, definition_type)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_definitions_namespace_type")
    op.execute("DROP INDEX IF EXISTS idx_concept_results_evaluated_at")
    op.execute("DROP INDEX IF EXISTS idx_decisions_evaluated_at")
    op.execute("DROP INDEX IF EXISTS idx_decisions_condition")
