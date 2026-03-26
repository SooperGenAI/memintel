"""Add application_context table and context_version column to tasks.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-26
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── application_context ────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE application_context (
            context_id          TEXT        PRIMARY KEY,
            version             VARCHAR(10) NOT NULL,
            domain_json         JSONB       NOT NULL,
            behavioural_json    JSONB       NOT NULL,
            semantic_hints_json JSONB       NOT NULL,
            calibration_bias_json JSONB     DEFAULT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            is_active           BOOLEAN     NOT NULL DEFAULT TRUE
        )
    """)
    op.execute("CREATE INDEX idx_context_is_active ON application_context (is_active)")

    # ── tasks: add context_version column ─────────────────────────────────────
    op.execute("ALTER TABLE tasks ADD COLUMN context_version VARCHAR(10) DEFAULT NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE tasks DROP COLUMN IF EXISTS context_version")
    op.execute("DROP TABLE IF EXISTS application_context")
