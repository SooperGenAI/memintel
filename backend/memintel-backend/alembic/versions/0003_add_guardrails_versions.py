"""Add guardrails_versions table and guardrails_version column to tasks.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-27
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── guardrails_versions ────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE guardrails_versions (
            guardrails_id   UUID        PRIMARY KEY,
            version         VARCHAR(10) NOT NULL,
            guardrails_json JSONB       NOT NULL,
            change_note     TEXT        DEFAULT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
            source          VARCHAR(10) NOT NULL DEFAULT 'api'
        )
    """)
    op.execute(
        "CREATE INDEX idx_guardrails_is_active ON guardrails_versions (is_active)"
    )

    # ── tasks: add guardrails_version column ───────────────────────────────────
    op.execute(
        "ALTER TABLE tasks ADD COLUMN guardrails_version VARCHAR(10) DEFAULT NULL"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE tasks DROP COLUMN IF EXISTS guardrails_version")
    op.execute("DROP TABLE IF EXISTS guardrails_versions")
