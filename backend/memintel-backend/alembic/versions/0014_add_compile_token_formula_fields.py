"""Add formula_summary and signal_bindings columns to compile_tokens.

These fields carry the Step 3 DAG Construction output (formula description
and signal role assignments) forward from Phase 1 (compile) to Phase 2
(register), so they can be persisted in the definitions.body JSONB column.

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-11
"""
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE compile_tokens
            ADD COLUMN formula_summary TEXT,
            ADD COLUMN signal_bindings  JSONB
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE compile_tokens
            DROP COLUMN IF EXISTS formula_summary,
            DROP COLUMN IF EXISTS signal_bindings
    """)
