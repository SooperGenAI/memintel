"""Add output_type column to compile_tokens for V7 two-phase concept registration.

POST /concepts/register (M-4) returns RegisterConceptResponse which includes
output_type. The output_type is declared at compile time (POST /concepts/compile)
and must be stored on the token so that Phase 2 can include it in the response
without requiring a second LLM call.

Revision ID: 0012
Revises: 0011
Create Date: 2026-04-05
"""
from alembic import op
import sqlalchemy as sa

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE compile_tokens
        ADD COLUMN output_type TEXT NOT NULL DEFAULT ''
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE compile_tokens
        DROP COLUMN output_type
    """)
