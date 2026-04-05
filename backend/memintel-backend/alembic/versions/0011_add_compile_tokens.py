"""Add compile_tokens table for V7 two-phase concept compilation.

compile_tokens stores short-lived single-use tokens issued by
POST /concepts/compile.  Each token is consumed atomically by
POST /concepts/register via a conditional UPDATE...WHERE used = FALSE,
guaranteeing exactly-once redemption even under concurrent requests.

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-05
"""
from alembic import op
import sqlalchemy as sa

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE compile_tokens (
            token_id     UUID        NOT NULL DEFAULT gen_random_uuid(),
            token_string TEXT        NOT NULL,
            identifier   TEXT        NOT NULL,
            ir_hash      TEXT        NOT NULL,
            expires_at   TIMESTAMPTZ NOT NULL,
            used         BOOLEAN     NOT NULL DEFAULT FALSE,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT compile_tokens_pkey        PRIMARY KEY (token_id),
            CONSTRAINT compile_tokens_token_string_key UNIQUE  (token_string)
        )
    """)

    op.execute("""
        CREATE INDEX compile_tokens_token_string_idx
            ON compile_tokens (token_string)
    """)

    op.execute("""
        CREATE INDEX compile_tokens_expires_at_idx
            ON compile_tokens (expires_at)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS compile_tokens")
