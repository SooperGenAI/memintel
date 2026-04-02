"""Add unique constraint on decisions(condition_id, condition_version, entity_id, evaluated_at).

Fixes BUG-B3: POST /evaluate/full with the same request timestamp now writes
exactly one decision record. The idempotency guarantee:
  - When req.timestamp is provided, evaluate_full sets DecisionRecord.evaluated_at
    to the parsed timestamp value instead of relying on DB DEFAULT NOW().
  - This unique constraint prevents a second row with the same
    (condition_id, condition_version, entity_id, evaluated_at) from being inserted.
  - DecisionStore.record() uses ON CONFLICT DO NOTHING so the second write
    is silently discarded — no error, no duplicate.
  - When req.timestamp is absent (snapshot mode), evaluated_at = NOW() and
    no conflict is expected.

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-02
"""
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_decisions_condition_entity_timestamp'
            ) THEN
                ALTER TABLE decisions
                ADD CONSTRAINT uq_decisions_condition_entity_timestamp
                UNIQUE (condition_id, condition_version, entity_id, evaluated_at);
            END IF;
        END $$
    """)


def downgrade() -> None:
    op.execute(
        "ALTER TABLE decisions "
        "DROP CONSTRAINT IF EXISTS uq_decisions_condition_entity_timestamp"
    )
