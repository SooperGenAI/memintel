"""Add decisions table, concept_results table, and output_text column.

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-31
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── concept_results (not in earlier migrations) ────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS concept_results (
            concept_id    TEXT             NOT NULL,
            version       TEXT             NOT NULL,
            entity        TEXT             NOT NULL,
            value         DOUBLE PRECISION,
            output_type   TEXT             NOT NULL,
            output_text   TEXT             DEFAULT NULL,
            evaluated_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_concept_results_lookup
            ON concept_results (concept_id, entity, evaluated_at DESC)
    """)

    # Add output_text if table already existed without it
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'concept_results' AND column_name = 'output_text'
            ) THEN
                ALTER TABLE concept_results ADD COLUMN output_text TEXT DEFAULT NULL;
            END IF;
        END $$;
    """)

    # ── decisions ──────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id       UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            concept_id        TEXT        NOT NULL,
            concept_version   TEXT        NOT NULL,
            condition_id      TEXT        NOT NULL,
            condition_version TEXT        NOT NULL,
            entity_id         TEXT        NOT NULL,
            evaluated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            fired             BOOLEAN     NOT NULL,
            concept_value     DOUBLE PRECISION,
            threshold_applied JSONB,
            ir_hash           TEXT,
            input_primitives  JSONB,
            signal_errors     JSONB,
            reason            TEXT,
            action_ids_fired  TEXT[],
            dry_run           BOOLEAN     NOT NULL DEFAULT FALSE
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_decisions_entity_concept
            ON decisions (entity_id, concept_id, evaluated_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS decisions")
    op.execute("DROP TABLE IF EXISTS concept_results")
