"""Initial schema — creates all six tables.

Revision ID: 0001
Revises:
Create Date: 2026-03-23
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── tasks ─────────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE tasks (
            task_id           TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::text,
            intent            TEXT        NOT NULL,
            concept_id        TEXT        NOT NULL,
            concept_version   TEXT        NOT NULL,
            condition_id      TEXT        NOT NULL,
            condition_version TEXT        NOT NULL,
            action_id         TEXT        NOT NULL,
            action_version    TEXT        NOT NULL,
            entity_scope      TEXT        NOT NULL,
            delivery          JSONB       NOT NULL,
            status            TEXT        NOT NULL DEFAULT 'active'
                                  CHECK (status IN ('active', 'paused', 'deleted', 'preview')),
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_triggered_at TIMESTAMPTZ          DEFAULT NULL,
            version           INTEGER     NOT NULL DEFAULT 1
        )
    """)
    op.execute("CREATE INDEX idx_tasks_status     ON tasks (status)")
    op.execute("CREATE INDEX idx_tasks_condition  ON tasks (condition_id, condition_version)")
    op.execute("CREATE INDEX idx_tasks_concept    ON tasks (concept_id, concept_version)")
    op.execute("CREATE INDEX idx_tasks_created_at ON tasks (created_at DESC)")

    # ── definitions ───────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE definitions (
            id              BIGSERIAL   PRIMARY KEY,
            definition_id   TEXT        NOT NULL,
            version         TEXT        NOT NULL,
            definition_type TEXT        NOT NULL
                                CHECK (definition_type IN ('concept', 'condition', 'action', 'primitive')),
            namespace       TEXT        NOT NULL
                                CHECK (namespace IN ('personal', 'team', 'org', 'global')),
            body            JSONB       NOT NULL,
            meaning_hash    TEXT                DEFAULT NULL,
            ir_hash         TEXT                DEFAULT NULL,
            deprecated      BOOLEAN     NOT NULL DEFAULT FALSE,
            deprecated_at   TIMESTAMPTZ          DEFAULT NULL,
            replacement_version TEXT             DEFAULT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_definition_version UNIQUE (definition_id, version)
        )
    """)
    op.execute("CREATE INDEX idx_definitions_id        ON definitions (definition_id)")
    op.execute("CREATE INDEX idx_definitions_type      ON definitions (definition_type)")
    op.execute("CREATE INDEX idx_definitions_namespace ON definitions (namespace)")
    op.execute("CREATE INDEX idx_definitions_hash      ON definitions (meaning_hash) WHERE meaning_hash IS NOT NULL")
    op.execute("CREATE INDEX idx_definitions_created   ON definitions (created_at DESC)")

    # ── feedback_records ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE feedback_records (
            feedback_id        TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::text,
            condition_id       TEXT        NOT NULL,
            condition_version  TEXT        NOT NULL,
            entity             TEXT        NOT NULL,
            decision_timestamp TIMESTAMPTZ NOT NULL,
            feedback           TEXT        NOT NULL
                                   CHECK (feedback IN ('false_positive', 'false_negative', 'correct')),
            note               TEXT                 DEFAULT NULL,
            recorded_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_feedback_decision
                UNIQUE (condition_id, condition_version, entity, decision_timestamp)
        )
    """)
    op.execute("CREATE INDEX idx_feedback_condition ON feedback_records (condition_id, condition_version)")
    op.execute("CREATE INDEX idx_feedback_entity    ON feedback_records (entity)")
    op.execute("CREATE INDEX idx_feedback_recorded  ON feedback_records (recorded_at DESC)")
    op.execute("""
        CREATE INDEX idx_feedback_lookup
            ON feedback_records (condition_id, condition_version, entity, decision_timestamp)
    """)

    # ── calibration_tokens ────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE calibration_tokens (
            id                  BIGSERIAL   PRIMARY KEY,
            token_string        TEXT        NOT NULL UNIQUE,
            condition_id        TEXT        NOT NULL,
            condition_version   TEXT        NOT NULL,
            recommended_params  JSONB       NOT NULL,
            expires_at          TIMESTAMPTZ NOT NULL,
            used_at             TIMESTAMPTZ          DEFAULT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_tokens_string    ON calibration_tokens (token_string)")
    op.execute("CREATE INDEX idx_tokens_expires   ON calibration_tokens (expires_at)")
    op.execute("CREATE INDEX idx_tokens_condition ON calibration_tokens (condition_id, condition_version)")

    # ── execution_graphs ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE execution_graphs (
            graph_id    TEXT        PRIMARY KEY,
            concept_id  TEXT        NOT NULL,
            version     TEXT        NOT NULL,
            ir_hash     TEXT        NOT NULL,
            graph_body  JSONB       NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_graph_concept_version UNIQUE (concept_id, version)
        )
    """)
    op.execute("CREATE INDEX idx_graphs_concept ON execution_graphs (concept_id, version)")
    op.execute("CREATE INDEX idx_graphs_ir_hash ON execution_graphs (ir_hash)")

    # ── jobs ──────────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE jobs (
            job_id          TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::text,
            job_type        TEXT        NOT NULL DEFAULT 'execute',
            status          TEXT        NOT NULL DEFAULT 'queued'
                                CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
            request_body    JSONB       NOT NULL,
            result_body     JSONB                DEFAULT NULL,
            error_body      JSONB                DEFAULT NULL,
            poll_interval_s INTEGER     NOT NULL DEFAULT 2,
            enqueued_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at      TIMESTAMPTZ          DEFAULT NULL,
            completed_at    TIMESTAMPTZ          DEFAULT NULL,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_jobs_status   ON jobs (status)")
    op.execute("CREATE INDEX idx_jobs_enqueued ON jobs (enqueued_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS jobs")
    op.execute("DROP TABLE IF EXISTS execution_graphs")
    op.execute("DROP TABLE IF EXISTS calibration_tokens")
    op.execute("DROP TABLE IF EXISTS feedback_records")
    op.execute("DROP TABLE IF EXISTS definitions")
    op.execute("DROP TABLE IF EXISTS tasks")
