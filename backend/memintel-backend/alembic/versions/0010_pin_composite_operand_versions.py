"""Pin composite condition operand versions in definition JSONB bodies.

Before this migration, composite conditions stored operands as plain
condition_id strings:

    "operands": ["org.high_churn_risk", "org.high_ltv_customer"]

After this migration, each operand is stored as a version-pinned OperandRef
dict so that the evaluation path never silently resolves to the latest version:

    "operands": [
        {"condition_id": "org.high_churn_risk",   "condition_version": "1.0"},
        {"condition_id": "org.high_ltv_customer",  "condition_version": "1.0"},
    ]

For each string operand the migration resolves the latest registered version
(by created_at DESC) of that condition_id from the definitions table.  If no
version is found the operand is replaced with a fallback OperandRef using
condition_version="1.0" to keep the body parseable.

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-02
"""
from alembic import op
import sqlalchemy as sa

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Convert string operands in composite condition bodies to OperandRef dicts."""
    op.execute("""
        DO $$
        DECLARE
            rec     RECORD;
            body    JSONB;
            strat   JSONB;
            params  JSONB;
            raw_ops JSONB;
            op_elem JSONB;
            op_id   TEXT;
            op_ver  TEXT;
            pinned  JSONB;
            i       INT;
        BEGIN
            FOR rec IN
                SELECT definition_id, version, body AS raw_body
                FROM definitions
                WHERE definition_type = 'condition'
                  AND body -> 'strategy' ->> 'type' = 'composite'
            LOOP
                body   := rec.raw_body;
                strat  := body -> 'strategy';
                params := strat -> 'params';
                raw_ops := params -> 'operands';

                -- Only migrate rows whose operands are plain strings (not already OperandRef dicts).
                IF raw_ops IS NULL OR jsonb_typeof(raw_ops) <> 'array' THEN
                    CONTINUE;
                END IF;

                -- Check whether first element is already an object (already migrated).
                IF jsonb_array_length(raw_ops) > 0
                   AND jsonb_typeof(raw_ops -> 0) = 'object' THEN
                    CONTINUE;
                END IF;

                -- Build pinned operands array.
                pinned := '[]'::JSONB;
                FOR i IN 0 .. jsonb_array_length(raw_ops) - 1 LOOP
                    op_elem := raw_ops -> i;
                    IF jsonb_typeof(op_elem) = 'object' THEN
                        -- Already a dict — keep as-is.
                        pinned := pinned || jsonb_build_array(op_elem);
                    ELSE
                        op_id := op_elem #>> '{}';
                        -- Resolve latest version of this condition_id.
                        SELECT d.version INTO op_ver
                        FROM definitions d
                        WHERE d.definition_id = op_id
                          AND d.definition_type = 'condition'
                        ORDER BY d.created_at DESC
                        LIMIT 1;

                        IF op_ver IS NULL THEN
                            op_ver := '1.0';  -- fallback for orphaned references
                        END IF;

                        pinned := pinned || jsonb_build_array(
                            jsonb_build_object(
                                'condition_id',      op_id,
                                'condition_version', op_ver
                            )
                        );
                    END IF;
                END LOOP;

                -- Write back the updated body.
                UPDATE definitions
                SET body = body
                        || jsonb_build_object(
                               'strategy',
                               strat || jsonb_build_object(
                                   'params',
                                   params || jsonb_build_object('operands', pinned)
                               )
                           )
                WHERE definition_id = rec.definition_id
                  AND version       = rec.version
                  AND definition_type = 'condition';
            END LOOP;
        END $$
    """)


def downgrade() -> None:
    """Convert OperandRef dicts back to plain condition_id strings."""
    op.execute("""
        DO $$
        DECLARE
            rec     RECORD;
            body    JSONB;
            strat   JSONB;
            params  JSONB;
            raw_ops JSONB;
            op_elem JSONB;
            flat    JSONB;
            i       INT;
        BEGIN
            FOR rec IN
                SELECT definition_id, version, body AS raw_body
                FROM definitions
                WHERE definition_type = 'condition'
                  AND body -> 'strategy' ->> 'type' = 'composite'
            LOOP
                body   := rec.raw_body;
                strat  := body -> 'strategy';
                params := strat -> 'params';
                raw_ops := params -> 'operands';

                IF raw_ops IS NULL OR jsonb_typeof(raw_ops) <> 'array' THEN
                    CONTINUE;
                END IF;

                IF jsonb_array_length(raw_ops) = 0
                   OR jsonb_typeof(raw_ops -> 0) <> 'object' THEN
                    CONTINUE;
                END IF;

                flat := '[]'::JSONB;
                FOR i IN 0 .. jsonb_array_length(raw_ops) - 1 LOOP
                    op_elem := raw_ops -> i;
                    flat := flat || jsonb_build_array(op_elem ->> 'condition_id');
                END LOOP;

                UPDATE definitions
                SET body = body
                        || jsonb_build_object(
                               'strategy',
                               strat || jsonb_build_object(
                                   'params',
                                   params || jsonb_build_object('operands', flat)
                               )
                           )
                WHERE definition_id = rec.definition_id
                  AND version       = rec.version
                  AND definition_type = 'condition';
            END LOOP;
        END $$
    """)
