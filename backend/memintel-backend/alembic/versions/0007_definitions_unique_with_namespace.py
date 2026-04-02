"""Widen definitions unique constraint to include namespace.

Fixes BUG-7-1: promote() inserts a new row for the target namespace, but the
existing constraint UNIQUE (definition_id, version) rejects any row that
shares a definition_id+version with an existing row — regardless of namespace.

Fix: replace the two-column constraint with a three-column constraint that
allows the same (definition_id, version) to exist in multiple namespaces.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-02
"""
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE definitions DROP CONSTRAINT uq_definition_version"
    )
    op.execute(
        "ALTER TABLE definitions "
        "ADD CONSTRAINT uq_definition_version "
        "UNIQUE (definition_id, version, namespace)"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE definitions DROP CONSTRAINT uq_definition_version"
    )
    # Remove any duplicate (definition_id, version) rows introduced by promote()
    # before restoring the narrower constraint — keep the earliest row (lowest id).
    op.execute(
        """
        DELETE FROM definitions a
        USING definitions b
        WHERE a.id > b.id
          AND a.definition_id = b.definition_id
          AND a.version = b.version
        """
    )
    op.execute(
        "ALTER TABLE definitions "
        "ADD CONSTRAINT uq_definition_version "
        "UNIQUE (definition_id, version)"
    )
