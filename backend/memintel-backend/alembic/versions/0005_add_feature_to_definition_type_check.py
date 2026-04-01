"""Add 'feature' to definition_type CHECK constraint.

Adds 'feature' as a valid value for the definition_type column in the
definitions table. Required by RegistryService.register_feature() which
inserts definition_type='feature'.

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-01
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the existing inline CHECK constraint (auto-named by PostgreSQL).
    op.execute(
        "ALTER TABLE definitions "
        "DROP CONSTRAINT IF EXISTS definitions_definition_type_check"
    )
    # Re-add with 'feature' included.
    op.execute(
        "ALTER TABLE definitions ADD CONSTRAINT definitions_definition_type_check "
        "CHECK (definition_type IN "
        "('concept', 'condition', 'action', 'primitive', 'feature'))"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE definitions "
        "DROP CONSTRAINT IF EXISTS definitions_definition_type_check"
    )
    op.execute(
        "ALTER TABLE definitions ADD CONSTRAINT definitions_definition_type_check "
        "CHECK (definition_type IN ('concept', 'condition', 'action', 'primitive'))"
    )
