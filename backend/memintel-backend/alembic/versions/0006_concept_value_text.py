"""Change decisions.concept_value from DOUBLE PRECISION to TEXT.

Fixes BUG-6-1 (bool True stored as 1.0) and BUG-6-2 (str concept values
crash asyncpg at INSERT time) by widening the column to TEXT.

After this migration:
  - Python True  → stored as "True"
  - Python False → stored as "False"
  - Python 1.87  → stored as "1.87"
  - Python "high_risk" → stored as "high_risk"
  - Python None  → stored as NULL

The DecisionStore serializes on INSERT (str(value)) and deserializes on
SELECT (float → bool → str fallback chain).

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-02
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PostgreSQL casts DOUBLE PRECISION → TEXT automatically; no USING needed.
    # Existing numeric rows (e.g. 1.0) are converted to their text representation.
    op.execute(
        "ALTER TABLE decisions ALTER COLUMN concept_value TYPE TEXT"
    )


def downgrade() -> None:
    # Rows containing non-numeric text (e.g. 'high_risk', 'True', 'False')
    # cannot be cast back to DOUBLE PRECISION; those values become NULL.
    op.execute(
        "ALTER TABLE decisions "
        "ALTER COLUMN concept_value TYPE DOUBLE PRECISION "
        "USING CASE "
        "  WHEN concept_value IS NULL THEN NULL "
        "  WHEN concept_value ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$' "
        "    THEN concept_value::DOUBLE PRECISION "
        "  ELSE NULL "
        "END"
    )
