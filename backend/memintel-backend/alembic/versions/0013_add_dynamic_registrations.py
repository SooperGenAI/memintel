"""Add registered_connectors and registered_primitives tables for dynamic runtime registration.

Dynamic primitives and connectors registered via the API are persisted here
so they survive restarts and are reloaded at startup.

Revision ID: 0013
Revises: 0012
Create Date: 2026-04-09
"""
from alembic import op
import sqlalchemy as sa

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE registered_connectors (
            id               SERIAL PRIMARY KEY,
            name             VARCHAR(255) NOT NULL UNIQUE,
            connector_type   VARCHAR(50)  NOT NULL,
            params_encrypted TEXT         NOT NULL,
            created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE registered_primitives (
            id              SERIAL PRIMARY KEY,
            name            VARCHAR(255) NOT NULL UNIQUE,
            primitive_type  VARCHAR(100) NOT NULL,
            connector_name  VARCHAR(255),
            query           TEXT,
            json_path       TEXT,
            created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE INDEX idx_registered_primitives_connector_name
            ON registered_primitives (connector_name)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS registered_primitives")
    op.execute("DROP TABLE IF EXISTS registered_connectors")
