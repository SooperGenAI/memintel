"""app/connectors — real data connector implementations."""
from app.connectors.postgres import PostgresConnector
from app.connectors.rest import RestConnector
from app.connectors.registry import ConnectorRegistry

__all__ = ["PostgresConnector", "RestConnector", "ConnectorRegistry"]
