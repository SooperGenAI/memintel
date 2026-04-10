"""
app/connectors/registry.py
──────────────────────────────────────────────────────────────────────────────
ConnectorRegistry — maps connector names to live connector instances.

Built at startup from memintel_config.yaml connectors: section via
ConnectorRegistry.from_config(). Stored on app.state.connector_registry.

Connector lifecycle:
  Build   — ConnectorRegistry.from_config() instantiates connectors
  Connect — connect() is lazy (PostgresConnector) or stateless (RestConnector)
  Health  — health_check() is called during startup unless skip_connector_health_check
  Close   — close_all() is called at shutdown

Thread safety: registry dict is read-only after construction.
"""
from __future__ import annotations

import logging
from typing import Any

from app.models.config import ConfigSchema, ConnectorConfig, PrimitiveSourceConfig
from app.runtime.data_resolver import ConnectorError

log = logging.getLogger(__name__)


class ConnectorRegistry:
    """
    Holds a name → connector mapping for async data connectors.

    get(name)    — returns the connector instance, raises ConnectorError if absent.
    close_all()  — gracefully closes all connector pools.
    """

    def __init__(self, registry: dict[str, Any]) -> None:
        self._registry = registry

    @classmethod
    async def from_config(cls, config: ConfigSchema) -> ConnectorRegistry:
        """
        Build a ConnectorRegistry from a resolved ConfigSchema.

        For each connector in config.connectors:
          postgres → PostgresConnector
          rest     → RestConnector
          mock     → skipped (MockConnector is instantiated separately for tests)
          unknown  → logged and skipped

        primitive_sources from config are grouped by connector name so each
        connector receives only the sources it is responsible for.
        """
        from app.connectors.postgres import PostgresConnector
        from app.connectors.rest import RestConnector

        # Group primitive_sources by connector name
        primitive_sources = config.primitive_sources or {}
        by_connector: dict[str, dict[str, PrimitiveSourceConfig]] = {}
        for prim_name, source_cfg in primitive_sources.items():
            if source_cfg.connector not in by_connector:
                by_connector[source_cfg.connector] = {}
            by_connector[source_cfg.connector][prim_name] = source_cfg

        registry: dict[str, Any] = {}
        for connector_name, connector_config in config.connectors.items():
            prim_sources = by_connector.get(connector_name, {})

            if connector_config.type == "postgres":
                conn = PostgresConnector(connector_config, prim_sources)
                registry[connector_name] = conn
                log.info("connector_registered name=%s type=postgres", connector_name)

            elif connector_config.type == "rest":
                conn = RestConnector(connector_config, prim_sources)
                registry[connector_name] = conn
                log.info("connector_registered name=%s type=rest", connector_name)

            elif connector_config.type == "mock":
                log.info("connector_skipped_mock name=%s", connector_name)

            else:
                log.warning(
                    "connector_unknown_type_skipped name=%s type=%s",
                    connector_name,
                    connector_config.type,
                )

        return cls(registry)

    def register(self, name: str, connector: Any) -> None:
        """
        Add or replace a connector in the live registry.

        Used by the dynamic registration API (POST /v1/connectors) to make a
        newly registered connector available without a restart.
        """
        self._registry[name] = connector

    def get(self, connector_name: str) -> Any:
        """
        Return the connector instance for connector_name.

        Raises ConnectorError if the connector is not registered.
        """
        conn = self._registry.get(connector_name)
        if conn is None:
            raise ConnectorError(
                f"Connector '{connector_name}' is not registered. "
                f"Available: {sorted(self._registry.keys())}"
            )
        return conn

    async def close_all(self) -> None:
        """Close all connector pools gracefully. Best-effort — logs failures."""
        for name, conn in self._registry.items():
            try:
                if hasattr(conn, "close"):
                    await conn.close()
                    log.info("connector_closed name=%s", name)
            except Exception as exc:
                log.warning("connector_close_failed name=%s error=%s", name, str(exc))
