"""
app/connectors/rest.py
──────────────────────────────────────────────────────────────────────────────
RestConnector — httpx-backed primitive data connector for REST APIs.

Fetches primitive values by making HTTP GET requests to a configured endpoint.
The endpoint path template is taken from PrimitiveSourceConfig.query and may
contain {entity_id} and {timestamp} placeholders (or :entity_id / :as_of
for consistency with the SQL connector style).

JSON path extraction (dot-notation):
  json_path="data.value" extracts response["data"]["value"].
  When json_path is not configured, the connector returns:
    - response["value"] if the response is a dict with that key
    - the raw response body if it is a scalar (int/float/str/bool)
    - None otherwise

Forward/backward fill is not supported — use PostgresConnector for fill strategies.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from app.models.config import ConnectorConfig, PrimitiveSourceConfig
from app.runtime.data_resolver import ConnectorError, PrimitiveValue

log = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def _resolve_value(value: str) -> str:
    """Resolve a ${ENV_VAR} reference or return the value as-is."""
    import os
    m = _ENV_VAR_RE.fullmatch(value)
    if m:
        resolved = os.environ.get(m.group(1))
        if resolved is None:
            raise ConnectorError(f"Environment variable {m.group(1)!r} is not set")
        return resolved
    return value


def _resolve_dict(d: dict[str, str]) -> dict[str, str]:
    """Resolve ${ENV_VAR} references in all values of a string dict."""
    return {k: _resolve_value(v) for k, v in d.items()}


def _extract_json_path(data: Any, json_path: str) -> Any:
    """Navigate dot-notation json_path through nested dicts. Returns None if missing."""
    parts = json_path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


class RestConnector:
    """
    httpx-backed connector for REST API primitives.

    Fetches values with HTTP GET using the path template from
    PrimitiveSourceConfig.query. Supports Bearer auth and custom headers.
    ${ENV_VAR} references in api_key and header values are resolved at
    construction time.

    fill strategies are not supported — forward_fill and backward_fill
    raise ConnectorError.
    """

    def __init__(
        self,
        config: ConnectorConfig,
        primitive_sources: dict[str, PrimitiveSourceConfig],
    ) -> None:
        self._base_url = (config.base_url or "").rstrip("/")
        # Resolve auth token from env var if present
        auth = config.auth or {}
        raw_api_key = auth.get("token") or auth.get("api_key") or ""
        self._api_key = _resolve_value(raw_api_key) if raw_api_key else ""
        self._timeout = (config.timeout_ms or 30000) / 1000.0
        self._ssl_verify = True  # ConnectorConfig has no ssl_verify; default True
        self._primitive_sources = primitive_sources
        # Additional headers (resolved)
        raw_headers: dict[str, str] = {}
        if isinstance(auth.get("headers"), dict):
            raw_headers = auth["headers"]
        self._extra_headers: dict[str, str] = _resolve_dict(raw_headers)

    async def fetch(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> PrimitiveValue:
        """
        Fetch the value of primitive_name for entity_id from the REST API.

        The endpoint path template is taken from PrimitiveSourceConfig.query.
        Supports {entity_id} / {timestamp} and :entity_id / :as_of placeholder
        styles.

        Extracts the return value using json_path (dot-notation) if configured,
        otherwise falls back to response["value"] or the raw scalar response.

        Raises ConnectorError on HTTP errors or connection failures.
        """
        source = self._primitive_sources.get(primitive_name)
        if source is None:
            raise ConnectorError(f"No endpoint configured for primitive '{primitive_name}'")

        # Substitute placeholders in path template
        path_template = source.query
        path = path_template.format_map({
            "entity_id": entity_id,
            "timestamp": timestamp or "",
        })
        # Also handle :entity_id / :as_of notation
        path = path.replace(":entity_id", entity_id).replace(":as_of", timestamp or "")

        url = f"{self._base_url}{path}"

        headers: dict[str, str] = dict(self._extra_headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(
                verify=self._ssl_verify,
                timeout=self._timeout,
            ) as client:
                resp = await client.get(url, headers=headers)
        except httpx.ConnectError as exc:
            raise ConnectorError(f"RestConnector: connection error: {exc}") from exc
        except Exception as exc:
            raise ConnectorError(f"RestConnector: request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise ConnectorError(
                f"RestConnector: HTTP {resp.status_code} for {url}"
            )

        try:
            body = resp.json()
        except Exception:
            # Non-JSON response — treat body as the raw value
            return PrimitiveValue(value=resp.text if resp.text else None)

        # Extract value from JSON response
        json_path = source.json_path if hasattr(source, 'json_path') else None
        if json_path:
            value = _extract_json_path(body, json_path)
        elif isinstance(body, dict):
            value = body.get("value")
        elif isinstance(body, (int, float, str, bool)):
            value = body
        else:
            value = None

        return PrimitiveValue(value=value)

    async def fetch_forward_fill(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> PrimitiveValue:
        """Not supported — raises ConnectorError."""
        raise ConnectorError(
            "forward_fill not supported by RestConnector. "
            "Use a PostgresConnector for fill strategies."
        )

    async def fetch_backward_fill(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> PrimitiveValue:
        """Not supported — raises ConnectorError."""
        raise ConnectorError(
            "backward_fill not supported by RestConnector. "
            "Use a PostgresConnector for fill strategies."
        )

    async def health_check(self) -> bool:
        """
        GET {base_url}/health (falling back to {base_url}/) with a 5-second timeout.
        Returns True if HTTP 200, False otherwise.
        """
        for path in ("/health", "/"):
            try:
                async with httpx.AsyncClient(
                    verify=self._ssl_verify,
                    timeout=5.0,
                ) as client:
                    resp = await client.get(f"{self._base_url}{path}")
                    if resp.status_code == 200:
                        return True
            except Exception:
                pass
        return False
