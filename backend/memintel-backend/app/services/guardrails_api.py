"""
app/services/guardrails_api.py
────────────────────────────────────────────────────────────────────────────────
GuardrailsApiService — guardrails lifecycle management via API.

Responsibilities
────────────────
  - Validate CreateGuardrailsRequest before saving:
      * All strategies in strategy_registry must be known
      * type_strategy_map may only reference registered strategies
      * bias_rules must only use valid severity levels
      * global_preferred_strategy must be in strategy_registry
      * global_default_strategy must be in strategy_registry
  - Delegate to GuardrailsStore for persistence.
  - After create, reload the config-level GuardrailsStore in memory so the
    new version takes effect without a server restart (synchronous guarantee).
  - Proxy read operations to GuardrailsStore, raising NotFoundError where
    appropriate.

Error mapping
─────────────
  create_guardrails()         → raises MemintelError(SEMANTIC_ERROR) on invalid definition.
  get_guardrails_version()    → raises NotFoundError if version not found.
"""
from __future__ import annotations

import structlog

from app.models.errors import ErrorType, MemintelError, NotFoundError
from app.models.guardrails_api import (
    CreateGuardrailsRequest,
    GuardrailsImpactResult,
    GuardrailsVersion,
)
from app.stores.guardrails import GuardrailsStore

log = structlog.get_logger(__name__)

# ── Known strategies and valid severity levels ────────────────────────────────

_KNOWN_STRATEGIES: frozenset[str] = frozenset({
    "threshold",
    "percentile",
    "z_score",
    "change",
    "equals",
    "composite",
})

_VALID_SEVERITY_LEVELS: frozenset[str] = frozenset({
    "high_severity",
    "medium_severity",
    "low_severity",
})


class GuardrailsApiService:
    """
    Business logic layer for API-managed guardrails.

    Public API
    ----------
    create_guardrails(req)          → GuardrailsVersion
    get_active_guardrails()         → GuardrailsVersion | None
    get_guardrails_version(ver)     → GuardrailsVersion   (raises NotFoundError)
    list_versions()                 → list[GuardrailsVersion]
    get_impact()                    → GuardrailsImpactResult
    """

    def __init__(self, store: GuardrailsStore, config_store: object | None = None) -> None:
        self._store = store
        self._config_store = config_store  # app.config.guardrails_store.GuardrailsStore

    async def create_guardrails(
        self, request: CreateGuardrailsRequest
    ) -> GuardrailsVersion:
        """
        Validate and persist a new guardrails version.

        Validation:
          - All strategy names in strategy_registry must be in _KNOWN_STRATEGIES.
          - type_strategy_map may only reference strategies in strategy_registry.
          - bias_rules values must be in _VALID_SEVERITY_LEVELS.
          - global_preferred_strategy must be in strategy_registry.
          - global_default_strategy must be in strategy_registry.

        After saving, reloads the config-level GuardrailsStore in memory so
        the new version is active immediately (before the 201 response returns).

        Raises:
          MemintelError(SEMANTIC_ERROR) — validation failure.
        """
        defn = request.guardrails
        registered = set(defn.strategy_registry)

        # Validate: all strategy names are known
        unknown = registered - _KNOWN_STRATEGIES
        if unknown:
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                f"Unknown strategies in strategy_registry: {sorted(unknown)}. "
                f"Known strategies: {sorted(_KNOWN_STRATEGIES)}.",
                location="guardrails.strategy_registry",
            )

        # Validate: type_strategy_map only references registered strategies
        for type_name, strategies in defn.type_strategy_map.items():
            unregistered = set(strategies) - registered
            if unregistered:
                raise MemintelError(
                    ErrorType.SEMANTIC_ERROR,
                    f"type_strategy_map['{type_name}'] references strategies not in "
                    f"strategy_registry: {sorted(unregistered)}.",
                    location=f"guardrails.type_strategy_map.{type_name}",
                )

        # Validate: bias_rules use valid severity levels
        for word, severity in defn.bias_rules.items():
            if severity not in _VALID_SEVERITY_LEVELS:
                raise MemintelError(
                    ErrorType.SEMANTIC_ERROR,
                    f"bias_rules['{word}'] has invalid severity level '{severity}'. "
                    f"Valid levels: {sorted(_VALID_SEVERITY_LEVELS)}.",
                    location=f"guardrails.bias_rules.{word}",
                )

        # Validate: global_preferred_strategy is registered
        if defn.global_preferred_strategy not in registered:
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                f"global_preferred_strategy '{defn.global_preferred_strategy}' is not "
                f"in strategy_registry.",
                location="guardrails.global_preferred_strategy",
            )

        # Validate: global_default_strategy is registered
        if defn.global_default_strategy not in registered:
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                f"global_default_strategy '{defn.global_default_strategy}' is not "
                f"in strategy_registry.",
                location="guardrails.global_default_strategy",
            )

        version = GuardrailsVersion(
            guardrails=defn,
            change_note=request.change_note,
            source="api",
        )

        created = await self._store.create(version)
        log.info(
            "guardrails_created",
            guardrails_id=created.guardrails_id,
            version=created.version,
        )

        # Reload config-level GuardrailsStore in memory — synchronous before response.
        if self._config_store is not None:
            reloaded = await self._config_store.reload_from_db(self._store)
            log.info("guardrails_reloaded_in_memory", reloaded=reloaded, version=created.version)

        return created

    async def get_active_guardrails(self) -> GuardrailsVersion | None:
        """Return the active guardrails version, or None if none exists."""
        return await self._store.get_active()

    async def get_guardrails_version(self, version: str) -> GuardrailsVersion:
        """
        Return the guardrails version for the given version string.

        Raises NotFoundError if the version does not exist.
        """
        v = await self._store.get_version(version)
        if v is None:
            raise NotFoundError(f"Guardrails version '{version}' not found.")
        return v

    async def list_versions(self) -> list[GuardrailsVersion]:
        """Return all guardrails versions, newest first."""
        return await self._store.list_versions()

    async def get_impact(self) -> GuardrailsImpactResult:
        """Return task distribution across guardrails versions."""
        return await self._store.get_impact()
