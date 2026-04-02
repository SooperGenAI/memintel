"""
app/config/guardrails_store.py
──────────────────────────────────────────────────────────────────────────────
In-memory store for the parsed Guardrails object loaded from
`memintel_guardrails.yaml`.

Lifecycle
─────────
load(path) is called exactly once during application startup by the lifespan
handler (app/main.py). After load() returns, the file-based guardrails are
available.

After the DB pool is established, reload_from_db() is called. If an active
API guardrails version exists in the database, it is loaded into memory and
takes precedence over the file-based version from that point forward.

All read methods raise RuntimeError if called before load().

Consumers
─────────
  LLM pipeline (POST /tasks):
    get_guardrails()           — full object for LLM context injection
    get_application_context()  — domain + instructions block
    get_strategy_registry()    — strategy descriptions and parameters
    get_threshold_bounds(s)    — per-strategy bounds (also used by Calibration)

  Compiler:
    get_strategy_registry()    — validates strategy references in definitions
    get_guardrails().constraints — disallowed strategies, max complexity

  CalibrationService:
    get_threshold_bounds(s)    — enforces [min, max] during calibration

  TaskAuthoringService:
    get_active_version()       — active GuardrailsVersion for task tracking

Startup invariant
─────────────────
GuardrailsStore.load() delegates to ConfigLoader.load_guardrails(), which
raises ConfigError if strategy_registry is empty. The lifespan handler
treats ConfigError as a fatal startup failure (sys.exit(1)).

API override
────────────
Once an admin posts to POST /guardrails, the API version takes precedence
over the file from that point forward. The file is the seed/fallback;
the API is the override. reload_from_db() is called after POST /guardrails
and at startup (after the DB pool is available) to load the active API
version into memory.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.models.config import ApplicationContext
from app.models.guardrails import Guardrails, StrategyRegistryEntry
from app.models.guardrails_api import GuardrailsVersion

if TYPE_CHECKING:
    from app.stores.guardrails import GuardrailsStore as GuardrailsVersionStore


class GuardrailsStore:
    """
    Thread-safe (single-writer) in-memory store for guardrails.

    _guardrails is populated from the YAML file at startup.
    _active_api_version tracks the active DB-sourced version (if any).
    _source reflects whether the current active version is from "file" or "api".

    All reads use the file-loaded Guardrails for runtime operations.
    get_active_version() returns the API version metadata for task tracking.
    """

    def __init__(self) -> None:
        self._guardrails: Guardrails | None = None
        self._active_api_version: GuardrailsVersion | None = None
        self._source: str = "file"

    # ── Startup ────────────────────────────────────────────────────────────────

    async def load(self, guardrails_path: str) -> None:
        """
        Parse memintel_guardrails.yaml, validate against the Guardrails schema,
        and store the result in memory.

        Raises RuntimeError if called after the store is already populated.
        Raises ConfigError (from ConfigLoader) if:
          - the file is missing
          - the YAML is malformed
          - the schema is invalid
          - strategy_registry is empty
        """
        if self._guardrails is not None:
            raise RuntimeError(
                "GuardrailsStore.load() called after the store is already "
                "populated. load() must be called exactly once at startup."
            )

        from app.config.config_loader import ConfigLoader

        loader = ConfigLoader()
        self._guardrails = loader.load_guardrails(guardrails_path)

    async def reload_from_db(self, db: "GuardrailsVersionStore") -> bool:
        """
        Check the database for an active API guardrails version and load it
        into memory if one exists.

        Called:
          1. During startup (after the DB pool is established).
          2. After POST /guardrails (synchronously before the 201 response).

        Returns True if an active API version was found and loaded.
        Returns False if no API version exists (file-based guardrails remain).

        Never raises — any DB error is silently absorbed (file-based guardrails
        continue to work if the DB is temporarily unavailable).
        """
        try:
            active = await db.get_active()
        except Exception:
            return False

        if active is None:
            return False

        self._active_api_version = active
        self._source = "api"
        return True

    # ── Read accessors ─────────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        """
        Return True if guardrails have been loaded via load(), False otherwise.

        Use this guard before calling get_guardrails() in contexts where the
        store may legitimately be unloaded (e.g. dependency injection during
        tests or graceful-degradation paths).
        """
        return self._guardrails is not None

    def get_guardrails(self) -> Guardrails:
        """
        Return the full parsed Guardrails object.

        Used by the LLM pipeline to inject the complete guardrails context
        into the system prompt at task authoring time.
        """
        return self._require_loaded()

    def get_application_context(self) -> ApplicationContext:
        """
        Return the application_context block.

        Injected into LLM system prompts during POST /tasks. Has zero
        influence on execution, evaluation, calibration, or feedback.
        """
        return self._require_loaded().application_context

    def get_strategy_registry(self) -> dict[str, StrategyRegistryEntry]:
        """
        Return the strategy registry dict keyed by strategy name.

        Used by the compiler to validate strategy references in definitions
        and by the LLM to understand available strategy types.
        """
        return self._require_loaded().strategy_registry

    def get_threshold_bounds(self, strategy: str) -> dict:
        """
        Return threshold bounds for a strategy as {'min': ..., 'max': ...}.

        Returns {} if no bounds are defined for the given strategy name.
        Used by CalibrationService and the compiler to enforce hard bounds
        on strategy parameters.
        """
        guardrails = self._require_loaded()
        bounds = guardrails.constraints.threshold_bounds.get(strategy)
        if bounds is None:
            return {}
        return bounds.model_dump()

    def get_active_version(self) -> GuardrailsVersion | None:
        """
        Return the active API guardrails version, or None if file-based only.

        Used by TaskAuthoringService to record which guardrails version was
        active when a task was created.

        Returns None when no API version has been posted — file-based
        guardrails are in use and tasks will have guardrails_version=None.
        """
        return self._active_api_version

    @property
    def source(self) -> str:
        """Return the current guardrails source: 'file' or 'api'."""
        return self._source

    # ── Internal ───────────────────────────────────────────────────────────────

    def _require_loaded(self) -> Guardrails:
        if self._guardrails is None:
            raise RuntimeError(
                "GuardrailsStore has not been loaded. "
                "Call load() during application startup before any reads."
            )
        return self._guardrails
