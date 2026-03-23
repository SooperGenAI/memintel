"""
app/config/guardrails_store.py
──────────────────────────────────────────────────────────────────────────────
In-memory store for the parsed Guardrails object loaded from
`memintel.guardrails.md`.

Lifecycle
─────────
load(path) is called exactly once during application startup by the lifespan
handler (app/main.py). After load() returns, the store is immutable — calling
load() a second time raises RuntimeError.

All other methods raise RuntimeError if called before load().

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

Startup invariant
─────────────────
GuardrailsStore.load() delegates to ConfigLoader.load_guardrails(), which
raises ConfigError if strategy_registry is empty. The lifespan handler
treats ConfigError as a fatal startup failure (sys.exit(1)).
"""
from __future__ import annotations

from app.models.config import ApplicationContext
from app.models.guardrails import Guardrails, StrategyRegistryEntry


class GuardrailsStore:
    """
    Thread-safe (single-writer) in-memory store for guardrails.

    Populated once at startup; immutable thereafter. All reads are O(1)
    after load — no file I/O on hot paths.
    """

    def __init__(self) -> None:
        self._guardrails: Guardrails | None = None

    # ── Startup ────────────────────────────────────────────────────────────────

    async def load(self, guardrails_path: str) -> None:
        """
        Parse memintel.guardrails.md, validate against the Guardrails schema,
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

    # ── Read accessors ─────────────────────────────────────────────────────────

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

    # ── Internal ───────────────────────────────────────────────────────────────

    def _require_loaded(self) -> Guardrails:
        if self._guardrails is None:
            raise RuntimeError(
                "GuardrailsStore has not been loaded. "
                "Call load() during application startup before any reads."
            )
        return self._guardrails
