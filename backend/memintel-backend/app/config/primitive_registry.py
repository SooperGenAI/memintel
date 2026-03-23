"""
app/config/primitive_registry.py
──────────────────────────────────────────────────────────────────────────────
In-memory registry of all primitives declared in `memintel.config.md`.

Lifecycle
─────────
load_from_config(config) is called during application startup, after the
ConfigLoader has validated and resolved the config. It replaces any existing
registry contents — it is idempotent (re-applying a config produces the same
state).

register(primitive) adds or replaces a single primitive. Used by the
ConfigApplyService step-by-step apply path.

Consumers
─────────
  DataResolutionService   — get(name) to look up source/access config at fetch time
  LLM prompt builder      — list_all() to enumerate available signals
  Compiler / TypeChecker  — get_type(name) to validate primitive type compatibility
  TaskAuthoringService    — list_all() to inject primitive registry into LLM context

Design
──────
No async I/O — all operations are synchronous in-memory dictionary lookups.
The registry is populated once at startup and read-only at runtime.
"""
from __future__ import annotations

from app.models.config import ConfigSchema, PrimitiveConfig


class PrimitiveRegistry:
    """
    In-memory registry of PrimitiveConfig objects keyed by primitive name.

    Primitive names follow the 'namespace.field' format (e.g. user.churn_score).
    All lookups are O(1).
    """

    def __init__(self) -> None:
        self._primitives: dict[str, PrimitiveConfig] = {}

    # ── Population ─────────────────────────────────────────────────────────────

    def load_from_config(self, config: ConfigSchema) -> None:
        """
        Populate the registry from a validated ConfigSchema.

        Replaces all existing entries. The config must already be validated
        and env vars resolved — this method performs no validation.
        """
        self._primitives = {p.name: p for p in config.primitives}

    def register(self, primitive: PrimitiveConfig) -> None:
        """
        Add or replace a single primitive in the registry.

        Used by ConfigApplyService when applying primitives one-by-one.
        If a primitive with the same name already exists, it is overwritten.
        """
        self._primitives[primitive.name] = primitive

    # ── Lookups ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> PrimitiveConfig | None:
        """
        Return the PrimitiveConfig for the given name, or None if not found.

        Used by DataResolutionService to look up source and access config
        before fetching data for a primitive at execution time.
        """
        return self._primitives.get(name)

    def list_all(self) -> list[PrimitiveConfig]:
        """
        Return all registered primitives as a list.

        Used by the LLM prompt builder to inject the full primitive registry
        into the system prompt context during task authoring.
        """
        return list(self._primitives.values())

    def get_type(self, name: str) -> str:
        """
        Return the Memintel type string for the named primitive.

        Used by the compiler and TypeChecker to validate that a concept
        references a primitive whose type is compatible with the strategy.

        Raises KeyError if the primitive is not registered.
        """
        primitive = self._primitives.get(name)
        if primitive is None:
            raise KeyError(
                f"Primitive '{name}' is not registered. "
                f"Available primitives: {sorted(self._primitives)}"
            )
        return primitive.type
