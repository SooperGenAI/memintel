"""
app/config/__init__.py
──────────────────────────────────────────────────────────────────────────────
Re-exports the public surface of the config layer.

Import from this package rather than individual modules:
    from app.config import ConfigError, ConfigLoader
    from app.config import GuardrailsStore
    from app.config import PrimitiveRegistry
"""

from app.config.config_loader import ConfigError, ConfigLoader
from app.config.guardrails_store import GuardrailsStore
from app.config.primitive_registry import PrimitiveRegistry

__all__ = [
    "ConfigError",
    "ConfigLoader",
    "GuardrailsStore",
    "PrimitiveRegistry",
]
