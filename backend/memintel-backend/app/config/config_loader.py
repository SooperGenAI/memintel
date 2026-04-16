"""
app/config/config_loader.py
──────────────────────────────────────────────────────────────────────────────
Parses and validates `memintel_config.yaml`, resolves ${ENV_VAR} references,
and loads `memintel_guardrails.yaml` into the Guardrails model.

Parsing rules
─────────────
1. Read the YAML file directly (yaml.safe_load)
2. Validate the parsed object against ConfigSchema
3. Resolve all ${ENV_VAR} references — raise ConfigError if any variable is unset
4. NEVER log resolved credential values at any log level

Config files must use the .yaml extension. Passing a .md file path raises
ConfigError immediately — the Markdown-embedded YAML format is no longer
supported.

Security invariants
───────────────────
- Resolved credential values are never written to logs at any level.
- ConfigError message includes the variable name but NOT the expected value.
- load() either returns a fully resolved config or raises ConfigError.
  A partially resolved config is never returned.

Guardrails loading
──────────────────
load_guardrails(path) parses the YAML file directly, strips the top-level
'guardrails:' wrapper if present, validates against the Guardrails model,
and enforces the non-empty strategy_registry invariant.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from app.models.config import ConfigSchema
from app.models.guardrails import Guardrails

log = logging.getLogger(__name__)

# Matches ${VARIABLE_NAME} — uppercase, alphanumeric + underscores
ENV_VAR_RE: re.Pattern[str] = re.compile(r'\$\{([A-Z][A-Z0-9_]*)\}')


class ConfigError(Exception):
    """
    Raised when the config is missing, structurally invalid, fails schema
    validation, or references an unset environment variable.

    The message always names the failing condition. Resolved credential values
    are NEVER included in the message.
    """
    pass


class ConfigLoader:
    """
    Loads and validates memintel_config.yaml (and memintel_guardrails.yaml).

    All public methods raise ConfigError on any failure. The system must not
    start with a partially applied configuration.

    Usage:
        loader = ConfigLoader()
        config = loader.load("/path/to/memintel_config.yaml")
        guardrails = loader.load_guardrails(config.guardrails_path)
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, config_path: str) -> ConfigSchema:
        """
        Parse, validate, and resolve a memintel_config.yaml file.

        Steps:
          1. Reject .md paths immediately (format no longer supported).
          2. Read and parse the YAML file directly.
          3. Scan client primitive yaml files and deep-merge any primitives: and
             primitive_sources: blocks found into the base dict. Directory is
             resolved from CLIENT_CONFIG_DIR env var, falling back to
             config/clients/ relative to the config file. Missing dir is skipped.
          4. Validate merged dict against ConfigSchema (raises ConfigError on failure).
          5. Resolve all ${ENV_VAR} references (raises ConfigError if any unset).
          6. Return the fully resolved, validated ConfigSchema.

        Raises ConfigError if the file has a .md extension, is missing,
        malformed, schema-invalid, or references an unset environment variable.
        """
        if config_path.endswith(".md"):
            raise ConfigError(
                "Config file must be .yaml — .md format is no longer supported. "
                "Rename your config file to .yaml (e.g. memintel_config.yaml)."
            )

        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {config_path}")

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"Cannot read config file '{config_path}': {e}") from e

        try:
            raw = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML parse error in config file: {e}") from e

        if not isinstance(raw, dict):
            raise ConfigError("Config file must contain a YAML mapping at the top level.")

        # Merge per-client primitive configs before validation so that cross-reference
        # checks (primitive → connector) see the full merged set in one pass.
        raw = self._merge_client_configs(raw, config_path)

        return self._validate_and_resolve(raw)

    def load_from_dict(self, raw: dict) -> ConfigSchema:
        """
        Validate and resolve a pre-parsed config dict.

        Used in tests to bypass file I/O. Applies the same schema validation
        and ${ENV_VAR} resolution as load().
        """
        return self._validate_and_resolve(raw)

    def load_guardrails(self, guardrails_path: str) -> Guardrails:
        """
        Parse, validate, and return the Guardrails from memintel_guardrails.yaml.

        The YAML file may have a top-level 'guardrails:' key (the spec wraps
        the entire content in it). If present, the wrapper is stripped before
        validation.

        Raises ConfigError if:
          - the path has a .md extension
          - the file is missing
          - the YAML is malformed
          - the Guardrails schema is invalid
          - strategy_registry is empty
        """
        if guardrails_path.endswith(".md"):
            raise ConfigError(
                "Config file must be .yaml — .md format is no longer supported. "
                "Rename your guardrails file to .yaml (e.g. memintel_guardrails.yaml)."
            )

        path = Path(guardrails_path)
        if not path.exists():
            raise ConfigError(f"Guardrails file not found: {guardrails_path}")

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(
                f"Cannot read guardrails file '{guardrails_path}': {e}"
            ) from e

        try:
            raw = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML parse error in guardrails file: {e}") from e

        if not isinstance(raw, dict):
            raise ConfigError("Guardrails file must contain a YAML mapping at the top level.")

        # Strip the top-level 'guardrails:' wrapper if present
        if "guardrails" in raw and isinstance(raw["guardrails"], dict):
            raw = raw["guardrails"]

        try:
            guardrails = Guardrails.model_validate(raw)
        except (PydanticValidationError, ValueError) as e:
            raise ConfigError(f"Guardrails validation failed: {e}") from e

        # Enforce startup invariant: strategy_registry must be non-empty
        if not guardrails.strategy_registry:
            raise ConfigError(
                "strategy_registry is empty — at least one strategy must be "
                "registered in the guardrails file."
            )

        return guardrails

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _merge_client_configs(self, raw: dict, config_path: str) -> dict:
        """
        Scan client primitive yaml files and append any primitives: and
        primitive_sources: entries found into raw.

        The directory to scan is resolved in this order:
          1. CLIENT_CONFIG_DIR environment variable (absolute path to a folder).
          2. Fallback: <config_dir>/config/clients/ relative to the config file.

        Rules:
          - Missing or empty directory → log and return raw unchanged.
          - Files are processed in sorted order for deterministic merging.
          - Only primitives: (list) and primitive_sources: (dict) keys are merged;
            all other keys in client files are ignored.
          - Duplicate primitive names across client files will be caught later by
            ConfigSchema._validate_unique_names.

        Raises ConfigError if a client file cannot be read or is not valid YAML.
        """
        client_dir_env = os.environ.get("CLIENT_CONFIG_DIR")
        if client_dir_env:
            clients_dir = Path(client_dir_env)
        else:
            clients_dir = Path(config_path).parent / "config" / "clients"

        if not clients_dir.is_dir():
            log.info("no_client_config_dir path=%s", str(clients_dir))
            return raw

        for client_file in sorted(clients_dir.glob("*.yaml")):
            try:
                text = client_file.read_text(encoding="utf-8")
            except OSError as e:
                raise ConfigError(
                    f"Cannot read client config '{client_file}': {e}"
                ) from e

            try:
                client_raw = yaml.safe_load(text) or {}
            except yaml.YAMLError as e:
                raise ConfigError(
                    f"YAML parse error in client config '{client_file}': {e}"
                ) from e

            if not isinstance(client_raw, dict):
                continue

            if "primitives" in client_raw:
                if not isinstance(raw.get("primitives"), list):
                    raw["primitives"] = []
                raw["primitives"].extend(client_raw["primitives"])

            if "primitive_sources" in client_raw:
                if not isinstance(raw.get("primitive_sources"), dict):
                    raw["primitive_sources"] = {}
                raw["primitive_sources"].update(client_raw["primitive_sources"])

        return raw

    def _validate_and_resolve(self, raw: dict) -> ConfigSchema:
        """Schema-validate, then resolve env vars, then re-validate."""
        # Pass 1: schema validation (catches field type errors, enum violations,
        # credential format checks, cross-reference checks)
        try:
            config = ConfigSchema.model_validate(raw)
        except (PydanticValidationError, ValueError) as e:
            raise ConfigError(f"Config validation failed: {e}") from e

        # Pass 2: resolve ${ENV_VAR} references on the serialised dict.
        # model_dump() gives plain Python types — safe to traverse recursively.
        resolved = self._resolve_env_vars(config.model_dump())

        # Pass 3: re-validate after resolution.
        # context={"resolved": True} tells the credential format validators
        # (ConnectorConfig._validate_password, LLMConfig._validate_api_key)
        # to skip the ${ENV_VAR} pattern check — the values are now plain strings.
        try:
            return ConfigSchema.model_validate(resolved, context={"resolved": True})
        except (PydanticValidationError, ValueError) as e:
            raise ConfigError(
                f"Config validation failed after env var resolution: {e}"
            ) from e

    def _resolve_env_vars(self, obj: Any) -> Any:
        """
        Recursively replace all ${VAR_NAME} occurrences with their env values.

        Raises ConfigError (naming the variable) if any referenced variable
        is not set. Never logs the resolved value.
        """
        if isinstance(obj, str):
            def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
                var = match.group(1)
                val = os.environ.get(var)
                if val is None:
                    raise ConfigError(
                        f"Required environment variable {var} is not set"
                    )
                # val is intentionally not logged
                return val

            return ENV_VAR_RE.sub(_replace, obj)

        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._resolve_env_vars(v) for v in obj]

        return obj
