"""
tests/unit/test_config.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the config bootstrapping layer:
  - ConfigLoader (load_from_dict, load, load_guardrails)
  - GuardrailsStore (load, accessors)
  - PrimitiveRegistry (load_from_config, get, list_all, get_type)

All tests are self-contained — no external files or services required.
GuardrailsStore.load() is tested via a tmp_path fixture that writes
minimal plain YAML files.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.config.config_loader import ConfigError, ConfigLoader
from app.config.guardrails_store import GuardrailsStore
from app.config.primitive_registry import PrimitiveRegistry
from app.models.config import ConfigSchema, PrimitiveConfig


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _minimal_connector() -> dict:
    """A postgres connector dict with an ${ENV_VAR} password."""
    return {
        "type": "postgres",
        "host": "${DB_HOST}",
        "port": 5432,
        "database": "testdb",
        "user": "${DB_USER}",
        "password": "${DB_PASSWORD}",
    }


def _minimal_primitive(connector_name: str = "postgres.test") -> dict:
    """A minimal float primitive referencing the given connector."""
    return {
        "name": "user.score",
        "type": "float",
        "missing_data_policy": "null",
        "source": {
            "type": "database",
            "identifier": connector_name,
            "field": "score",
            "access": {
                "method": "sql",
                "query": "SELECT score FROM t WHERE user_id = :entity_id",
            },
        },
    }


def _minimal_llm() -> dict:
    """A minimal LLM config dict with an ${ENV_VAR} api_key."""
    return {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "api_key": "${ANTHROPIC_API_KEY}",
        "endpoint": "https://api.anthropic.com",
    }


def minimal_valid_config_dict(connector_name: str = "postgres.test") -> dict:
    """
    The smallest possible config dict that passes ConfigSchema validation
    (before env var resolution).
    """
    return {
        "primitives": [_minimal_primitive(connector_name)],
        "connectors": {connector_name: _minimal_connector()},
        "llm": _minimal_llm(),
        "guardrails_path": "memintel_guardrails.yaml",
    }


def _minimal_guardrails_yaml() -> str:
    """
    A minimal memintel_guardrails.yaml plain YAML string with one strategy.
    The YAML is wrapped under a top-level 'guardrails:' key as the spec
    defines.
    """
    return textwrap.dedent("""\
        guardrails:
          application_context:
            name: Test App
            description: Test application context
            instructions:
              - Prefer simpler strategies
            default_entity_scope: user
            action_preferences: {}

          strategy_registry:
            threshold:
              version: "1.0"
              description: Threshold strategy
              input_types:
                - float
                - int
              output_type: boolean
              parameters:
                threshold:
                  type: float
                  required: true
                direction:
                  type: string
                  required: false

          type_compatibility:
            float:
              valid_strategies:
                - threshold
              invalid_strategies: []
    """)


def _minimal_config_yaml(extra_env_var: str | None = None) -> str:
    """
    Minimal memintel_config.yaml plain YAML string.

    If extra_env_var is provided, it is inserted as a value in the connector
    host field so env var resolution will attempt to resolve it.
    """
    host_val = f"${{{extra_env_var}}}" if extra_env_var else "localhost"
    return textwrap.dedent(f"""\
        primitives:
          - name: user.score
            type: float
            missing_data_policy: "null"
            source:
              type: database
              identifier: postgres.test
              field: score
              access:
                method: sql
                query: "SELECT score FROM t WHERE user_id = :entity_id"

        connectors:
          postgres.test:
            type: postgres
            host: {host_val}
            port: 5432
            database: testdb
            user: admin
            password: "${{DB_PASSWORD}}"

        llm:
          provider: anthropic
          model: claude-sonnet-4-20250514
          api_key: "${{ANTHROPIC_API_KEY}}"
          endpoint: https://api.anthropic.com

        environment:
          namespace: org
          log_level: INFO

        guardrails_path: memintel_guardrails.yaml
    """)


# ── ConfigLoader.load_from_dict ────────────────────────────────────────────────

class TestConfigLoaderLoadFromDict:

    def test_valid_config_loads_without_error(self, monkeypatch):
        """A well-formed config dict with resolved env vars should load cleanly."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        config = ConfigLoader().load_from_dict(minimal_valid_config_dict())

        assert isinstance(config, ConfigSchema)
        assert len(config.primitives) == 1
        assert config.primitives[0].name == "user.score"
        assert config.primitives[0].type == "float"

    def test_missing_llm_key_raises_config_error(self):
        raw = {
            "primitives": [],
            "connectors": {},
            # llm is intentionally omitted
        }
        with pytest.raises(ConfigError, match="Config validation failed"):
            ConfigLoader().load_from_dict(raw)

    def test_invalid_primitive_type_raises_config_error(self, monkeypatch):
        """'float_array' is not a valid Memintel type — must raise ConfigError."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-key")

        raw = minimal_valid_config_dict()
        raw["primitives"][0]["type"] = "float_array"

        with pytest.raises(ConfigError, match="Invalid Memintel type"):
            ConfigLoader().load_from_dict(raw)

    def test_plaintext_password_raises_config_error(self):
        """A plaintext password must raise ConfigError (not a warning)."""
        raw = minimal_valid_config_dict()
        raw["connectors"]["postgres.test"]["password"] = "my-plaintext-password"

        with pytest.raises(ConfigError, match="Credential fields must use"):
            ConfigLoader().load_from_dict(raw)

    def test_plaintext_api_key_raises_config_error(self):
        """A plaintext LLM api_key must raise ConfigError."""
        raw = minimal_valid_config_dict()
        raw["llm"]["api_key"] = "sk-abc123"

        with pytest.raises(ConfigError, match="api_key must be"):
            ConfigLoader().load_from_dict(raw)

    def test_invalid_llm_provider_raises_config_error(self):
        """An unsupported provider string must raise ConfigError."""
        raw = minimal_valid_config_dict()
        raw["llm"]["provider"] = "gemini"

        with pytest.raises(ConfigError, match="Invalid LLM provider"):
            ConfigLoader().load_from_dict(raw)

    def test_duplicate_primitive_names_raises_config_error(self, monkeypatch):
        """Duplicate primitive names in the same config must raise ConfigError."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-key")

        raw = minimal_valid_config_dict()
        raw["primitives"].append(_minimal_primitive())  # same name: user.score

        with pytest.raises(ConfigError, match="Primitive names must be unique"):
            ConfigLoader().load_from_dict(raw)

    def test_unknown_connector_reference_raises_config_error(self, monkeypatch):
        """A primitive referencing a non-existent connector must raise ConfigError."""
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-key")

        raw = minimal_valid_config_dict()
        raw["primitives"][0]["source"]["identifier"] = "postgres.missing"

        with pytest.raises(ConfigError, match="references unknown connector"):
            ConfigLoader().load_from_dict(raw)


# ── ${ENV_VAR} resolution ─────────────────────────────────────────────────────

class TestEnvVarResolution:

    def test_missing_env_var_raises_config_error(self, monkeypatch, tmp_path):
        """Referencing an unset env var must raise ConfigError naming the variable."""
        monkeypatch.delenv("DB_PASSWORD", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-key")

        config_file = tmp_path / "memintel_config.yaml"
        config_file.write_text(_minimal_config_yaml())

        with pytest.raises(ConfigError, match="DB_PASSWORD is not set"):
            ConfigLoader().load(str(config_file))

    def test_error_names_variable_not_value(self, monkeypatch, tmp_path):
        """
        ConfigError message must include the variable name but must NOT
        contain any hint of the expected value.
        """
        monkeypatch.delenv("DB_PASSWORD", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-key")

        config_file = tmp_path / "memintel_config.yaml"
        config_file.write_text(_minimal_config_yaml())

        with pytest.raises(ConfigError) as exc_info:
            ConfigLoader().load(str(config_file))

        message = str(exc_info.value)
        assert "DB_PASSWORD" in message
        assert "secret" not in message.lower()
        assert "password" not in message.lower() or "DB_PASSWORD" in message

    def test_partially_resolved_config_not_returned(self, monkeypatch, tmp_path):
        """
        If one env var resolves but another is missing, no partial config
        is returned — ConfigError must be raised before load() returns.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-key")
        monkeypatch.delenv("DB_PASSWORD", raising=False)

        config_file = tmp_path / "memintel_config.yaml"
        config_file.write_text(_minimal_config_yaml())

        with pytest.raises(ConfigError):
            ConfigLoader().load(str(config_file))

    def test_extra_env_var_in_host_resolved(self, monkeypatch, tmp_path):
        """An ${ENV_VAR} reference in a non-credential field resolves correctly."""
        monkeypatch.setenv("MY_DB_HOST", "db.example.com")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-key")

        config_file = tmp_path / "memintel_config.yaml"
        config_file.write_text(_minimal_config_yaml(extra_env_var="MY_DB_HOST"))

        # Should not raise — MY_DB_HOST is set
        config = ConfigLoader().load(str(config_file))
        assert config is not None


# ── ConfigLoader.load (file-based) ────────────────────────────────────────────

class TestConfigLoaderLoad:

    def test_missing_config_file_raises(self):
        with pytest.raises(ConfigError, match="Config file not found"):
            ConfigLoader().load("/nonexistent/path/memintel_config.yaml")

    def test_valid_config_file_loads(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-key")

        config_file = tmp_path / "memintel_config.yaml"
        config_file.write_text(_minimal_config_yaml())

        config = ConfigLoader().load(str(config_file))

        assert isinstance(config, ConfigSchema)
        assert config.llm.provider == "anthropic"

    def test_direct_yaml_file_loads_correct_guardrails_path(self, monkeypatch, tmp_path):
        """A plain YAML config file is parsed directly with the expected values."""
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-key")

        config_file = tmp_path / "memintel_config.yaml"
        config_file.write_text(_minimal_config_yaml())

        config = ConfigLoader().load(str(config_file))
        assert config.guardrails_path == "memintel_guardrails.yaml"

    def test_md_extension_raises_config_error(self, tmp_path):
        """Passing a .md path to load() raises ConfigError about .yaml requirement."""
        md_file = tmp_path / "memintel.config.md"
        md_file.write_text("# some markdown")

        with pytest.raises(ConfigError, match=r"\.md format is no longer supported"):
            ConfigLoader().load(str(md_file))

    def test_valid_yaml_resolves_env_vars(self, monkeypatch, tmp_path):
        """A .yaml config file correctly resolves ${ENV_VAR} references."""
        monkeypatch.setenv("DB_PASSWORD", "resolved-secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-resolved")
        monkeypatch.setenv("MY_DB_HOST", "db.prod.example.com")

        config_file = tmp_path / "memintel_config.yaml"
        config_file.write_text(_minimal_config_yaml(extra_env_var="MY_DB_HOST"))

        config = ConfigLoader().load(str(config_file))
        assert config is not None
        assert config.llm.provider == "anthropic"


# ── GuardrailsStore ────────────────────────────────────────────────────────────

class TestGuardrailsStore:

    @pytest.fixture
    def guardrails_file(self, tmp_path) -> Path:
        """Write a minimal guardrails file and return the path."""
        f = tmp_path / "memintel_guardrails.yaml"
        f.write_text(_minimal_guardrails_yaml())
        return f

    @pytest.mark.asyncio
    async def test_load_populates_strategy_registry(self, guardrails_file):
        store = GuardrailsStore()
        await store.load(str(guardrails_file))

        registry = store.get_strategy_registry()
        assert "threshold" in registry

    @pytest.mark.asyncio
    async def test_get_application_context_returns_correct_name(self, guardrails_file):
        store = GuardrailsStore()
        await store.load(str(guardrails_file))

        ctx = store.get_application_context()
        assert ctx.name == "Test App"
        assert ctx.description == "Test application context"
        assert "Prefer simpler strategies" in ctx.instructions

    @pytest.mark.asyncio
    async def test_get_guardrails_returns_full_object(self, guardrails_file):
        from app.models.guardrails import Guardrails

        store = GuardrailsStore()
        await store.load(str(guardrails_file))

        guardrails = store.get_guardrails()
        assert isinstance(guardrails, Guardrails)

    @pytest.mark.asyncio
    async def test_get_threshold_bounds_returns_correct_values(self, tmp_path):
        """Threshold bounds defined in constraints are returned correctly."""
        yaml_text = textwrap.dedent("""\
            guardrails:
              application_context:
                name: Test
                description: Test
                instructions: []

              strategy_registry:
                threshold:
                  version: "1.0"
                  description: Threshold
                  input_types: [float]
                  output_type: boolean
                  parameters:
                    threshold:
                      type: float
                      required: true

              constraints:
                threshold_bounds:
                  threshold:
                    min: 0.01
                    max: 0.99
        """)

        f = tmp_path / "guardrails_with_bounds.yaml"
        f.write_text(yaml_text)

        store = GuardrailsStore()
        await store.load(str(f))

        bounds = store.get_threshold_bounds("threshold")
        assert bounds == {"min": 0.01, "max": 0.99}

    @pytest.mark.asyncio
    async def test_get_threshold_bounds_returns_empty_for_unknown_strategy(
        self, guardrails_file
    ):
        store = GuardrailsStore()
        await store.load(str(guardrails_file))

        bounds = store.get_threshold_bounds("nonexistent_strategy")
        assert bounds == {}

    @pytest.mark.asyncio
    async def test_load_twice_raises_runtime_error(self, guardrails_file):
        store = GuardrailsStore()
        await store.load(str(guardrails_file))

        with pytest.raises(RuntimeError, match="already populated"):
            await store.load(str(guardrails_file))

    @pytest.mark.asyncio
    async def test_missing_guardrails_file_raises_config_error(self):
        store = GuardrailsStore()

        with pytest.raises(ConfigError, match="not found"):
            await store.load("/nonexistent/path/memintel_guardrails.yaml")

    @pytest.mark.asyncio
    async def test_empty_strategy_registry_raises_config_error(self, tmp_path):
        """System must refuse to start if strategy_registry is empty."""
        yaml_text = textwrap.dedent("""\
            guardrails:
              application_context:
                name: Test
                description: Test
                instructions: []
              strategy_registry: {}
        """)
        f = tmp_path / "empty_registry.yaml"
        f.write_text(yaml_text)

        store = GuardrailsStore()
        with pytest.raises(ConfigError, match="strategy_registry is empty"):
            await store.load(str(f))

    def test_read_before_load_raises_runtime_error(self):
        """Accessing store methods before load() raises RuntimeError."""
        store = GuardrailsStore()

        with pytest.raises(RuntimeError, match="has not been loaded"):
            store.get_guardrails()

        with pytest.raises(RuntimeError, match="has not been loaded"):
            store.get_application_context()

        with pytest.raises(RuntimeError, match="has not been loaded"):
            store.get_strategy_registry()


# ── PrimitiveRegistry ─────────────────────────────────────────────────────────

class TestPrimitiveRegistry:

    def _make_config(self) -> ConfigSchema:
        """Build a ConfigSchema without env var resolution (host is plaintext)."""
        from app.models.config import (
            AccessConfig,
            ConnectorConfig,
            EnvironmentConfig,
            LLMConfig,
            PrimitiveConfig,
            SourceConfig,
        )

        return ConfigSchema(
            primitives=[
                PrimitiveConfig(
                    name="user.score",
                    type="float",
                    missing_data_policy="null",
                    source=SourceConfig(
                        type="database",
                        identifier="postgres.test",
                        field="score",
                        access=AccessConfig(
                            method="sql",
                            query="SELECT score FROM t WHERE user_id = :entity_id",
                        ),
                    ),
                ),
                PrimitiveConfig(
                    name="user.activity",
                    type="time_series<int>",
                    missing_data_policy="zero",
                    source=SourceConfig(
                        type="database",
                        identifier="postgres.test",
                        field="activity",
                        access=AccessConfig(
                            method="sql",
                            query=(
                                "SELECT recorded_at, activity FROM t "
                                "WHERE user_id = :entity_id"
                            ),
                        ),
                    ),
                ),
            ],
            connectors={
                "postgres.test": ConnectorConfig(
                    type="postgres",
                    host="localhost",
                    user="admin",
                    password="${DB_PASSWORD}",
                )
            },
            llm=LLMConfig(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                api_key="${ANTHROPIC_API_KEY}",
                endpoint="https://api.anthropic.com",
            ),
            environment=EnvironmentConfig(),
        )

    def test_load_from_config_registers_all_primitives(self):
        config = self._make_config()
        reg = PrimitiveRegistry()
        reg.load_from_config(config)

        assert len(reg.list_all()) == 2

    def test_get_returns_correct_primitive(self):
        config = self._make_config()
        reg = PrimitiveRegistry()
        reg.load_from_config(config)

        p = reg.get("user.score")
        assert p is not None
        assert p.name == "user.score"
        assert p.type == "float"

    def test_get_returns_none_for_unknown(self):
        reg = PrimitiveRegistry()
        assert reg.get("unknown.primitive") is None

    def test_list_all_returns_all_primitives(self):
        config = self._make_config()
        reg = PrimitiveRegistry()
        reg.load_from_config(config)

        names = {p.name for p in reg.list_all()}
        assert names == {"user.score", "user.activity"}

    def test_get_type_returns_memintel_type_string(self):
        config = self._make_config()
        reg = PrimitiveRegistry()
        reg.load_from_config(config)

        assert reg.get_type("user.score") == "float"
        assert reg.get_type("user.activity") == "time_series<int>"

    def test_get_type_raises_key_error_for_unknown(self):
        reg = PrimitiveRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get_type("unknown.signal")

    def test_register_adds_single_primitive(self):
        from app.models.config import AccessConfig, PrimitiveConfig, SourceConfig

        reg = PrimitiveRegistry()
        prim = PrimitiveConfig(
            name="event.clicks",
            type="int",
            source=SourceConfig(
                type="database",
                identifier="postgres.test",
                field="clicks",
                access=AccessConfig(method="sql", query="SELECT clicks FROM t"),
            ),
        )
        reg.register(prim)

        assert reg.get("event.clicks") is not None
        assert reg.get_type("event.clicks") == "int"

    def test_load_from_config_replaces_existing(self):
        """Calling load_from_config a second time replaces all existing entries."""
        from app.models.config import AccessConfig, PrimitiveConfig, SourceConfig

        reg = PrimitiveRegistry()

        # Register something first
        prim = PrimitiveConfig(
            name="old.signal",
            type="int",
            source=SourceConfig(
                type="database",
                identifier="postgres.test",
                field="x",
                access=AccessConfig(method="sql", query="SELECT x FROM t"),
            ),
        )
        reg.register(prim)
        assert reg.get("old.signal") is not None

        # load_from_config should replace everything
        config = self._make_config()
        reg.load_from_config(config)

        assert reg.get("old.signal") is None
        assert reg.get("user.score") is not None
