"""
app/models/config.py
──────────────────────────────────────────────────────────────────────────────
Configuration domain models for `memintel apply` and runtime data resolution.

Covers:
  - AccessConfig / SourceConfig / PrimitiveConfig   — data signal declarations
  - ConnectorConfig                                  — database and API connections
  - LLMConfig                                        — LLM provider for task authoring
  - RateLimitConfig / ExecutionConfig / EnvironmentConfig — runtime settings
  - ConfigSchema                                     — top-level merged YAML object
  - ApplicationContext                               — guardrails LLM injection block
  - ApplyResult                                      — ConfigApplyService.apply() response
  - PrimitiveValue                                   — DataResolutionService.fetch() response

Design notes
────────────
ENV_VAR_PATTERN is the canonical regex for credential field validation.
  Credential fields (password, api_key, auth tokens) MUST match ${ENV_VAR}
  syntax — never plaintext. The pattern is a module-level constant so that
  ConnectorConfig, LLMConfig, and any future credential-bearing models share
  the same enforcement rule.

PrimitiveConfig.type validation accepts all Memintel types from
  memintel_type_system.md v1.1 including nullable variants (float?).
  The exact set is defined here as VALID_PRIMITIVE_TYPES rather than
  delegating to MemintelType.NODE_OUTPUT_TYPES — primitive types include
  time_series and list containers that are not valid node output types.

ConnectorConfig is a union-style model — different connector types (postgres,
  mysql, rest, kafka) use different subsets of fields. All type-specific
  optional fields are listed explicitly rather than using model_config extra='allow',
  so that the validator layer rejects unknown field names at config load time.

ConfigSchema validators run in dependency order:
  1. connectors validated first (no cross-references)
  2. primitives validated second (references connectors for identifier check)
  3. Cross-reference validation uses info.data['connectors'] from the previous step.

ApplicationContext is extracted from the guardrails YAML block and stored
  separately by ConfigApplyService for injection into LLM prompts. It has
  zero influence on execution, evaluation, or calibration — only POST /tasks.

PrimitiveValue carries a raw value (Any) because the actual Python type
  depends on the Memintel type: float/int/bool/str for scalars, list[dict]
  for time_series (each entry has 'timestamp' and 'value' keys), list[float]
  or list[int] for list types, and None for nullable types with missing data.
  The type string is preserved so the runtime can validate before use.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


# ── Credential validation constant ────────────────────────────────────────────

#: All credential fields MUST match this pattern — never plaintext.
#: Syntax: ${VARIABLE_NAME} where VARIABLE_NAME is uppercase alphanumeric + underscores.
ENV_VAR_PATTERN: re.Pattern[str] = re.compile(r'^\$\{[A-Z][A-Z0-9_]*\}$')

#: All valid primitive type strings per memintel_type_system.md v1.1.
#: Includes nullable variants. Used by PrimitiveConfig.validate_memintel_type.
VALID_PRIMITIVE_TYPES: frozenset[str] = frozenset({
    "float", "int", "boolean", "string", "categorical",
    "time_series<float>", "time_series<int>",
    "list<float>", "list<int>",
    # Nullable variants
    "float?", "int?", "boolean?", "string?", "categorical?",
    "time_series<float>?", "time_series<int>?",
    "list<float>?", "list<int>?",
})


# ── Access and source config ───────────────────────────────────────────────────

class AccessConfig(BaseModel):
    """
    How the connector fetches data for a primitive.

    method determines the access pattern:
      sql   — SQL query with :entity_id and :as_of placeholders
      rest  — HTTP request; path uses :entity_id substitution
      kafka — stream consumer (for streaming primitives)

    SQL query placeholders:
      :entity_id — replaced with the entity string at fetch time
      :as_of     — replaced with the timestamp (ISO 8601 UTC) or NOW() in
                   snapshot mode. All fetches within one evaluation call use
                   the same mode — mixing is not allowed.
    """
    method: str                         # sql | rest | kafka
    query: str | None = None            # SQL with :entity_id and :as_of
    path: str | None = None             # REST path, e.g. /metrics/:entity_id
    entity_param: str | None = None     # REST query param name for entity
    timestamp_param: str | None = None  # REST query param name for as_of


class SourceConfig(BaseModel):
    """
    Data source binding for a primitive.

    type describes the source category (database, api, stream).
    identifier is the connector name declared in the connectors block —
    format: type.label (e.g. postgres.analytics, rest.billing_api).
    field is the column or response field to extract from the raw data.
    """
    type: str           # database | api | stream
    identifier: str     # connector name — must exist in ConfigSchema.connectors
    field: str
    access: AccessConfig


class PrimitiveConfig(BaseModel):
    """
    A named data signal available to the LLM for concept generation.

    Primitive names follow namespace.field format:
      Examples: user.churn_score, payment.failure_rate, events.page_views
      - namespace is a domain label (user, payment, order, events)
      - Do NOT use definition namespaces (org, team, personal, global)
        as primitive namespaces — they serve different purposes

    missing_data_policy determines behaviour when the fetch returns no data:
      null          — return T? (nullable); propagates null through the DAG
      zero          — substitute 0; non-nullable
      forward_fill  — last known value before the timestamp; non-nullable
      backward_fill — next known value after the timestamp; non-nullable
    """
    name: str                               # namespace.field format
    type: str                               # Memintel type string
    missing_data_policy: str = "null"       # null | zero | forward_fill | backward_fill
    source: SourceConfig

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if v.count(".") != 1:
            raise ValueError(
                f"Primitive name must be 'namespace.field', got: {v!r}. "
                f"Example: user.churn_score"
            )
        namespace, field = v.split(".", 1)
        if not re.match(r'^[a-z][a-z0-9_]*$', namespace):
            raise ValueError(
                f"Primitive namespace must be lowercase alphanumeric + underscores, "
                f"got: {namespace!r}"
            )
        if not re.match(r'^[a-z][a-z0-9_]*$', field):
            raise ValueError(
                f"Primitive field name must be lowercase alphanumeric + underscores, "
                f"got: {field!r}"
            )
        return v

    @field_validator("missing_data_policy")
    @classmethod
    def _validate_policy(cls, v: str) -> str:
        valid = {"null", "zero", "forward_fill", "backward_fill"}
        if v not in valid:
            raise ValueError(
                f"missing_data_policy must be one of {sorted(valid)}, got: {v!r}"
            )
        return v

    @field_validator("type")
    @classmethod
    def _validate_memintel_type(cls, v: str) -> str:
        if v not in VALID_PRIMITIVE_TYPES:
            raise ValueError(
                f"Invalid Memintel type: {v!r}. "
                f"Must be a type defined in memintel_type_system.md v1.1. "
                f"Valid scalars: float, int, boolean, string, categorical. "
                f"Valid containers: time_series<float>, time_series<int>, "
                f"list<float>, list<int>. Append '?' for nullable. "
                f"Invalid examples: float_array, string_list."
            )
        return v


# ── Connector config ───────────────────────────────────────────────────────────

class ConnectorConfig(BaseModel):
    """
    A named database or API connection. Credentials must use ${ENV_VAR} references.

    Connector names follow type.label format:
      Examples: postgres.analytics, rest.payments_api, kafka.event_stream
      The label is free-form but must be unique within the config.

    All type-specific optional fields are listed explicitly to reject unknown
    field names at config load time. Fields that don't apply to a connector
    type default to None and are ignored at runtime.

    Postgres/MySQL fields: host, port, database, user, password, pool_min, pool_max,
                           connect_timeout_ms
    REST fields:           base_url, auth, timeout_ms, retry_max
    Kafka fields:          brokers, consumer_group, topics, auth,
                           auto_offset_reset, fetch_timeout_ms
    Shared fields:         timeout_ms, retry_max

    Security rule: password and any auth.token / auth.password values MUST be
    ${ENV_VAR} references. ConfigLoader validates this after schema validation.
    Resolved values are never logged at any level.
    """
    type: str                                   # postgres | mysql | rest | kafka

    # Relational database fields (postgres, mysql)
    host: str | None = None
    port: int | None = None
    database: str | None = None
    user: str | None = None
    password: str | None = None                 # MUST be ${ENV_VAR} if present
    pool_min: int = 2
    pool_max: int = 10
    connect_timeout_ms: int | None = None

    # REST connector fields
    base_url: str | None = None                 # REST connectors
    auth: dict[str, Any] | None = None          # type: bearer|api_key|basic + credentials

    # Kafka connector fields
    brokers: list[str] | None = None            # list of broker addresses
    consumer_group: str | None = None
    topics: list[str] | None = None
    auto_offset_reset: str | None = None        # latest | earliest
    fetch_timeout_ms: int | None = None

    # Shared tuning fields
    timeout_ms: int = 10000
    retry_max: int = 3

    @field_validator("password", mode="before")
    @classmethod
    def _validate_password(
        cls, v: str | None, info: ValidationInfo
    ) -> str | None:
        # Skip format check when re-validating after env var resolution
        if info.context and info.context.get("resolved"):
            return v
        if v is not None and not ENV_VAR_PATTERN.match(str(v)):
            raise ValueError(
                "Credential fields must use ${ENV_VAR} references. "
                "Found plaintext value in password field."
            )
        return v


# ── LLM config ────────────────────────────────────────────────────────────────

#: Valid LLM provider identifiers.
VALID_LLM_PROVIDERS: frozenset[str] = frozenset({
    "anthropic",
    "openai",
    "azure_openai",
    "ollama",
})


class LLMConfig(BaseModel):
    """
    LLM provider configuration for task authoring (POST /tasks only).

    The LLM is NEVER called on any execution path. This config applies
    exclusively to TaskAuthoringService and POST /agents/* endpoints.

    temperature: 0 is strongly recommended — higher values introduce
      non-determinism in strategy selection and parameter generation that
      the guardrails system cannot fully compensate for.

    max_retries: how many times the refinement loop retries when the LLM
      produces a definition that fails compiler validation. After max_retries
      exhausted, POST /tasks returns HTTP 422 semantic_error.

    api_key MUST be an ${ENV_VAR} reference — never plaintext.
    """
    provider: str       # anthropic | openai | azure_openai | ollama
    model: str
    api_key: str        # MUST be ${ENV_VAR}
    endpoint: str
    timeout_ms: int = 30000
    max_retries: int = 3
    temperature: float = 0

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        if v not in VALID_LLM_PROVIDERS:
            raise ValueError(
                f"Invalid LLM provider: {v!r}. "
                f"Must be one of: {sorted(VALID_LLM_PROVIDERS)}. "
                f"For other providers, implement a provider adapter."
            )
        return v

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, v: str, info: ValidationInfo) -> str:
        # Skip format check when re-validating after env var resolution
        if info.context and info.context.get("resolved"):
            return v
        if not ENV_VAR_PATTERN.match(v):
            raise ValueError(
                "LLM api_key must be an ${ENV_VAR} reference, got plaintext value."
            )
        return v


# ── Environment config ─────────────────────────────────────────────────────────

class RateLimitConfig(BaseModel):
    """
    Per-API-key rate limit settings.

    Scope is per API key — NOT global, NOT per IP address.
    burst allows short spikes within the same key scope.
    HTTP 429 responses include Retry-After header; SDK exposes retryAfterSeconds.
    """
    requests_per_minute: int = 600
    burst: int = 60


class ExecutionConfig(BaseModel):
    """
    Runtime execution tuning parameters.

    sync_timeout_ms — maximum duration for synchronous execution
      (evaluateFull, execute). Returns HTTP 504 execution_timeout on breach.

    async_poll_interval_ms — default poll interval hint returned in
      Job.poll_interval_seconds for async execution jobs.

    max_batch_size — maximum number of entities in a single
      evaluateConditionBatch call.
    """
    sync_timeout_ms: int = 30000
    async_poll_interval_ms: int = 2000
    max_batch_size: int = 100


class EnvironmentConfig(BaseModel):
    """
    Runtime environment settings applied globally at startup.

    namespace — default definition namespace for task creation.
      personal  — visible only to the creating API key
      team      — shared within a team scope
      org       — shared across the organisation (recommended default)
      global    — platform-wide (requires elevated API key to promote to)

    log_format:
      json — structured JSON (recommended for production)
      text — human-readable (useful in development)

    skip_connector_health_check — set True in production if connector
      health checks during startup cause unacceptable latency.
    """
    namespace: str = "org"              # personal | team | org | global
    log_level: str = "INFO"             # DEBUG | INFO | WARNING | ERROR
    log_format: str = "json"            # json | text
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    skip_connector_health_check: bool = False


# ── Top-level config schema ────────────────────────────────────────────────────

class ConfigSchema(BaseModel):
    """
    Top-level validated config object. Produced by ConfigLoader from
    `memintel_config.yaml`.

    Validation is two-pass:
      Pass 1 — schema validation (field types, enum values, credential format)
      Pass 2 — cross-reference validation (primitive → connector, env var resolution)

    ConfigLoader raises ConfigError on any failure. The system must not
    start with a partially valid config — all-or-nothing.

    guardrails_path defaults to 'memintel_guardrails.yaml' in the same
    directory as the config file. ConfigLoader resolves this to an absolute
    path before loading the guardrails.
    """
    primitives: list[PrimitiveConfig]
    connectors: dict[str, ConnectorConfig]
    llm: LLMConfig
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    guardrails_path: str = "memintel_guardrails.yaml"

    @field_validator("primitives")
    @classmethod
    def _validate_unique_names(
        cls, primitives: list[PrimitiveConfig]
    ) -> list[PrimitiveConfig]:
        names = [p.name for p in primitives]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Primitive names must be unique. "
                f"Duplicates found: {sorted(duplicates)}"
            )
        return primitives

    @model_validator(mode="after")
    def _validate_primitive_connectors(self) -> ConfigSchema:
        """Verify every primitive's source.identifier exists in connectors."""
        connector_names = set(self.connectors.keys())
        for p in self.primitives:
            if p.source.identifier not in connector_names:
                raise ValueError(
                    f"Primitive '{p.name}' references unknown connector "
                    f"'{p.source.identifier}'. "
                    f"Available connectors: {sorted(connector_names)}"
                )
        return self


# ── ApplicationContext ─────────────────────────────────────────────────────────

class ApplicationContext(BaseModel):
    """
    Domain description and LLM behavioral instructions extracted from the
    guardrails file (memintel.guardrails.md §2).

    Stored separately by ConfigApplyService for injection into LLM system
    prompts during task authoring. Has zero influence on execution,
    evaluation, calibration, or feedback — POST /tasks only.

    action_preferences maps severity labels to action type strings:
      Example: {"high_severity": "workflow", "medium_severity": "notification"}
    This is position 3 in the action binding resolution order, after
    user-explicit and guardrails default, and subject to constraints (§10
    of memintel.guardrails.md).
    """
    name: str
    description: str
    instructions: list[str]
    default_entity_scope: str | None = None
    action_preferences: dict[str, Any] = Field(default_factory=dict)


# ── ConfigApplyService result ──────────────────────────────────────────────────

class ApplyResult(BaseModel):
    """
    Response from ConfigApplyService.apply().

    Summarises what was registered during `memintel apply`.
    All fields reflect the state after a successful, atomic apply.
    If apply failed, ConfigError was raised and no partial state was written.

    warnings lists advisory messages that did not prevent apply from
    succeeding (e.g. a time_series primitive with a SUM aggregate query,
    or a deprecated connector option).
    """
    primitives_registered: int
    connectors_registered: int
    guardrails_loaded: bool = True
    warnings: list[str] = Field(default_factory=list)


# ── PrimitiveValue ─────────────────────────────────────────────────────────────

class PrimitiveValue(BaseModel):
    """
    The resolved value of a primitive fetch. Returned by DataResolutionService.fetch().

    value carries the Python representation of the fetched data:
      float / int / bool / str     — for scalar types (float, int, boolean, categorical)
      list[dict[str, Any]]         — for time_series types; each entry has
                                     'timestamp' (ISO 8601 str) and 'value' keys
      list[float] | list[int]      — for list<float> / list<int>
      None                         — for nullable types (T?) when data is missing
                                     and missing_data_policy='null'

    type is the Memintel type string from the primitive declaration.
    The executor validates the returned type before feeding it into the
    execution graph.

    deterministic reflects the fetch mode:
      True  — timestamp was provided; same entity + timestamp always returns
              the same value. Result is cacheable indefinitely.
      False — snapshot mode (no timestamp); reflects NOW(). MUST NOT be
              cached across requests.
    """
    value: Any
    type: str         # Memintel type string, e.g. 'float', 'time_series<int>'
    deterministic: bool
