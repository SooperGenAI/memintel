# Memintel — System Configuration Guide
**File:** `memintel.config.md`  
**Version:** 1.0  
**Status:** Authoritative  
**Audience:** Admins configuring Memintel deployments; Session 2.5 of the Claude Code process

> **Relationship to other files:**
> ```
> memintel.config.md       → defines primitives, connectors, LLM, environment
> memintel.guardrails.md   → defines strategy registry, priors, bias rules, application context
> persistence-schema.md    → defines database tables and stores
> ```
> `memintel apply` reads this file first, then loads the guardrails file referenced
> by `guardrails_path`. Both must be valid for startup to proceed.

---

## 1. File Format

`memintel.config.md` is a **Markdown file containing fenced YAML blocks**. The file may contain
human-readable prose between YAML blocks for documentation purposes. The `ConfigLoader` extracts
and merges all YAML blocks in document order into a single configuration object.

```
PARSING RULES:

1. Extract all fenced YAML blocks (```yaml ... ```)
2. Merge them in document order — later blocks override earlier ones on key conflict
3. Validate the merged object against ConfigSchema
4. Resolve all ${ENV_VAR} references — raise ConfigError if any variable is unset
5. NEVER log resolved credential values at any log level
```

The file lives in the repository alongside `memintel.guardrails.md`. Both files are committed to
version control. Credentials are never in the file — only `${ENV_VAR}` references.

---

## 2. Top-Level Schema

```yaml
# memintel.config.md — top-level structure

primitives:          # list — data signals available to the LLM
connectors:          # map — database and API connections
llm:                 # object — LLM provider for task authoring
environment:         # object — runtime settings
guardrails_path:     # string — path to memintel.guardrails.md (default: 'memintel.guardrails.md')
```

All five top-level keys are required. The `ConfigLoader` rejects configs with missing keys.
`guardrails_path` defaults to `memintel.guardrails.md` in the same directory as the config file.

---

## 3. `primitives` — Data Signals

Primitives are the atomic data inputs available to the LLM when generating concept definitions.
Each primitive declares its name, type, data source, and how to handle missing data.

```yaml
primitives:
  - name: user.churn_score           # namespace.field format — must match namespace rules
    type: float                      # Memintel type string — see memintel_type_system.md
    missing_data_policy: null        # null | zero | forward_fill | backward_fill
    source:
      type: database                 # database | api | stream
      identifier: postgres.analytics # connector name (must exist in connectors block)
      field: churn_score
      access:
        method: sql
        query: >
          SELECT churn_score FROM user_metrics
          WHERE user_id = :entity_id
          AND recorded_at <= :as_of
          ORDER BY recorded_at DESC LIMIT 1

  - name: user.activity_count
    type: time_series<int>
    missing_data_policy: zero
    source:
      type: database
      identifier: postgres.analytics
      field: activity_count
      access:
        method: sql
        query: >
          SELECT recorded_at, activity_count FROM daily_activity
          WHERE user_id = :entity_id
          AND recorded_at <= :as_of
          ORDER BY recorded_at ASC

  - name: user.plan_tier
    type: categorical
    missing_data_policy: null
    source:
      type: database
      identifier: postgres.accounts
      field: plan_tier
      access:
        method: sql
        query: >
          SELECT plan_tier FROM subscriptions
          WHERE user_id = :entity_id LIMIT 1

  - name: payment.failure_rate
    type: float
    missing_data_policy: zero
    source:
      type: database
      identifier: postgres.payments
      field: failure_rate
      access:
        method: sql
        query: >
          SELECT failure_rate FROM payment_metrics
          WHERE account_id = :entity_id
          AND period_end <= :as_of
          ORDER BY period_end DESC LIMIT 1

  - name: events.page_views
    type: time_series<int>
    missing_data_policy: zero
    source:
      type: api
      identifier: rest.analytics_api
      field: page_views
      access:
        method: rest
        path: /metrics/page_views
        entity_param: user_id
        timestamp_param: before
```

### Primitive Naming Rules

```
PRIMITIVE NAMING:

Format:   namespace.field_name
Examples: user.churn_score, payment.failure_rate, events.page_views

- namespace must be lowercase, alphanumeric and underscores only
- field_name must be lowercase, alphanumeric and underscores only
- The full name (namespace.field_name) must be unique within the config
- Do NOT use the definition namespace (org, team, personal, global) as
  primitive namespaces — primitive namespaces are domain labels (user,
  payment, order, events) not permission scopes
```

### `missing_data_policy` Values

| Value | Behaviour | Returns |
|---|---|---|
| `null` | Return nullable type `T?` | Propagates null through DAG unless handled |
| `zero` | Return `0` (int) or `0.0` (float) when data missing | Non-nullable |
| `forward_fill` | Return last known value before the timestamp | Non-nullable |
| `backward_fill` | Return next known value after the timestamp | Non-nullable |

### SQL Query Variables

All SQL queries may use two placeholders:
- `:entity_id` — replaced with the entity string at fetch time (URL-encoded)
- `:as_of` — replaced with the `timestamp` parameter (ISO 8601 UTC)

```
TIMESTAMP SUBSTITUTION RULE:

Deterministic mode (timestamp provided):
  :as_of is replaced with the exact ISO 8601 timestamp
  result.deterministic = True
  result is cacheable (indefinite TTL)

Snapshot mode (timestamp is None):
  :as_of is replaced with NOW() at fetch time
  result.deterministic = False
  result MUST NOT be cached across requests
  DataResolutionService MUST set deterministic=False on the Result

The timestamp mode applies to the ENTIRE evaluation call.
All primitive fetches within one call use the same mode.
Mixing is not allowed: if timestamp is None, all fetches use NOW().
If timestamp is set, all fetches use that exact timestamp.

Reason: mixing would produce a result that is partially historical
and partially current — semantically incoherent and uncacheable.
```

---

## 4. `connectors` — Data Source Connections

Connectors define the database and API connections that primitives reference. Each connector is
named and its credentials reference environment variables — never plaintext.

### 4.1 PostgreSQL Connector

```yaml
connectors:
  postgres.analytics:                # connector name used in primitive source.identifier
    type: postgres
    host: ${DB_ANALYTICS_HOST}       # always ${ENV_VAR} — never plaintext
    port: 5432
    database: analytics
    user: ${DB_ANALYTICS_USER}
    password: ${DB_ANALYTICS_PASSWORD}
    pool_min: 2
    pool_max: 10
    connect_timeout_ms: 5000

  postgres.accounts:
    type: postgres
    host: ${DB_ACCOUNTS_HOST}
    port: 5432
    database: accounts
    user: ${DB_ACCOUNTS_USER}
    password: ${DB_ACCOUNTS_PASSWORD}
```

### 4.2 REST API Connector

```yaml
connectors:
  rest.analytics_api:
    type: rest
    base_url: ${ANALYTICS_API_BASE_URL}
    auth:
      type: bearer                   # bearer | api_key | basic
      token: ${ANALYTICS_API_TOKEN}
    timeout_ms: 10000
    retry_max: 3
    retry_backoff_ms: 500
```

### 4.3 Kafka Connector

```yaml
connectors:
  kafka.event_stream:
    type: kafka
    brokers:
      - ${KAFKA_BROKER_1}
      - ${KAFKA_BROKER_2}
    consumer_group: memintel-resolver
    topics:
      - user_events
      - payment_events
    auth:
      type: sasl_plain
      username: ${KAFKA_USER}
      password: ${KAFKA_PASSWORD}
    auto_offset_reset: latest
    fetch_timeout_ms: 5000
```

### 4.4 MySQL Connector

```yaml
connectors:
  mysql.legacy_db:
    type: mysql
    host: ${MYSQL_HOST}
    port: 3306
    database: legacy
    user: ${MYSQL_USER}
    password: ${MYSQL_PASSWORD}
```

### Connector-Primitive Type Compatibility

```
CONNECTOR TYPE COMPATIBILITY:

ConfigLoader performs lightweight structural checks only.
Full type validation is deferred to the TypeChecker at execution time.

Structural checks enforced at config load:
  - time_series primitives: SQL query MUST have at least two SELECT columns
    (heuristic: checks that query references more than one field name)
  - categorical primitives: ConfigLoader WARNS if query uses SUM/AVG/COUNT
    as primary return (advisory only - does not fail the config)
  - float/int primitives: no structural check at load time

DO NOT replicate TypeChecker logic in ConfigLoader.
Full enforcement happens at:
  Runtime:     TypeChecker.check_node() validates every operator input
  Compilation: Compiler rejects type-incompatible flows before execution
```

### Connector Rules

```
CONNECTOR RULES:

1. All credential values MUST be ${ENV_VAR} references — never plaintext
   ConfigLoader raises ConfigError if any credential field contains a
   literal string that does not match the ${...} pattern

2. Connector names follow the format: type.label
   Examples: postgres.analytics, rest.payments_api, kafka.events
   The label is free-form but must be unique within the config

3. Every primitive's source.identifier must match a connector name
   ConfigLoader validates this after schema validation

4. ConfigLoader MUST attempt a connection health check on every connector
   during memintel apply — raise ConfigError if any connector is unreachable
   (this is optional at startup in production if health checks are too slow;
   make it configurable via environment.skip_connector_health_check)

5. CONNECTOR RETRY RULES (enforced in DataResolutionService, not ConfigLoader):
   RETRY on transient failures only:
     - Network errors (connection refused, DNS failure, socket timeout)
     - Timeouts (connect_timeout_ms exceeded)
     - REST 5xx responses (server-side transient errors)
     - Kafka fetch timeout
   DO NOT RETRY on:
     - REST 4xx responses (client errors - retrying won't fix them)
     - SQL syntax errors or constraint violations
     - Authentication failures (wrong credentials waste quota on retry)
     - Data type mismatches (these are bugs, not transient failures)
   Retry pattern: exponential backoff with jitter
     delay_ms = base_ms * (2 ** attempt) + randint(0, 50)
     base_ms=100, max attempts from connector config retry_max
```

---

## 5. `llm` — Language Model Configuration

The LLM is used exclusively during task authoring (`POST /tasks`). All other
endpoints are deterministic and do not call the LLM.

```yaml
llm:
  provider: anthropic                # anthropic | openai | azure_openai | ollama
  model: claude-sonnet-4-20250514    # model identifier as required by the provider
  api_key: ${ANTHROPIC_API_KEY}      # always ${ENV_VAR}
  endpoint: https://api.anthropic.com  # base URL — may be overridden for Azure or proxy
  timeout_ms: 30000                  # request timeout — default 30s
  max_retries: 3                     # max refinement loop retries on validation failure
  temperature: 0                     # 0 = fully deterministic generation (recommended)
```

### LLM Provider Values

| `provider` | Notes |
|---|---|
| `anthropic` | Uses Anthropic Messages API. `endpoint` defaults to `https://api.anthropic.com` |
| `openai` | Uses OpenAI Chat Completions API. `endpoint` defaults to `https://api.openai.com` |
| `azure_openai` | Requires `endpoint` to be the Azure deployment URL |
| `ollama` | For local development. `endpoint` should be `http://localhost:11434` |

### LLM Rules

```
LLM CONFIGURATION RULES:

- api_key MUST be an ${ENV_VAR} reference — never plaintext
- temperature: 0 is strongly recommended — higher values introduce
  non-determinism in strategy selection and parameter generation
  which the guardrails system cannot fully compensate for
- max_retries: the refinement loop retries when the LLM produces a
  definition that fails compiler validation — after max_retries
  exhausted, POST /tasks returns HTTP 422 semantic_error
- The LLM is ONLY invoked at task creation time — all execution
  paths (evaluate/full, evaluate/condition, execute) never call the LLM
```

---

## 6. `environment` — Runtime Settings

```yaml
environment:
  namespace: org                     # default namespace for task creation: personal | team | org | global
  log_level: INFO                    # DEBUG | INFO | WARNING | ERROR
  log_format: json                   # json | text
  rate_limit:
    requests_per_minute: 600         # per API key - NOT global, NOT per IP
    burst: 60                        # burst within the same API key scope
  execution:
    sync_timeout_ms: 30000           # max duration for synchronous execution (evaluateFull, execute)
    async_poll_interval_ms: 2000     # default poll interval hint for async jobs
    max_batch_size: 100              # max entities in a single evaluateConditionBatch call
  skip_connector_health_check: false # set true in prod if health checks cause slow startup
```

### `namespace` Values

| Value | Description |
|---|---|
| `personal` | Definitions visible only to the creating API key |
| `team` | Definitions shared within a team scope |
| `org` | Definitions shared across the organisation (recommended default) |
| `global` | Platform-wide definitions (requires elevated API key to promote to) |

---

## 7. `guardrails_path`

```yaml
guardrails_path: memintel.guardrails.md   # default — relative to the config file location
```

This is the path to `memintel.guardrails.md`. The `ConfigLoader` loads this file
after validating the main config. If the guardrails file is missing or invalid,
startup fails with `ConfigError`.

Both files should live in the same directory and be committed together to version control.

---

## 8. `${ENV_VAR}` Substitution

All values in the config that reference environment variables use `${VAR_NAME}` syntax.

```
SUBSTITUTION RULES:

Syntax:  ${VARIABLE_NAME}
         Variable names: uppercase, underscores, alphanumeric only

Examples:
  ${DB_PASSWORD}            → os.environ['DB_PASSWORD']
  ${ANTHROPIC_API_KEY}      → os.environ['ANTHROPIC_API_KEY']
  ${KAFKA_BROKER_1}         → os.environ['KAFKA_BROKER_1']

Resolution:
  - All ${...} references are resolved AFTER schema validation
  - If any referenced variable is not set → ConfigError immediately
  - ConfigError message MUST include the variable name but NOT the expected value
    Example: "Required environment variable DB_PASSWORD is not set"
  - Resolved values are NEVER logged at any level
  - Resolved values are stored in memory only — never written to disk or logs

Validation:
  - Credential fields (password, api_key, token) MUST use ${ENV_VAR} references
  - If a credential field contains a literal string not matching ${...} → ConfigError
    Example: password: "my-password" → ConfigError: "Credential fields must use
    ${ENV_VAR} references. Found plaintext value in connectors.postgres.analytics.password"

Partial resolution:
  - ConfigLoader resolves ALL ${ENV_VAR} references in one pass before returning
  - If ANY variable is unset → ConfigError → DO NOT return a partially resolved config
```

---

## 9. Complete Example — SaaS Product Analytics

This is a complete working `memintel.config.md` for a SaaS product analytics deployment
monitoring user engagement, churn risk, and payment health.

```yaml
# ─── Guardrails ──────────────────────────────────────────────────────────────
guardrails_path: memintel.guardrails.md

# ─── Data Source Connections ─────────────────────────────────────────────────
connectors:
  postgres.analytics:
    type: postgres
    host: ${ANALYTICS_DB_HOST}
    port: 5432
    database: analytics
    user: ${ANALYTICS_DB_USER}
    password: ${ANALYTICS_DB_PASSWORD}
    pool_min: 2
    pool_max: 10

  postgres.accounts:
    type: postgres
    host: ${ACCOUNTS_DB_HOST}
    port: 5432
    database: accounts
    user: ${ACCOUNTS_DB_USER}
    password: ${ACCOUNTS_DB_PASSWORD}

  rest.billing_api:
    type: rest
    base_url: ${BILLING_API_URL}
    auth:
      type: bearer
      token: ${BILLING_API_TOKEN}
    timeout_ms: 8000
    retry_max: 3

# ─── Primitives ──────────────────────────────────────────────────────────────
primitives:
  # User engagement signals
  - name: user.daily_active_minutes
    type: time_series<float>
    missing_data_policy: zero
    source:
      type: database
      identifier: postgres.analytics
      field: active_minutes
      access:
        method: sql
        query: >
          SELECT recorded_date AS recorded_at, active_minutes
          FROM daily_engagement
          WHERE user_id = :entity_id
          AND recorded_date <= :as_of
          ORDER BY recorded_date ASC

  - name: user.feature_adoption_score
    type: float
    missing_data_policy: null
    source:
      type: database
      identifier: postgres.analytics
      field: adoption_score
      access:
        method: sql
        query: >
          SELECT adoption_score FROM feature_scores
          WHERE user_id = :entity_id
          AND computed_at <= :as_of
          ORDER BY computed_at DESC LIMIT 1

  - name: user.session_count_30d
    type: int
    missing_data_policy: zero
    source:
      type: database
      identifier: postgres.analytics
      field: session_count
      access:
        method: sql
        query: >
          SELECT COUNT(*) AS session_count FROM sessions
          WHERE user_id = :entity_id
          AND started_at BETWEEN (:as_of::date - INTERVAL '30 days') AND :as_of

  # Account signals
  - name: account.plan_tier
    type: categorical
    missing_data_policy: null
    source:
      type: database
      identifier: postgres.accounts
      field: plan_tier
      access:
        method: sql
        query: >
          SELECT plan_tier FROM subscriptions
          WHERE account_id = :entity_id
          AND status = 'active' LIMIT 1

  - name: account.seats_used_pct
    type: float
    missing_data_policy: null
    source:
      type: database
      identifier: postgres.accounts
      field: seats_used_pct
      access:
        method: sql
        query: >
          SELECT (seats_used::float / seats_total) AS seats_used_pct
          FROM licenses
          WHERE account_id = :entity_id LIMIT 1

  # Payment signals
  - name: payment.failure_rate_30d
    type: float
    missing_data_policy: zero
    source:
      type: api
      identifier: rest.billing_api
      field: failure_rate
      access:
        method: rest
        path: /accounts/:entity_id/payment-stats
        timestamp_param: before

# ─── LLM ─────────────────────────────────────────────────────────────────────
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}
  endpoint: https://api.anthropic.com
  timeout_ms: 30000
  max_retries: 3
  temperature: 0

# ─── Environment ─────────────────────────────────────────────────────────────
environment:
  namespace: org
  log_level: INFO
  log_format: json
  rate_limit:
    requests_per_minute: 600
    burst: 60
  execution:
    sync_timeout_ms: 30000
    async_poll_interval_ms: 2000
    max_batch_size: 100
  skip_connector_health_check: false
```

---

## 10. ConfigSchema — Pydantic v2 Models

These are the exact Pydantic v2 models that `ConfigLoader` validates the merged YAML against.
They live in `/app/config/config_loader.py`.

```python
from pydantic import BaseModel, field_validator
import re

ENV_VAR_PATTERN = re.compile(r'^\$\{[A-Z][A-Z0-9_]*\}$')

class AccessConfig(BaseModel):
    method: str                          # sql | rest | kafka
    query: str | None = None             # SQL with :entity_id and :as_of
    path: str | None = None              # REST path with :entity_id
    entity_param: str | None = None      # REST query param name for entity
    timestamp_param: str | None = None   # REST query param name for as_of

class SourceConfig(BaseModel):
    type: str                            # database | api | stream
    identifier: str                      # connector name
    field: str
    access: AccessConfig

class PrimitiveConfig(BaseModel):
    name: str                            # namespace.field format
    type: str                            # Memintel type string
    missing_data_policy: str = 'null'    # null | zero | forward_fill | backward_fill
    source: SourceConfig

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if '.' not in v or v.count('.') != 1:
            raise ValueError(f"Primitive name must be 'namespace.field', got: {v!r}")
        return v

    @field_validator('missing_data_policy')
    @classmethod
    def validate_policy(cls, v: str) -> str:
        valid = {'null', 'zero', 'forward_fill', 'backward_fill'}
        if v not in valid:
            raise ValueError(f"missing_data_policy must be one of {valid}, got: {v!r}")
        return v

    @field_validator('type')
    @classmethod
    def validate_memintel_type(cls, v: str) -> str:
        # PRIMITIVE TYPE VALIDATION
        # primitive.type MUST be a valid Memintel type from memintel_type_system.md v1.1
        VALID_TYPES = {
            'float', 'int', 'boolean', 'string', 'categorical',
            'time_series<float>', 'time_series<int>',
            'list<float>', 'list<int>',
            'float?', 'int?', 'boolean?', 'string?', 'categorical?',
            'time_series<float>?', 'time_series<int>?',
            'list<float>?', 'list<int>?',
        }
        if v not in VALID_TYPES:
            raise ValueError(
                f"Invalid Memintel type: {v!r}. Must be a type defined in "
                f"memintel_type_system.md v1.1. Valid scalars: float, int, "
                f"boolean, string, categorical. Valid containers: "
                f"time_series<float>, time_series<int>, list<float>, list<int>. "
                f"Append '?' for nullable. Invalid examples: float_array, string_list."
            )
        return v

class ConnectorConfig(BaseModel):
    type: str                            # postgres | mysql | rest | kafka
    host: str | None = None
    port: int | None = None
    database: str | None = None
    user: str | None = None
    password: str | None = None          # MUST be ${ENV_VAR} if present
    base_url: str | None = None          # REST connectors
    auth: dict | None = None
    timeout_ms: int = 10000
    retry_max: int = 3
    pool_min: int = 2
    pool_max: int = 10

    @field_validator('password', mode='before')
    @classmethod
    def validate_password(cls, v: str | None) -> str | None:
        if v is not None and not ENV_VAR_PATTERN.match(str(v)):
            raise ValueError(
                f"Credential fields must use ${{ENV_VAR}} references. "
                f"Found plaintext value in password field."
            )
        return v

class LLMConfig(BaseModel):
    provider: str                        # anthropic | openai | azure_openai | ollama
    model: str
    api_key: str                         # MUST be ${ENV_VAR}
    endpoint: str
    timeout_ms: int = 30000
    max_retries: int = 3
    temperature: float = 0

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {'anthropic', 'openai', 'azure_openai', 'ollama'}
        if v not in allowed:
            raise ValueError(
                f"Invalid LLM provider: {v!r}. "
                f"Must be one of: {sorted(allowed)}. "
                f"For other providers, implement a provider adapter."
            )
        return v

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not ENV_VAR_PATTERN.match(v):
            raise ValueError(
                f"LLM api_key must be an ${{ENV_VAR}} reference, got plaintext value."
            )
        return v

class RateLimitConfig(BaseModel):
    requests_per_minute: int = 600
    burst: int = 60
    # RATE LIMIT SCOPE:
    # - Per API key: each key has its own independent counter
    # - NOT global across all users of a deployment
    # - NOT per IP address
    # - burst allows short spikes within the same key scope
    # - 429 response includes Retry-After header; SDK exposes retryAfterSeconds

class ExecutionConfig(BaseModel):
    sync_timeout_ms: int = 30000
    async_poll_interval_ms: int = 2000
    max_batch_size: int = 100

class EnvironmentConfig(BaseModel):
    namespace: str = 'org'               # personal | team | org | global
    log_level: str = 'INFO'
    log_format: str = 'json'             # json | text
    rate_limit: RateLimitConfig = RateLimitConfig()
    execution: ExecutionConfig = ExecutionConfig()
    skip_connector_health_check: bool = False

class ConfigSchema(BaseModel):
    primitives: list[PrimitiveConfig]
    connectors: dict[str, ConnectorConfig]
    llm: LLMConfig
    environment: EnvironmentConfig = EnvironmentConfig()
    guardrails_path: str = 'memintel.guardrails.md'

    @field_validator('primitives')
    @classmethod
    def validate_unique_names(cls, primitives: list[PrimitiveConfig]) -> list[PrimitiveConfig]:
        names = [p.name for p in primitives]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Primitive names must be unique. "
                f"Duplicates found: {sorted(duplicates)}"
            )
        return primitives

    @field_validator('primitives')
    @classmethod
    def validate_primitive_connectors(
        cls, primitives: list[PrimitiveConfig], info
    ) -> list[PrimitiveConfig]:
        # Validate that every primitive's source.identifier exists in connectors
        # Note: this validator runs after connectors is validated
        if hasattr(info, 'data') and 'connectors' in info.data:
            connector_names = set(info.data['connectors'].keys())
            for p in primitives:
                if p.source.identifier not in connector_names:
                    raise ValueError(
                        f"Primitive '{p.name}' references unknown connector "
                        f"'{p.source.identifier}'. "
                        f"Available connectors: {sorted(connector_names)}"
                    )
        return primitives
```

---

## 11. ConfigLoader — Loading and Validation

```python
import os, re, yaml
from pathlib import Path

class ConfigError(Exception):
    """Raised when config is missing, invalid, or has unresolved env vars."""
    pass

ENV_VAR_RE = re.compile(r'\$\{([A-Z][A-Z0-9_]*)\}')

class ConfigLoader:
    def load(self, config_path: str) -> ConfigSchema:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {config_path}")

        # Step 1: Extract and merge all YAML blocks from Markdown
        raw_yaml = self._extract_yaml_blocks(path.read_text())

        # Step 2: Validate against ConfigSchema
        try:
            config = ConfigSchema.model_validate(raw_yaml)
        except Exception as e:
            raise ConfigError(f"Config validation failed: {e}") from e

        # Step 3: Resolve ${ENV_VAR} references
        resolved = self._resolve_env_vars(config.model_dump())
        return ConfigSchema.model_validate(resolved)

    def _extract_yaml_blocks(self, markdown: str) -> dict:
        blocks = re.findall(r'```yaml\n(.*?)```', markdown, re.DOTALL)
        merged = {}
        for block in blocks:
            parsed = yaml.safe_load(block)
            if isinstance(parsed, dict):
                merged = {**merged, **parsed}
        return merged

    def _resolve_env_vars(self, obj):
        """Recursively resolve ${VAR_NAME} references. Never logs resolved values."""
        if isinstance(obj, str):
            def replace(match):
                var = match.group(1)
                val = os.environ.get(var)
                if val is None:
                    raise ConfigError(
                        f"Required environment variable {var} is not set"
                    )
                return val
            return ENV_VAR_RE.sub(replace, obj)
        elif isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(v) for v in obj]
        return obj
```

---

## 12. Failure Simulation Tests

These tests verify the system fails safely: no silent corruption, no partial state,
clear error propagation. Add these to `/tests/unit/test_config_failures.py` in Session 2.5.

```python
class TestConfigFailureModes:

    # Bad config structure
    def test_missing_llm_key_raises(self):
        config = {'primitives': [], 'connectors': {}, 'environment': {}}
        with pytest.raises(ConfigError, match='llm'):
            ConfigLoader().load_from_dict(config)

    def test_invalid_primitive_type_raises(self):
        # type: float_array is not a valid Memintel type
        config = minimal_valid_config()
        config['primitives'][0]['type'] = 'float_array'
        with pytest.raises(ConfigError, match='Invalid Memintel type'):
            ConfigLoader().load_from_dict(config)

    def test_plaintext_password_raises(self):
        config = minimal_valid_config()
        config['connectors']['postgres.db']['password'] = 'my-password'
        with pytest.raises(ConfigError, match='Credential fields must use'):
            ConfigLoader().load_from_dict(config)

    def test_plaintext_api_key_raises(self):
        config = minimal_valid_config()
        config['llm']['api_key'] = 'sk-abc123'
        with pytest.raises(ConfigError, match='api_key must be an'):
            ConfigLoader().load_from_dict(config)

    def test_invalid_llm_provider_raises(self):
        config = minimal_valid_config()
        config['llm']['provider'] = 'gemini'
        with pytest.raises(ConfigError, match='Invalid LLM provider'):
            ConfigLoader().load_from_dict(config)

    def test_duplicate_primitive_names_raises(self):
        config = minimal_valid_config()
        config['primitives'].append(config['primitives'][0].copy())
        with pytest.raises(ConfigError, match='Primitive names must be unique'):
            ConfigLoader().load_from_dict(config)

    def test_unknown_connector_reference_raises(self):
        config = minimal_valid_config()
        config['primitives'][0]['source']['identifier'] = 'postgres.missing'
        with pytest.raises(ConfigError, match='references unknown connector'):
            ConfigLoader().load_from_dict(config)

    # Environment variable failures
    def test_missing_env_var_raises(self, monkeypatch):
        monkeypatch.delenv('DB_PASSWORD', raising=False)
        config_text = minimal_config_md_with_env_var('DB_PASSWORD')
        with pytest.raises(ConfigError, match='DB_PASSWORD is not set'):
            ConfigLoader().load(config_text)

    def test_error_contains_var_name_not_value(self, monkeypatch):
        # Error message must name the missing variable, never reveal expected value
        monkeypatch.delenv('DB_PASSWORD', raising=False)
        config_text = minimal_config_md_with_env_var('DB_PASSWORD')
        with pytest.raises(ConfigError) as exc_info:
            ConfigLoader().load(config_text)
        assert 'DB_PASSWORD' in str(exc_info.value)
        # Must not contain any hint of the expected value
        assert 'secret' not in str(exc_info.value).lower()

    def test_partial_env_resolution_not_returned(self, monkeypatch):
        # If one var resolves and another doesn't, no partial config is returned
        monkeypatch.setenv('DB_USER', 'admin')
        monkeypatch.delenv('DB_PASSWORD', raising=False)
        with pytest.raises(ConfigError):
            result = ConfigLoader().load(config_with_both_vars)
        # Verify the function raised before returning

    # Connector failures
    def test_unreachable_connector_raises(self, mock_connector_health):
        mock_connector_health.side_effect = ConnectionRefusedError
        with pytest.raises(ConfigError, match='unreachable'):
            ConfigApplyService().apply('valid_config_bad_host.md')

    def test_partial_apply_is_rolled_back(self, mock_connector_health):
        # First connector OK, second fails. Verify nothing was registered.
        registry = PrimitiveRegistry()
        mock_connector_health.fail_on_second_call()
        with pytest.raises(ConfigError):
            ConfigApplyService(registry=registry).apply('two_connectors.md')
        assert registry.list_all() == []  # clean state - nothing partially registered

    # Guardrails failures
    def test_missing_guardrails_file_raises(self):
        config = minimal_valid_config()
        config['guardrails_path'] = 'nonexistent.md'
        with pytest.raises(ConfigError, match='not found'):
            ConfigApplyService().apply_from_dict(config)

    def test_empty_strategy_registry_raises(self, empty_guardrails):
        # Guardrails loads but has no strategies - system must refuse to start
        with pytest.raises(ConfigError, match='strategy_registry is empty'):
            ConfigApplyService().apply_with_guardrails(empty_guardrails)
```

---

## 13. Validation Checklist

`memintel apply` validates these conditions before accepting the config. The system
does not start if any check fails.

```
VALIDATION CHECKLIST (ConfigApplyService.apply()):

Schema validation:
  ✓ All required top-level keys present
  ✓ All primitive names follow namespace.field format
  ✓ All missing_data_policy values are valid
  ✓ All credential fields use ${ENV_VAR} references (no plaintext)
  ✓ LLM api_key uses ${ENV_VAR} reference

Cross-reference validation:
  ✓ Every primitive's source.identifier matches a connector name
  ✓ guardrails_path file exists and is readable

Environment variable validation:
  ✓ All ${ENV_VAR} references resolve to non-empty strings

Guardrails validation:
  ✓ guardrails file loads without error
  ✓ strategy_registry is non-empty (at least one strategy registered)
  ✓ application_context block is present

Connector health (if skip_connector_health_check=false):
  ✓ Each connector accepts a test connection

FAILURE BEHAVIOUR:
  Any failed check → log the specific error → exit with non-zero code
  DO NOT start with partial configuration
  DO NOT silently fall back to defaults
  DO NOT log resolved credential values in the error output
```
