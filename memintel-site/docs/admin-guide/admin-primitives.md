---
id: admin-primitives
title: Configuring Primitives
sidebar_label: Primitives
---

# Configuring Primitives

Primitives are the signal vocabulary of your Memintel deployment. They declare what signals exist, how they are typed, and where they come from. The compiler can only use primitives that are registered — and users can only monitor signals that have been declared as primitives.

Getting primitives right is the most consequential configuration decision in a Memintel deployment. This page explains how to design them well and what to avoid.

---

## What a Primitive Is

A primitive is a **typed, normalised declaration of a single observable signal**. It is not a formula. It is not a concept. It is not a derived score. It is a direct representation of one thing that can be measured.

```yaml
# This is a primitive — one observable fact, strictly typed
- id: customer.days_since_last_login
  type: int
  source: auth_pipeline
  entity: customer_id
  description: Days since the customer last authenticated successfully

# This is NOT a primitive — it combines two things
- id: customer.engagement_and_health  # Wrong — two concepts bundled together
  type: float
```

The rule: if you find yourself describing a primitive with "and" or "or", it is not a primitive — it is a concept. Break it apart.

---

## The Primitives Config File

Primitives are defined in `memintel_primitives.yaml`. This file is loaded at server startup via the `MEMINTEL_CONFIG_PATH` environment variable. Changes require a server restart.

### File Structure

```yaml
# memintel_primitives.yaml

primitives:
  - id: <string>           # required — unique identifier, dot-namespaced
    type: <type>           # required — see type system below
    source: <string>       # required — which data pipeline provides this
    entity: <string>       # required — the entity ID field this maps to
    description: <string>  # required — plain English, one sentence
    nullable: <bool>       # optional — default false
```

### Naming Convention

Use dot-namespaced IDs in the format `entity.signal_name`. This makes the registry browsable and prevents naming collisions as it grows.

```yaml
# Good — clear, domain-readable, unambiguous
customer.days_since_last_login
account.active_user_rate_30d
transaction.value_vs_baseline_ratio
provider.license_expiry_days
service.error_rate_trend_1h

# Bad — too generic, not domain-readable
feature_3
engagement_score
metric_a
```

---

## The Type System

Every primitive must declare a type. The type determines which evaluation strategies are available at compile time.

### Scalar Types

| Type | Description | Strategies Available |
|---|---|---|
| `float` | Continuous numeric value | `threshold`, `percentile`, `z_score`, `change` |
| `int` | Integer numeric value | `threshold`, `percentile`, `change` |
| `boolean` | True or false | `equals` |
| `string` | Free text | `equals` |
| `categorical` | Enumerated value from a defined set | `equals` |

### Container Types

| Type | Description | Strategies Available |
|---|---|---|
| `time_series<float>` | Ordered sequence of float values over time | `z_score`, `change`, `percentile` |
| `time_series<int>` | Ordered sequence of integer values over time | `z_score`, `change`, `percentile` |

### Nullable Types

Any type can be declared nullable by appending `?`. A nullable primitive may return `null` when no value is available — for example, a `float?` sentiment score for a customer with no recent emails.

```yaml
- id: deal.last_call_sentiment_score
  type: float?     # null if no calls recorded in window
  description: Sentiment score from last call recording — null if no calls
```

:::warning
Do not use `float` for a signal that sometimes has no value. Use `float?`. An unexpected `null` on a non-nullable primitive causes a runtime evaluation error. Always declare nullability explicitly.
:::

### Choosing the Right Type

```yaml
# A ratio or percentage → float
- id: account.active_user_rate_30d
  type: float
  description: Ratio of active users to total seats, 0-1

# A count or duration → int
- id: deal.thread_stalled_days
  type: int
  description: Days since last email reply in deal thread

# A true/false flag → boolean
- id: customer.payment_failed_flag
  type: boolean
  description: True if most recent payment attempt failed

# A status from a fixed set → categorical
- id: customer.risk_tier
  type: categorical
  description: Risk classification — standard | elevated | high | blocked

# A historical sequence for trend analysis → time_series<float>
- id: service.error_rate_trend_1h
  type: time_series<float>
  description: Error rate sampled every 5 minutes over last hour

# A value that may not always exist → float?
- id: borrower.management_sentiment_score
  type: float?
  description: LLM-extracted sentiment from last management commentary — null if no commentary available
```

---

## Primitive Design Principles

### 1 — One signal per primitive

Never bundle two signals into one primitive. If the description contains "and" or "or", it is not a primitive.

```yaml
# Wrong — two signals combined
- id: deal.engagement_and_sentiment
  type: float

# Right — two separate primitives
- id: deal.engagement_score
  type: float
  description: Composite engagement activity score, 0-1

- id: deal.sentiment_score
  type: float
  description: LLM-extracted sentiment from recent communications, 0-1
```

### 2 — Observable facts, not interpretations

A primitive should be directly measurable or computable from raw data without requiring domain interpretation. "Customer health" is not a primitive — it is a concept derived from primitives. "Days since last login" is a primitive.

```yaml
# Wrong — this is a concept, not a primitive
- id: customer.health_score
  type: float
  description: Overall customer health

# Right — this is an observable fact
- id: customer.days_since_last_login
  type: int
  description: Days since the customer last authenticated

- id: customer.feature_adoption_score
  type: float
  description: Ratio of activated features to total available features, 0-1
```

### 3 — Register time-series variants for trend detection

If a signal changes meaningfully over time and you want the compiler to detect trends, register a time-series variant alongside the scalar.

```yaml
# Point-in-time — current value only
- id: borrower.dscr
  type: float
  description: Current debt service coverage ratio

# Time-series — last 4 quarters, enables trend and z_score strategies
- id: borrower.dscr_trend_4q
  type: time_series<float>
  description: DSCR over last 4 quarters, oldest to newest
```

When a user says "alert me when DSCR is declining significantly", the compiler can use `change` strategy on `borrower.dscr_trend_4q` to detect trajectory — not just current level.

### 4 — Pair LLM-extracted signals with confidence scores

When a signal is extracted by an LLM or ML model, register the confidence score alongside the signal value. The compiler can use this to weight the signal appropriately.

```yaml
- id: deal.sentiment_score
  type: float
  description: LLM-extracted sentiment from recent deal communications, 0-1

- id: deal.sentiment_confidence
  type: float
  description: Model confidence for the sentiment extraction, 0-1
```

### 5 — Separate external state from internal state

External signals — regulatory data, market data, peer benchmarks — should be registered as explicitly as internal signals. This makes the dual memory structure visible in the primitive registry.

```yaml
# Internal — the company's own data
- id: filing.deprecated_tag_count
  type: int
  source: filing_history_pipeline
  entity: filing_id
  description: Number of deprecated XBRL tags in this draft filing

# External — regulatory environment data
- id: taxonomy.deprecated_in_current_version
  type: boolean
  source: sec_taxonomy_pipeline
  entity: xbrl_tag_id
  description: True if this tag is deprecated in the current SEC taxonomy version
```

---

## Complete Example — SaaS Churn Detection

```yaml
# memintel_primitives_saas.yaml

primitives:

  # User engagement signals
  - id: user.days_since_last_login
    type: int
    source: auth_pipeline
    entity: user_id
    description: Days since the user last authenticated successfully

  - id: user.core_actions_30d
    type: int
    source: activity_pipeline
    entity: user_id
    description: Count of core workflow actions performed in last 30 days

  - id: user.feature_breadth_score
    type: float
    source: activity_pipeline
    entity: user_id
    description: Ratio of distinct features used to total available features, 0-1

  - id: user.session_frequency_trend_8w
    type: time_series<float>
    source: activity_pipeline
    entity: user_id
    description: Weekly session count over last 8 weeks — enables trend detection

  # Account health signals
  - id: account.active_user_rate_30d
    type: float
    source: activity_pipeline
    entity: account_id
    description: Ratio of active users to total licensed seats, 0-1

  - id: account.seat_utilization_rate
    type: float
    source: billing_pipeline
    entity: account_id
    description: Ratio of seats in use to seats licensed, 0-1

  - id: account.support_ticket_rate_30d
    type: float
    source: support_pipeline
    entity: account_id
    description: Support tickets per user per 30 days — elevated rate indicates friction

  - id: account.nps_score
    type: float?
    source: survey_pipeline
    entity: account_id
    description: Most recent NPS score, 0-10 — null if no survey response in last 180d

  - id: account.days_to_renewal
    type: int
    source: billing_pipeline
    entity: account_id
    description: Days until next renewal date — negative if past due

  # Payment signals
  - id: account.payment_failed_flag
    type: boolean
    source: billing_pipeline
    entity: account_id
    description: True if most recent payment attempt failed

  - id: account.invoice_overdue_days
    type: int
    source: billing_pipeline
    entity: account_id
    description: Days the current invoice is overdue — 0 if not overdue

  # Product signals
  - id: account.api_call_volume_trend_8w
    type: time_series<int>
    source: api_pipeline
    entity: account_id
    description: Weekly API call count over last 8 weeks

  - id: account.integration_active_count
    type: int
    source: product_pipeline
    entity: account_id
    description: Number of active third-party integrations configured

  # External signals
  - id: account.company_headcount_change_pct
    type: float?
    source: firmographic_pipeline
    entity: account_id
    description: Percentage change in employee headcount in last 6 months — from external firmographic data. Null if not available.
```

---

## Complete Example — AML Transaction Monitoring

```yaml
# memintel_primitives_aml.yaml

primitives:

  # Customer behavior baseline
  - id: customer.avg_transaction_value_90d
    type: float
    source: transaction_pipeline
    entity: customer_id
    description: Average transaction value over trailing 90 days

  - id: customer.transaction_velocity_30d
    type: time_series<int>
    source: transaction_pipeline
    entity: customer_id
    description: Daily transaction count over last 30 days — enables velocity change detection

  - id: customer.counterparty_jurisdiction_risk
    type: float
    source: risk_pipeline
    entity: customer_id
    description: Weighted average jurisdiction risk score of recent counterparties, 0-1

  - id: customer.risk_tier
    type: categorical
    source: kyc_pipeline
    entity: customer_id
    description: Current KYC risk classification — standard | elevated | high

  # Transaction signals
  - id: transaction.value_vs_baseline_ratio
    type: float
    source: transaction_pipeline
    entity: transaction_id
    description: This transaction's value divided by customer's 90-day average — 1.0 = at baseline

  - id: transaction.structuring_signal
    type: float
    source: transaction_pipeline
    entity: transaction_id
    description: Probability score for structuring pattern — multiple sub-threshold transactions, 0-1

  - id: transaction.narrative_risk_score
    type: float
    source: nlp_pipeline
    entity: transaction_id
    description: LLM-extracted risk score from transaction narrative and reference fields, 0-1

  - id: transaction.narrative_confidence
    type: float
    source: nlp_pipeline
    entity: transaction_id
    description: Model confidence for narrative risk extraction, 0-1

  # External regulatory signals
  - id: customer.watchlist_match_score
    type: float
    source: sanctions_pipeline
    entity: customer_id
    description: Highest fuzzy match score against current OFAC/UN/EU watchlists, 0-1

  - id: customer.jurisdiction_fatf_status
    type: categorical
    source: regulatory_pipeline
    entity: customer_id
    description: FATF status of customer primary jurisdiction — clean | grey | black

  - id: typology.recent_match_score
    type: float
    source: regulatory_pipeline
    entity: customer_id
    description: Similarity of customer transaction pattern to recently published AML typologies, 0-1
```

---

## Loading and Validating

Primitives are loaded at startup from the file path set in `MEMINTEL_CONFIG_PATH`. The server validates the file on startup and refuses to start if any primitive definition is malformed.

```bash
# Set the config path
export MEMINTEL_CONFIG_PATH="/etc/memintel/memintel_config.yaml"

# Start the server — will validate primitives on startup
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Verify primitives loaded correctly
curl http://localhost:8000/definitions?type=primitive
```

To validate without restarting:

```bash
python -c "
from app.config.loader import ConfigLoader
config = ConfigLoader('/etc/memintel/memintel_config.yaml')
config.validate()
print(f'{len(config.primitives)} primitives loaded successfully')
"
```

### Startup Validation Checks

The server performs these checks on every primitive at startup:

| Check | What it verifies |
|---|---|
| ID format | Dot-namespaced, lowercase, no spaces |
| Type validity | Type is in the declared type system |
| Uniqueness | No duplicate IDs |
| Description present | Non-empty description |
| Source declared | Source pipeline is named |
| Entity declared | Entity field is named |

If any check fails, the server logs the specific primitive ID and the validation error, then refuses to start.

---

## Common Mistakes

**Defining concepts as primitives.** `customer.health_score`, `deal.risk_level`, `account.engagement` — these are concepts derived from multiple signals, not primitives. Register the underlying signals instead and let the compiler derive the concept.

**Missing nullable declarations.** If a signal sometimes has no value, declare it as `type?`. A runtime null on a non-nullable primitive causes evaluation failures that are hard to diagnose.

**Not registering time-series variants.** If you want the compiler to detect trends and trajectories, register `time_series<float>` variants alongside scalar types. A user saying "alert me when X is declining" cannot be resolved without a time-series primitive.

**Overly generic descriptions.** "A metric for the user" is not a useful description. Write descriptions that would allow a new developer to implement the resolver without asking questions.

**Not pairing LLM signals with confidence scores.** An LLM-extracted score without a confidence score cannot be appropriately weighted by the compiler. Always register both.

---

## Next Step

[Configure Guardrails →](/docs/admin-guide/admin-guardrails)
