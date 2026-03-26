---
id: admin-primitives
title: Step 2 — Primitives
sidebar_label: Step 2 — Primitives
---

# Step 2 — Primitives

A primitive is a single signal that you want Memintel to be able to monitor. Think of primitives as the **vocabulary of measurable things** in your domain — the building blocks that all monitoring tasks are made from.

Before you can create a monitoring task for "deal engagement" or "patient adverse event severity", those concepts need to be broken down into their underlying measurable signals. Those signals are your primitives.

:::note The key distinction
**Primitives are raw, observable facts. They are not interpretations.**

- ✓ "Days since last email reply" — a primitive (directly measurable)
- ✗ "Deal health" — not a primitive (an interpretation of multiple signals)
- ✓ "Transaction amount divided by customer 90-day average" — a primitive (computable)
- ✗ "Transaction risk" — not a primitive (a concept derived from multiple signals)

The compiler derives concepts from primitives. Your job is to define the primitives.
:::

---

## Where Primitives Live

Primitives are defined in the `primitives:` section of your `memintel_config.yaml` file:

```yaml
# memintel_config.yaml

primitives:
  - id: account.active_user_rate_30d
    type: float
    source: activity_pipeline
    entity: account_id
    description: "Ratio of active users to total licensed seats over last 30 days, 0-1"

  - id: account.days_to_renewal
    type: int
    source: billing_pipeline
    entity: account_id
    description: "Days until the next subscription renewal date"
```

Each primitive is a list item (starting with `-`) with four required fields.

---

## The Four Required Fields

### id — the signal's name

A unique identifier for this signal. Use the format `entity.signal_name` — lowercase, with a dot separating the entity type from the signal name, and underscores between words.

```yaml
id: account.active_user_rate_30d
id: customer.days_since_last_login
id: borrower.debt_service_coverage_ratio
id: patient.adverse_event_severity_score
id: service.error_rate_5m
```

:::tip Naming convention
The part before the dot is the **entity type** — the thing being measured (account, customer, patient, deal).
The part after the dot is the **signal name** — what is being measured, often including a time window.
This makes the registry easy to browse as it grows.
:::

### type — what kind of data it contains

The type tells the system what kind of values this signal produces and which evaluation strategies are available for it.

| Type | What it means | Example signals |
|---|---|---|
| `float` | A decimal number, usually between 0 and 1 or a ratio | Sentiment score, engagement rate, LTV ratio |
| `int` | A whole number | Days since login, count of events, number of calls |
| `boolean` | True or false only | Payment failed flag, OIG exclusion flag, license active |
| `categorical` | One value from a fixed list | Risk tier (low/medium/high), status (active/paused/closed) |
| `time_series<float>` | A sequence of decimal values over time | Error rate over last hour, DSCR over last 4 quarters |
| `time_series<int>` | A sequence of whole numbers over time | Daily transaction count over last 30 days |

**When to use `time_series` vs a plain number:**

Use a time series when you want the system to be able to detect **trends and trajectories** — not just the current value. For example:

- `borrower.dscr` (type: `float`) — the current DSCR value right now
- `borrower.dscr_trend_4q` (type: `time_series<float>`) — the DSCR across the last 4 quarters, enabling detection of a declining trend

If a user might say "alert me when X is declining" or "alert me when X is trending upward", register a time series variant.

**Nullable signals:**

If a signal sometimes has no value — for example, a sentiment score for a customer who has never sent an email — add `?` to make it nullable:

```yaml
- id: deal.last_call_sentiment
  type: float?      # null if no calls have been recorded
  description: "Sentiment score from the last call recording — null if no calls"
```

### source — which data pipeline provides this

The name of the data pipeline or system that this signal comes from. This is used by your data engineer to know which resolver function to write. It does not affect how the signal is evaluated — it is a label for the technical team.

```yaml
source: billing_pipeline
source: activity_pipeline
source: crm_system
source: safety_database
source: sec_taxonomy_feed
```

Use a short, descriptive name that matches how your data team refers to that data source.

### description — plain English explanation

A one-sentence description of what this signal measures. Write it clearly enough that a new team member could implement the data fetch without asking questions.

```yaml
# Too vague
description: "A metric for the user"

# Good
description: "Days since the customer last authenticated to the platform"
description: "Ratio of active users to total licensed seats in the last 30 days, expressed as a value between 0 and 1"
description: "True if the most recent payment attempt for this account failed"
```

---

## One Signal Per Primitive

The most important design rule: **each primitive measures exactly one thing**.

If you find yourself writing "and" or "or" in a description, you are probably trying to combine two signals into one primitive. Split them.

```yaml
# Wrong — two signals combined
- id: deal.engagement_and_sentiment
  type: float
  description: "Combined engagement and sentiment score"

# Right — two separate primitives
- id: deal.engagement_score
  type: float
  description: "Composite activity score based on email, call, and meeting frequency, 0-1"

- id: deal.sentiment_score
  type: float
  description: "LLM-extracted sentiment from recent deal communications, 0-1"
```

The system combines primitives into concepts automatically. Your job is to provide the raw signals, not pre-combine them.

---

## Internal vs External Signals

One of Memintel's key capabilities is evaluating **your internal data against external signals** — regulatory changes, market data, peer benchmarks. Both types are registered as primitives.

```yaml
primitives:

  # Internal signal — your own data
  - id: filing.deprecated_tag_count
    type: int
    source: filing_history_pipeline
    entity: filing_id
    description: "Number of XBRL tags in this draft that are deprecated in the new taxonomy"

  # External signal — regulatory environment data
  - id: taxonomy.tag_deprecated_flag
    type: boolean
    source: sec_taxonomy_feed
    entity: xbrl_tag_id
    description: "True if this tag is deprecated in the current SEC GAAP taxonomy version"

  # External signal — peer benchmark data
  - id: provider.peer_deviation_percentile
    type: float
    source: benchmarking_pipeline
    entity: provider_id
    description: "This provider's billing deviation percentile within their specialty peer group, 0-100"
```

Register external signals the same way as internal ones. Your data engineer connects them to the appropriate external data source.

---

## Complete Examples by Domain

### SaaS Churn Detection

```yaml
primitives:

  # User engagement
  - id: user.days_since_last_login
    type: int
    source: auth_pipeline
    entity: user_id
    description: "Days since this user last authenticated to the platform"

  - id: user.core_actions_30d
    type: int
    source: activity_pipeline
    entity: user_id
    description: "Count of core workflow actions (create, edit, share, export) in last 30 days"

  - id: user.session_frequency_trend_8w
    type: time_series<float>
    source: activity_pipeline
    entity: user_id
    description: "Weekly session count over last 8 weeks, oldest to newest — enables trend detection"

  # Account health
  - id: account.active_user_rate_30d
    type: float
    source: activity_pipeline
    entity: account_id
    description: "Ratio of active users to total licensed seats in last 30 days, 0-1"

  - id: account.seat_utilization_rate
    type: float
    source: billing_pipeline
    entity: account_id
    description: "Ratio of seats currently in use to seats licensed, 0-1"

  - id: account.days_to_renewal
    type: int
    source: billing_pipeline
    entity: account_id
    description: "Days until next renewal date — negative if past due"

  - id: account.payment_failed_flag
    type: boolean
    source: billing_pipeline
    entity: account_id
    description: "True if the most recent payment attempt for this account failed"

  - id: account.nps_score
    type: float?
    source: survey_pipeline
    entity: account_id
    description: "Most recent NPS score, 0-10 — null if no survey response in last 180 days"

  - id: account.support_ticket_rate_30d
    type: float
    source: support_pipeline
    entity: account_id
    description: "Support tickets per user per 30 days — elevated rate indicates friction"
```

### Credit Risk Monitoring

```yaml
primitives:

  # Borrower financial health
  - id: borrower.dscr
    type: float
    source: financial_analysis_pipeline
    entity: borrower_id
    description: "Debt service coverage ratio — EBITDA divided by total debt service. Below 1.0 means insufficient cash flow to service debt."

  - id: borrower.dscr_trend_4q
    type: time_series<float>
    source: financial_analysis_pipeline
    entity: borrower_id
    description: "DSCR across last 4 quarters, oldest to newest — enables declining trend detection"

  - id: borrower.leverage_ratio
    type: float
    source: financial_analysis_pipeline
    entity: borrower_id
    description: "Total debt divided by EBITDA — higher values indicate more leverage"

  - id: borrower.management_sentiment_score
    type: float?
    source: nlp_pipeline
    entity: borrower_id
    description: "LLM-extracted sentiment from most recent management commentary, 0-1 — null if no commentary available"

  # Loan and covenant signals
  - id: loan.covenant_headroom_pct
    type: float
    source: covenant_monitoring_pipeline
    entity: loan_id
    description: "Distance to nearest covenant threshold as a percentage — negative means breach has occurred"

  - id: loan.days_since_financial_submission
    type: int
    source: covenant_monitoring_pipeline
    entity: loan_id
    description: "Days since borrower last submitted required financial statements"
```

### Clinical Trial Safety

```yaml
primitives:

  # Patient adverse event signals
  - id: patient.ae_severity_score
    type: float
    source: safety_database
    entity: patient_id
    description: "Composite adverse event severity score based on MedDRA grades, 0-1. Higher = more severe."

  - id: patient.ae_relatedness_signal
    type: float
    source: nlp_pipeline
    entity: patient_id
    description: "LLM-extracted probability that the most recent adverse event is related to the study drug, 0-1"

  - id: patient.ae_relatedness_confidence
    type: float
    source: nlp_pipeline
    entity: patient_id
    description: "Confidence score for the relatedness assessment, 0-1"

  - id: patient.sae_count_30d
    type: int
    source: safety_database
    entity: patient_id
    description: "Number of serious adverse events reported for this patient in the last 30 days"

  # Trial-level signals
  - id: trial.treatment_vs_comparator_ratio
    type: float
    source: edc_pipeline
    entity: trial_id
    description: "Ratio of AE incidence rate in treatment arm to AE incidence rate in comparator arm"

  - id: trial.stopping_rule_proximity_score
    type: float
    source: safety_monitoring_pipeline
    entity: trial_id
    description: "How close current safety data is to the pre-specified stopping thresholds, 0-1. Values above 0.75 warrant DSMB review."

  # External regulatory signals
  - id: compound.faers_signal_score
    type: float
    source: faers_pipeline
    entity: compound_id
    description: "Disproportionality signal score from FDA FAERS database for this compound class, 0-1"

  - id: compound.fda_class_safety_alert_flag
    type: boolean
    source: fda_guidance_pipeline
    entity: compound_id
    description: "True if FDA has issued a safety communication for this compound class in the last 90 days"
```

---

## Working with Your Data Engineer

Your data engineer's job is to connect each primitive you declare to the actual data source. For each primitive you add:

1. **You define it** — write the entry in `memintel_config.yaml` with the id, type, source, and description
2. **They implement it** — write a resolver function that fetches the value for a given entity at a given point in time
3. **You test it** — verify that monitoring tasks using this primitive produce sensible results

A clear description makes the data engineer's job much easier. The better you describe what the signal measures and what its value range means, the faster they can implement it correctly.

---

## Common Mistakes

**Defining concepts as primitives.** "Deal health score", "customer risk level", "account engagement" — these are concepts that the compiler derives from primitive signals. Register the underlying signals instead.

**Forgetting to mark nullable signals.** If a signal sometimes has no value (a customer with no calls, a borrower with no commentary), declare it as `type?`. An unexpected null on a non-nullable primitive causes evaluation errors.

**Missing time-series variants.** If you want to detect trends — "declining over the last 4 quarters", "increasing over the last 8 weeks" — register a `time_series<float>` or `time_series<int>` variant. The plain `float` version only tells you the current value.

**Writing vague descriptions.** Your data engineer will implement exactly what you describe. "A metric for the account" will result in a question back to you. "Ratio of active users to total licensed seats in the last 30 days, expressed as a decimal between 0 and 1" will not.

---

## Next Step

→ [Step 3: Guardrails](/docs/admin-guide/admin-guardrails)
