---
id: admin-guardrails
title: Configuring Guardrails
sidebar_label: Guardrails
---

# Configuring Guardrails

Guardrails are the policy layer that constrains how the compiler resolves user intent. They define which evaluation strategies are valid for which primitive types, what parameter ranges are acceptable, how severity language maps to numeric thresholds, and which strategy the compiler should prefer when intent is ambiguous.

Without guardrails, the compiler has no domain policy. With well-configured guardrails, the same intent expression produces consistently appropriate compiled conditions across every task — regardless of who created the task or when.

---

## What Guardrails Control

```
User says: "Alert me when a customer shows significant churn risk"
                          ↓
Guardrails answers:
  1. What strategies are valid for float primitives?
     → threshold, percentile, z_score, change
  2. What does "significant" mean?
     → bias_rules maps it to medium_severity
  3. What threshold represents medium severity for churn signals?
     → parameter_priors defines this per primitive
  4. If multiple strategies are valid, which should the compiler prefer?
     → global_preferred_strategy breaks the tie
                          ↓
Compiler produces a specific, deterministic condition
```

The compiler never freely invents strategies or parameters. It resolves within the constraints the admin has defined.

---

## The Guardrails Config File

Guardrails are defined in `memintel_guardrails.yaml`. Like primitives, this file is loaded at startup and changes require a restart.

### File Structure

```yaml
# memintel_guardrails.yaml

type_strategy_map:        # which strategies are valid for which types
parameter_priors:         # per-primitive threshold priors by severity
bias_rules:               # how natural language severity maps to severity levels
global_default_strategy:  # fallback strategy when no other rule matches
global_preferred_strategy: # preferred strategy when multiple are valid
threshold_directions:     # whether threshold fires above or below (optional)
```

---

## type_strategy_map

Declares which evaluation strategies are valid for each primitive type. The compiler will only select strategies listed here — any strategy not declared for a type is rejected at compile time.

```yaml
type_strategy_map:
  float:                [threshold, percentile, z_score, change]
  int:                  [threshold, percentile, change]
  boolean:              [equals]
  string:               [equals]
  categorical:          [equals]
  time_series<float>:   [z_score, change, percentile]
  time_series<int>:     [z_score, change, percentile]
  float?:               [threshold]
  int?:                 [threshold]
```

:::note
Nullable types (`float?`, `int?`) should generally only support `threshold` — other strategies require a complete value history that null values would corrupt.
:::

**Restricting strategies for specific domains:**

In a clinical trial context, you may want to restrict float primitives to only `threshold` and `percentile` — preventing z_score strategies that require statistical baselines that may not be appropriate for small patient populations:

```yaml
# Clinical trials — restrict to simpler strategies
type_strategy_map:
  float:    [threshold, percentile]
  int:      [threshold, percentile]
  boolean:  [equals]
```

In a real-time fraud detection context, you may want to enable all strategies on time_series types because velocity pattern detection is central:

```yaml
# Fraud detection — full strategy set on time series
type_strategy_map:
  time_series<float>: [z_score, change, percentile, threshold]
  time_series<int>:   [z_score, change, percentile, threshold]
```

---

## parameter_priors

Defines the numeric parameter values that correspond to each severity level for each primitive. This is where domain expertise is encoded most precisely.

When a user says "significant" or "high risk" or "approaching limit", the compiler maps the language to a severity level via `bias_rules`, then looks up the corresponding parameter value from `parameter_priors`.

```yaml
parameter_priors:
  <primitive_id>:
    low_severity:     { <strategy_param>: <value> }
    medium_severity:  { <strategy_param>: <value> }
    high_severity:    { <strategy_param>: <value> }
```

### Threshold Parameters

For `threshold` strategy, the parameter is `value` — the numeric level at which the condition fires.

```yaml
parameter_priors:
  # A DSCR below 1.25 is a covenant breach — set priors approaching that floor
  borrower.dscr:
    low_severity:    { value: 1.80 }   # early warning — plenty of headroom
    medium_severity: { value: 1.50 }   # approaching concern territory
    high_severity:   { value: 1.30 }   # near covenant floor (1.25)

  # Transaction value vs customer baseline
  transaction.value_vs_baseline_ratio:
    low_severity:    { value: 3.0  }   # 3x baseline — worth noting
    medium_severity: { value: 7.0  }   # 7x baseline — investigate
    high_severity:   { value: 15.0 }   # 15x baseline — strong anomaly signal

  # Error budget burn rate (multiples of sustainable rate)
  service.error_budget_burn_rate_1h:
    low_severity:    { value: 2.0  }   # 2x sustainable — early warning
    medium_severity: { value: 5.0  }   # consuming budget in 6 days
    high_severity:   { value: 14.4 }   # consuming budget in 2 days
```

### Percentile Parameters

For `percentile` strategy, the parameter is `value` — the percentile rank within the population at which the condition fires.

```yaml
parameter_priors:
  # Site deviation rate vs trial peers
  site.peer_deviation_percentile:
    low_severity:    { value: 70 }   # above 70th percentile of peers
    medium_severity: { value: 80 }   # above 80th percentile
    high_severity:   { value: 90 }   # in top 10% of sites by deviation rate

  # Stage duration vs pipeline norms
  deal.stage_duration_days:
    low_severity:    { value: 60 }
    medium_severity: { value: 75 }
    high_severity:   { value: 90 }
```

### Z-Score Parameters

For `z_score` strategy, the parameter is `threshold` — the number of standard deviations from the mean at which the condition fires.

```yaml
parameter_priors:
  # Provider billing volume anomaly
  provider.procedure_volume_30d:
    low_severity:    { threshold: 2.0 }
    medium_severity: { threshold: 2.5 }
    high_severity:   { threshold: 3.0 }
```

### Change Parameters

For `change` strategy, the parameter includes `value` (magnitude of change) and `window` (time window over which to measure it).

```yaml
parameter_priors:
  # Error rate increase over time window
  service.error_rate_trend_1h:
    low_severity:    { value: 0.002, window: "30m" }  # 0.2pp rise in 30 min
    medium_severity: { value: 0.005, window: "20m" }  # 0.5pp rise in 20 min
    high_severity:   { value: 0.010, window: "15m" }  # 1.0pp rise in 15 min

  # DSCR quarterly decline
  borrower.dscr_trend_4q:
    low_severity:    { value: 0.20, window: "2q" }   # 20% decline over 2 quarters
    medium_severity: { value: 0.30, window: "3q" }   # 30% decline over 3 quarters
    high_severity:   { value: 0.40, window: "4q" }   # 40% decline over 4 quarters
```

---

## bias_rules

Maps natural language severity expressions to severity levels. When a user uses a word or phrase in their intent that carries a severity implication, the compiler looks it up in `bias_rules` to determine which severity level to apply to `parameter_priors`.

```yaml
bias_rules:
  # Words that map to high_severity — strict, sensitive thresholds
  urgent:         high_severity
  critical:       high_severity
  immediately:    high_severity
  conservative:   high_severity
  page:           high_severity    # "alert me / page me when..."

  # Words that map to medium_severity — standard thresholds
  significant:    medium_severity
  material:       medium_severity
  elevated:       medium_severity
  notable:        medium_severity

  # Words that map to low_severity — early warning thresholds
  early:          low_severity
  proactive:      low_severity
  monitor:        low_severity
  approaching:    low_severity
  trending:       low_severity
```

### Domain-Specific Bias Rules

Add domain-specific terms that carry severity meaning in your context:

```yaml
# Financial services additions
bias_rules:
  breach:         high_severity    # "approaching breach" → high
  covenant:       high_severity
  sar:            high_severity    # "SAR-level activity" → high
  enhanced:       medium_severity  # "enhanced due diligence" → medium
  watchlist:      high_severity

# Clinical trial additions
bias_rules:
  stopping:       high_severity    # "approaching stopping rule" → high
  serious:        high_severity    # "serious adverse event" → high
  susar:          high_severity
  unexpected:     medium_severity
  possibly:       low_severity     # "possibly related" → low

# DevOps / SRE additions
bias_rules:
  outage:         high_severity
  slo:            medium_severity
  degradation:    medium_severity
  leak:           medium_severity  # "memory leak" → medium
  latency:        low_severity
```

---

## global_default_strategy and global_preferred_strategy

When the compiler has resolved a primitive type but multiple strategies are valid (per `type_strategy_map`), it uses these settings to break the tie.

```yaml
global_default_strategy:   threshold    # fallback when no other rule matches
global_preferred_strategy: percentile   # preferred when multiple strategies are valid
```

**Choosing the right defaults:**

| Domain | Recommended default | Recommended preferred | Reason |
|---|---|---|---|
| SaaS / product analytics | `threshold` | `percentile` | Relative comparison to population is usually more meaningful than absolute thresholds |
| Financial risk | `threshold` | `threshold` | Regulatory thresholds are often absolute, not relative |
| AML / fraud | `threshold` | `z_score` | Anomaly detection against individual baseline is central |
| DevOps / SRE | `threshold` | `change` | Trend detection is more valuable than current-level thresholds |
| Clinical trials | `threshold` | `threshold` | Protocol-defined absolute thresholds are the norm |

---

## threshold_directions

By default, a `threshold` condition fires when the value is **above** the threshold. For signals where the concern is values that are **too low** (ratios, coverage ratios, budget remaining), declare the direction explicitly.

```yaml
threshold_directions:
  bank.cet1_ratio:                below    # fires when ratio falls below threshold
  bank.lcr:                       below
  account.active_user_rate_30d:   below    # fires when active rate falls below threshold
  trial.stopping_rule_proximity:  above    # fires when approaching stopping rule (default)
  borrower.dscr:                  below    # fires when DSCR falls below threshold
```

---

## Complete Examples

### SaaS Churn Detection

```yaml
# memintel_guardrails_saas.yaml

type_strategy_map:
  float:                [threshold, percentile, z_score, change]
  int:                  [threshold, percentile, change]
  boolean:              [equals]
  categorical:          [equals]
  time_series<float>:   [z_score, change, percentile]
  time_series<int>:     [z_score, change, percentile]
  float?:               [threshold]

parameter_priors:
  account.active_user_rate_30d:
    low_severity:     { value: 0.60 }   # 60% of seats active — early concern
    medium_severity:  { value: 0.45 }   # 45% active — material drop
    high_severity:    { value: 0.30 }   # 30% active — serious risk

  account.seat_utilization_rate:
    low_severity:     { value: 0.65 }
    medium_severity:  { value: 0.50 }
    high_severity:    { value: 0.35 }

  account.days_to_renewal:
    low_severity:     { value: 90 }     # 90 days to renewal — start monitoring
    medium_severity:  { value: 60 }
    high_severity:    { value: 30 }     # 30 days — urgent engagement needed

  user.session_frequency_trend_8w:
    low_severity:     { value: 0.20, window: "4w" }  # 20% decline over 4 weeks
    medium_severity:  { value: 0.35, window: "4w" }  # 35% decline
    high_severity:    { value: 0.50, window: "4w" }  # 50% decline — strong signal

bias_rules:
  urgent:       high_severity
  critical:     high_severity
  significant:  medium_severity
  material:     medium_severity
  early:        low_severity
  proactive:    low_severity
  approaching:  low_severity

threshold_directions:
  account.active_user_rate_30d:   below
  account.seat_utilization_rate:  below

global_default_strategy:   threshold
global_preferred_strategy: percentile
```

### Credit Risk Monitoring

```yaml
# memintel_guardrails_credit.yaml

type_strategy_map:
  float:                [threshold, percentile, z_score, change]
  int:                  [threshold, percentile, change]
  float?:               [threshold]
  boolean:              [equals]
  categorical:          [equals]
  time_series<float>:   [change, z_score, percentile]

parameter_priors:
  borrower.dscr:
    low_severity:     { value: 1.80 }
    medium_severity:  { value: 1.50 }
    high_severity:    { value: 1.30 }   # covenant floor: 1.25

  borrower.leverage_ratio:
    low_severity:     { value: 3.0 }    # total debt / EBITDA
    medium_severity:  { value: 4.0 }
    high_severity:    { value: 5.5 }

  borrower.dscr_trend_4q:
    low_severity:     { value: 0.20, window: "2q" }
    medium_severity:  { value: 0.30, window: "3q" }
    high_severity:    { value: 0.40, window: "4q" }

  loan.covenant_headroom_pct:
    low_severity:     { value: 25 }     # 25% headroom remaining
    medium_severity:  { value: 15 }
    high_severity:    { value: 8  }     # 8% — covenant breach imminent

bias_rules:
  breach:       high_severity
  covenant:     high_severity
  deteriorating: medium_severity
  stressed:     medium_severity
  declining:    medium_severity
  watch:        low_severity
  early:        low_severity
  proactive:    low_severity

threshold_directions:
  borrower.dscr:            below
  borrower.current_ratio:   below
  loan.covenant_headroom_pct: below

global_default_strategy:   threshold
global_preferred_strategy: threshold   # regulatory thresholds are absolute in this domain
```

### DevOps / SRE

```yaml
# memintel_guardrails_sre.yaml

type_strategy_map:
  float:                [threshold, percentile, z_score, change]
  int:                  [threshold, percentile, change]
  boolean:              [equals]
  time_series<float>:   [change, z_score, percentile, threshold]
  time_series<int>:     [change, z_score, percentile]

parameter_priors:
  service.error_budget_burn_rate_1h:
    low_severity:     { value: 2.0  }
    medium_severity:  { value: 5.0  }
    high_severity:    { value: 14.4 }

  service.p99_latency_vs_baseline_ratio:
    low_severity:     { value: 1.5 }    # 50% above baseline
    medium_severity:  { value: 2.5 }    # 150% above baseline
    high_severity:    { value: 4.0 }    # 300% above baseline

  service.error_rate_trend_1h:
    low_severity:     { value: 0.002, window: "30m" }
    medium_severity:  { value: 0.005, window: "20m" }
    high_severity:    { value: 0.010, window: "15m" }

  deployment.similar_deployment_incident_rate:
    low_severity:     { value: 0.10 }
    medium_severity:  { value: 0.20 }
    high_severity:    { value: 0.35 }

bias_rules:
  page:         high_severity
  outage:       high_severity
  critical:     high_severity
  degradation:  medium_severity
  slo:          medium_severity
  early:        low_severity
  proactive:    low_severity
  trending:     low_severity

global_default_strategy:   threshold
global_preferred_strategy: change      # trend detection preferred in SRE context
```

---

## Validation

The server validates the guardrails config at startup:

```bash
# Verify guardrails load correctly
curl http://localhost:8000/health

# List registered strategies
curl http://localhost:8000/guardrails/strategies

# Check that a specific primitive has valid strategy mappings
curl http://localhost:8000/guardrails/validate?primitive_id=borrower.dscr
```

### Startup Validation Checks

| Check | What it verifies |
|---|---|
| All strategies in type_strategy_map exist in strategy registry | No undefined strategy names |
| All primitives in parameter_priors exist in primitive registry | No orphaned prior definitions |
| All severity levels present for each prior | low, medium, and high all declared |
| Parameter values within declared bounds | No out-of-range priors |
| No conflicting threshold directions | A primitive cannot be both above and below |

---

## Common Mistakes

**Setting parameter priors that are too wide apart.** If low_severity threshold is 0.3 and high_severity is 0.9, any user whose intent maps to medium_severity gets a threshold of 0.6 — which may be completely wrong for the domain. Keep severity levels proportional and meaningful.

**Not customising bias_rules for domain terms.** Generic bias rules miss the domain-specific severity language your users will naturally use. A credit risk analyst will say "watch list", "covenant", "deteriorating" — add these to bias_rules so they resolve correctly.

**Using the same global_preferred_strategy for all domains.** `percentile` is the right default for population comparison use cases (who is in the bottom quartile). `change` is the right default for trajectory monitoring (what is getting worse). `threshold` is the right default for regulatory compliance (is the ratio above the regulatory minimum). Pick the one that matches your primary use case.

**Forgetting threshold_directions for below-threshold signals.** Coverage ratios, active user rates, DSCR, LCR — these all fire when they fall below a threshold, not above. Forgetting to declare `below` means the condition never fires for the right reason.

---

## Next Step

[Configure Actions →](/docs/admin-guide/admin-actions)
