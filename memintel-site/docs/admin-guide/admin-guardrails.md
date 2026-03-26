---
id: admin-guardrails
title: Step 3 — Guardrails
sidebar_label: Step 3 — Guardrails
---

# Step 3 — Guardrails

Guardrails are the rules the system follows when interpreting what your team members ask for. When someone says "alert me when churn risk is **significant**" — the guardrails define what "significant" means in numbers.

Think of guardrails as your domain policy encoded in configuration. The system will always operate within these rules when compiling monitoring tasks.

---

## Why Guardrails Matter

Without guardrails, when a user says "alert me when the active user rate is significantly low", the system has no idea whether "significant" means 60%, 40%, or 20%. It would have to guess.

With guardrails, you have defined — in advance, based on your domain expertise — that in your application, "significant" applied to `account.active_user_rate_30d` means below 45%. The user gets exactly the threshold that makes operational sense, without ever having to specify a number.

---

## Where Guardrails Live

Guardrails are defined in the `guardrails:` section of your `memintel_config.yaml`:

```yaml
# memintel_config.yaml

guardrails:
  type_strategy_map:
    float:      [threshold, percentile, z_score, change]
    int:        [threshold, percentile, change]
    boolean:    [equals]
    categorical: [equals]

  parameter_priors:
    account.active_user_rate_30d:
      low_severity:    { value: 0.60 }
      medium_severity: { value: 0.45 }
      high_severity:   { value: 0.30 }

  bias_rules:
    significant: medium_severity
    urgent:      high_severity
    early:       low_severity

  global_default_strategy:   threshold
  global_preferred_strategy: percentile
```

There are four sub-sections. You will spend most of your time on `parameter_priors` and `bias_rules` — the other two usually need little or no customisation.

---

## Sub-section 1 — type_strategy_map

This declares which evaluation methods (strategies) are valid for each signal type. In most cases, you can copy this block directly and leave it unchanged.

```yaml
guardrails:
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

**When you might change this:**

You might want to restrict certain strategies in highly regulated domains. For example, in a clinical trial context where patient populations are small, `z_score` strategies (which require a large baseline dataset) may not be appropriate:

```yaml
# Clinical trials — restrict to simpler strategies
type_strategy_map:
  float:    [threshold, percentile]
  int:      [threshold, percentile]
  boolean:  [equals]
```

If you are unsure, use the default block above and leave it unchanged.

---

## Sub-section 2 — parameter_priors

This is the most important part of the guardrails configuration. For each signal, you define what threshold values correspond to low, medium, and high severity.

```yaml
parameter_priors:
  <signal_id>:
    low_severity:    { value: <number> }
    medium_severity: { value: <number> }
    high_severity:   { value: <number> }
```

When a user says "alert me when active user rate drops significantly", the system:
1. Identifies the relevant signal: `account.active_user_rate_30d`
2. Maps "significantly" to `medium_severity` via `bias_rules`
3. Looks up the `medium_severity` threshold: `0.45`
4. Compiles the condition: fires when active user rate drops below 45%

### Choosing threshold values

Set the three levels based on your operational experience and domain knowledge:

- **low_severity** — early warning. Worth monitoring but not yet a reason to act. A proactive signal.
- **medium_severity** — material concern. Action should be considered. A standard monitoring threshold.
- **high_severity** — urgent. Immediate action required.

### Direction matters

For most signals, the condition fires when the value goes **above** the threshold (e.g. error rate above 5%). But some signals fire when they go **below** a threshold. For those, add a `threshold_directions` section (covered below).

### Examples by signal type

**Account active user rate (lower is worse):**
```yaml
account.active_user_rate_30d:
  low_severity:    { value: 0.60 }  # 60% — start paying attention
  medium_severity: { value: 0.45 }  # 45% — take action
  high_severity:   { value: 0.30 }  # 30% — urgent engagement needed
```

**Days to renewal (fewer days = more urgent):**
```yaml
account.days_to_renewal:
  low_severity:    { value: 90 }  # 90 days out — begin monitoring
  medium_severity: { value: 60 }  # 60 days — start outreach
  high_severity:   { value: 30 }  # 30 days — urgent
```

**Transaction value vs customer baseline (higher = more suspicious):**
```yaml
transaction.value_vs_baseline_ratio:
  low_severity:    { value: 3.0  }  # 3x baseline — worth noting
  medium_severity: { value: 7.0  }  # 7x baseline — investigate
  high_severity:   { value: 15.0 }  # 15x baseline — strong anomaly
```

**Adverse event severity score (higher = more serious):**
```yaml
patient.ae_severity_score:
  low_severity:    { value: 0.5 }  # Grade 2+ events
  medium_severity: { value: 0.7 }  # Grade 3+ events
  high_severity:   { value: 0.9 }  # Grade 4/5 events
```

**Covenant headroom (lower = closer to breach):**
```yaml
loan.covenant_headroom_pct:
  low_severity:    { value: 25 }  # 25% headroom — start monitoring
  medium_severity: { value: 15 }  # 15% — review with borrower
  high_severity:   { value: 8  }  # 8% — breach imminent
```

**DSCR quarterly trend (detecting rate of decline):**

For time-series signals, the parameters include a `window` (how many periods to look back):
```yaml
borrower.dscr_trend_4q:
  low_severity:    { value: 0.20, window: "2q" }  # 20% decline over 2 quarters
  medium_severity: { value: 0.30, window: "3q" }  # 30% decline over 3 quarters
  high_severity:   { value: 0.40, window: "4q" }  # 40% decline over 4 quarters
```

---

## Sub-section 3 — bias_rules

This maps the natural language words your team uses when creating monitoring tasks to the severity levels you defined in `parameter_priors`.

```yaml
bias_rules:
  <word or phrase>: <severity level>
```

When a user says "alert me when churn risk is **significant**", the system looks up "significant" in `bias_rules` and finds `medium_severity`. It then uses the `medium_severity` threshold from `parameter_priors`.

### Standard bias rules

These work for most domains — copy them as a starting point:

```yaml
bias_rules:
  # High severity words
  urgent:         high_severity
  critical:       high_severity
  immediately:    high_severity
  page:           high_severity

  # Medium severity words
  significant:    medium_severity
  material:       medium_severity
  elevated:       medium_severity
  notable:        medium_severity

  # Low severity words
  early:          low_severity
  proactive:      low_severity
  monitor:        low_severity
  approaching:    low_severity
  trending:       low_severity
```

### Adding domain-specific words

Add any terms your team commonly uses that carry a severity implication in your domain:

```yaml
# Financial services
bias_rules:
  breach:         high_severity
  covenant:       high_severity
  sar:            high_severity
  enhanced:       medium_severity

# Clinical trials
bias_rules:
  stopping:       high_severity
  serious:        high_severity
  susar:          high_severity
  unexpected:     medium_severity
  possibly:       low_severity

# DevOps / SRE
bias_rules:
  outage:         high_severity
  slo:            medium_severity
  degradation:    medium_severity
  leak:           medium_severity
```

---

## Sub-section 4 — global strategies

These two lines tell the system which evaluation method to use as a default and which to prefer when multiple methods are valid.

```yaml
global_default_strategy:   threshold    # use this when nothing else matches
global_preferred_strategy: percentile   # prefer this when multiple options are valid
```

**Which to choose:**

| Your domain | Recommended default | Recommended preferred |
|---|---|---|
| SaaS / product analytics | `threshold` | `percentile` |
| Financial risk / compliance | `threshold` | `threshold` |
| AML / fraud detection | `threshold` | `z_score` |
| DevOps / SRE | `threshold` | `change` |
| Clinical trials | `threshold` | `threshold` |

If you are unsure, `threshold` for both is a safe default. You can always change it later.

---

## Sub-section 5 — threshold_directions (optional)

By default, a threshold condition fires when a value goes **above** the threshold. But some signals fire when they go **below** a threshold — coverage ratios, active user rates, DSCR.

Add this section for any signal where "below the threshold" is the concern:

```yaml
threshold_directions:
  account.active_user_rate_30d:   below  # fires when rate FALLS below threshold
  account.seat_utilization_rate:  below
  borrower.dscr:                  below  # fires when DSCR FALLS below threshold
  bank.cet1_ratio:                below
  loan.covenant_headroom_pct:     below
```

A simple rule: if your `parameter_priors` values for a signal get **smaller** as severity increases, that signal probably fires `below`. If they get **larger** as severity increases, it fires `above` (the default).

```yaml
# Getting smaller as severity increases → fires below
account.active_user_rate_30d:
  low_severity:    { value: 0.60 }  ← largest
  medium_severity: { value: 0.45 }
  high_severity:   { value: 0.30 }  ← smallest
# → add to threshold_directions as "below"

# Getting larger as severity increases → fires above (default, no entry needed)
transaction.value_vs_baseline_ratio:
  low_severity:    { value: 3.0  }  ← smallest
  medium_severity: { value: 7.0  }
  high_severity:   { value: 15.0 }  ← largest
# → no entry needed in threshold_directions
```

---

## Complete Guardrails Examples

### SaaS Churn Detection

```yaml
guardrails:
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
      low_severity:    { value: 0.60 }
      medium_severity: { value: 0.45 }
      high_severity:   { value: 0.30 }

    account.seat_utilization_rate:
      low_severity:    { value: 0.65 }
      medium_severity: { value: 0.50 }
      high_severity:   { value: 0.35 }

    account.days_to_renewal:
      low_severity:    { value: 90 }
      medium_severity: { value: 60 }
      high_severity:   { value: 30 }

    user.session_frequency_trend_8w:
      low_severity:    { value: 0.20, window: "4w" }
      medium_severity: { value: 0.35, window: "4w" }
      high_severity:   { value: 0.50, window: "4w" }

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
guardrails:
  type_strategy_map:
    float:                [threshold, percentile, z_score, change]
    int:                  [threshold, percentile, change]
    float?:               [threshold]
    boolean:              [equals]
    categorical:          [equals]
    time_series<float>:   [change, z_score, percentile]

  parameter_priors:
    borrower.dscr:
      low_severity:    { value: 1.80 }
      medium_severity: { value: 1.50 }
      high_severity:   { value: 1.30 }

    borrower.leverage_ratio:
      low_severity:    { value: 3.0 }
      medium_severity: { value: 4.0 }
      high_severity:   { value: 5.5 }

    borrower.dscr_trend_4q:
      low_severity:    { value: 0.20, window: "2q" }
      medium_severity: { value: 0.30, window: "3q" }
      high_severity:   { value: 0.40, window: "4q" }

    loan.covenant_headroom_pct:
      low_severity:    { value: 25 }
      medium_severity: { value: 15 }
      high_severity:   { value: 8  }

  bias_rules:
    breach:        high_severity
    covenant:      high_severity
    deteriorating: medium_severity
    stressed:      medium_severity
    declining:     medium_severity
    watch:         low_severity
    early:         low_severity
    proactive:     low_severity

  threshold_directions:
    borrower.dscr:              below
    borrower.current_ratio:     below
    loan.covenant_headroom_pct: below

  global_default_strategy:   threshold
  global_preferred_strategy: threshold
```

---

## Common Mistakes

**Setting threshold values that are too far apart.** If `low_severity` is 0.9 and `high_severity` is 0.1, the `medium_severity` value of 0.5 may be too blunt for most tasks. Keep the three levels proportional and operationally meaningful.

**Not adding domain-specific bias rules.** If your team will say "covenant breach risk" or "SUSAR-level event" or "SLO degradation", add those terms to `bias_rules`. Without them, the compiler falls back to the global default severity.

**Forgetting `threshold_directions` for below-threshold signals.** If a signal gets worse as it goes lower — DSCR, active user rate, budget remaining — it needs a `below` entry in `threshold_directions`. Without it, the condition will never fire because the values will never go above the threshold.

---

## Next Step

→ [Step 4: Actions](/docs/admin-guide/admin-actions)
