---
id: admin-guardrails
title: Step 2 — Guardrails
sidebar_label: Step 2 — Guardrails
---

# Step 2 — Guardrails (memintel_guardrails.yaml)

The guardrails file is your policy configuration. It defines the rules the compiler follows when interpreting what your team members ask for — which evaluation strategies are permitted, what parameter ranges are valid, and how natural language severity words map to numeric thresholds.

**This is your file.** Your data engineer sets up `memintel_config.yaml`. You own and maintain `memintel_guardrails.yaml`.

---

## What This File Is

`memintel_guardrails.yaml` is a plain YAML text file that lives on the server at a path declared inside `memintel_config.yaml`. Ask your data engineer:

1. Where the file is located on the server
2. How to access and edit it (directly via terminal, or through your hosting platform's file editor)

:::warning
Changes to this file require a **server restart** to take effect. After editing, ask your data engineer to restart the server. Application context (Step 1) does not require a restart — only this file does.
:::

---

## What is YAML?

YAML is a plain text format for configuration. It uses indentation and colons to organise information. The most important rule: **indentation matters** — lines that are indented further belong to the line above them.

```yaml
# This is a comment — ignored by the system
parent_section:
  child_item: value          # colon separates key from value
  another_item: other_value

list_section:
  - first_item               # dash means list item
  - second_item
```

You will mostly be editing values (the parts after colons and dashes) rather than creating structure from scratch. The examples in this guide show exactly what everything should look like.

---

## File Structure

Your `memintel_guardrails.yaml` has five sections:

```yaml
# memintel_guardrails.yaml

strategy_registry:     # which evaluation methods are available
  - threshold
  - percentile
  - z_score
  - change
  - equals
  - composite

type_strategy_map:     # which methods are valid for each signal type
  float:    [threshold, percentile, z_score, change]
  int:      [threshold, percentile, change]
  boolean:  [equals]

parameter_priors:      # what thresholds mean low/medium/high for each signal
  <signal_id>:
    low_severity:    { value: <number> }
    medium_severity: { value: <number> }
    high_severity:   { value: <number> }

bias_rules:            # how natural language severity words map to severity levels
  urgent:      high_severity
  significant: medium_severity
  early:       low_severity

global_preferred_strategy: percentile   # which method to prefer when multiple are valid
global_default_strategy:   threshold    # fallback when no other rule matches
```

The sections you will edit most often are **`parameter_priors`** and **`bias_rules`**. The others rarely need changing once set up.

---

## Section 1 — strategy_registry

Lists the evaluation methods (strategies) available in your deployment. In most cases, copy this block exactly and do not change it.

```yaml
strategy_registry:
  - threshold    # fires when a value crosses a numeric threshold
  - percentile   # fires when a value is above/below a percentile rank
  - z_score      # fires when a value is statistically unusual vs baseline
  - change       # fires when a value is trending in a significant direction
  - equals       # fires when a value matches a specific category or flag
  - composite    # fires when a combination of conditions is met
```

---

## Section 2 — type_strategy_map

Declares which strategies are valid for each type of signal. Copy this block and leave it unchanged unless you have a specific reason to restrict certain strategies.

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

**When you might restrict this:**

In a clinical trial context where patient populations are small, `z_score` (which needs a large dataset to be statistically meaningful) may not be appropriate:

```yaml
# Clinical trials — restrict to simpler strategies
type_strategy_map:
  float:    [threshold, percentile]
  int:      [threshold, percentile]
  boolean:  [equals]
```

---

## Section 3 — parameter_priors

**This is the most important section.** For each signal your team will monitor, define what numeric threshold corresponds to low, medium, and high severity.

```yaml
parameter_priors:
  <signal_id>:
    low_severity:    { value: <number> }
    medium_severity: { value: <number> }
    high_severity:   { value: <number> }
```

When a user says "alert me when active user rate is **significantly** low":
1. The compiler identifies the signal: `account.active_user_rate_30d`
2. It maps "significantly" → `medium_severity` (via `bias_rules`)
3. It looks up `medium_severity` for this signal: `0.45`
4. It compiles: fire when active user rate falls below 45%

The user never specified 45%. That came from your guardrails.

### Choosing threshold values

Set three levels based on your operational experience:

| Level | Meaning | When to act |
|---|---|---|
| `low_severity` | Early warning — worth watching | Proactive monitoring, no urgent action needed |
| `medium_severity` | Material concern — action should be considered | Standard alert threshold |
| `high_severity` | Urgent — immediate action required | Drop everything |

### Direction: above vs below

Most signals fire when they go **above** the threshold (e.g. error rate above 5%). But some signals fire when they go **below** — coverage ratios, active user rates, DSCR. For those, you will need the `threshold_directions` section (see below).

A simple guide: if your threshold values get **smaller** as severity increases, the signal fires `below`. If they get **larger**, it fires `above` (the default — no extra config needed).

```yaml
# Smaller as severity increases → fires BELOW (needs threshold_directions entry)
account.active_user_rate_30d:
  low_severity:    { value: 0.60 }   # ← largest
  medium_severity: { value: 0.45 }
  high_severity:   { value: 0.30 }   # ← smallest

# Larger as severity increases → fires ABOVE (default, no extra config needed)
transaction.value_vs_baseline_ratio:
  low_severity:    { value: 3.0  }   # ← smallest
  medium_severity: { value: 7.0  }
  high_severity:   { value: 15.0 }   # ← largest
```

### Examples by domain

**SaaS — active user rate (fires below):**
```yaml
account.active_user_rate_30d:
  low_severity:    { value: 0.60 }   # 60% active — start watching
  medium_severity: { value: 0.45 }   # 45% active — take action
  high_severity:   { value: 0.30 }   # 30% active — urgent
```

**SaaS — days to renewal (fires below):**
```yaml
account.days_to_renewal:
  low_severity:    { value: 90 }   # 90 days — begin monitoring
  medium_severity: { value: 60 }   # 60 days — start outreach
  high_severity:   { value: 30 }   # 30 days — urgent
```

**AML — transaction value vs customer baseline (fires above):**
```yaml
transaction.value_vs_baseline_ratio:
  low_severity:    { value: 3.0  }   # 3x baseline
  medium_severity: { value: 7.0  }   # 7x baseline
  high_severity:   { value: 15.0 }   # 15x baseline — strong anomaly
```

**Credit — debt service coverage ratio (fires below):**
```yaml
borrower.dscr:
  low_severity:    { value: 1.80 }   # early warning
  medium_severity: { value: 1.50 }   # material concern
  high_severity:   { value: 1.30 }   # near covenant floor
```

**Credit — DSCR declining trend (time-series — uses value + window):**
```yaml
borrower.dscr_trend_4q:
  low_severity:    { value: 0.20, window: "2q" }   # 20% decline over 2 quarters
  medium_severity: { value: 0.30, window: "3q" }
  high_severity:   { value: 0.40, window: "4q" }
```

**Clinical — adverse event severity score (fires above):**
```yaml
patient.ae_severity_score:
  low_severity:    { value: 0.50 }   # Grade 2+ events
  medium_severity: { value: 0.70 }   # Grade 3+ events
  high_severity:   { value: 0.90 }   # Grade 4/5 events — serious
```

**DevOps — error budget burn rate (fires above):**
```yaml
service.error_budget_burn_rate_1h:
  low_severity:    { value: 2.0  }   # consuming budget 2x sustainable rate
  medium_severity: { value: 5.0  }   # consuming 30-day budget in 6 days
  high_severity:   { value: 14.4 }   # consuming 30-day budget in 2 days
```

---

## Section 4 — bias_rules

Maps the natural language words your team uses in monitoring requests to the severity levels you defined in `parameter_priors`.

```yaml
bias_rules:
  <word or phrase>: <severity level>
```

### Standard bias rules — copy as your starting point

```yaml
bias_rules:
  # High severity
  urgent:         high_severity
  critical:       high_severity
  immediately:    high_severity
  page:           high_severity

  # Medium severity
  significant:    medium_severity
  material:       medium_severity
  elevated:       medium_severity
  notable:        medium_severity

  # Low severity
  early:          low_severity
  proactive:      low_severity
  monitor:        low_severity
  approaching:    low_severity
  trending:       low_severity
```

### Add domain-specific words

Think about the words your team will naturally use when creating monitoring requests. Any word that carries a severity implication in your domain should have an entry:

```yaml
# Financial services
bias_rules:
  breach:       high_severity
  covenant:     high_severity
  sar:          high_severity
  enhanced:     medium_severity   # "enhanced due diligence"
  watchlist:    high_severity

# Clinical trials
bias_rules:
  stopping:     high_severity   # "approaching stopping rule"
  serious:      high_severity   # "serious adverse event"
  susar:        high_severity
  unexpected:   medium_severity
  possibly:     low_severity    # "possibly related"

# DevOps / SRE
bias_rules:
  outage:       high_severity
  slo:          medium_severity
  degradation:  medium_severity
  leak:         medium_severity
```

---

## Section 5 — threshold_directions (optional)

Add this section for signals where the condition fires when the value goes **below** the threshold rather than above.

```yaml
threshold_directions:
  account.active_user_rate_30d:   below
  account.seat_utilization_rate:  below
  borrower.dscr:                  below
  bank.cet1_ratio:                below
  loan.covenant_headroom_pct:     below
```

If you do not include this section, all threshold conditions fire when the value goes above the threshold.

---

## Section 6 — global strategies

```yaml
global_preferred_strategy: percentile   # prefer this when multiple strategies are valid
global_default_strategy:   threshold    # use this when nothing else matches
```

| Your domain | Recommended preferred | Reason |
|---|---|---|
| SaaS / product analytics | `percentile` | Relative comparison to population is usually more meaningful |
| Financial risk / compliance | `threshold` | Regulatory thresholds are typically absolute |
| AML / fraud detection | `z_score` | Anomaly against individual baseline is central |
| DevOps / SRE | `change` | Trend detection is more valuable than current level |
| Clinical trials | `threshold` | Protocol-defined absolute thresholds are the norm |

---

## Complete Examples

### SaaS Churn Detection

```yaml
# memintel_guardrails.yaml

strategy_registry:
  - threshold
  - percentile
  - z_score
  - change
  - equals
  - composite

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

global_preferred_strategy: percentile
global_default_strategy:   threshold
```

### Credit Risk

```yaml
# memintel_guardrails.yaml

strategy_registry:
  - threshold
  - percentile
  - z_score
  - change
  - equals
  - composite

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

threshold_directions:
  borrower.dscr:              below
  borrower.current_ratio:     below
  loan.covenant_headroom_pct: below

global_preferred_strategy: threshold
global_default_strategy:   threshold
```

---

## Applying Your Changes

After editing `memintel_guardrails.yaml`:

1. **Save the file** on the server
2. **Ask your data engineer to restart the server** — changes are not picked up until restart
3. **Verify** by asking your data engineer to check server startup logs for any YAML errors

To check the file is valid YAML before restarting (ask your data engineer to run):
```bash
python3 -c "import yaml; yaml.safe_load(open('memintel_guardrails.yaml'))" && echo "Valid"
```

---

## Common Mistakes

**Setting threshold values too far apart.** If `low_severity` is 0.9 and `high_severity` is 0.1, the `medium_severity` of 0.5 may not match any real-world scenario. Keep the three levels proportional and operationally meaningful.

**Missing domain-specific bias rules.** If your team says "covenant breach risk", "SUSAR-level event", or "SLO degradation" in their monitoring requests, add these to `bias_rules`. Without them, the compiler falls back to the global default severity.

**Forgetting `threshold_directions` for below-threshold signals.** If a signal gets worse as it goes lower (DSCR, active user rate, budget remaining), add it to `threshold_directions` as `below`. Without this, the condition will never fire.

**Editing the file without a server restart.** Changes to this file do not take effect until the server is restarted. Always ask your data engineer to restart after you edit.

---

## Setup Complete

You have now completed the two admin setup steps:

1. ✓ **Application Context** — domain briefing submitted via `POST /context`
2. ✓ **Guardrails** — policy configuration defined in `memintel_guardrails.yaml`

Ask your data engineer to restart the server and run a smoke test to verify everything is working.

After that, your team can begin creating monitoring tasks. Point them to the [Quickstart](/docs/intro/quickstart).
