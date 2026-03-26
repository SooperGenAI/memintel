---
id: financial-risk-monitoring
title: Financial Risk Monitoring
sidebar_label: Financial Risk Monitoring
---

# Tutorial: Financial Risk Monitoring

A walkthrough of three of the most critical risk monitoring use cases for financial institutions — Anti-Money Laundering transaction monitoring, credit risk and loan portfolio surveillance, and capital adequacy monitoring under Basel. Each follows the same Memintel architecture but with domain-specific primitives, concepts, and conditions.

:::note What you'll build
Three deterministic monitoring systems that continuously evaluate a financial institution's risk state against an evolving regulatory and market environment — with every decision auditable, reproducible, and explainable.
:::

---

## Why Financial Risk Needs Memintel

Financial institutions face a specific class of risk problem that conventional systems handle badly. The risk is not just about what is happening now — it is about how the current state of a customer, portfolio, or balance sheet relates to an evolving set of external signals, regulatory thresholds, and historical patterns.

Current approaches share a common failure mode: they are **reactive and rule-based**. A transaction monitoring system fires when a transaction crosses a threshold. A credit system flags when a loan-to-value ratio breaches a limit. A capital system reports when a ratio falls below a floor. All of these are point-in-time checks against static rules.

What they cannot do:

- Detect gradual deterioration before a threshold is breached
- Evaluate context — a transaction that looks suspicious in isolation may be normal given a customer's established pattern
- React to external signals — a regulatory update, a market event, a peer institution enforcement action
- Maintain an audit trail that is legally defensible in enforcement proceedings

Memintel solves this by separating the **discovery of risk** (which requires contextual, temporal, and cross-memory reasoning) from the **execution of risk decisions** (which must be deterministic, auditable, and reproducible).

---

## The Three Roles

The same role structure applies across all three use cases.

| Role | Who they are | What they do |
|---|---|---|
| **Data Engineer** | Risk technology / data team | Builds data pipelines, writes resolver functions, delivers typed primitives |
| **Admin** | Chief Risk Officer / Compliance Officer | Maintains config files — primitives, guardrails — and governs task visibility and calibration |
| **User** | Risk analyst / compliance officer | Expresses intent via dashboard or bot — creates, monitors, and manages tasks |

---

## Use Case 1 — AML Transaction Monitoring

### The Problem

AML compliance requires continuous monitoring of customer transactions for patterns indicating money laundering, terrorist financing, or other financial crimes. The challenge is both scale — large banks process millions of transactions daily — and contextual complexity. A transaction that looks suspicious in isolation may be completely normal given a customer's established behavior. Conversely, a series of individually unremarkable transactions may reveal a sophisticated laundering pattern when evaluated collectively over time.

Current rule-based systems generate enormous false positive rates — industry estimates suggest 90–95% of AML alerts are false positives. This consumes vast compliance analyst time and paradoxically reduces the effectiveness of genuine detection.

### The Architecture Boundary

```
Raw transactions  →  Signal Extraction  →  Primitives  →  Memintel
(payment flows,       (pattern analysis,    (typed,          (deterministic
 account data,         LLM on narratives,    normalised)       evaluation)
 counterparty info)    risk scoring)
```

### Primitive Design

*From transaction signals:*

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
    description: Daily transaction count over last 30 days — enables change detection

  - id: customer.counterparty_jurisdiction_risk
    type: float
    source: risk_pipeline
    entity: customer_id
    description: Weighted average jurisdiction risk of counterparties, 0-1

  # Current transaction signals
  - id: transaction.value_vs_baseline_ratio
    type: float
    source: transaction_pipeline
    entity: transaction_id
    description: Current transaction value divided by customer 90-day average

  - id: transaction.structuring_signal
    type: float
    source: transaction_pipeline
    entity: transaction_id
    description: Probability score that transaction pattern suggests structuring, 0-1

  - id: transaction.narrative_risk_score
    type: float
    source: nlp_pipeline
    entity: transaction_id
    description: LLM-extracted risk score from transaction narrative/reference, 0-1

  - id: transaction.narrative_confidence
    type: float
    source: nlp_pipeline
    entity: transaction_id
    description: Confidence of narrative risk extraction, 0-1

  # External regulatory signals
  - id: customer.watchlist_match_score
    type: float
    source: sanctions_pipeline
    entity: customer_id
    description: Fuzzy match score against current watchlists, 0-1

  - id: customer.jurisdiction_fatf_status
    type: categorical
    source: regulatory_pipeline
    entity: customer_id
    description: FATF status of customer's primary jurisdiction — clean/grey/black

  - id: typology.recent_match_score
    type: float
    source: regulatory_pipeline
    entity: customer_id
    description: Similarity of customer pattern to recently published typologies, 0-1
```

### Resolver Examples

```python
# resolvers/aml_resolvers.py

@registry.resolver("customer.avg_transaction_value_90d")
async def resolve_avg_txn_value(entity_id: str, timestamp: datetime) -> float:
    result = await db.execute("""
        SELECT AVG(amount)
        FROM transactions
        WHERE customer_id = $1
          AND transaction_date BETWEEN $2 - INTERVAL '90 days' AND $2
    """, entity_id, timestamp)
    return float(result.scalar() or 0)


@registry.resolver("transaction.value_vs_baseline_ratio")
async def resolve_value_ratio(entity_id: str, timestamp: datetime) -> float:
    # entity_id is transaction_id here
    txn = await db.execute("""
        SELECT amount, customer_id FROM transactions WHERE id = $1
    """, entity_id)
    row = txn.fetchone()
    if not row:
        return 0.0

    baseline = await db.execute("""
        SELECT AVG(amount)
        FROM transactions
        WHERE customer_id = $1
          AND transaction_date BETWEEN $2 - INTERVAL '90 days' AND $2
          AND id != $3
    """, row.customer_id, timestamp, entity_id)

    avg = float(baseline.scalar() or 1)
    return float(row.amount) / avg if avg > 0 else 1.0


@registry.resolver("customer.watchlist_match_score")
async def resolve_watchlist_score(entity_id: str, timestamp: datetime) -> float:
    # Watchlist state as of timestamp — not current state
    result = await sanctions_api.fuzzy_match(
        customer_id=entity_id,
        lists=["OFAC_SDN", "UN_SECURITY_COUNCIL", "EU_CONSOLIDATED"],
        as_of=timestamp
    )
    return result.highest_score
```

### Guardrails

```yaml
# memintel_guardrails_aml.yaml

parameter_priors:
  transaction.value_vs_baseline_ratio:
    low_severity:     { threshold: 3.0  }   # 3x baseline
    medium_severity:  { threshold: 7.0  }   # 7x baseline
    high_severity:    { threshold: 15.0 }   # 15x baseline

  transaction.structuring_signal:
    low_severity:     { threshold: 0.4 }
    medium_severity:  { threshold: 0.65 }
    high_severity:    { threshold: 0.85 }

  typology.recent_match_score:
    low_severity:     { threshold: 0.5 }
    medium_severity:  { threshold: 0.7 }
    high_severity:    { threshold: 0.85 }

bias_rules:
  conservative:   high_severity
  enhanced:       high_severity
  standard:       medium_severity
  monitor:        low_severity
```

### User Creates a Task

```typescript
// Compliance officer via AML monitoring dashboard
const task = await client.tasks.create({
    intent: "Alert me when a transaction shows high risk of structuring or unusual pattern relative to customer baseline",
    entityScope: "all_active_transactions",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/aml-alert"
    },
    dryRun: true
});

// System resolves to composite condition:
// structuring_signal > 0.85 OR value_vs_baseline > 15x
// weighted by customer jurisdiction risk and typology match
```

### What the Alert Looks Like

```
🚨 AML Alert — Unusual Transaction Pattern

Customer:    ABC Trading Ltd (customer_id: cust_7823)
Transaction: $847,000  (ref: TXN_92847)
Filed:       2024-03-15 14:23 UTC

Signals:
  • Transaction value: 12.4x customer 90-day average
  • Structuring signal: 0.71 (threshold: 0.65)
  • Counterparty jurisdiction: Grey-listed (FATF)
  • Narrative risk: 0.63 — reference contains unusual routing language

Risk score: 0.84 / threshold: 0.65

→ Open SAR workflow   → Mark reviewed   → Escalate to MLRO
```

### The AML Advantage

The critical difference from rule-based systems: the `value_vs_baseline_ratio` primitive means that a $10,000 transaction from a customer whose baseline is $100,000 scores very differently from a $10,000 transaction from a customer whose baseline is $800. The same transaction value can be perfectly normal or highly suspicious — context determines which. Memintel evaluates this contextually and deterministically.

---

## Use Case 2 — Credit Risk and Loan Portfolio Monitoring

### The Problem

Credit risk monitoring requires continuous surveillance of loan portfolios for deteriorating borrower health, changing collateral values, covenant breaches, and concentration risks. The challenge is that credit deterioration is typically gradual — a borrower who will default in six months often shows subtle warning signs months earlier that individually look unremarkable but collectively indicate stress.

Current systems flag covenant breaches reactively, when it is often too late for meaningful intervention. Memintel enables proactive early warning — detecting the trajectory toward a breach before it occurs.

### The Architecture Boundary

```
Raw loan data  →  Signal Extraction  →  Primitives  →  Memintel
(financials,      (ratio computation,    (typed,          (deterministic
 collateral,       LLM on mgmt notes,    normalised)       evaluation)
 market data)      market pricing)
```

### Primitive Design

```yaml
# memintel_primitives_credit.yaml

primitives:

  # Borrower financial health
  - id: borrower.debt_service_coverage_ratio
    type: float
    source: financial_analysis_pipeline
    entity: borrower_id
    description: EBITDA / total debt service — below 1.0 indicates inability to service debt

  - id: borrower.dscr_trend_4q
    type: time_series<float>
    source: financial_analysis_pipeline
    entity: borrower_id
    description: DSCR over last 4 quarters — enables trend and z_score detection

  - id: borrower.leverage_ratio
    type: float
    source: financial_analysis_pipeline
    entity: borrower_id
    description: Total debt / EBITDA

  - id: borrower.current_ratio
    type: float
    source: financial_analysis_pipeline
    entity: borrower_id
    description: Current assets / current liabilities — liquidity indicator

  - id: borrower.management_sentiment_score
    type: float?
    source: nlp_pipeline
    entity: borrower_id
    description: LLM-extracted sentiment from management commentary, 0-1

  # Collateral signals
  - id: collateral.ltv_ratio
    type: float
    source: collateral_pipeline
    entity: loan_id
    description: Outstanding loan balance / current collateral value

  - id: collateral.value_change_pct_90d
    type: float
    source: collateral_pipeline
    entity: loan_id
    description: Percentage change in collateral value over trailing 90 days

  # Covenant tracking
  - id: loan.covenant_headroom_pct
    type: float
    source: covenant_monitoring_pipeline
    entity: loan_id
    description: Distance to nearest covenant threshold as percentage — negative = breach

  - id: loan.days_since_financial_submission
    type: int
    source: covenant_monitoring_pipeline
    entity: loan_id
    description: Days since borrower last submitted required financial statements

  # Portfolio concentration
  - id: portfolio.sector_concentration_pct
    type: float
    source: portfolio_analytics_pipeline
    entity: sector_id
    description: Percentage of total portfolio exposure in this sector

  - id: sector.external_stress_signal
    type: float
    source: market_data_pipeline
    entity: sector_id
    description: Market-based stress signal for this sector, 0-1
```

### Resolver Examples

```python
# resolvers/credit_resolvers.py

@registry.resolver("borrower.debt_service_coverage_ratio")
async def resolve_dscr(entity_id: str, timestamp: datetime) -> float:
    result = await db.execute("""
        SELECT ebitda / NULLIF(total_debt_service, 0)
        FROM borrower_financials
        WHERE borrower_id = $1
          AND period_end_date <= $2
        ORDER BY period_end_date DESC
        LIMIT 1
    """, entity_id, timestamp)
    return float(result.scalar() or 0)


@registry.resolver("loan.covenant_headroom_pct")
async def resolve_covenant_headroom(entity_id: str, timestamp: datetime) -> float:
    # Returns the smallest headroom across all covenants for this loan
    # Negative value means breach
    result = await db.execute("""
        SELECT MIN(
            (current_value - covenant_threshold) / NULLIF(ABS(covenant_threshold), 0) * 100
        )
        FROM loan_covenant_tracking
        WHERE loan_id = $1
          AND measurement_date <= $2
          AND is_active = true
        ORDER BY measurement_date DESC
    """, entity_id, timestamp)
    return float(result.scalar() or 100)


@registry.resolver("borrower.dscr_trend_4q")
async def resolve_dscr_trend(entity_id: str, timestamp: datetime) -> list[float]:
    result = await db.execute("""
        SELECT ebitda / NULLIF(total_debt_service, 0) AS dscr
        FROM borrower_financials
        WHERE borrower_id = $1
          AND period_end_date <= $2
        ORDER BY period_end_date DESC
        LIMIT 4
    """, entity_id, timestamp)
    return [float(row.dscr or 0) for row in result.fetchall()]
```

### User Creates Tasks

```typescript
// Risk analyst creates three monitoring tasks

// Task 1 — Early warning: DSCR trending down
const earlyWarning = await client.tasks.create({
    intent: "Alert me when a borrower's debt service coverage ratio is declining significantly quarter over quarter",
    entityScope: "all_active_borrowers",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/credit-early-warning" }
});
// Resolves to: change strategy on dscr_trend_4q — fires on significant downward trend

// Task 2 — Covenant proximity alert
const covenantAlert = await client.tasks.create({
    intent: "Alert me when a loan is approaching a covenant breach",
    entityScope: "all_active_loans",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/covenant-alert" }
});
// Resolves to: covenant_headroom_pct < 15% (medium severity threshold)

// Task 3 — Concentration risk
const concentrationAlert = await client.tasks.create({
    intent: "Alert me when sector concentration in the portfolio becomes significant",
    entityScope: "all_active_sectors",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/concentration-alert" }
});
// Resolves to: sector_concentration_pct > 15% AND sector.external_stress_signal > 0.6
```

### What the Alert Looks Like

```
⚠️ Credit Early Warning — DSCR Deterioration

Borrower:  Acme Manufacturing Ltd
Loan:      $24M term loan (loan_id: LN_4821)
Officer:   J. Martinez

DSCR Trend (last 4 quarters):
  Q1 2023: 2.41  Q2 2023: 2.18  Q3 2023: 1.87  Q4 2023: 1.52

Change: -0.89 over 4 quarters (-37%)  |  Covenant floor: 1.25
Current headroom to covenant: 21.6%   |  Trajectory: 2 quarters to breach

Contributing signals:
  • Revenue down 12% YoY (from financial submission)
  • Management sentiment score: 0.31 (cautious tone in Q3 commentary)
  • Sector stress signal: 0.58 (manufacturing headwinds)

→ Schedule borrower review   → Update watch list   → Model covenant scenarios
```

### The Credit Risk Advantage

The `dscr_trend_4q` primitive registered as `time_series<float>` is what enables the early warning capability. A point-in-time DSCR of 1.52 looks fine — it is above the covenant floor of 1.25. But evaluated as a time series using a `change` strategy, the -37% decline over four quarters is a strong deterioration signal. Memintel detects the trajectory, not just the current position. This is the difference between reactive covenant monitoring and genuine early warning.

---

## Use Case 3 — Capital Adequacy Monitoring (Basel)

### The Problem

Basel III and its successors impose complex capital adequacy requirements on banks. They must hold sufficient capital against risk-weighted assets, maintain appropriate liquidity coverage ratios, and meet leverage requirements. These ratios are sensitive to both internal portfolio composition and external market conditions — a credit rating downgrade of a single large counterparty can materially shift risk-weighted assets and capital requirements across the entire portfolio.

Current capital systems are predominantly retrospective — they calculate yesterday's position and report it. Memintel enables forward-looking capital monitoring — detecting when a combination of internal positions and external conditions is creating capital pressure before a ratio breach occurs.

### The Architecture Boundary

```
Portfolio data  →  Signal Extraction  →  Primitives  →  Memintel
(positions,         (risk weighting,      (typed,          (deterministic
 counterparties,     LCR computation,      normalised)       evaluation)
 market data)        stress modelling)
```

### Primitive Design

```yaml
# memintel_primitives_capital.yaml

primitives:

  # Capital ratios
  - id: bank.cet1_ratio
    type: float
    source: capital_calculation_pipeline
    entity: entity_id
    description: Common Equity Tier 1 ratio — regulatory minimum 4.5%, typical buffer target 10%+

  - id: bank.cet1_ratio_trend_4q
    type: time_series<float>
    source: capital_calculation_pipeline
    entity: entity_id
    description: CET1 ratio over last 4 quarters

  - id: bank.lcr
    type: float
    source: liquidity_pipeline
    entity: entity_id
    description: Liquidity Coverage Ratio — regulatory minimum 100%

  - id: bank.leverage_ratio
    type: float
    source: capital_calculation_pipeline
    entity: entity_id
    description: Tier 1 capital / total exposure — regulatory minimum 3%

  # RWA sensitivity signals
  - id: counterparty.rating_change_flag
    type: boolean
    source: market_data_pipeline
    entity: counterparty_id
    description: True if counterparty credit rating changed in last 30 days

  - id: counterparty.rating_direction
    type: categorical
    source: market_data_pipeline
    entity: counterparty_id
    description: Direction of most recent rating change — upgrade/downgrade/stable

  - id: counterparty.exposure_pct_of_rwa
    type: float
    source: capital_calculation_pipeline
    entity: counterparty_id
    description: This counterparty's contribution to total RWA as percentage

  # Market stress signals
  - id: market.credit_spread_index
    type: time_series<float>
    source: market_data_pipeline
    entity: asset_class_id
    description: Credit spread index for this asset class — last 60 trading days

  - id: market.var_utilisation_pct
    type: float
    source: risk_pipeline
    entity: asset_class_id
    description: VaR limit utilisation for this asset class, 0-1

  # Regulatory environment
  - id: regulation.rwa_methodology_change_flag
    type: boolean
    source: regulatory_pipeline
    entity: asset_class_id
    description: True if risk weight methodology for this asset class changed in current Basel update

  - id: regulation.supervisory_add_on_pct
    type: float
    source: regulatory_pipeline
    entity: entity_id
    description: Supervisory capital add-on as percentage of RWA — from SREP outcome
```

### Resolver Examples

```python
# resolvers/capital_resolvers.py

@registry.resolver("bank.cet1_ratio")
async def resolve_cet1(entity_id: str, timestamp: datetime) -> float:
    result = await db.execute("""
        SELECT cet1_ratio
        FROM capital_reports
        WHERE entity_id = $1
          AND report_date <= $2
        ORDER BY report_date DESC
        LIMIT 1
    """, entity_id, timestamp)
    return float(result.scalar() or 0)


@registry.resolver("counterparty.rating_change_flag")
async def resolve_rating_change(entity_id: str, timestamp: datetime) -> bool:
    result = await db.execute("""
        SELECT COUNT(*) > 0
        FROM credit_rating_events
        WHERE counterparty_id = $1
          AND event_date BETWEEN $2 - INTERVAL '30 days' AND $2
    """, entity_id, timestamp)
    return bool(result.scalar())


@registry.resolver("counterparty.exposure_pct_of_rwa")
async def resolve_exposure_pct(entity_id: str, timestamp: datetime) -> float:
    total_rwa = await db.execute("""
        SELECT total_rwa FROM capital_reports
        WHERE report_date <= $1 ORDER BY report_date DESC LIMIT 1
    """, timestamp)

    counterparty_rwa = await db.execute("""
        SELECT SUM(risk_weighted_exposure)
        FROM rwa_breakdown
        WHERE counterparty_id = $1
          AND calculation_date <= $2
    """, entity_id, timestamp)

    total = float(total_rwa.scalar() or 1)
    cp_rwa = float(counterparty_rwa.scalar() or 0)
    return (cp_rwa / total) * 100 if total > 0 else 0
```

### Guardrails

```yaml
# memintel_guardrails_capital.yaml

parameter_priors:
  bank.cet1_ratio:
    low_severity:     { threshold: 0.115 }  # approaching management buffer
    medium_severity:  { threshold: 0.105 }  # inside management buffer
    high_severity:    { threshold: 0.095 }  # approaching regulatory minimum

  bank.lcr:
    low_severity:     { threshold: 1.15  }  # 15% above minimum
    medium_severity:  { threshold: 1.08  }  # 8% above minimum
    high_severity:    { threshold: 1.02  }  # near minimum

  counterparty.exposure_pct_of_rwa:
    low_severity:     { threshold: 5.0  }   # 5% of total RWA
    medium_severity:  { threshold: 8.0  }   # 8% of total RWA
    high_severity:    { threshold: 12.0 }   # 12% of total RWA

bias_rules:
  proactive:      low_severity
  early warning:  low_severity
  approaching:    medium_severity
  critical:       high_severity

# Note: capital ratios use direction: "below" — threshold fires when ratio
# falls below the parameter value, not above
threshold_directions:
  bank.cet1_ratio:    below
  bank.lcr:           below
  bank.leverage_ratio: below
```

### User Creates Tasks

```typescript
// Capital management team creates monitoring tasks

// Task 1 — CET1 ratio deterioration trend
const cet1Task = await client.tasks.create({
    intent: "Alert me when CET1 ratio shows a significant declining trend over the last 4 quarters",
    entityScope: "all_regulated_entities",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/capital-cet1" }
});
// Resolves to: change strategy on cet1_ratio_trend_4q — fires on significant decline

// Task 2 — Large counterparty downgrade impact
const downgradTask = await client.tasks.create({
    intent: "Alert me immediately when a large counterparty is downgraded",
    entityScope: "all_active_counterparties",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/capital-downgrade" }
});
// Resolves to: rating_change_flag = true AND rating_direction = downgrade
//              AND exposure_pct_of_rwa > 5%

// Task 3 — LCR approaching minimum
const lcrTask = await client.tasks.create({
    intent: "Alert me when LCR is approaching the regulatory minimum",
    entityScope: "all_regulated_entities",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/capital-lcr" }
});
// Resolves to: lcr < 1.08 (medium severity — inside management buffer)
```

### What the Alert Looks Like

```
🔴 Capital Alert — Counterparty Downgrade Impact

Counterparty:  GlobalTech Finance SA
Event:         Rating downgraded A- → BBB+ (S&P, 2024-03-15)
Exposure:      $340M (8.2% of total RWA)

Estimated RWA Impact:
  Pre-downgrade RWA:   $234M  (risk weight: 50%)
  Post-downgrade RWA:  $340M  (risk weight: 100%)
  RWA increase:        +$106M (+0.31% CET1 impact)

Current CET1 ratio:    11.2%
Post-event estimate:   10.89%  (management buffer: 11.0%)

→ Review hedging options   → Update capital plan   → Notify ALCO
```

### The Capital Adequacy Advantage

The `counterparty.exposure_pct_of_rwa` primitive is what makes this genuinely intelligent. When a counterparty is downgraded, the system does not just note the rating event — it immediately evaluates the capital impact by combining the downgrade signal with the counterparty's contribution to total RWA. A downgrade of a counterparty representing 0.3% of RWA is a routine monitoring item. A downgrade of a counterparty representing 8.2% of RWA may require immediate capital action. The Concept layer computes this distinction automatically.

---

## The System Response Loop (All Three Use Cases)

The evaluation loop is identical across all three use cases — only the primitives, concepts, and thresholds differ.

```
Trigger: Schedule (hourly/daily) OR event-driven
  (new transaction, rating event, regulatory update)
          ↓
For each entity in scope:
  Memintel calls resolvers with (entity_id, timestamp)
  Resolvers fetch point-in-time values from data sources
  Concept computation runs
  Condition evaluates
          ↓
  If condition fires:
    Action triggered — webhook, notification, SAR workflow
    Decision logged with full audit trail
    Alert delivered to analyst / officer
          ↓
  If not: Decision logged. No action.
          ↓
Next trigger
```

### Event-driven vs scheduled evaluation

AML is best run **event-driven** — evaluate at transaction time, not on a schedule. Delays in AML monitoring create regulatory exposure.

Credit risk is best run **daily or on financial submission** — borrower financials change slowly, but covenant proximity should be checked daily as market conditions affect collateral values.

Capital adequacy is best run **event-driven for rating changes** and **daily for ratio monitoring** — a rating event should trigger immediate evaluation; ratio trends are monitored daily.

```typescript
// AML — event-driven, fires on each transaction
const amlTask = await client.tasks.create({
    intent: "Alert on high-risk transactions",
    schedule: { type: "event", trigger: "transaction.created" },
    ...
});

// Credit — daily monitoring
const creditTask = await client.tasks.create({
    intent: "Early warning on DSCR deterioration",
    schedule: { frequency: "daily", time: "06:00" },
    ...
});

// Capital — event-driven for rating changes
const capitalTask = await client.tasks.create({
    intent: "Alert on large counterparty downgrade",
    schedule: { type: "event", trigger: "rating.changed" },
    ...
});
```

---

## Task Management

### User controls (same across all use cases)

```typescript
// View all active risk monitoring tasks
const tasks = await client.tasks.list({ owner: "current_user" });

// Pause AML task during scheduled system maintenance
await client.tasks.pause("tsk_aml_watch", {
    reason: "Transaction pipeline maintenance window 02:00-04:00"
});

// Check task performance metrics
const taskDetail = await client.tasks.get("tsk_credit_early_warning");
console.log(`Fired ${taskDetail.fire_count_30d} times in last 30 days`);
console.log(`False positive rate: ${taskDetail.false_positive_rate_30d}`);
```

### Admin visibility

```typescript
// Admin view — all tasks across all risk types
const allTasks = await adminClient.tasks.list();

// Filter by domain
const amlTasks     = await adminClient.tasks.list({ tag: "aml" });
const creditTasks  = await adminClient.tasks.list({ tag: "credit_risk" });
const capitalTasks = await adminClient.tasks.list({ tag: "capital" });

// Version distribution — critical for regulatory audits
// Regulator may ask: "which condition version was active on date X?"
const metrics = await adminClient.tasks.metrics({
    conditionId: "transaction.high_aml_risk"
});
console.log(metrics.version_distribution);
// { "1.0": 2, "1.1": 5, "1.2": 8 }
```

---

## Calibration

### AML calibration — driven by SAR outcomes

```typescript
// A transaction was flagged but SAR was not filed — false positive
await client.feedback.decision({
    conditionId: "transaction.high_aml_risk",
    conditionVersion: "1.1",
    entity: "txn_92847",
    timestamp: "2024-03-15T14:23:00Z",
    feedback: "false_positive",
    note: "Transaction consistent with known business pattern — documentary evidence collected"
});
```

### Credit calibration — driven by default events

```typescript
// A borrower defaulted that was not flagged — false negative
await client.feedback.decision({
    conditionId: "borrower.early_warning",
    conditionVersion: "1.0",
    entity: "borrower_4821",
    timestamp: "2023-09-30T00:00:00Z",
    feedback: "false_negative",
    note: "Borrower defaulted Q1 2024 — DSCR deterioration was not flagged early enough"
});
```

### Capital calibration — driven by supervisory findings

```typescript
// Supervisory review identified capital pressure that system missed
await client.feedback.decision({
    conditionId: "bank.capital_pressure",
    conditionVersion: "1.0",
    entity: "entity_001",
    timestamp: "2024-01-31T00:00:00Z",
    feedback: "false_negative",
    note: "SREP outcome identified capital concern — threshold too conservative"
});
```

The calibration request, impact review, and approval flow is identical across all three domains — see the [Deal Intelligence tutorial](/docs/tutorials/deal-intelligence#step-9----the-feedback-and-calibration-cycle) for the full flow.

---

## Role Summary

| Step | Who | AML | Credit Risk | Capital |
|---|---|---|---|---|
| Data pipeline | **Data Eng.** | Transaction feeds, NLP scoring | Financial analysis, collateral valuation | Capital calc engine, rating feeds |
| Resolvers | **Data Eng.** | Transaction, watchlist, typology APIs | Borrower financials, covenant tracking | Capital ratios, RWA breakdown |
| Primitives config | **Admin** | `memintel_primitives_aml.yaml` | `memintel_primitives_credit.yaml` | `memintel_primitives_capital.yaml` |
| Guardrails config | **Admin** | `memintel_guardrails_aml.yaml` | `memintel_guardrails_credit.yaml` | `memintel_guardrails_capital.yaml` |
| Task creation | **User** | Compliance officer | Credit analyst | Capital management team |
| Feedback | **User** | SAR outcomes | Default events | Supervisory findings |
| Calibration | **Admin** | Approves threshold adjustments | Approves threshold adjustments | Approves threshold adjustments |

---

## Why This Architecture Fits Financial Risk

All three use cases share the same structural property: **the risk signal requires evaluating internal state against an evolving external environment.**

AML requires evaluating a transaction against both the customer's historical pattern (internal) and current watchlists and typologies (external). A rule-based system sees the transaction in isolation. Memintel sees it in context.

Credit risk requires evaluating a borrower's financials against both their own historical trajectory (internal) and current market conditions and sector stress (external). A covenant monitoring system fires when a breach occurs. Memintel fires when a breach is approaching.

Capital adequacy requires evaluating portfolio positions against both current capital ratios (internal) and evolving risk weights, rating events, and regulatory requirements (external). A capital reporting system tells you where you are. Memintel tells you where you are heading.

The auditability requirement is the other shared property. In all three domains, every decision must be defensible — to auditors, to regulators, to courts. AML SAR filings require documented decision basis. Credit decisions require documented risk assessment trails. Capital calculations require reproducible methodology. Memintel's deterministic, immutable audit trail — logging every evaluation with its exact concept version, condition version, primitive values, and timestamp — produces this documentation automatically.

---

## Next Steps

- [Core Concepts](/docs/intro/core-concepts) — understand the ψ → φ → α model in depth
- [XBRL Compliance Tutorial](/docs/tutorials/xbrl-compliance) — the same architecture applied to SEC filing compliance
- [Deal Intelligence Tutorial](/docs/tutorials/deal-intelligence) — the same architecture applied to sales
- [Guardrails System](/docs/intro/guardrails) — how admins configure the policy layer
- [API Reference](/docs/api-reference/overview) — full endpoint documentation
