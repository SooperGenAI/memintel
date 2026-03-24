---
id: snippets
title: Code Snippets
sidebar_label: Code Snippets
---

# Code Snippet Reference

20 production-ready examples · TypeScript SDK · ψ → φ → α pattern

:::note API Alignment
These snippets have been verified against the Memintel App Developer API v2.1.
Methods marked **[Python SDK]** are available in the Python backend SDK only
(`POST /execute`, `POST /execute/batch`, `POST /execute/range`, `POST /execute/async`,
`POST /evaluate/condition`, `POST /evaluate/condition/batch`).
The TypeScript SDK uses `evaluateFull()` as the primary execution path.
:::

## How to Read These Examples

Every snippet follows the same three-layer structure:

| Layer | Role |
|---|---|
| **ψ Concept** | Computes a signal — what is measured and how |
| **φ Condition** | Interprets significance — what the signal means in context |
| **α Action** | Defines the automated response — what happens when φ fires |

Each example shows: the SDK call, the response outputs, why the pattern works, and variations to adapt it.

---

## Categories

- **A. Risk & Monitoring** — 1–5: Churn, Fraud, Credit Risk, Payment Failures, Inactivity
- **B. Business Metrics** — 6–10: Revenue Growth, Conversion, LTV, Cohorts, KPI Anomalies
- **C. Time-Series & Trends** — 11–14: MA Crossover, Volatility, Momentum, Rolling Percentile
- **D. Trading & Finance** — 15–18: Stock Momentum, Narrative Divergence, Sector Rotation, Event-Driven
- **E. Operations & Alerts** — 19–20: SLA Breach, System Anomaly

---

## A. Risk & Monitoring

### 1. Churn Risk Detection

| | |
|---|---|
| **ψ Concept** | Compute churn probability score (0–1) for a user |
| **φ Condition** | Score > 0.8 over last 30 days = high churn risk |
| **α Action** | Send retention email when condition fires |

```typescript
import Memintel from "@memintel/sdk";

const client = new Memintel({ apiKey: process.env.MEMINTEL_API_KEY });

const result = await client.evaluateFull({
  concept_id: "org.churn_risk",
  concept_version: "1.2",
  condition_id: "org.high_churn",
  condition_version: "1.0",
  entity: "user_abc123",
  timestamp: "2024-03-15T09:00:00Z", // deterministic
});

result.result.value                              // 0.87 (churn score)
result.decision.value                            // true (condition fired)
result.decision.actions_triggered[0].status      // "triggered"
result.decision.actions_triggered[0].action_id   // "org.send_retention_email"
```

**Why this works:** Timestamp pins execution — the same call on any date returns the same result. Action fires automatically with no conditional branching in your code.

**Variations:**
- Lower threshold to `0.65` → catch users earlier in churn trajectory
- Use `percentile` condition instead → flag top 10% churners relative to cohort
- Add `dry_run: true` → simulate without sending emails during testing

---

### 2. Fraud Risk Flagging

| | |
|---|---|
| **ψ Concept** | Compute fraud probability score for a transaction |
| **φ Condition** | Score > 0.9 = high fraud risk |
| **α Action** | Freeze account and alert risk team |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.fraud_score",
  concept_version: "3.1",
  condition_id: "org.fraud_above_threshold",
  condition_version: "2.0",
  entity: "txn_9f8e7d",
  timestamp: new Date().toISOString(),
  explain: true, // see which signals drove the score
});

const score = result.result.value;  // 0.94
const isFraud = result.decision.value;  // true
const actions = result.decision.actions_triggered;
// [{ action_id: "org.freeze_account", status: "triggered" },
//  { action_id: "org.alert_risk_team", status: "triggered" }]

// Inspect signal contributions
const contributions = result.result.explanation?.contributions;
// { velocity: 0.55, geo_mismatch: 0.28, device_risk: 0.17 }
```

**Why this works:** `explain: true` reveals which input signals drove the score. Two actions trigger in one atomic call — no orchestration code needed.

**Variations:**
- Adjust threshold to `0.75` for higher recall / lower precision
- Use `z_score` strategy on the condition to flag relative spikes vs. baseline
- Use `dry_run: true` in staging to verify action payloads before going live

---

### 3. Credit Risk Monitoring

| | |
|---|---|
| **ψ Concept** | Compute credit risk score for an account |
| **φ Condition** | Score crosses > 0.75 threshold (change strategy) |
| **α Action** | Trigger credit review workflow |

```typescript
// Evaluate the full pipeline — concept + condition + action
const result = await client.evaluateFull({
  concept_id: "org.credit_risk",
  concept_version: "1.0",
  condition_id: "org.credit_risk_elevated",
  condition_version: "1.0",
  entity: "account_456",
  timestamp: "2024-06-01T00:00:00Z",
});

if (result.decision.value) {
  console.log("Review triggered for:", result.decision.entity);
  console.log("Actions:", result.decision.actions_triggered);
  // [{ action_id: "org.trigger_credit_review", status: "triggered" }]
}
```

:::note Python SDK
If the concept result is already cached, you can evaluate the condition alone
using `POST /evaluate/condition` via the Python SDK — faster on repeated calls.
Pre-warm with `POST /execute` to ensure the first condition evaluation hits cache.
:::

**Variations:**
- Use `change` strategy with `direction: 'up'` to only fire on deterioration
- Batch-evaluate a portfolio via `POST /evaluate/condition/batch` (Python SDK)

---

### 4. Payment Failure Spike Detection

| | |
|---|---|
| **ψ Concept** | Compute payment failure rate for a merchant over a rolling window |
| **φ Condition** | Z-score > 3.0 vs. historical baseline = anomalous spike |
| **α Action** | Page on-call team and open incident ticket |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.payment_failure_rate",
  concept_version: "1.0",
  condition_id: "org.failure_rate_spike",
  condition_version: "1.0",
  entity: "merchant_abc",
  timestamp: "2024-06-15T14:00:00Z",
});

// condition uses z_score strategy — detects deviation from baseline
console.log(result.result.value);   // 0.23 (23% failure rate)
console.log(result.decision.value); // true (z-score = 3.8)
result.decision.actions_triggered[0].action_id  // "org.page_oncall"
```

**Why this works:** Z-score detects anomalies relative to baseline — not just absolute value. A 23% failure rate may be normal for some merchants; z-score cuts through that.

**Variations:**
- Lower z-score threshold to `2.0` → more sensitive, higher alert rate
- Change window from `1h` to `15m` → faster detection of fast-moving spikes
- Add a second condition on raw rate `> 0.15` for absolute floor protection

---

### 5. User Inactivity Alert

| | |
|---|---|
| **ψ Concept** | Compute days since last meaningful engagement event |
| **φ Condition** | Days since activity > 14 = inactive |
| **α Action** | Send re-engagement push notification |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.days_since_activity",
  concept_version: "1.0",
  condition_id: "org.user_inactive_14d",
  condition_version: "1.0",
  entity: "user_xyz",
  // no timestamp → snapshot mode, reflects current state
});

console.log(result.result.value);   // 17 (days since last session)
console.log(result.decision.value); // true
result.decision.actions_triggered[0].action_id  // "org.send_reengagement_push"
```

**Why this works:** Snapshot mode (no timestamp) reflects real-time state — ideal for operational monitoring where you want current data on every call.

**Variations:**
- Segment by user tier: 7d for free users, 30d for paid users
- Chain with a churn risk concept to suppress re-engagement if churn is already high

---

## B. Business Metrics

### 6. Revenue Growth Tracking

| | |
|---|---|
| **ψ Concept** | Compute revenue growth rate (month-over-month) for a company |
| **φ Condition** | Growth rate > 20% MoM = accelerating |
| **α Action** | Add company to high-growth outreach list |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.revenue_growth_mom",
  concept_version: "1.0",
  condition_id: "org.high_growth_threshold",
  condition_version: "1.0",
  entity: "company_xyz",
  timestamp: "2024-03-31T23:59:59Z",
  explain: true,
});

result.result.value   // 0.34 (34% MoM growth)
result.decision.value // true (> 20% threshold)
result.decision.actions_triggered[0].action_id  // "org.add_to_outreach_list"

// Inspect signal contributions
const contributions = result.result.explanation?.contributions;
// { recurring_revenue: 0.68, new_logos: 0.22, expansion: 0.10 }
```

:::note Python SDK
For concept-only execution with `explain_mode: "summary"`, use `POST /execute`
via the Python SDK. Both `evaluateFull()` and `POST /execute` share the same
result cache — a concept executed by one is available to the other.
:::

**Variations:**
- Switch to a `percentile` condition to target the top 25% growers
- Use `POST /execute/range` (Python SDK) to track growth trajectory over 12 months

---

### 7. Conversion Rate Drop

| | |
|---|---|
| **ψ Concept** | Compute signup-to-paid conversion rate over a 7-day rolling window |
| **φ Condition** | Conversion rate drops > 20% vs. 30-day historical average (change strategy) |
| **α Action** | Notify product channel and create a tracking issue |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.signup_conversion_rate",
  concept_version: "2.0",
  condition_id: "org.conversion_drop_alert",
  condition_version: "1.0",
  entity: "funnel_main",
  timestamp: "2024-07-10T00:00:00Z",
});

// change strategy — fires only on deterioration, not just low absolute value
console.log(result.result.value);   // 0.031 (3.1% conversion)
console.log(result.decision.value); // true (down from 4.2% baseline)
result.decision.actions_triggered[0].action_id  // "org.notify_product_channel"
```

**Why this works:** Change strategy is direction-aware — only fires on drops, not on a low absolute value that has been stable.

**Variations:**
- Add `direction: 'up'` for a separate condition to detect conversion improvements
- Segment by traffic source: `entity = 'funnel_paid'` vs `'funnel_organic'`

---

### 8. Customer LTV Threshold

| | |
|---|---|
| **ψ Concept** | Compute predicted lifetime value for a customer |
| **φ Condition** | Predicted LTV > $10,000 |
| **α Action** | Assign dedicated success manager and upgrade support tier |

```typescript
// Evaluate LTV condition across a customer segment
// Note: evaluateConditionBatch() is available via the Python SDK.
// With the TypeScript SDK, evaluate each entity with evaluateFull().

const entities = ["cust_001", "cust_002", "cust_003", "cust_004"];

const decisions = await Promise.all(
  entities.map(entity =>
    client.evaluateFull({
      concept_id: "org.customer_ltv",
      concept_version: "1.0",
      condition_id: "org.high_value_customer",
      condition_version: "1.0",
      entity,
      timestamp: "2024-04-01T00:00:00Z",
    })
  )
);

const highValue = decisions.filter(d => d.decision.value === true);
console.log(highValue.map(d => d.decision.entity));
// ["cust_001", "cust_004"]
// Actions auto-triggered for matched entities
```

:::note Python SDK
`POST /evaluate/condition/batch` (Python SDK) parallelises this natively in a
single network call — more efficient for large segments.
:::

**Variations:**
- Use `percentile` condition to dynamically flag top 5% by LTV within a cohort
- Scale to hundreds of entities using the Python SDK batch endpoint

---

### 9. Cohort Performance Comparison

| | |
|---|---|
| **ψ Concept** | Compute 90-day retention rate per acquisition cohort |
| **φ Condition** | Retention rate in the bottom 25th percentile vs. all cohorts |
| **α Action** | Flag cohort for growth team review |

```typescript
const cohorts = ["cohort_jan24", "cohort_feb24", "cohort_mar24"];

// Step 1: Get concept values for all cohorts
// Note: executeBatch() is available in the Python SDK.
// With the TypeScript SDK, use parallel evaluateFull() calls.
const results = await Promise.all(
  cohorts.map(entity =>
    client.evaluateFull({
      concept_id: "org.cohort_90d_retention",
      concept_version: "1.0",
      condition_id: "org.low_retention_cohort",
      condition_version: "1.0",
      entity,
      timestamp: "2024-04-01T00:00:00Z",
    })
  )
);

results.forEach(r => {
  console.log(r.decision.entity, r.result.value, r.decision.value);
  // cohort_jan24  0.72  false
  // cohort_feb24  0.58  true  ← flagged
  // cohort_mar24  0.69  false
});
```

**Why this works:** Percentile condition is cross-sectional — ranks relative to peers, not an absolute cut.

---

### 10. KPI Anomaly Detection

| | |
|---|---|
| **ψ Concept** | Compute daily active users (DAU) for the platform |
| **φ Condition** | DAU deviates > 2.5 standard deviations from 90-day rolling mean |
| **α Action** | Trigger executive alert and start anomaly investigation workflow |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.daily_active_users",
  concept_version: "1.0",
  condition_id: "org.dau_anomaly_zscore",
  condition_version: "1.0",
  entity: "platform_main",
  timestamp: "2024-03-22T00:00:00Z",
  explain: true,
});

console.log(result.result.value);   // 84200 (DAU)
console.log(result.decision.value); // true (z-score = -3.1)

const contributions = result.result.explanation?.contributions;
// { mobile_sessions: 0.62, web_sessions: 0.38 }
result.decision.actions_triggered[0].action_id  // "org.exec_alert"
```

**Why this works:** Z-score adapts to seasonal variation — no manual baseline management. `explain: true` shows which channels drove the anomaly.

---

## C. Time-Series & Trends

### 11. Moving Average Crossover

| | |
|---|---|
| **ψ Concept** | Compute spread between 10-day and 50-day moving averages |
| **φ Condition** | Spread crosses from negative to positive (change strategy, direction: up) |
| **α Action** | Generate buy signal and notify strategy engine |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.ma_spread_10_50",
  concept_version: "1.0",
  condition_id: "org.bullish_ma_crossover",
  condition_version: "1.0",
  entity: "AAPL",
  timestamp: "2024-03-15T16:00:00Z", // market close
});

// change strategy with direction: 'up' fires only on the crossover event
// — not on every day that 10d > 50d
console.log(result.result.value);   // 0.42 (spread: +$0.42)
console.log(result.decision.value); // true (just crossed zero)
result.decision.actions_triggered[0].action_id  // "org.generate_buy_signal"
```

**Variations:**
- Reverse: `direction: 'down'` → detect bearish death cross
- Use MACD spread instead of simple MA for momentum-adjusted signal
- Evaluate across a basket of tickers using parallel `evaluateFull()` calls

---

### 12. Volatility Spike Detection

| | |
|---|---|
| **ψ Concept** | Compute 10-day realised volatility (annualised) for an asset |
| **φ Condition** | Volatility > 2.5x the 90-day median (percentile strategy) |
| **α Action** | Alert risk desk and reduce position limits |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.realised_volatility_10d",
  concept_version: "1.0",
  condition_id: "org.vol_spike_percentile",
  condition_version: "1.0",
  entity: "BTC-USD",
  timestamp: "2024-03-12T16:00:00Z",
});

console.log(result.result.value);   // 0.87 (87% annualised vol)
console.log(result.decision.value); // true (above 95th percentile)

const actions = result.decision.actions_triggered;
// [{ action_id: "org.alert_risk_desk", status: "triggered" },
//  { action_id: "org.reduce_position_limits", status: "triggered" }]
```

**Why this works:** Percentile condition adapts to the asset's own history — BTC at 87% vol is normal; SPX at 87% is extreme. Entity-relative comparison handles this automatically.

---

### 13. Momentum Acceleration

| | |
|---|---|
| **ψ Concept** | Compute rate-of-change of 14-day RSI momentum |
| **φ Condition** | RSI momentum accelerating: week-over-week change > 15 points |
| **α Action** | Flag asset for entry watchlist review |

```typescript
// executeRange() for time-series history is available via the Python SDK.
// With the TypeScript SDK, use sequential evaluateFull() calls at each interval.

// Step 1: evaluate at weekly intervals to build the history
const dates = [
  "2024-02-01T00:00:00Z",
  "2024-02-08T00:00:00Z",
  "2024-02-15T00:00:00Z",
  "2024-03-01T00:00:00Z",
];

const history = await Promise.all(
  dates.map(timestamp =>
    client.evaluateFull({
      concept_id: "org.rsi_momentum_14d",
      concept_version: "1.0",
      condition_id: "org.momentum_accelerating",
      condition_version: "1.0",
      entity: "ETH-USD",
      timestamp,
    })
  )
);

history.forEach(r =>
  console.log(r.result.timestamp, r.result.value)
);
// 2024-02-01  42.1
// 2024-02-08  48.7
// 2024-02-15  56.3
// 2024-03-01  72.8  ← acceleration detected

// Step 2: evaluate the condition at the current decision point
const signal = await client.evaluateFull({
  concept_id: "org.rsi_momentum_14d",
  concept_version: "1.0",
  condition_id: "org.momentum_accelerating",
  condition_version: "1.0",
  entity: "ETH-USD",
  timestamp: "2024-03-15T16:00:00Z",
});

signal.decision.value  // true (acceleration confirmed)
signal.decision.actions_triggered[0].action_id  // "org.flag_for_watchlist"
```

:::note Python SDK
`POST /execute/range` (Python SDK) handles the time-series loop natively and
is more efficient for large date ranges.
:::

---

### 14. Rolling Percentile Detection

| | |
|---|---|
| **ψ Concept** | Compute 30-day cumulative return for a strategy |
| **φ Condition** | Return is in top 10th percentile vs. the strategy's own 1-year history |
| **α Action** | Promote strategy to primary allocation |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.strategy_30d_return",
  concept_version: "1.0",
  condition_id: "org.top_decile_performer",
  condition_version: "1.0",
  entity: "strategy_momentum_v3",
  timestamp: "2024-03-31T23:59:59Z",
});

console.log(result.result.value);   // 0.187 (18.7% 30d return)
console.log(result.decision.value); // true (top decile)
result.decision.actions_triggered[0].action_id  // "org.promote_to_primary"
```

---

## D. Trading & Finance

### 15. Stock Momentum Signal

| | |
|---|---|
| **ψ Concept** | Compute composite momentum score combining price ROC and analyst sentiment |
| **φ Condition** | Composite score > 0.75 = strong momentum alignment |
| **α Action** | Generate trade signal and log to strategy ledger |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.composite_momentum",
  concept_version: "2.0",
  condition_id: "org.strong_momentum_signal",
  condition_version: "1.0",
  entity: "NVDA",
  timestamp: "2024-03-15T16:00:00Z",
  explain: true,
});

const score = result.result.value;  // 0.81
const fired = result.decision.value; // true

const contrib = result.result.explanation?.contributions;
// { price_roc: 0.48, sentiment_score: 0.33 }
// When sentiment contribution > price: narrative is leading — early signal
```

**Why this works:** `explain: true` reveals whether price or narrative is leading. Composite concept fuses two distinct signals; condition acts on the combined output.

---

### 16. Narrative vs Price Divergence

| | |
|---|---|
| **ψ Concept** | Compute divergence between sentiment score and normalised price momentum |
| **φ Condition** | Divergence > 0.3 with sentiment leading price |
| **α Action** | Alert research desk and flag for further analysis |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.narrative_price_divergence",
  concept_version: "1.0",
  condition_id: "org.narrative_leading_price",
  condition_version: "1.0",
  entity: "TSLA",
  timestamp: "2024-03-10T16:00:00Z",
  explain: true,
});

const divergence = result.result.value;  // 0.42
const leading = result.decision.value;    // true

const breakdown = result.result.explanation?.contributions;
// { sentiment_component: 0.67, price_component: 0.25, vol_adj: 0.08 }
// High sentiment contribution: market hasn't caught up yet
```

**Why this works:** `explain: true` is essential here — the contribution breakdown IS the signal. When `sentiment_component > price_component`, the market hasn't priced in the narrative yet.

---

### 17. Sector Rotation Detection

| | |
|---|---|
| **ψ Concept** | Compute relative strength spread between two sector ETFs (XLK vs. XLE) |
| **φ Condition** | Spread shifts by more than 0.15 over 5 days (change strategy) |
| **α Action** | Trigger sector rotation alert and rebalance workflow |

```typescript
// Execute concept for both sectors in parallel
const [tech, energy] = await Promise.all([
  client.evaluateFull({
    concept_id: "org.sector_momentum",
    concept_version: "1.0",
    condition_id: "org.sector_active",
    condition_version: "1.0",
    entity: "XLK",
    timestamp: "2024-03-15T16:00:00Z",
  }),
  client.evaluateFull({
    concept_id: "org.sector_momentum",
    concept_version: "1.0",
    condition_id: "org.sector_active",
    condition_version: "1.0",
    entity: "XLE",
    timestamp: "2024-03-15T16:00:00Z",
  }),
]);

console.log("Tech momentum:", tech.result.value);    // 0.72
console.log("Energy momentum:", energy.result.value); // 0.29

// Detect the rotation on the spread concept
const rotation = await client.evaluateFull({
  concept_id: "org.sector_spread_xlk_xle",
  concept_version: "1.0",
  condition_id: "org.sector_rotation_signal",
  condition_version: "1.0",
  entity: "sector_pair_xlk_xle",
  timestamp: "2024-03-15T16:00:00Z",
});

rotation.result.value   // 0.43 (spread, widening)
rotation.decision.value // true (rotation detected)
```

---

### 18. Event-Driven Trade Signal

| | |
|---|---|
| **ψ Concept** | Compute post-earnings sentiment score for a ticker |
| **φ Condition** | Sentiment score > 0.65 within 2h of earnings release |
| **α Action** | Generate long signal and set position sizing parameters |

```typescript
// Earnings dropped at 16:00 — evaluate at 16:15 snapshot
const result = await client.evaluateFull({
  concept_id: "org.post_earnings_sentiment",
  concept_version: "1.0",
  condition_id: "org.strong_post_earnings_bullish",
  condition_version: "1.0",
  entity: "META",
  timestamp: "2024-02-01T21:15:00Z", // 15m after report
  explain: true,
});

// Deterministic: can re-evaluate this exact moment later for audit
console.log(result.result.value);   // 0.78 (post-earnings sentiment)
console.log(result.decision.value); // true (bullish signal)

const contrib = result.result.explanation?.contributions;
// { guidance_tone: 0.45, beat_magnitude: 0.32, mgmt_language: 0.23 }
```

---

## E. Operations & Alerts

### 19. SLA Breach Detection

| | |
|---|---|
| **ψ Concept** | Compute p99 API response latency over a 5-minute rolling window |
| **φ Condition** | p99 latency > 500ms = SLA breach |
| **α Action** | Page on-call engineer and open PagerDuty incident |

```typescript
const result = await client.evaluateFull({
  concept_id: "org.api_p99_latency_5m",
  concept_version: "1.0",
  condition_id: "org.sla_breach_500ms",
  condition_version: "1.0",
  entity: "service_payments_api",
  // no timestamp → snapshot mode for real-time ops monitoring
});

if (result.decision.value) {
  console.log("SLA BREACH:", result.result.value, "ms p99 latency");
  // SLA BREACH: 723 ms p99 latency
  console.log("Actions:", result.decision.actions_triggered);
  // [{ action_id: "org.page_oncall", status: "triggered" },
  //  { action_id: "org.open_pagerduty_incident", status: "triggered" }]
}
```

**Why this works:** Snapshot mode (no timestamp) ensures real-time monitoring reflects current state. Two actions trigger atomically — page + incident creation with no coordination code.

**Variations:**
- Add a warning condition at 300ms for early alerting before SLA breach
- Segment by endpoint: `entity = 'service_payments_api/checkout'`

---

### 20. System Anomaly Detection

| | |
|---|---|
| **ψ Concept** | Compute composite anomaly score from CPU, memory, and error rate signals |
| **φ Condition** | Composite z-score > 3.0 = system anomaly |
| **α Action** | Trigger auto-remediation workflow and create incident |

```typescript
// For lightweight concepts, evaluateFull() is the right default.
// For compute-heavy multi-signal concepts that may exceed 30s,
// use POST /execute/async + GET /jobs/{job_id} via the Python SDK.

const result = await client.evaluateFull({
  concept_id: "org.system_anomaly_composite",
  concept_version: "1.0",
  condition_id: "org.system_anomaly_detected",
  condition_version: "1.0",
  entity: "cluster_prod_us_east_1",
  // no timestamp → snapshot mode for real-time ops
});

console.log(result.result.value);   // 0.91 (composite anomaly score)
console.log(result.decision.value); // true (z-score = 3.7)
result.decision.actions_triggered[0].action_id  // "org.trigger_auto_remediation"
result.decision.actions_triggered[1].action_id  // "org.create_incident"
```

:::note Python SDK — Async Execution
For heavy multi-signal concepts expected to exceed 30 seconds, use
`POST /execute/async` + `GET /jobs/{job_id}` via the Python SDK.
Poll until `status === "completed"`, then evaluate the condition using
`POST /evaluate/condition` — it will hit the cached result from the async job.
Cancel a running job with `DELETE /jobs/{job_id}`.
:::

---

## SDK Method Quick Reference

| Method | Description |
|---|---|
| `client.evaluateFull()` | ψ + φ + α in one atomic call — **recommended default** (TypeScript SDK) |
| `POST /execute` | ψ only, with optional `explain` — Python SDK |
| `POST /evaluate/condition` | φ only, uses cached ψ result — Python SDK |
| `POST /evaluate/condition/batch` | φ for multiple entities in parallel — Python SDK |
| `POST /execute/batch` | ψ for multiple entities in parallel — Python SDK |
| `POST /execute/range` | ψ over a time range (time-series) — Python SDK |
| `POST /execute/async` | ψ as a background job (> 30s workloads) — Python SDK |
| `GET /jobs/{job_id}` | Poll an async job for result — Python SDK |

**Base URL:** `https://api.memsdl.ai/v1` · **Auth:** `X-API-Key` header · **Install:** `npm install @memintel/sdk`
