---
id: devops-observability
title: DevOps and Observability
sidebar_label: DevOps / Observability
---

# Tutorial: DevOps and Observability

A walkthrough of two critical SRE use cases — deployment risk monitoring and incident early warning with SLO breach prediction. Together they cover the two most consequential moments in a production system's lifecycle: when something changes, and when something is going wrong.

:::note What you'll build
Two deterministic monitoring systems that continuously evaluate your production environment — detecting deployment risk before it materialises and predicting SLO breaches before they occur — with every decision auditable, consistent, and reproducible across environments.
:::

---

## Why Observability Needs Memintel

SRE teams already have observability tooling — metrics, traces, logs, dashboards. The problem is not data. The problem is **signal extraction from noise at scale**.

Current alerting systems are rules-based: if error rate exceeds 1%, page someone. This approach has two well-known failure modes that every SRE team lives with:

**Alert fatigue** — rules fire constantly on thresholds that are technically breached but contextually normal. Engineers learn to ignore alerts. The one that matters gets missed.

**Late detection** — by the time a static threshold is crossed, the degradation has been underway for hours. The intervention window has already narrowed. The incident is no longer preventable — it is only recoverable.

Both failures have the same root cause: **the rules were written against a specific understanding of what "bad" looks like that may not match the current system's behaviour, and nobody updated them.**

Memintel addresses this differently. Instead of asking an engineer to specify what failure looks like, you express what reliability means — and the system continuously evaluates whether the trajectory of your environment is consistent with that.

---

## The Three Roles

| Role | Who they are | What they do |
|---|---|---|
| **Data Engineer** | Platform / observability team | Builds metric pipelines, writes resolver functions, delivers typed primitives from observability stack |
| **Admin** | SRE lead / platform owner | Maintains config files — primitives, guardrails — governs strategy registry and severity priors |
| **User** | SRE / on-call engineer | Expresses intent via internal tooling — creates, monitors, and manages monitoring tasks |

---

## Use Case 1 — Deployment Risk Monitoring

### The Problem

Every deployment is a risk event. Even with CI/CD, canary releases, and feature flags, deployments introduce the most common source of production incidents. The challenge is that deployment risk is not binary — it is a function of what changed, what the current system state is, and what patterns previous deployments with similar characteristics have shown.

Current approaches are either too conservative (block everything) or too permissive (ship and watch). What is missing is a continuous assessment of deployment risk that accounts for the specific characteristics of this deployment, this service, and this moment in the production environment.

### The Architecture Boundary

```
Raw signals   →  Signal Extraction  →  Primitives  →  Memintel
(CI/CD events,   (diff analysis,       (typed,          (deterministic
 metrics,         change scoring,       normalised)       risk evaluation)
 test results)    LLM on changelogs)
```

### Primitive Design

```yaml
# memintel_primitives_deployment.yaml

primitives:

  # Change characteristics
  - id: deployment.lines_changed
    type: int
    source: ci_pipeline
    entity: deployment_id
    description: Total lines of code changed in this deployment

  - id: deployment.files_changed_count
    type: int
    source: ci_pipeline
    entity: deployment_id
    description: Number of files modified in this deployment

  - id: deployment.has_db_migration
    type: boolean
    source: ci_pipeline
    entity: deployment_id
    description: True if deployment includes a database schema migration

  - id: deployment.has_dependency_update
    type: boolean
    source: ci_pipeline
    entity: deployment_id
    description: True if deployment updates external dependencies

  - id: deployment.changelog_risk_score
    type: float
    source: nlp_pipeline
    entity: deployment_id
    description: LLM-extracted risk score from PR description and commit messages, 0-1

  - id: deployment.changelog_confidence
    type: float
    source: nlp_pipeline
    entity: deployment_id
    description: Confidence of changelog risk extraction, 0-1

  # Service state at time of deployment
  - id: service.current_error_rate
    type: float
    source: metrics_pipeline
    entity: service_id
    description: Current error rate — deploying into an already-degraded service increases risk

  - id: service.current_p99_latency_ms
    type: float
    source: metrics_pipeline
    entity: service_id
    description: Current p99 latency in milliseconds

  - id: service.active_incident_flag
    type: boolean
    source: incident_management_pipeline
    entity: service_id
    description: True if service has an active open incident

  - id: service.deployment_failure_rate_30d
    type: float
    source: ci_pipeline
    entity: service_id
    description: Ratio of failed deployments in last 30 days, 0-1

  # Timing and environment signals
  - id: deployment.is_peak_traffic_window
    type: boolean
    source: traffic_analysis_pipeline
    entity: deployment_id
    description: True if deployment falls within peak traffic hours for this service

  - id: deployment.days_since_last_deploy
    type: int
    source: ci_pipeline
    entity: service_id
    description: Days since the last successful deployment — longer gap increases risk

  - id: deployment.test_coverage_delta
    type: float
    source: ci_pipeline
    entity: deployment_id
    description: Change in test coverage — negative means coverage decreased

  - id: deployment.similar_deployment_incident_rate
    type: float
    source: historical_analysis_pipeline
    entity: deployment_id
    description: Incident rate for historically similar deployments to this service, 0-1
```

### Resolver Examples

```python
# resolvers/deployment_resolvers.py

@registry.resolver("deployment.changelog_risk_score")
async def resolve_changelog_risk(entity_id: str, timestamp: datetime) -> float:
    # Pull PR description, commit messages, and diff summary
    # Pass to LLM for risk extraction — happens once at deployment creation
    deployment = await ci_api.get_deployment(entity_id)
    changelog_text = f"""
    PR title: {deployment.pr_title}
    PR description: {deployment.pr_description}
    Commits: {deployment.commit_messages}
    Changed files: {deployment.changed_files_summary}
    """
    result = await llm_client.extract_risk_score(
        text=changelog_text,
        domain="software_deployment",
        as_of=timestamp
    )
    return result.risk_score


@registry.resolver("deployment.similar_deployment_incident_rate")
async def resolve_similar_incident_rate(entity_id: str, timestamp: datetime) -> float:
    # Find historically similar deployments and compute their incident rate
    deployment = await ci_api.get_deployment(entity_id)
    result = await db.execute("""
        SELECT
            SUM(CASE WHEN d.caused_incident THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0)
        FROM deployments d
        WHERE d.service_id = $1
          AND d.has_db_migration = $2
          AND d.has_dependency_update = $3
          AND d.deployed_at BETWEEN $4 - INTERVAL '90 days' AND $4
          AND d.deployment_id != $5
    """, deployment.service_id, deployment.has_db_migration,
        deployment.has_dependency_update, timestamp, entity_id)
    return float(result.scalar() or 0)


@registry.resolver("service.current_error_rate")
async def resolve_error_rate(entity_id: str, timestamp: datetime) -> float:
    # Point-in-time error rate from metrics store
    result = await metrics_api.query(
        metric="http_error_rate",
        service=entity_id,
        at=timestamp,
        window="5m"
    )
    return float(result.value or 0)


@registry.resolver("deployment.is_peak_traffic_window")
async def resolve_peak_traffic(entity_id: str, timestamp: datetime) -> bool:
    deployment = await ci_api.get_deployment(entity_id)
    # Check if deployment time falls within service's historically peak hours
    result = await db.execute("""
        SELECT EXISTS (
            SELECT 1 FROM service_traffic_patterns
            WHERE service_id = $1
              AND day_of_week = EXTRACT(DOW FROM $2)
              AND hour_of_day = EXTRACT(HOUR FROM $2)
              AND is_peak = true
        )
    """, deployment.service_id, timestamp)
    return bool(result.scalar())
```

### Guardrails

```yaml
# memintel_guardrails_deployment.yaml

type_strategy_map:
  int:      [threshold, percentile]
  float:    [threshold, percentile, z_score]
  boolean:  [equals]

parameter_priors:
  deployment.lines_changed:
    low_severity:     { threshold: 200  }
    medium_severity:  { threshold: 500  }
    high_severity:    { threshold: 1500 }

  deployment.changelog_risk_score:
    low_severity:     { threshold: 0.4 }
    medium_severity:  { threshold: 0.6 }
    high_severity:    { threshold: 0.8 }

  deployment.similar_deployment_incident_rate:
    low_severity:     { threshold: 0.10 }
    medium_severity:  { threshold: 0.20 }
    high_severity:    { threshold: 0.35 }

  service.deployment_failure_rate_30d:
    low_severity:     { threshold: 0.10 }
    medium_severity:  { threshold: 0.20 }
    high_severity:    { threshold: 0.35 }

bias_rules:
  cautious:     high_severity
  standard:     medium_severity
  fast:         low_severity
  hotfix:       low_severity     # hotfixes bypass standard caution

global_default_strategy:   threshold
global_preferred_strategy: threshold
```

### User Creates Tasks

```typescript
// SRE lead creates deployment monitoring tasks via internal deployment portal

// Task 1 — High-risk deployment gate
const deploymentRisk = await client.tasks.create({
    intent: "Alert me when a deployment carries significant risk based on change size, service history, and current environment state",
    entityScope: "all_production_deployments",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/deployment-risk"
    },
    dryRun: true
});

// System resolves to composite condition:
// changelog_risk_score + similar_incident_rate + service_error_rate
// weighted by has_db_migration, is_peak_traffic_window, days_since_last_deploy

// Task 2 — Immediate block: deploying into active incident
const activeIncidentBlock = await client.tasks.create({
    intent: "Alert me immediately when someone attempts to deploy to a service with an active incident",
    entityScope: "all_production_deployments",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/deployment-block",
        priority: "immediate"
    }
});
// Resolves to: service.active_incident_flag = true (immediate — no threshold)

// Task 3 — Peak traffic deployment warning
const peakDeployment = await client.tasks.create({
    intent: "Alert me when a significant deployment is scheduled during peak traffic",
    entityScope: "all_production_deployments",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/deployment-timing"
    }
});
// Resolves to: is_peak_traffic_window = true AND lines_changed > 500
```

### What the Alert Looks Like

```
⚠️ Deployment Risk Alert — High Risk Detected

Service:      payment-service  (v2.4.1 → v2.4.2)
Deployment:   deploy_9f3k2m
Triggered by: @sarah.chen  |  14:23 UTC

Risk Factors:
  • DB migration included          — schema change detected
  • 847 lines changed              — above medium threshold (500)
  • Changelog risk score: 0.71     — "refactored payment retry logic"
  • Similar deployments:           — 28% caused incidents in last 90 days
  • Current error rate: 0.41%      — elevated baseline (normal: 0.12%)
  • Peak traffic window:           — 14:00-16:00 is historically high traffic

Overall risk score:  0.79  (threshold: 0.65)

Recommendation: Consider delaying to off-peak window (after 18:00 UTC)
or enabling canary release with 5% traffic split.

→ Proceed anyway   → Delay deployment   → Switch to canary   → Cancel
```

### The Deployment Risk Advantage

The `deployment.similar_deployment_incident_rate` primitive is what makes this genuinely intelligent rather than just threshold-based. A deployment with a database migration to a service that has had 28% of similar deployments cause incidents is categorically different from a deployment with a database migration to a service that has never had a migration-related incident. The system distinguishes between these automatically — not because a rule was written for each service, but because the historical pattern is encoded in the primitive and the intent "significant risk" incorporates it.

---

## Use Case 2 — Incident Early Warning and SLO Breach Prediction

### The Problem

SLO breaches are almost never sudden. They are the end state of a gradual degradation that began hours or days before the threshold was crossed. By the time a static alert fires, the error budget may already be significantly consumed, the on-call engineer is paged into an already-critical situation, and the investigation starts from scratch.

The SRE team does not need to be told the SLO was breached. They need to be told the SLO is *going to be* breached — with enough lead time to investigate and intervene before users are impacted.

### The Architecture Boundary

```
Raw telemetry  →  Signal Extraction  →  Primitives  →  Memintel
(metrics,          (aggregation,         (typed,          (deterministic
 traces,            percentile calc,      normalised)       trajectory eval)
 logs, events)      anomaly scoring)
```

### Primitive Design

```yaml
# memintel_primitives_slo.yaml

primitives:

  # SLO and error budget signals
  - id: service.error_rate_5m
    type: float
    source: metrics_pipeline
    entity: service_id
    description: Error rate over last 5 minutes

  - id: service.error_rate_trend_1h
    type: time_series<float>
    source: metrics_pipeline
    entity: service_id
    description: Error rate sampled every 5 minutes over last hour — enables change detection

  - id: service.error_budget_remaining_pct
    type: float
    source: slo_pipeline
    entity: service_id
    description: Percentage of monthly error budget remaining, 0-1

  - id: service.error_budget_burn_rate_1h
    type: float
    source: slo_pipeline
    entity: service_id
    description: Current error budget burn rate — how many months of budget consumed per hour

  # Latency signals
  - id: service.p99_latency_ms
    type: float
    source: metrics_pipeline
    entity: service_id
    description: Current p99 latency in milliseconds

  - id: service.p99_latency_trend_1h
    type: time_series<float>
    source: metrics_pipeline
    entity: service_id
    description: p99 latency sampled every 5 minutes over last hour

  - id: service.p99_latency_vs_baseline_ratio
    type: float
    source: metrics_pipeline
    entity: service_id
    description: Current p99 vs historical baseline for same day/hour pattern

  # Infrastructure signals
  - id: service.memory_utilization_pct
    type: float
    source: metrics_pipeline
    entity: service_id
    description: Current memory utilisation percentage

  - id: service.memory_trend_2h
    type: time_series<float>
    source: metrics_pipeline
    entity: service_id
    description: Memory utilisation over last 2 hours — detects memory leaks

  - id: service.cpu_utilization_pct
    type: float
    source: metrics_pipeline
    entity: service_id
    description: Current CPU utilisation percentage

  - id: service.saturation_score
    type: float
    source: metrics_pipeline
    entity: service_id
    description: Composite resource saturation across CPU, memory, connections, 0-1

  # Dependency signals
  - id: dependency.error_rate
    type: float
    source: metrics_pipeline
    entity: dependency_id
    description: Error rate of upstream dependency — enables cascade detection

  - id: dependency.latency_ms
    type: float
    source: metrics_pipeline
    entity: dependency_id
    description: Current latency of upstream dependency

  - id: service.downstream_impact_score
    type: float
    source: dependency_graph_pipeline
    entity: service_id
    description: Estimated blast radius if this service degrades, 0-1

  # Historical context
  - id: service.recent_deployment_flag
    type: boolean
    source: ci_pipeline
    entity: service_id
    description: True if a deployment occurred in the last 2 hours

  - id: service.similar_incident_pattern_score
    type: float
    source: historical_analysis_pipeline
    entity: service_id
    description: Similarity of current signal pattern to patterns that preceded past incidents, 0-1
```

### Resolver Examples

```python
# resolvers/slo_resolvers.py

@registry.resolver("service.error_budget_burn_rate_1h")
async def resolve_burn_rate(entity_id: str, timestamp: datetime) -> float:
    # Burn rate = how fast we are consuming the monthly error budget
    # A burn rate of 1.0 = consuming budget at exactly the sustainable rate
    # A burn rate of 14.4 = consuming the entire monthly budget in 2 days
    result = await db.execute("""
        WITH error_counts AS (
            SELECT
                COUNT(CASE WHEN status >= 500 THEN 1 END)::float AS errors,
                COUNT(*)::float AS total
            FROM request_log
            WHERE service_id = $1
              AND request_time BETWEEN $2 - INTERVAL '1 hour' AND $2
        )
        SELECT
            (errors / NULLIF(total, 0)) / $3 * (30 * 24)
        FROM error_counts
    """, entity_id, timestamp, await get_slo_threshold(entity_id))
    return float(result.scalar() or 0)


@registry.resolver("service.error_rate_trend_1h")
async def resolve_error_rate_trend(entity_id: str, timestamp: datetime) -> list[float]:
    # Return 12 data points — one per 5 minutes over the last hour
    result = await metrics_api.query_range(
        metric="http_error_rate",
        service=entity_id,
        start=timestamp - timedelta(hours=1),
        end=timestamp,
        step="5m"
    )
    return [float(point.value) for point in result.data_points]


@registry.resolver("service.similar_incident_pattern_score")
async def resolve_pattern_score(entity_id: str, timestamp: datetime) -> float:
    # Compare current signal fingerprint against historical pre-incident patterns
    current_signals = await get_signal_fingerprint(entity_id, timestamp)
    result = await pattern_matcher.similarity_score(
        service_id=entity_id,
        current_fingerprint=current_signals,
        compare_against="pre_incident_windows",
        lookback_days=90
    )
    return float(result.score)


@registry.resolver("service.p99_latency_vs_baseline_ratio")
async def resolve_latency_vs_baseline(entity_id: str, timestamp: datetime) -> float:
    current = await metrics_api.query(
        metric="http_p99_latency",
        service=entity_id,
        at=timestamp,
        window="5m"
    )
    # Compare against same hour same day-of-week over last 4 weeks
    baseline = await metrics_api.query(
        metric="http_p99_latency",
        service=entity_id,
        at=timestamp,
        window="5m",
        historical_baseline=True,
        baseline_weeks=4
    )
    return float(current.value / baseline.value) if baseline.value > 0 else 1.0
```

### Guardrails

```yaml
# memintel_guardrails_slo.yaml

type_strategy_map:
  float:                [threshold, percentile, z_score]
  time_series<float>:   [change, z_score, percentile]
  boolean:              [equals]

parameter_priors:
  service.error_budget_burn_rate_1h:
    low_severity:     { threshold: 2.0  }   # consuming budget 2x faster than sustainable
    medium_severity:  { threshold: 5.0  }   # consuming 30-day budget in 6 days
    high_severity:    { threshold: 14.4 }   # consuming 30-day budget in 2 days

  service.p99_latency_vs_baseline_ratio:
    low_severity:     { threshold: 1.5 }    # 50% above baseline
    medium_severity:  { threshold: 2.5 }    # 150% above baseline
    high_severity:    { threshold: 4.0 }    # 300% above baseline

  service.similar_incident_pattern_score:
    low_severity:     { threshold: 0.55 }
    medium_severity:  { threshold: 0.70 }
    high_severity:    { threshold: 0.85 }

  service.saturation_score:
    low_severity:     { threshold: 0.70 }
    medium_severity:  { threshold: 0.82 }
    high_severity:    { threshold: 0.92 }

  # Change strategy thresholds for trend detection
  service.error_rate_trend_1h:
    low_severity:     { value: 0.002, window: "30m" }   # 0.2pp increase in 30 mins
    medium_severity:  { value: 0.005, window: "20m" }   # 0.5pp increase in 20 mins
    high_severity:    { value: 0.010, window: "15m" }   # 1.0pp increase in 15 mins

bias_rules:
  proactive:    low_severity
  early:        low_severity
  standard:     medium_severity
  critical:     high_severity
  page:         high_severity

global_default_strategy:   threshold
global_preferred_strategy: change    # trend detection is preferred for SLO monitoring
```

### User Creates Tasks

```typescript
// SRE creates monitoring tasks via internal platform or Slack bot

// Task 1 — Error budget burn rate — the primary SLO signal
const burnRateTask = await client.tasks.create({
    intent: "Alert me early when error budget is burning significantly faster than sustainable",
    entityScope: "all_production_services",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/slo-burn-rate"
    }
});
// Resolves to: burn_rate > 5.0 (medium severity — consuming budget in 6 days)
// Fires well before the SLO is breached

// Task 2 — Error rate trending up
const errorTrendTask = await client.tasks.create({
    intent: "Alert me when error rate is rising steadily — before it breaches the SLO threshold",
    entityScope: "all_production_services",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/slo-error-trend"
    }
});
// Resolves to: change strategy on error_rate_trend_1h
// Fires on sustained upward trajectory, not just current level

// Task 3 — Memory leak detection
const memoryLeakTask = await client.tasks.create({
    intent: "Alert me when a service shows signs of a memory leak",
    entityScope: "all_production_services",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/slo-memory-leak"
    }
});
// Resolves to: monotonic increase strategy on memory_trend_2h
// Detects gradual linear growth that indicates a leak, not a spike

// Task 4 — Pattern matches past incidents
const patternTask = await client.tasks.create({
    intent: "Alert me when current service signals resemble patterns that preceded past incidents",
    entityScope: "all_production_services",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/slo-pattern-match"
    }
});
// Resolves to: similar_incident_pattern_score > 0.70
// This is the most powerful early warning — fires on familiar danger signs

// Task 5 — Latency degradation relative to baseline
const latencyTask = await client.tasks.create({
    intent: "Alert me when p99 latency is significantly above the historical baseline for this time of day",
    entityScope: "all_production_services",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/slo-latency"
    }
});
// Resolves to: p99_latency_vs_baseline_ratio > 2.5
// Baseline-adjusted — handles services with variable traffic patterns
```

### What the Alerts Look Like

**Burn rate alert — early warning:**
```
⚠️ SLO Early Warning — Error Budget Burning Fast

Service:       checkout-service
SLO:           99.9% availability  |  Monthly budget: 43.8 minutes
Triggered:     14:47 UTC  (proactive — SLO not yet breached)

Error budget status:
  Remaining:    67.3%  (29.4 minutes)
  Burn rate:    6.2x   (consuming 30-day budget in 4.8 days)
  At this rate: SLO breach in approximately 18.3 hours

Current signals:
  Error rate:     0.41%  (SLO threshold: 0.10%)
  p99 latency:    284ms  (baseline: 112ms — 2.5x elevated)
  Saturation:     0.71   (approaching high threshold)
  Recent deploy:  Yes    (payment-service v2.4.2 — 47 mins ago)

Pattern match:  0.68 similarity to pre-incident patterns (last incident: 2023-11-14)

→ Open investigation   → Check recent deployment   → Rollback   → Silence 30m
```

**Memory leak alert:**
```
⚠️ SLO Early Warning — Memory Leak Detected

Service:      user-session-service
Triggered:    09:23 UTC

Memory trend (last 2 hours):
  07:23: 61.2%  →  08:23: 68.7%  →  09:23: 76.1%
  Rate of increase: +7.4% per hour (linear — consistent with leak)

At current rate:
  OOM likely:    ~3.2 hours  (12:30 UTC)
  Service impact: downstream auth-service and api-gateway affected

Recent changes:
  Last deployment: user-session-service v1.8.3  (yesterday 16:45 UTC)
  Changed: session cache TTL logic refactored

→ Profile memory   → Check heap dump   → Rollback to v1.8.2   → Scale horizontally
```

### The SLO Monitoring Advantage

Two primitives make this qualitatively different from threshold alerting:

**`service.error_budget_burn_rate_1h`** converts the abstract SLO into a concrete rate of consumption. A burn rate of 6.2x tells the SRE that at the current pace, the entire monthly error budget will be consumed in 4.8 days — not that the error rate is 0.41%. This is the framing that drives action. Nobody acts on "error rate is 0.41% and the SLO is 0.1%." Everyone acts on "we will breach our SLO in 18 hours at this rate."

**`service.similar_incident_pattern_score`** is the most powerful primitive in this system. Instead of writing rules for every failure mode, the system learns from historical incident data what the pre-incident signal fingerprint looks like — and fires when the current fingerprint resembles it. The SRE does not need to enumerate every possible failure pattern. They express "alert me when this looks like it's going wrong" and the system matches against the evidence of what "going wrong" has looked like before.

---

## The System Response Loop

```
Trigger: Every 5 minutes (metrics) OR event-driven (deployment events)
          ↓
For each service in scope:
  Resolvers fetch point-in-time values:
    → Metrics API: error_rate, latency, memory, CPU
    → SLO pipeline: budget remaining, burn rate
    → CI pipeline: recent deployment flag
    → Pattern matcher: similarity to past incidents
  Concept computation runs:
    → Composite saturation score
    → Burn rate calculation
    → Trend analysis on time-series primitives
  Condition evaluates
          ↓
  If fired:
    Alert delivered with context — not just "threshold breached"
    Decision logged with full audit trail
    Runbook link attached based on condition type
          ↓
  If not: Decision logged. No alert.
          ↓
Next 5-minute tick
```

### Event-driven vs scheduled

| Task | Best trigger | Reason |
|---|---|---|
| Deployment risk | Event-driven on deployment creation | Risk assessment needed before deployment proceeds |
| Active incident block | Event-driven on deployment creation | Immediate — must fire before deployment starts |
| Error budget burn rate | Every 5 minutes | Budget consumption is continuous |
| Error rate trend | Every 5 minutes | Trend needs consistent sampling interval |
| Memory leak | Every 5 minutes | Linear growth requires regular sampling to detect slope |
| Pattern match | Every 5 minutes | Pattern similarity changes as signals evolve |

---

## Task Management

### User controls

```typescript
// View all active SLO monitoring tasks
const tasks = await client.tasks.list({ owner: "current_user" });

// Silence a task during a known maintenance window
await client.tasks.pause("tsk_slo_burn_rate_checkout", {
    reason: "Planned maintenance window 02:00-04:00 UTC — expected elevated errors"
});

// Check alert performance — are we getting too many false positives?
const taskDetail = await client.tasks.get("tsk_error_rate_trend");
console.log(`Fired ${taskDetail.fire_count_7d} times in last 7 days`);
console.log(`False positive rate: ${taskDetail.false_positive_rate_7d}`);
console.log(`Avg lead time before incident: ${taskDetail.avg_lead_time_minutes} mins`);
```

### Admin visibility

```typescript
// Admin view — alert volume and quality across all services
const metrics = await adminClient.tasks.metrics({
    tag: "slo_monitoring"
});

console.log(metrics.total_alerts_7d);          // 847
console.log(metrics.false_positive_rate);       // 0.23 — 23% false positive rate
console.log(metrics.avg_lead_time_minutes);     // 34 — average 34 mins before incident
console.log(metrics.incidents_caught_early);    // 12 of 14 incidents had prior warning

// Version distribution — are teams on the latest calibrated conditions?
console.log(metrics.version_distribution);
// { "1.0": 8, "1.1": 23, "1.2": 41 }  — most teams on latest version

// Suspend tasks if metrics pipeline is unreliable
await adminClient.tasks.suspend("tsk_memory_leak_all_services", {
    reason: "Memory metrics pipeline outage — data unreliable until 15:00 UTC"
});
```

---

## Calibration

### Deployment risk — driven by post-deployment incident data

```typescript
// A deployment was flagged as high-risk but deployed without incident — false positive
await client.feedback.decision({
    conditionId: "deployment.high_risk",
    conditionVersion: "1.1",
    entity: "deploy_9f3k2m",
    timestamp: "2024-03-15T14:23:00Z",
    feedback: "false_positive",
    note: "Deployment completed without incident — DB migrations for this service are low risk"
});
```

### SLO monitoring — driven by incident retrospectives

```typescript
// An incident occurred that was not predicted — false negative
// Memory was not flagged even though leak had been running for 3 hours
await client.feedback.decision({
    conditionId: "service.memory_leak_risk",
    conditionVersion: "1.0",
    entity: "user-session-service",
    timestamp: "2024-03-12T07:00:00Z",
    feedback: "false_negative",
    note: "Memory leak started at 07:00 — OOM occurred at 11:30 — not caught early enough"
});

// Admin reviews calibration recommendation
const cal = await adminClient.conditions.calibrate({
    conditionId: "service.memory_leak_risk",
    conditionVersion: "1.0",
});

console.log(cal.recommended_params);
// { monotonic_increase_threshold: 5.0 }  — reduced from 7.4% per hour to 5%
// Catches the leak earlier in its progression

console.log(cal.impact.delta_alerts);   // +3 per week
// Slightly more alerts but catches leaks 90 mins earlier on average
```

---

## Full Lifecycle Diagram

```
SETUP (one time)
──────────────────────────────────────────────────────────
Data Engineer:   Configure metrics pipeline resolvers
Data Engineer:   Configure CI/CD event pipeline resolvers
Data Engineer:   Build pattern matcher against incident history
Admin:           Define primitives → memintel_primitives_deployment.yaml
                                     memintel_primitives_slo.yaml
Admin:           Define guardrails  → memintel_guardrails_deployment.yaml
                                      memintel_guardrails_slo.yaml
System:          Load config at startup


DEPLOYMENT RISK EVALUATION
──────────────────────────────────────────────────────────
Deployment event created in CI/CD
  → Memintel evaluates immediately
  → Risk score computed from change size, service history, environment state
  → If high risk: alert fires before deployment proceeds
  → Engineer decides: proceed, delay, switch to canary, or cancel


ONGOING SLO MONITORING LOOP
──────────────────────────────────────────────────────────
Every 5 minutes:
  → Resolvers fetch current metrics, SLO state, pattern scores
  → Concept computation: burn rate, trend analysis, pattern matching
  → Conditions evaluate
  → If fired: contextual alert with lead time estimate and runbook link
  → If not: decision logged, no alert


TASK MANAGEMENT
──────────────────────────────────────────────────────────
User:    View / pause / resume own tasks
         (e.g. pause during planned maintenance windows)
Admin:   View alert volume and quality across all services
Admin:   Suspend tasks during metrics pipeline outages
Admin:   Deprecate condition versions after calibration


CALIBRATION CYCLE
──────────────────────────────────────────────────────────
Ground truth: incident retrospectives, post-deployment reviews
User:    Submit feedback (false positive / false negative)
Admin:   Review calibration recommendation + lead time impact
Admin:   Approve → new condition version (immutable)
User:    Rebind task to new version (explicit, never automatic)
         ↓
Back to ONGOING LOOP with improved detection sensitivity
```

---

## Role Summary

| Step | Who | Deployment Risk | SLO Monitoring |
|---|---|---|---|
| Data pipeline | **Data Eng.** | CI/CD events, diff analysis, NLP on changelogs | Metrics pipeline, SLO engine, pattern matcher |
| Resolvers | **Data Eng.** | Changelog risk, historical incident rate, environment state | Burn rate, trend series, baseline comparison |
| Primitives | **Admin** | `memintel_primitives_deployment.yaml` | `memintel_primitives_slo.yaml` |
| Guardrails | **Admin** | `memintel_guardrails_deployment.yaml` | `memintel_guardrails_slo.yaml` |
| Task creation | **User** | SRE lead / release manager | On-call SRE |
| Feedback | **User** | Post-deployment incident reviews | Incident retrospectives |
| Calibration | **Admin** | Approves risk threshold adjustments | Approves sensitivity adjustments |

---

## Why This Architecture Fits SRE

Both use cases share the same structural property: **the risk signal requires evaluating current state against historical pattern, not just against a static threshold.**

Deployment risk requires evaluating this deployment against the history of similar deployments to this service — with similar change characteristics, at similar times, in similar environment states. A threshold on lines changed is noise. A threshold on "lines changed to a service whose similar deployments caused incidents 28% of the time" is signal.

SLO breach prediction requires evaluating current metrics against the trajectory — not the current value. An error rate of 0.41% below a 1.0% SLO threshold is fine in isolation. An error rate that has risen from 0.08% to 0.41% in 90 minutes, with a burn rate of 6.2x, while memory is climbing linearly and the signal pattern resembles the last three incidents — that is a page that should have gone out 30 minutes ago.

Static thresholds fire when something is already wrong. Memintel fires when something is **becoming** wrong — which is the only framing that gives an SRE team enough time to prevent an incident rather than just respond to one.

---

## Next Steps

- [Core Concepts](/docs/intro/core-concepts) — understand the ψ → φ → α model in depth
- [Why not SQL and rules?](/docs/intro/why-not-rules) — the architectural case for intent-based monitoring
- [Financial Risk Monitoring](/docs/tutorials/financial-risk-monitoring) — the same architecture in finance
- [API Reference](/docs/api-reference/overview) — full endpoint documentation
