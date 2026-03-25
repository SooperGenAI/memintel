---
id: xbrl-compliance
title: XBRL Compliance Intelligence
sidebar_label: XBRL Compliance
---

# Tutorial: XBRL Compliance Intelligence for SEC Filings

A complete end-to-end walkthrough of building a deterministic XBRL compliance intelligence system on Memintel — from raw filing data and regulatory sources through to proactive pre-filing alerts. This tutorial covers the full architecture, the division of roles and responsibilities, and the design principles that make this system genuinely reliable in a high-stakes regulatory environment.

:::note What you'll build
A system that continuously monitors a company's filing state against evolving SEC requirements — detecting taxonomy changes, cross-period inconsistencies, comment letter risk, and filing pipeline issues before they become regulatory problems.
:::

---

## The Problem This Solves

Filing an SEC report is not just a matter of preparing numbers. Every 10-K and 10-Q requires a company to continuously reconcile:

1. **Their own history** — what was reported last quarter, what tags were used, what disclosures were made
2. **Changing external requirements** — new XBRL taxonomy versions, updated accounting standards, SEC comment letter patterns from peer companies

Today this reconciliation is manual, fragmented across teams and tools, done very late in the filing process, and error-prone. Companies only discover issues when validation fails, when auditors flag them, or — worst of all — when the SEC responds after filing.

**Memintel makes this continuous, proactive, and deterministic.** Instead of "prepare → validate → fix", it enables "continuously monitor → detect early → guide corrections."

---

## The Right Mental Model

The same three-layer architecture from the deal intelligence tutorial applies here — but the data sources and signals are different.

```
Layer 1           Layer 2                   Layer 3
──────────        ──────────────────────    ──────────────────────────
Raw Data     →    Signal Extraction    →    Memintel
(filings,         (parsers + LLMs,          (typed primitives →
 taxonomies,       structured signals)        deterministic decisions)
 comment ltrs)
```

### Layer 1 — Data Sources

**Internal (company's own data):**
- Historical XBRL filings — every element tagged in every prior filing
- Accounting policies in effect at each period
- Custom taxonomy extensions created for company-specific line items
- Prior SEC comment letters received and their responses
- Current filing pipeline state — milestones, dependencies, deadlines

**External (regulatory environment):**
- SEC US GAAP taxonomy versions — current and all historical versions
- IFRS taxonomy updates
- SEC comment letters across all public companies (available on EDGAR)
- FASB and IASB accounting standard updates and exposure drafts
- EDGAR filing deadlines and regulatory calendars

### Layer 2 — Signal Extraction

Parsers and LLMs convert raw filing data and regulatory documents into structured signals.

*From XBRL filings:*
- `tag_usage_count` — how many times a specific element has been used across prior filings
- `tag_last_used_period` — most recent period in which a tag appeared
- `calculation_relationship_hash` — fingerprint of calculation relationships between tags
- `prior_period_value` — reported value for a tag in the comparative period
- `current_period_value` — reported value for a tag in the current draft

*From taxonomy updates:*
- `deprecated_tag_flag` — boolean, is this tag deprecated in the new taxonomy version?
- `replacement_tag_id` — what tag replaces a deprecated one
- `calculation_change_flag` — boolean, have calculation relationships changed?
- `effective_date` — when the taxonomy change takes effect

*From SEC comment letters:*
- `comment_topic` — categorical, what disclosure area was questioned
- `comment_frequency_12m` — how many times this topic appeared in letters in the last 12 months
- `peer_similarity_score` — how similar the company's disclosure is to ones that attracted comments
- `trend_direction` — is SEC focus on this topic increasing or decreasing

*From filing pipeline:*
- `milestone_completion_rate` — ratio of completed to scheduled milestones
- `days_to_deadline` — calendar days until filing deadline
- `open_audit_issues` — count of unresolved audit questions
- `dependency_blockers` — count of upstream dependencies not yet resolved

### Layer 3 — Primitives (where Memintel begins)

```json
{ "name": "deprecated_tag_flag",          "type": "boolean"     }
{ "name": "tag_usage_count",              "type": "int"         }
{ "name": "prior_period_value",           "type": "float?"      }
{ "name": "current_period_value",         "type": "float?"      }
{ "name": "comment_frequency_12m",        "type": "int"         }
{ "name": "peer_similarity_score",        "type": "float"       }
{ "name": "days_to_deadline",             "type": "int"         }
{ "name": "milestone_completion_rate",    "type": "float"       }
{ "name": "open_audit_issues",            "type": "int"         }
```

:::warning The critical insight
Memintel's determinism guarantee begins at the primitive layer. The taxonomy comparison logic, the comment letter parsing, the pipeline dependency tracking — all of that happens upstream in your data pipeline. What crosses into Memintel must be clean, typed, and normalised. If a deprecated tag flag is sometimes `null` and sometimes `false`, the system cannot distinguish between "not deprecated" and "not yet evaluated." Type your primitives strictly.
:::

---

## The Four Intelligence Problems

This system addresses four distinct compliance problems, each corresponding to a different concept and condition in Memintel.

| Problem | What it detects | When it matters |
|---|---|---|
| **Taxonomy change impact** | Deprecated tags, changed calculations | Annually when SEC updates taxonomy |
| **Cross-period consistency** | Tags and values inconsistent with prior periods | Any time a policy change or acquisition occurs |
| **Comment letter risk** | Disclosures similar to those that attracted SEC scrutiny | Pre-filing review |
| **Filing pipeline risk** | Deadline risk from missed milestones or dependencies | Throughout the filing preparation cycle |

---

## Who Defines What

| Layer | Who | Nature |
|---|---|---|
| Data sources | Data Engineer | Raw — filings, taxonomy files, EDGAR corpus |
| Signal extraction | Data Engineer + parsers/LLMs | Semi-structured signals |
| Primitives | Admin (config file) | Clean, typed |
| Guardrails | Admin (config file) | Policy layer — strategies, priors, bias rules |
| Intent | User (compliance officer) | Plain language |
| Concepts + Conditions | Memintel (compiler) | Auto-resolved from intent + guardrails |
| Actions | Admin | Configured alert endpoints |

---

## Designing Good Primitives for This Domain

The XBRL compliance domain has some specific primitive design considerations worth highlighting.

### Time-indexed primitives are essential

Almost every meaningful signal in this domain is about change over time — a tag that was valid last year is deprecated this year, a value that was consistent across three prior periods has suddenly diverged. Register time-series variants of key primitives.

```yaml
# Point-in-time — current state
- id: filing.deprecated_tag_count
  type: int

# Time-series — enables change and trend detection
- id: filing.deprecated_tag_count_8q
  type: time_series<int>
  description: Deprecated tag count across last 8 quarters
```

### Separate existence from value

A `prior_period_value` that is null because there was no prior period is different from one that is null because the data pipeline failed. Be explicit.

```yaml
- id: tag.prior_period_value
  type: float?
  description: Value in comparative period — null if first filing or no comparative data

- id: tag.prior_period_exists
  type: boolean
  description: True if a prior period value exists for comparison
```

### External state is a first-class primitive type

The regulatory environment is external memory. Register its key signals as explicitly as internal ones.

```yaml
- id: taxonomy.deprecated_in_current_version
  type: boolean
  source: sec_taxonomy_pipeline
  entity: xbrl_tag_id

- id: sec_comments.topic_frequency_12m
  type: int
  source: edgar_comment_pipeline
  entity: disclosure_topic_id
```

---

## Step 1 — Define Primitives (Config File)

*Who does this: **Admin**, working with the Data Engineer.*

```yaml
# memintel_primitives_xbrl.yaml

primitives:

  # Taxonomy change signals
  - id: tag.deprecated_flag
    type: boolean
    source: sec_taxonomy_pipeline
    entity: xbrl_tag_id
    description: True if this tag is deprecated in the current taxonomy version

  - id: tag.replacement_available
    type: boolean
    source: sec_taxonomy_pipeline
    entity: xbrl_tag_id
    description: True if a replacement tag exists for this deprecated tag

  - id: tag.calculation_changed_flag
    type: boolean
    source: sec_taxonomy_pipeline
    entity: xbrl_tag_id
    description: True if calculation relationships for this tag changed in current taxonomy

  - id: tag.usage_count_prior_filings
    type: int
    source: filing_history_pipeline
    entity: xbrl_tag_id
    description: Number of prior filings in which this tag was used

  # Cross-period consistency signals
  - id: tag.prior_period_value
    type: float?
    source: filing_history_pipeline
    entity: xbrl_tag_id
    description: Value reported for this tag in the comparative period

  - id: tag.current_draft_value
    type: float?
    source: draft_filing_pipeline
    entity: xbrl_tag_id
    description: Value in the current draft filing

  - id: tag.value_change_pct
    type: float?
    source: filing_history_pipeline
    entity: xbrl_tag_id
    description: Percentage change from prior period value

  - id: tag.peer_median_change_pct
    type: float?
    source: edgar_peer_pipeline
    entity: xbrl_tag_id
    description: Median percentage change for this tag across industry peers

  # Comment letter risk signals
  - id: disclosure.comment_frequency_12m
    type: int
    source: edgar_comment_pipeline
    entity: disclosure_topic_id
    description: Times this topic appeared in SEC comment letters in last 12 months

  - id: disclosure.peer_similarity_score
    type: float
    source: edgar_comment_pipeline
    entity: disclosure_topic_id
    description: Similarity of company disclosure to disclosures that attracted comments, 0-1

  - id: disclosure.sec_focus_trend
    type: time_series<int>
    source: edgar_comment_pipeline
    entity: disclosure_topic_id
    description: Monthly comment frequency for this topic — last 24 months

  # Filing pipeline signals
  - id: pipeline.days_to_deadline
    type: int
    source: filing_pipeline_system
    entity: filing_id
    description: Calendar days until filing deadline

  - id: pipeline.milestone_completion_rate
    type: float
    source: filing_pipeline_system
    entity: filing_id
    description: Ratio of completed to scheduled milestones at this point in cycle, 0-1

  - id: pipeline.open_audit_issues
    type: int
    source: audit_system
    entity: filing_id
    description: Count of unresolved audit questions

  - id: pipeline.dependency_blockers
    type: int
    source: filing_pipeline_system
    entity: filing_id
    description: Count of upstream dependencies blocking current milestone
```

---

## Step 2 — Write Primitive Resolvers (Application Code)

*Who does this: **Data Engineer**.*

Resolvers are application code that the data engineer writes. They live in your application, not in Memintel. Memintel calls them at evaluation time with an entity ID and a timestamp, and expects a typed value back.

```python
# resolvers/xbrl_resolvers.py — your code, your codebase

from memintel import registry
from datetime import datetime
from db import db
from taxonomy_client import taxonomy_api

@registry.resolver("tag.deprecated_flag")
async def resolve_deprecated_flag(entity_id: str, timestamp: datetime) -> bool:
    # entity_id is an xbrl_tag_id e.g. "us-gaap:Revenues"
    # Check if this tag is deprecated in the taxonomy version
    # effective as of the given timestamp
    taxonomy_version = await taxonomy_api.get_effective_version(timestamp)
    return await taxonomy_api.is_deprecated(entity_id, taxonomy_version)


@registry.resolver("tag.usage_count_prior_filings")
async def resolve_usage_count(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT COUNT(DISTINCT filing_period)
        FROM xbrl_tag_usage
        WHERE tag_id = $1
          AND filing_date < $2
    """, entity_id, timestamp)
    return result.scalar() or 0


@registry.resolver("tag.prior_period_value")
async def resolve_prior_period_value(entity_id: str, timestamp: datetime) -> float | None:
    result = await db.execute("""
        SELECT reported_value
        FROM xbrl_filing_values
        WHERE tag_id = $1
          AND filing_date = (
            SELECT MAX(filing_date)
            FROM xbrl_filing_values
            WHERE tag_id = $1
              AND filing_date < $2
          )
    """, entity_id, timestamp)
    return result.scalar()  # returns None if no prior period exists


@registry.resolver("disclosure.comment_frequency_12m")
async def resolve_comment_frequency(entity_id: str, timestamp: datetime) -> int:
    # entity_id is a disclosure_topic_id e.g. "revenue_recognition"
    result = await db.execute("""
        SELECT COUNT(*)
        FROM sec_comment_letters
        WHERE topic_id = $1
          AND letter_date BETWEEN $2 - INTERVAL '12 months' AND $2
    """, entity_id, timestamp)
    return result.scalar() or 0


@registry.resolver("pipeline.days_to_deadline")
async def resolve_days_to_deadline(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT EXTRACT(DAY FROM deadline_date - $2)::int
        FROM filing_deadlines
        WHERE filing_id = $1
    """, entity_id, timestamp)
    return max(result.scalar() or 0, 0)
```

### Point-in-time correctness is especially important here

The XBRL domain has a particularly subtle point-in-time requirement: **what the taxonomy said at a specific date is different from what it says today.** A tag that is deprecated now may have been perfectly valid when it was used in a 2022 filing. Resolvers must be capable of answering "as-of" questions — not just current state.

```python
# Wrong — returns current taxonomy state, not as-of state
@registry.resolver("tag.deprecated_flag")
async def resolve_deprecated_flag(entity_id: str, timestamp: datetime) -> bool:
    return await taxonomy_api.is_deprecated_today(entity_id)  # ignores timestamp!

# Right — returns taxonomy state as of the given timestamp
@registry.resolver("tag.deprecated_flag")
async def resolve_deprecated_flag(entity_id: str, timestamp: datetime) -> bool:
    taxonomy_version = await taxonomy_api.get_effective_version(timestamp)
    return await taxonomy_api.is_deprecated(entity_id, taxonomy_version)
```

This requires your taxonomy pipeline to maintain a complete history of every taxonomy version and its effective dates — not just the current version. The EDGAR taxonomy archive is publicly available and provides this history.

### Data infrastructure for this domain

| Source | Pattern | Notes |
|---|---|---|
| SEC taxonomy versions | Versioned archive with effective dates | Use EDGAR taxonomy archive |
| Company filing history | Event log — one row per tag per filing period | Never overwrite historical values |
| SEC comment letters | Append-only corpus indexed by date and topic | EDGAR public corpus |
| Draft filing values | Snapshot table — refreshed as draft evolves | Include draft timestamp |
| Pipeline milestones | Event log — one row per milestone completion | Track planned vs actual dates |

---

## Step 3 — Configure Guardrails (Config File)

*Who does this: **Admin**.*

```yaml
# memintel_guardrails_xbrl.yaml

type_strategy_map:
  int:                  [threshold, percentile, change]
  float:                [threshold, percentile, z_score, change]
  float?:               [threshold]       # nullable — only threshold applicable
  time_series<int>:     [z_score, change, percentile]
  boolean:              [equals]

parameter_priors:
  tag.usage_count_prior_filings:
    low_severity:     { threshold: 1  }   # used at least once
    medium_severity:  { threshold: 4  }   # used in at least 4 filings
    high_severity:    { threshold: 8  }   # deeply embedded — 8+ filings

  tag.value_change_pct:
    low_severity:     { threshold: 0.10 }  # 10% change
    medium_severity:  { threshold: 0.25 }  # 25% change
    high_severity:    { threshold: 0.50 }  # 50% change

  disclosure.comment_frequency_12m:
    low_severity:     { threshold: 3   }
    medium_severity:  { threshold: 8   }
    high_severity:    { threshold: 15  }

  disclosure.peer_similarity_score:
    low_severity:     { threshold: 0.5 }
    medium_severity:  { threshold: 0.7 }
    high_severity:    { threshold: 0.85 }

  pipeline.milestone_completion_rate:
    low_severity:     { threshold: 0.85 }  # slightly behind
    medium_severity:  { threshold: 0.70 }  # materially behind
    high_severity:    { threshold: 0.55 }  # serious deadline risk

bias_rules:
  proactive:          low_severity
  early warning:      low_severity
  significant:        medium_severity
  critical:           high_severity
  urgent:             high_severity

global_default_strategy:   threshold
global_preferred_strategy: threshold
```

---

## Step 4 — Memintel Compiles Concepts and Conditions

*Who does this: **Memintel** (automatically).*

Four distinct compilation paths correspond to the four intelligence problems.

**Taxonomy change impact:**
```
User intent: "Alert me when a tag we use heavily is deprecated"
                     ↓
Guardrails maps primitives:
  tag.deprecated_flag        (boolean → equals)
  tag.usage_count_prior_filings (int → threshold)
                     ↓
Compiler produces:
  Concept:   deprecated AND heavily_used
             (deprecated_flag = true AND usage_count >= 8)
  Condition: fires when concept = true
  Action:    webhook → compliance team
```

**Comment letter risk:**
```
User intent: "Alert me when a disclosure is at significant risk of attracting SEC scrutiny"
                     ↓
Guardrails maps primitives:
  disclosure.comment_frequency_12m   (int → threshold, medium_severity = 8)
  disclosure.peer_similarity_score   (float → threshold, medium_severity = 0.70)
                     ↓
Compiler produces:
  Concept:   comment_risk_score = weighted_sum(frequency_signal, similarity_signal)
  Condition: comment_risk_score > 0.70
  Action:    webhook → pre-filing review queue
```

---

## Step 5 — User Creates a Task

*Who does this: **User** (Chief Compliance Officer, Controller, or financial reporting manager), via a bot or internal tool.*

The compliance officer never calls the Memintel API directly. They interact with an application interface — a Slack bot, a compliance dashboard, or an internal reporting tool — that calls Memintel on their behalf.

### The dryRun → confirm → activate flow

```typescript
// Compliance officer types in dashboard:
// "Alert me when a tag we've used in more than 8 filings is deprecated"

const preview = await client.tasks.create({
    intent: "Alert me when a tag we have used heavily is deprecated in the new taxonomy",
    entityScope: "all_active_xbrl_tags",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/xbrl-taxonomy-alert"
    },
    dryRun: true
});

// Bot translates compiled condition back to plain English
console.log(preview.condition.strategy);
// { type: "composite",
//   operands: [
//     { primitive: "tag.deprecated_flag",         equals: true },
//     { primitive: "tag.usage_count_prior_filings", above: 8   }
//   ]
// }

// Bot shows compliance officer:
// "I'll alert you when a tag used in more than 8 prior filings is marked
//  deprecated in the current taxonomy version. This would currently flag
//  3 tags in your active filing. Shall I activate this?"

// Officer confirms → activate
const task = await client.tasks.create({
    intent: "Alert me when a tag we have used heavily is deprecated in the new taxonomy",
    entityScope: "all_active_xbrl_tags",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/xbrl-taxonomy-alert"
    }
});
```

---

## Step 6 — Task Types

### Type 1 — One-off evaluation

Immediate answer. No ongoing monitoring.

```
User: "Which of our current tags are deprecated in the 2024 taxonomy?"
Bot:  Evaluates all active tags against current taxonomy
      Returns: "7 tags in your current draft are deprecated.
               3 have direct replacements. 4 require manual review."
```

```typescript
const activeTagIds = await filingSystem.getActiveTags({ draftId: "10k_q4_2024" });

const results = await Promise.all(
    activeTagIds.map(tagId =>
        client.evaluateFull({
            concept_id: "tag.taxonomy_risk",
            concept_version: "1.0",
            condition_id: "tag.deprecated_and_heavily_used",
            condition_version: "1.0",
            entity: tagId,
            timestamp: new Date().toISOString(),
            explain: true
        })
    )
);

const flagged = results.filter(r => r.decision.value === true);
console.log(`${flagged.length} tags flagged for taxonomy review`);
```

### Type 2 — Ongoing monitoring (primary use case)

Persistent task that runs continuously. Fires alerts as conditions are met.

```
User: "Continuously monitor our filing for comment letter risk"
Bot:  Creates task → monitors all disclosures in active draft
      Alerts compliance team as new SEC comment letters are published
      that increase risk for our specific disclosures
```

This is the most powerful use case. When the SEC publishes a new comment letter to a peer company on revenue recognition, the system immediately evaluates the company's own revenue recognition disclosure and fires an alert if the similarity score exceeds the threshold — before the company files.

### Type 3 — Batch evaluation

Full pipeline risk assessment across all active filing items at once.

```typescript
// Generate complete pre-filing risk report
const allDisclosureTopics = await filingSystem.getDisclosureTopics({ draftId: "10k_q4_2024" });

const commentRiskResults = await Promise.all(
    allDisclosureTopics.map(topic =>
        client.evaluateFull({
            concept_id: "disclosure.comment_risk_score",
            concept_version: "1.0",
            condition_id: "disclosure.high_comment_risk",
            condition_version: "1.0",
            entity: topic.id,
            timestamp: new Date().toISOString(),
            explain: true
        })
    )
);

const highRisk = commentRiskResults
    .filter(r => r.decision.value === true)
    .sort((a, b) => b.result.value - a.result.value);

console.log(`Pre-filing report: ${highRisk.length} disclosures flagged for enhanced review`);
highRisk.forEach(r => {
    console.log(`${r.decision.entity}: ${(r.result.value * 100).toFixed(0)}% comment risk`);
    console.log(`  Comment frequency 12m: ${r.result.explanation.contributions.comment_frequency}`);
    console.log(`  Peer similarity: ${r.result.explanation.contributions.peer_similarity}`);
});
```

### Type 4 — Historical replay

What was the taxonomy risk state of a specific filing at a specific date? Fully reproducible.

```typescript
// Reconstruct filing state as of a specific date
// Useful for audit trail, regulatory inquiry responses
const result = await client.evaluateFull({
    concept_id: "tag.taxonomy_risk",
    concept_version: "1.0",
    condition_id: "tag.deprecated_and_heavily_used",
    condition_version: "1.0",
    entity: "us-gaap:Revenues",
    timestamp: "2023-12-31T00:00:00Z",  // as of year-end
    explain: true
});
// Returns the exact evaluation as it would have been on Dec 31, 2023
// Using the taxonomy version effective on that date
```

---

## Step 7 — The System Response Loop

*What happens after an ongoing task is activated.*

```
Task activated
      ↓
Trigger fires — either on schedule (hourly/daily)
OR on event (new taxonomy version published,
             new SEC comment letter released,
             pipeline milestone status changes)
      ↓
For each entity in scope (tag, disclosure, filing):
  Memintel calls resolvers with (entity_id, timestamp)
  Resolvers fetch point-in-time values:
    → taxonomy_api: is this tag deprecated as of now?
    → filing_history_db: how many times was this tag used?
    → edgar_comment_corpus: comment frequency last 12 months?
    → pipeline_system: days to deadline?
  Concept computation runs on primitive values
  Condition evaluates: does risk score cross threshold?
      ↓
  If YES → Action triggered
    Webhook delivers structured alert
    Decision logged with full audit trail
    Compliance team receives actionable notification
      ↓
  If NO  → No action. Decision logged.
      ↓
Next trigger
```

### What the compliance team receives

When a condition fires, the alert should be actionable — not just a flag:

```
🚨 Taxonomy Risk Alert — 3 tags deprecated in 2024 taxonomy

Tags requiring action before filing:

  us-gaap:OtherComprehensiveIncomeLossNetOfTaxPortionAttributableToParent
  Used in 11 prior filings  |  Replacement: available
  Impact: affects 4 calculation relationships

  us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic
  Used in 8 prior filings   |  Replacement: available
  Impact: affects 2 calculation relationships

  us-gaap:CashAndDueFromBanks
  Used in 9 prior filings   |  Replacement: requires manual review
  Impact: custom extension may be affected

→ View full impact assessment   → Assign to preparer   → Mark as reviewed
```

### The dual memory advantage

Unlike point-in-time validation tools, this system evaluates every alert in the context of **both** the company's internal history and the current external regulatory environment. A deprecated tag is only a high-priority alert if the company has used it repeatedly — if it appears in 11 prior filings, that is a deeply embedded dependency that requires careful migration. If it appears in only 1 filing, it may be trivially replaceable. The Concept layer computes this distinction automatically.

### What gets logged

Every evaluation is logged with full audit trail — critical in a regulatory context where every compliance decision must be defensible:

```json
{
  "decision_id":       "dec_xbrl_7k3m",
  "task_id":           "tsk_taxonomy_watch",
  "entity":            "us-gaap:Revenues",
  "timestamp":         "2024-01-15T08:00:00Z",
  "concept_id":        "tag.taxonomy_risk",
  "concept_version":   "1.0",
  "condition_id":      "tag.deprecated_and_heavily_used",
  "condition_version": "1.0",
  "result_value":      0.91,
  "decision_value":    true,
  "action_triggered":  true,
  "contributions": {
    "deprecated_flag":        1.0,
    "usage_count_prior":      0.82,
    "calculation_change":     0.73
  },
  "taxonomy_version_evaluated": "2024-us-gaap-v1.0",
  "effective_as_of":            "2024-01-15T00:00:00Z"
}
```

---

## Step 8 — Task Management

### User controls

*Compliance officers can view and manage their own tasks.*

```typescript
// View all active compliance monitoring tasks
const tasks = await client.tasks.list({ owner: "current_user" });

tasks.forEach(task => {
    console.log(`${task.id}: ${task.intent}`);
    console.log(`  Status:      ${task.status}`);
    console.log(`  Last fired:  ${task.last_fired_at}`);
    console.log(`  Fired ${task.fire_count_30d} times in last 30 days`);
    console.log(`  Next run:    ${task.next_run_at}`);
});

// Pause during filing blackout period
await client.tasks.pause("tsk_taxonomy_watch", {
    reason: "Filing blackout — resuming after 10-K submission"
});

// Resume after filing
await client.tasks.resume("tsk_taxonomy_watch");

// Rebind to new condition version after calibration
await client.tasks.update("tsk_taxonomy_watch", {
    conditionVersion: "1.1"
});
```

### Admin visibility

*Admins can see and manage all tasks across all users and filing cycles.*

```typescript
// View all tasks across all users
const allTasks = await adminClient.tasks.list();

// Version distribution — how many tasks on each condition version
const metrics = await adminClient.tasks.metrics({
    conditionId: "tag.deprecated_and_heavily_used"
});

console.log(metrics.active_task_count);       // 8
console.log(metrics.firing_rate_per_day);     // 1.4
console.log(metrics.version_distribution);
// { "1.0": 5, "1.1": 3 }

// Suspend tasks if a data source is unreliable
// e.g. SEC comment letter pipeline outage
await adminClient.tasks.suspend("tsk_comment_risk", {
    reason: "EDGAR comment letter pipeline outage — data unreliable"
});

// Deprecate a condition version
await adminClient.conditions.deprecateVersion({
    conditionId: "disclosure.high_comment_risk",
    version: "1.0",
    message: "Recalibrated peer similarity weights — please rebind to v1.1"
});
```

---

## Step 9 — The Feedback and Calibration Cycle

*Continuous loop — not a one-time step.*

The XBRL compliance domain has a specific calibration consideration: **external ground truth is publicly available.** When the SEC issues a comment letter to your company, that is definitive feedback on which disclosures attracted scrutiny. When a deprecated tag causes a validation failure at submission, that is definitive feedback on taxonomy risk detection. This external ground truth makes calibration more objective than in many other domains.

```typescript
// Step 1 — Compliance team submits feedback
// A comment letter was received on revenue recognition disclosure
// that the system had not flagged — false negative
await client.feedback.decision({
    conditionId: "disclosure.high_comment_risk",
    conditionVersion: "1.0",
    entity: "revenue_recognition",
    timestamp: "2024-03-01T00:00:00Z",
    feedback: "false_negative",
    note: "SEC issued comment letter on revenue recognition timing — was not flagged"
});

// Step 2 — Admin requests calibration
const cal = await adminClient.conditions.calibrate({
    conditionId: "disclosure.high_comment_risk",
    conditionVersion: "1.0",
});

// Step 3 — Review impact
console.log(cal.recommended_params);
// { peer_similarity_threshold: 0.65 }  — lowered from 0.70
// catches more potential comments at the cost of more reviews
console.log(cal.impact.delta_alerts);   // +2 per filing cycle
console.log(cal.feedback_summary);
// { false_positives: 3, false_negatives: 2, correct: 18 }

// Step 4 — Admin approves — new immutable version created
const applied = await adminClient.conditions.applyCalibration({
    calibrationToken: cal.calibration_token,
});

// Step 5 — Tasks explicitly rebound by users
// Nothing changes automatically
await client.tasks.update("tsk_comment_risk", {
    conditionVersion: applied.new_version,
});
```

---

## Full Lifecycle Diagram

```
SETUP (one time)
──────────────────────────────────────────────────────────
Data Engineer:   Configure taxonomy version archive pipeline
Data Engineer:   Configure EDGAR comment letter corpus pipeline
Data Engineer:   Write resolver functions (application code)
Admin:           Define primitives → memintel_primitives_xbrl.yaml
Admin:           Define guardrails  → memintel_guardrails_xbrl.yaml
System:          Load config at startup


TASK CREATION
──────────────────────────────────────────────────────────
User (CCO/Controller): Express intent via compliance dashboard
Bot:              POST /tasks/create?dryRun=true
Memintel:         Compile intent → concept + condition
Bot:              Show plain-English preview (tags affected, estimated firings)
User:             Confirm
Bot:              POST /tasks/create → task activated


ONGOING EVALUATION LOOP
──────────────────────────────────────────────────────────
Trigger: Schedule (hourly/daily) OR event (new taxonomy, new comment letter)
  → Resolvers fetch point-in-time values:
      Taxonomy API (effective version as of timestamp)
      Filing history DB (prior usage counts, prior values)
      EDGAR comment corpus (recent comment frequency)
      Pipeline system (milestone completion, days to deadline)
  → Concept computation runs
  → Condition evaluates
  → If fired: alert delivered to compliance team, decision logged
  → If not:   decision logged, no action


TASK MANAGEMENT
──────────────────────────────────────────────────────────
User:   View / pause / resume own tasks
        (e.g. pause during filing blackout period)
Admin:  View all tasks across all users and filing cycles
Admin:  Suspend tasks if data source unreliable
Admin:  Deprecate condition versions after recalibration


CALIBRATION CYCLE
──────────────────────────────────────────────────────────
Ground truth arrives: SEC comment letter, validation failure, audit finding
User:   Submit feedback (false positive / false negative / correct)
Admin:  Request calibration recommendation
Admin:  Review impact — more alerts vs fewer alerts tradeoff
Admin:  Approve → new condition version created (immutable)
User:   Rebind task to new version (explicit, never automatic)
        ↓
Back to ONGOING EVALUATION LOOP with recalibrated thresholds
```

---

## Role Summary

| Step | Who | What |
|---|---|---|
| Configure data pipelines | **Data Engineer** | Taxonomy archive, EDGAR corpus, filing history |
| Write resolvers | **Data Engineer** | Application code — point-in-time data fetching |
| Define primitives | **Admin** | `memintel_primitives_xbrl.yaml` — loaded at startup |
| Configure guardrails | **Admin** | `memintel_guardrails_xbrl.yaml` — loaded at startup |
| Compile intent | **Memintel** | Auto-resolves concepts + conditions |
| Create task | **User** | Via compliance dashboard — never directly via API |
| Ongoing evaluation | **System** | Event-triggered and scheduled, fully logged |
| View own tasks | **User** | Status, last fired, filing cycle context |
| View all tasks | **Admin** | Cross-user, version distribution, suspension controls |
| Submit feedback | **User** | SEC comment letters and validation failures as ground truth |
| Approve calibration | **Admin** | Reviews impact, creates new immutable version |
| Rebind task | **User** | Explicit — nothing changes automatically |

---

## Why This Problem Specifically Needs Memintel

This domain is worth pausing on — it is one of the strongest structural fits for Memintel's architecture.

**The problem is explicitly about time-indexed state.** A tag valid in 2022 may be deprecated in 2024. A disclosure consistent with SEC expectations in 2021 may attract scrutiny in 2023 as SEC focus shifts. SQL can tell you the current state. Memintel tells you whether the current state is correct given how both the company's filing history and the regulatory environment have evolved.

**The auditability requirement is legally mandated.** Every compliance decision must be defensible to regulators and auditors. Memintel's deterministic, immutable audit trail — logging every evaluation with the exact concept version, condition version, primitive values, and timestamp — produces this documentation automatically as a byproduct of operation.

**The dual memory structure is the core of the problem.** Every meaningful signal requires evaluating internal state (what the company has filed, what tags they use, what disclosures they make) against external state (what the current taxonomy requires, what the SEC is scrutinising, what peer companies are being asked about). Neither memory alone is sufficient. Memintel's architecture is explicitly designed for this kind of continuous cross-memory reconciliation.

**The proactive requirement is what existing tools cannot satisfy.** Current XBRL validation software is reactive — it tells you what is wrong after you submit. Memintel continuously evaluates the evolving internal and external state and tells you what will become wrong before you file. That shift from reactive to proactive is the core value proposition, and it requires exactly the kind of state tracking and continuous evaluation that Memintel provides.

---

## Next Steps

- [Core Concepts](/docs/intro/core-concepts) — understand the ψ → φ → α model in depth
- [Guardrails System](/docs/intro/guardrails) — how admins configure the policy layer
- [Deal Intelligence Tutorial](/docs/tutorials/deal-intelligence) — the same architecture applied to sales
- [API Reference](/docs/api-reference/overview) — full endpoint documentation
