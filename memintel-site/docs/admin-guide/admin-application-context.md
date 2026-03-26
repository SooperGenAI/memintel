---
id: admin-application-context
title: Configuring Application Context
sidebar_label: Application Context
---

# Configuring Application Context

Application context is the first thing an admin defines — before primitives, before guardrails, before any user creates a task. It gives the LLM the domain knowledge it needs to compile accurate, domain-aware definitions from user intent.

---

## What It Does

Without context, when a user says *"alert me when a deal is at high risk of stalling"*, the compiler has no way to know:
- What signals are relevant to "stalling" in this domain
- What "high risk" means relative to the industry and use case
- What domain-specific terms like "engaged" or "high value" mean
- Whether missing a stalled deal is more costly than a false alarm

With context, the compiler has answers to all of these questions before it resolves a single intent. The first compiled condition is materially more accurate, and the system requires fewer calibration cycles to reach production quality.

:::tip
Define context before onboarding any users. Tasks created without context receive a `context_warning` in their response and will produce generic definitions.
:::

---

## The Context Schema

Context is defined via `POST /context`. The full schema:

```json
{
  "domain": {
    "description": "string — required. What the application does.",
    "entities": [
      { "name": "string", "description": "string" }
    ],
    "decisions": ["string"]
  },
  "behavioural": {
    "data_cadence": "batch | streaming | mixed",
    "meaningful_windows": { "min": "30d", "max": "90d" },
    "regulatory": ["string"]
  },
  "semantic_hints": [
    { "term": "string", "definition": "string" }
  ],
  "calibration_bias": {
    "false_negative_cost": "high | medium | low",
    "false_positive_cost": "high | medium | low"
  }
}
```

### domain.description

The most important field. Write this as a precise, information-dense paragraph — not a marketing tagline. The compiler reads this to understand what the application does and calibrates its interpretation of all user intent against it.

```json
// Too vague — the compiler learns almost nothing
"description": "A platform for monitoring business metrics."

// Good — tells the compiler exactly what domain it is working in
"description": "B2B SaaS churn detection for mid-market software companies. We monitor user engagement signals and account-level health metrics to identify accounts at risk of not renewing their annual subscription."
```

### domain.entities

Declare the things being monitored — the primary and secondary objects in your domain. The compiler uses these to understand what `entity_id` values represent when users create tasks.

```json
"entities": [
  {
    "name": "account",
    "description": "A company-level subscription — the billing and contract unit. The primary entity for churn monitoring."
  },
  {
    "name": "user",
    "description": "An individual platform user within an account. Secondary entity — user behaviour signals roll up to account health."
  }
]
```

### domain.decisions

The decision types your application makes. These help the compiler understand what categories of conditions are meaningful and how to group similar intent expressions.

```json
"decisions": ["churn_risk", "expansion_opportunity", "support_escalation", "payment_risk"]
```

### behavioural.data_cadence

How data arrives. This directly influences which evaluation strategies the compiler prefers:

| Cadence | Effect on compilation |
|---|---|
| `batch` | Compiler prefers longer windows, tolerates latency in signal freshness |
| `streaming` | Compiler prefers shorter windows and event-driven strategies |
| `mixed` | Compiler selects based on individual primitive types |

```json
"behavioural": {
  "data_cadence": "batch"
}
```

### behavioural.meaningful_windows

The minimum and maximum time windows that are operationally meaningful for this domain. The compiler uses these to bound window parameters during compilation, and the calibration engine clamps window recommendations to this range.

```json
// B2B SaaS — monthly billing cycle means <30d windows are not actionable
"meaningful_windows": { "min": "30d", "max": "90d" }

// Real-time fraud detection — hours matter, not months
"meaningful_windows": { "min": "1h", "max": "24h" }

// Clinical trial safety — weekly data cuts, annual trial duration
"meaningful_windows": { "min": "7d", "max": "180d" }
```

### behavioural.regulatory

The regulatory frameworks that apply to this application. This signals to the compiler that certain terms have precise legal definitions that must not be interpreted loosely.

```json
// Financial services
"regulatory": ["BSA", "FinCEN", "FATF"]

// Healthcare
"regulatory": ["HIPAA", "CMS", "ICH-E6"]

// SaaS / general
"regulatory": ["GDPR", "SOC2"]
```

When regulatory frameworks are declared, the compiler treats severity language like "significant", "material", and "elevated" more conservatively — because in a regulated context, those words carry legal weight.

### semantic_hints

The most powerful and most underused field. Semantic hints let the admin define precisely what domain-specific terms mean — resolving ambiguity that would otherwise cause the compiler to produce generic definitions.

Each hint is a term-definition pair. The definition should be unambiguous and operational — something a new analyst could act on without further clarification.

```json
"semantic_hints": [
  {
    "term": "active user",
    "definition": "logged in AND performed at least one core action (create, edit, or share) in the last 14 days. Login alone does not count as active."
  },
  {
    "term": "high value account",
    "definition": "ARR above $50,000, regardless of seat count or plan type."
  },
  {
    "term": "churned",
    "definition": "subscription cancelled or not renewed within 30 days of expiry date."
  },
  {
    "term": "at risk",
    "definition": "showing two or more negative signals in the last 30 days with no positive signal in the last 7 days."
  }
]
```

:::tip
Add a semantic hint whenever a business term appears in user intent expressions that has a specific, non-obvious meaning in your domain. "Active user" means different things in different products. The hint resolves this permanently for the lifetime of this context version.
:::

### calibration_bias

Declares the cost asymmetry between false negatives (missing a real signal) and false positives (firing on a non-event). The compiler uses this to bias threshold resolution, and the calibration engine uses it to adjust recommendations away from the raw statistical optimum.

```json
// Missing a churning account is worse than an unnecessary outreach
"calibration_bias": {
  "false_negative_cost": "high",
  "false_positive_cost": "medium"
}

// A false block of a legitimate payment is worse than a missed fraud
"calibration_bias": {
  "false_negative_cost": "medium",
  "false_positive_cost": "high"
}

// Patient safety — never miss a safety signal
"calibration_bias": {
  "false_negative_cost": "high",
  "false_positive_cost": "low"
}
```

`bias_direction` is auto-derived and cannot be set manually:
- `false_negative_cost > false_positive_cost` → `recall` bias (lower thresholds)
- `false_positive_cost > false_negative_cost` → `precision` bias (higher thresholds)
- Equal → `balanced`

---

## Complete Examples

### SaaS Churn Detection

```json
{
  "domain": {
    "description": "B2B SaaS churn detection for mid-market software companies. We monitor user engagement and account health signals to identify accounts at risk of not renewing their subscription before the renewal window.",
    "entities": [
      {
        "name": "account",
        "description": "A company-level subscription — the unit of billing and the primary entity for churn monitoring."
      },
      {
        "name": "user",
        "description": "An individual platform user within an account. User-level signals aggregate to account health scores."
      }
    ],
    "decisions": ["churn_risk", "expansion_opportunity", "support_escalation"]
  },
  "behavioural": {
    "data_cadence": "batch",
    "meaningful_windows": { "min": "30d", "max": "90d" },
    "regulatory": ["GDPR", "SOC2"]
  },
  "semantic_hints": [
    {
      "term": "active user",
      "definition": "logged in AND performed at least one core action (create, edit, share, or export) in the last 14 days. Login alone does not qualify."
    },
    {
      "term": "high value account",
      "definition": "ARR above $50,000 regardless of seat count."
    },
    {
      "term": "core feature",
      "definition": "the primary workflow action the account subscribed for — document creation, data export, or API integration depending on plan type."
    }
  ],
  "calibration_bias": {
    "false_negative_cost": "high",
    "false_positive_cost": "medium"
  }
}
```

### AML Transaction Monitoring

```json
{
  "domain": {
    "description": "Anti-money laundering transaction monitoring for a mid-size commercial bank. We monitor customer transaction patterns in real time to detect structuring, layering, and other financial crime indicators in compliance with BSA and FinCEN requirements.",
    "entities": [
      {
        "name": "customer",
        "description": "A registered bank customer with an established transaction history and a risk classification on file."
      },
      {
        "name": "transaction",
        "description": "An individual payment, wire transfer, cash deposit, or withdrawal event. The primary unit of real-time evaluation."
      }
    ],
    "decisions": ["sar_required", "enhanced_due_diligence", "transaction_hold", "account_review"]
  },
  "behavioural": {
    "data_cadence": "streaming",
    "meaningful_windows": { "min": "1d", "max": "90d" },
    "regulatory": ["BSA", "FinCEN", "FATF"]
  },
  "semantic_hints": [
    {
      "term": "unusual",
      "definition": "materially different from the customer's established 90-day rolling baseline — more than 3 standard deviations from their mean transaction value or velocity."
    },
    {
      "term": "structuring",
      "definition": "a pattern of multiple transactions designed to remain below the $10,000 CTR reporting threshold within any 24-hour window."
    },
    {
      "term": "high risk jurisdiction",
      "definition": "a country on the current FATF grey or black list, or subject to OFAC sanctions programme."
    }
  ],
  "calibration_bias": {
    "false_negative_cost": "high",
    "false_positive_cost": "medium"
  }
}
```

### Clinical Trial Safety Monitoring

```json
{
  "domain": {
    "description": "Clinical trial safety monitoring and pharmacovigilance for a Phase 3 oncology programme. We continuously evaluate adverse event patterns against both internal trial data and external FDA safety signals to protect patient safety and ensure ICH E2A compliance.",
    "entities": [
      {
        "name": "patient",
        "description": "An enrolled trial participant with an active safety monitoring obligation."
      },
      {
        "name": "trial",
        "description": "The clinical study with a defined safety monitoring plan and pre-specified stopping rules."
      }
    ],
    "decisions": ["sae_assessment_required", "dsmb_notification", "stopping_rule_proximity", "susar_reporting"]
  },
  "behavioural": {
    "data_cadence": "batch",
    "meaningful_windows": { "min": "7d", "max": "180d" },
    "regulatory": ["FDA-21CFR", "ICH-E6", "ICH-E2A", "GCP"]
  },
  "semantic_hints": [
    {
      "term": "serious",
      "definition": "results in death, is life-threatening, requires inpatient hospitalisation, results in persistent disability, or is a congenital anomaly. Per ICH E2A definition."
    },
    {
      "term": "unexpected",
      "definition": "the nature, severity, or frequency is not consistent with the current Investigator Brochure or reference safety information."
    },
    {
      "term": "related",
      "definition": "there is a reasonable possibility that the investigational medicinal product caused the adverse event — based on temporal relationship, known pharmacology, and absence of alternative explanation."
    }
  ],
  "calibration_bias": {
    "false_negative_cost": "high",
    "false_positive_cost": "low"
  }
}
```

---

## Deploying a Context Update

Context is versioned. Every `POST /context` creates a new immutable version (`v1`, `v2`, `v3`...) and deactivates the previous one.

```bash
# Deploy updated context
curl -X POST https://your-domain/context \
  -H "Content-Type: application/json" \
  -d @updated-context.json

# Check which tasks are now on an older version
curl https://your-domain/context/impact
# {
#   "current_version": "v2",
#   "tasks_on_current_version": 14,
#   "tasks_on_older_versions": [{ "version": "v1", "task_count": 8 }],
#   "total_stale_tasks": 8
# }
```

Tasks on older versions continue running correctly — they are pinned to their compiled version and fully reproducible. But they do not benefit from the updated context. Trigger recompilation for affected tasks to incorporate the new domain understanding.

```bash
# Recompile tasks on older context versions
await adminClient.tasks.recompile(
  task_ids: ["tsk_001", "tsk_002", ...],
  reason: "Context updated — new semantic hints for high value account"
)
```

---

## Common Mistakes

**Writing a vague description.** The compiler cannot derive useful context from "a business intelligence platform." Be specific about what the application does, what domain it operates in, and what decisions it makes.

**Omitting semantic hints for ambiguous terms.** Any term that appears in user intent and has a non-obvious domain-specific meaning should have a hint. "Active", "high risk", "stalled", "at risk", "material" — these all mean different things in different domains.

**Setting the wrong calibration bias.** The bias directly affects the thresholds the compiler selects. If your application takes automated action on every alert (blocking a transaction, suspending an account), `false_positive_cost` should be `high`. If alerts go to a human reviewer who can apply judgement, `false_positive_cost` can be `medium` or `low`.

**Forgetting to check context impact after an update.** After every context version bump, run `GET /context/impact` to see how many tasks are on older versions. Tasks compiled under v1 do not automatically benefit from v2.

---

## Next Step

[Configure Primitives →](/docs/admin-guide/admin-primitives)
