---
id: admin-application-context
title: Step 1 — Application Context
sidebar_label: Step 1 — Context
---

# Step 1 — Application Context

Application context is a description of your application that you provide to Memintel before anything else. Think of it as the briefing you give a new analyst before they start work — background on what the business does, who the customers are, what specific terms mean, and what matters most.

Without this briefing, when one of your team members says "alert me when a high-value account shows churn risk", the system has no idea what "high-value" means in your context. Is that $10k ARR? $500k ARR? With context defined, it knows exactly.

:::tip Why this is the most important step
Every monitoring task your team creates is compiled using this context as background knowledge. Tasks created without context produce generic definitions that need many more calibration cycles to become accurate. Define context first — before any users create any tasks.
:::

---

## What You Are Defining

Application context has four parts. You only need to fill in what is relevant to your domain.

| Part | What it is | Required? |
|---|---|---|
| **Domain description** | A plain English paragraph describing what your application does | Yes |
| **Entities** | The things being monitored — customers, deals, providers, patients | Recommended |
| **Behavioural settings** | How data flows and what time windows are meaningful | Recommended |
| **Semantic hints** | Definitions of specific terms that have precise meanings in your domain | Strongly recommended |
| **Calibration bias** | Whether it is worse to miss a real signal or to fire a false alarm | Recommended |

---

## How to Define Context

Unlike the other three sections, application context is **not** defined in `memintel_config.yaml`. It is submitted via an API call — a request you send to the Memintel server. This means you can update it at any time without restarting the server.

The easiest way to do this is via a `curl` command from a terminal, or using a tool like [Postman](https://www.postman.com) or [Insomnia](https://insomnia.rest) if you prefer a graphical interface.

```bash
curl -X POST https://your-memintel-domain/context \
  -H "Content-Type: application/json" \
  -d @context.json
```

Where `context.json` is a file containing your context definition. The sections below explain exactly what to put in that file.

---

## Part 1 — Domain Description

Write one paragraph describing what your application does. Be specific. The system uses this to understand the purpose of every monitoring task your team creates.

```json
{
  "domain": {
    "description": "B2B SaaS churn detection for mid-market software companies. We monitor user engagement and account health signals to identify accounts at risk of not renewing their subscription before the renewal window."
  }
}
```

**Good descriptions are:**
- Specific about the industry and use case
- Clear about what is being monitored and why
- Written as if briefing an intelligent analyst who knows nothing about your company

**Avoid:**
- Generic descriptions like "a business intelligence platform"
- Marketing language — be precise, not promotional
- Descriptions longer than 3-4 sentences — keep it focused

---

## Part 2 — Entities

Declare the things your system monitors. An entity is the subject of a monitoring task — the thing a task evaluates.

```json
{
  "domain": {
    "description": "...",
    "entities": [
      {
        "name": "account",
        "description": "A company-level subscription — the billing and contract unit. The primary subject of churn monitoring."
      },
      {
        "name": "user",
        "description": "An individual platform user within an account. User behaviour signals roll up to account-level health."
      }
    ]
  }
}
```

Most domains have one primary entity (the main thing being monitored) and one or two secondary entities. Examples:

| Domain | Primary entity | Secondary entities |
|---|---|---|
| SaaS churn | account | user |
| AML compliance | customer | transaction |
| Credit risk | borrower | loan |
| Clinical trial safety | patient | trial |
| Sales pipeline | deal | contact |
| Healthcare network | provider | claim |

---

## Part 3 — Behavioural Settings

Tell the system how data flows and what time windows make sense for your domain.

### data_cadence

How frequently does your data update?

```json
"behavioural": {
  "data_cadence": "batch"
}
```

| Value | Use when |
|---|---|
| `batch` | Data updates daily, weekly, or on a fixed schedule (most common for B2B SaaS, credit risk, clinical trials) |
| `streaming` | Data arrives in real time, second by second (fraud detection, transaction monitoring, DevOps) |
| `mixed` | Different signals update at different rates |

### meaningful_windows

What is the shortest and longest time window that is operationally meaningful in your domain? This prevents the system from recommending windows that are too short to be actionable or too long to be useful.

```json
"behavioural": {
  "data_cadence": "batch",
  "meaningful_windows": {
    "min": "30d",
    "max": "90d"
  }
}
```

Examples:

| Domain | Min window | Max window | Reason |
|---|---|---|---|
| B2B SaaS churn | `30d` | `90d` | Monthly billing cycles — shorter windows are not actionable |
| Real-time fraud | `1h` | `24h` | Fraud patterns emerge and resolve within hours |
| Credit risk | `30d` | `365d` | Quarterly financial submissions, annual review cycles |
| Clinical trial safety | `7d` | `180d` | Weekly data cuts, trial duration months to years |
| DevOps / SRE | `5m` | `24h` | Incidents develop and resolve within hours |

### regulatory

List any regulatory frameworks that apply to your application. This signals to the system that certain terms have precise legal or regulatory definitions.

```json
"behavioural": {
  "data_cadence": "batch",
  "meaningful_windows": { "min": "30d", "max": "90d" },
  "regulatory": ["GDPR", "SOC2"]
}
```

Common frameworks by domain:

| Domain | Frameworks |
|---|---|
| Financial services / AML | `BSA`, `FinCEN`, `FATF` |
| Healthcare (US) | `HIPAA`, `CMS`, `OIG` |
| Clinical trials | `FDA-21CFR`, `ICH-E6`, `ICH-E2A`, `GCP` |
| SaaS / general | `GDPR`, `SOC2` |
| Capital markets | `MiFID2`, `Basel-III` |

Leave this empty if no specific regulatory framework applies.

---

## Part 4 — Semantic Hints

This is the most powerful part of the context and the most commonly skipped — which is a mistake.

A semantic hint defines precisely what a specific term means in your domain. Without hints, the system uses its general understanding of words. With hints, it uses your exact definition.

```json
"semantic_hints": [
  {
    "term": "active user",
    "definition": "logged in AND performed at least one core action (create, edit, share, or export) in the last 14 days. Login alone does not count as active."
  },
  {
    "term": "high value account",
    "definition": "ARR above $50,000 regardless of seat count or plan type."
  },
  {
    "term": "at risk",
    "definition": "showing two or more negative engagement signals in the last 30 days with no positive signal in the last 7 days."
  }
]
```

**How to decide what to add:**

Go through the monitoring requests your team is likely to make. Any word or phrase that has a specific meaning in your organisation — not the common English meaning — should have a hint.

Common candidates:

| Term | Why it needs a hint |
|---|---|
| "active" | Does login count? Does viewing count? Only creating/editing? |
| "high value" | What is the ARR / revenue threshold? |
| "at risk" | How many signals? Over what time window? |
| "engaged" | What actions count as engagement in your product? |
| "stalled" | How many days of inactivity? Does it apply to both sides? |
| "serious" (clinical) | The ICH E2A legal definition — not the everyday meaning |
| "related" (clinical) | Reasonable possibility of causality — a specific standard |
| "unusual" (AML) | Deviation from the customer's 90-day baseline, not absolute value |

Write definitions the way you would explain the term to a new analyst on their first day — unambiguous, operational, no room for interpretation.

---

## Part 5 — Calibration Bias

This tells the system which type of error is worse in your domain: missing a real signal (false negative) or firing when nothing is wrong (false positive).

```json
"calibration_bias": {
  "false_negative_cost": "high",
  "false_positive_cost": "medium"
}
```

Both values can be `high`, `medium`, or `low`. The system automatically derives the direction:
- If missing signals is worse → system leans toward **sensitivity** (catches more, may have more false alarms)
- If false alarms are worse → system leans toward **precision** (fewer false alarms, may miss more)
- If equal → **balanced**

**Which to use:**

| Scenario | false_negative_cost | false_positive_cost | Reason |
|---|---|---|---|
| SaaS churn detection | `high` | `medium` | Missing a churning account costs more than an extra customer success call |
| Automated fraud blocking | `medium` | `high` | Blocking a legitimate customer transaction is highly costly |
| AML / compliance monitoring | `high` | `medium` | Regulators expect you to catch suspicious activity |
| Clinical trial safety | `high` | `low` | Never miss a safety signal — patient safety is paramount |
| Credit risk early warning | `high` | `low` | Missing deterioration leads to unexpected defaults |
| DevOps SLO monitoring | `high` | `medium` | A missed incident is worse than a brief false page |

---

## Complete Context Examples

### SaaS Churn Detection

```json
{
  "domain": {
    "description": "B2B SaaS churn detection for mid-market software companies. We monitor user engagement and account health signals to identify accounts at risk of not renewing their subscription before the renewal window.",
    "entities": [
      {
        "name": "account",
        "description": "A company-level subscription — the billing and contract unit. The primary subject of churn monitoring."
      },
      {
        "name": "user",
        "description": "An individual platform user within an account."
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
      "definition": "logged in AND performed at least one core action (create, edit, share, or export) in the last 14 days. Login alone does not count."
    },
    {
      "term": "high value account",
      "definition": "ARR above $50,000 regardless of seat count."
    },
    {
      "term": "core action",
      "definition": "creating, editing, sharing, or exporting a document — not just logging in or viewing."
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
        "description": "An individual payment, wire transfer, cash deposit, or withdrawal event."
      }
    ],
    "decisions": ["sar_required", "enhanced_due_diligence", "account_review"]
  },
  "behavioural": {
    "data_cadence": "streaming",
    "meaningful_windows": { "min": "1d", "max": "90d" },
    "regulatory": ["BSA", "FinCEN", "FATF"]
  },
  "semantic_hints": [
    {
      "term": "unusual",
      "definition": "more than 3 standard deviations from the customer's rolling 90-day average transaction value or velocity — not unusual in absolute terms, but unusual relative to that specific customer's history."
    },
    {
      "term": "structuring",
      "definition": "a pattern of multiple transactions intentionally kept below the $10,000 CTR reporting threshold within any 24-hour window."
    },
    {
      "term": "high risk jurisdiction",
      "definition": "a country currently on the FATF grey or black list, or subject to an active OFAC sanctions programme."
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
    "description": "Clinical trial safety monitoring for a Phase 3 oncology programme. We continuously evaluate adverse event patterns to protect patient safety and ensure compliance with ICH E2A and FDA 21 CFR reporting requirements.",
    "entities": [
      {
        "name": "patient",
        "description": "An enrolled trial participant with an active safety monitoring obligation."
      },
      {
        "name": "trial",
        "description": "The clinical study with defined safety monitoring plan and pre-specified stopping rules."
      }
    ],
    "decisions": ["sae_assessment_required", "dsmb_notification", "susar_reporting"]
  },
  "behavioural": {
    "data_cadence": "batch",
    "meaningful_windows": { "min": "7d", "max": "180d" },
    "regulatory": ["FDA-21CFR", "ICH-E6", "ICH-E2A", "GCP"]
  },
  "semantic_hints": [
    {
      "term": "serious",
      "definition": "results in death, is life-threatening, requires inpatient hospitalisation, results in persistent or significant disability, or is a congenital anomaly. Per ICH E2A definition — not the everyday meaning of serious."
    },
    {
      "term": "unexpected",
      "definition": "the nature, severity, or frequency is not consistent with the current Investigator Brochure or reference safety information for this compound."
    },
    {
      "term": "related",
      "definition": "there is a reasonable possibility that the investigational medicinal product caused the adverse event — assessed on temporal relationship, known pharmacology, and absence of alternative explanation."
    }
  ],
  "calibration_bias": {
    "false_negative_cost": "high",
    "false_positive_cost": "low"
  }
}
```

---

## Submitting Your Context

Once you have written your context JSON:

```bash
# Save your context to a file, e.g. context.json
# Then submit it:
curl -X POST https://your-memintel-domain/context \
  -H "Content-Type: application/json" \
  -d @context.json
```

You should receive a response like:

```json
{
  "context_id": "ctx_8f3k2m",
  "version": "v1",
  "is_active": true
}
```

The `version: v1` confirms it was received. Every time you update context, the version number increments (`v2`, `v3`...) and the previous version is kept for audit purposes.

---

## Updating Context Later

You can update context at any time without restarting the server. Just submit a new `POST /context` call with your updated content. The new version immediately becomes active.

After updating, ask your data engineer to recompile any existing tasks so they benefit from the updated context.

---

---

## Worked Example — With vs Without Context

The difference context makes is most visible when you compare what the compiler produces with and without it.

### Without context

A user says: *"Alert me when an account is at risk of churning."*

The compiler has no domain knowledge. It produces something generic:

```json
{
  "concept": "account_activity_score",
  "strategy": "threshold",
  "params": { "value": 0.5 },
  "window": "30d",
  "context_version": null,
  "context_warning": "No active application context exists. Task compiled without domain context — definitions may be less accurate."
}
```

The threshold of 0.5 is a generic midpoint. The window of 30d is a default. Nothing is grounded in the actual domain.

### With context active

The same intent, with context v1 active, compiles to:

```json
{
  "concept": "account_churn_risk",
  "strategy": "composite",
  "components": [
    { "primitive": "account.active_user_rate_30d",    "weight": 0.40 },
    { "primitive": "account.core_action_freq_30d",    "weight": 0.30 },
    { "primitive": "account.seat_utilization_rate",   "weight": 0.20 },
    { "primitive": "account.support_ticket_rate_30d", "weight": 0.10 }
  ],
  "params": { "value": 0.38 },
  "window": "30d",
  "context_version": "v1",
  "context_warning": null
}
```

The composite now reflects what "churn risk" actually means in this domain. The threshold of 0.38 reflects the recall bias — lower than the generic 0.5 midpoint, catching more at-risk accounts. The `semantic_hint` for "active user" is why `account.active_user_rate_30d` was selected — the compiler knew that "active" means "logged in AND performed core action", not just "logged in."

---

## How Context Affects Calibration

When `calibration_bias` is defined, the calibration engine adjusts its recommendations away from the raw statistical optimum.

### Recall bias — missing signals is worse

Context has `false_negative_cost: high`. The statistically optimal threshold from feedback is 0.78 — but the system recommends lower:

```json
{
  "statistically_optimal": 0.78,
  "context_adjusted": 0.702,
  "recommended": 0.702,
  "adjustment_explanation": "Threshold adjusted from 0.78 to 0.702 toward recall based on application context (false_negative_cost=high)"
}
```

### Precision bias — false alarms are worse

Context has `false_positive_cost: high`. The statistically optimal threshold is 0.71 — but the system recommends higher:

```json
{
  "statistically_optimal": 0.71,
  "context_adjusted": 0.781,
  "recommended": 0.781,
  "adjustment_explanation": "Threshold adjusted from 0.71 to 0.781 toward precision based on application context (false_positive_cost=high)"
}
```

### When to use each bias

| Bias | Use when | Examples |
|---|---|---|
| **Recall** | Missing a true signal is worse than a false alarm | Churn detection, AML, medical alerts, safety monitoring |
| **Precision** | False alarms are costly — especially for automated actions | Fraud blocking, automated suspensions, clinical stopping rules |
| **Balanced** | Roughly equal cost on both sides | Internal dashboards, advisory alerts with human review |

---

## Window Clamping

If `behavioural.meaningful_windows` is defined, calibration window recommendations are automatically clamped to that range. If the calibration algorithm recommends a 14-day window for a SaaS application with `min: 30d`, the response will note:

```json
{
  "adjustment_explanation": "Window clamped to 30d minimum per application context (meaningful_windows.min=30d). Statistically optimal window was 14d."
}
```

This prevents the calibration engine from recommending windows that are operationally meaningless in your domain.

---

## Checking Context Impact After an Update

After updating context, use `GET /context/impact` to see how many tasks were compiled under older versions:

```bash
GET /context/impact
```

```json
{
  "current_version": "v2",
  "tasks_on_current_version": 14,
  "tasks_on_older_versions": [{ "version": "v1", "task_count": 8 }],
  "total_stale_tasks": 8
}
```

Tasks on older versions continue running correctly — they are pinned to their compiled version. But they do not benefit from the updated context. Trigger recompilation to incorporate the new domain understanding.


## Common Mistakes

**Writing a vague description.** "A monitoring platform" tells the system nothing useful. Be specific about industry, use case, and what is being monitored.

**Skipping semantic hints.** This is the most common mistake. If your team uses terms like "active user", "high value", "at risk", or "unusual" in their monitoring requests — and those terms have specific meanings in your organisation — the system will guess what they mean. Define them explicitly.

**Setting the wrong calibration bias.** If your system takes automated action on every alert (blocks a transaction, suspends an account), false positives are very costly — set `false_positive_cost: high`. If alerts go to a human reviewer who can apply judgement, false positives are more tolerable.

---

## Next Step

→ [Step 2: Primitives](/docs/admin-guide/admin-primitives)
