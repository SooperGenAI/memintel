---
id: admin-overview
title: Admin Guide
sidebar_label: Overview
---

# Admin Guide

This guide is for the **admin** — the person responsible for setting up and maintaining Memintel for their organisation. You do not need to be a software developer to follow this guide, but you do need to know your domain well: what signals matter, what thresholds are meaningful, and what should happen when something is detected.

---

## What the Admin Does

Memintel has three types of people who interact with it:

| Role | Who they are | What they do |
|---|---|---|
| **Data Engineer** | Your technical team | Connects Memintel to your data sources. Writes the code that fetches signal values. |
| **Admin** | You | Defines what signals exist, what the rules are, and what happens when something is detected. |
| **User** | Your team members | Types plain English monitoring requests like "alert me when a deal is at risk." |

As admin, you are the bridge between the technical team and the users. You do not write code. You define the vocabulary and the rules that the system works within.

---

## How Configuration Works

Everything you configure as admin lives in a single file called **`memintel_config.yaml`**. This file sits on your server and is read by Memintel when it starts up.

:::note What is a YAML file?
YAML is a simple text format for configuration. It uses indentation and colons to organise information — similar to a structured list. You can edit it in any text editor. The most important rule: **indentation matters**. Lines that are indented further are "inside" the lines above them. We will show you exactly what everything should look like — you will mostly be copying and editing examples rather than writing from scratch.
:::

Your `memintel_config.yaml` has four sections:

```
memintel_config.yaml
│
├── context          ← What your application does (optional but strongly recommended)
├── primitives       ← What signals you want to monitor
├── guardrails       ← The rules the system follows when interpreting requests
└── actions          ← What happens when something is detected
```

Each section is covered in detail in the pages that follow. You will build the file up one section at a time.

---

## Where the File Lives

The file can live anywhere on your server. The location is set by your technical team via an environment variable called `MEMINTEL_CONFIG_PATH`. A common location is:

```
/etc/memintel/memintel_config.yaml
```

Ask your data engineer where they have set this path before you start editing. You need to edit the file at that exact location for Memintel to pick up your changes.

:::warning
Every time you change `memintel_config.yaml`, the server needs to be restarted to pick up the changes. Ask your data engineer to restart the server after you finish editing. Application context (the first section) is the exception — it is updated via a separate process and does not require a restart.
:::

---

## Setup Order

Work through the sections in this order. Each one builds on the previous.

### Step 1 — Application Context
*What your application does, who the users are, what terms mean in your domain.*

This is the most important step for accuracy. It gives the system the background knowledge it needs to understand what your team members are asking for when they create monitoring tasks.

→ [Set up Application Context](/docs/admin-guide/admin-application-context)

### Step 2 — Primitives
*The signals you want to monitor.*

A primitive is a single measurable signal — "days since last login", "transaction amount vs baseline", "adverse event severity score". You declare what signals exist and what type of data they contain. Your data engineer then connects each signal to the actual data source.

→ [Set up Primitives](/docs/admin-guide/admin-primitives)

### Step 3 — Guardrails
*The rules the system follows when interpreting monitoring requests.*

When a user says "alert me when something is significantly elevated", the guardrails tell the system what "significantly" means in numbers. You define this mapping based on your domain knowledge.

→ [Set up Guardrails](/docs/admin-guide/admin-guardrails)

### Step 4 — Actions
*What happens when a condition fires.*

An action is the delivery mechanism — where does the alert go? A Slack message, an email, a webhook to another system, or just a log entry. You define the available actions here and your team members choose which one to use when they create a task.

→ [Set up Actions](/docs/admin-guide/admin-actions)

---

## The Complete File Structure

Here is what a complete `memintel_config.yaml` looks like with all four sections. Do not worry about understanding every line right now — each section is explained in detail on its own page. This is just to show you the overall shape.

```yaml
# memintel_config.yaml

# ─────────────────────────────────────────────
# SECTION 1: APPLICATION CONTEXT
# ─────────────────────────────────────────────
context:
  domain:
    description: "B2B SaaS churn detection for mid-market software companies."
    entities:
      - name: account
        description: "A company-level subscription"
      - name: user
        description: "An individual platform user"
    decisions:
      - churn_risk
      - expansion_opportunity
  behavioural:
    data_cadence: batch
    meaningful_windows:
      min: 30d
      max: 90d
    regulatory:
      - GDPR
      - SOC2
  semantic_hints:
    - term: "active user"
      definition: "logged in AND performed a core action in last 14 days"
    - term: "high value account"
      definition: "ARR above $50,000"
  calibration_bias:
    false_negative_cost: high
    false_positive_cost: medium

# ─────────────────────────────────────────────
# SECTION 2: PRIMITIVES
# ─────────────────────────────────────────────
primitives:
  - id: account.active_user_rate_30d
    type: float
    source: activity_pipeline
    entity: account_id
    description: "Ratio of active users to total licensed seats, 0-1"

  - id: account.days_to_renewal
    type: int
    source: billing_pipeline
    entity: account_id
    description: "Days until next renewal date"

  - id: account.payment_failed_flag
    type: boolean
    source: billing_pipeline
    entity: account_id
    description: "True if most recent payment attempt failed"

# ─────────────────────────────────────────────
# SECTION 3: GUARDRAILS
# ─────────────────────────────────────────────
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

    account.days_to_renewal:
      low_severity:    { value: 90 }
      medium_severity: { value: 60 }
      high_severity:   { value: 30 }

  bias_rules:
    urgent:      high_severity
    significant: medium_severity
    early:       low_severity
    approaching: low_severity

  global_default_strategy:   threshold
  global_preferred_strategy: percentile

# ─────────────────────────────────────────────
# SECTION 4: ACTIONS
# ─────────────────────────────────────────────
actions:
  - id: slack_alert
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_WEBHOOK
    description: "Sends alert to #customer-success Slack channel"

  - id: webhook_crm
    type: webhook
    endpoint: https://myapp.com/hooks/churn-alert
    description: "Posts alert to CRM workflow system"
```

---

## Before You Start

Before editing the config file, check these three things:

**1. Know where the file is.** Ask your data engineer: "Where is `memintel_config.yaml` on the server?" You need the exact path.

**2. Know how to edit it.** If the server is remote (e.g. on Railway, Render, or AWS), ask your data engineer how to access and edit the file. On some platforms you can edit environment config through a dashboard. On others you need to edit the file directly via a terminal.

**3. Have a text editor ready.** Any plain text editor works — VS Code, Notepad++, nano, vim. Do not use Microsoft Word or Google Docs — they add hidden formatting that will break the YAML.

---

## Ready to Start?

→ [Step 1: Application Context](/docs/admin-guide/admin-application-context)
