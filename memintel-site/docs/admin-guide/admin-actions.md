---
id: admin-actions
title: Step 4 — Actions
sidebar_label: Step 4 — Actions
---

# Step 4 — Actions

An action is what happens when a monitoring condition fires. It is the delivery mechanism — where does the alert go, and in what form?

You define a library of available actions here. When your team members create monitoring tasks, they choose which action to attach. You can have as many actions as you need — a Slack alert for one team, an email for another, a webhook to your CRM for a third.

---

## Where Actions Live

Actions are defined in the `actions:` section of `memintel_config.yaml`:

```yaml
# memintel_config.yaml

actions:
  - id: slack_sales_ops
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_WEBHOOK
    description: "Sends alert to the #sales-ops Slack channel"

  - id: crm_webhook
    type: webhook
    endpoint: https://myapp.com/hooks/alert
    description: "Posts alert payload to CRM workflow system"

  - id: audit_log_only
    type: log_only
    description: "Records the decision to the audit log — no external delivery"
```

---

## Three Action Types

### notification — formatted alerts to messaging channels

Use this for Slack messages, emails, or in-app notifications. Memintel formats the alert into a readable message — your team receives plain English, not raw data.

```yaml
- id: slack_compliance
  type: notification
  channel: slack    # slack | email | in_app
  endpoint: https://hooks.slack.com/services/$SLACK_COMPLIANCE_WEBHOOK
  description: "AML alert to #compliance Slack channel"
```

For email:
```yaml
- id: credit_risk_email
  type: notification
  channel: email
  endpoint: credit-risk-team@mycompany.com
  description: "Credit risk early warning email to credit risk team"
```

### webhook — raw data to another system

Use this when you want to send the full alert payload to another system — your CRM, ticketing system, PagerDuty, incident management tool, or any application with an API endpoint.

```yaml
- id: pagerduty_page
  type: webhook
  endpoint: https://events.pagerduty.com/v2/enqueue
  headers:
    Authorization: "Token token=$PAGERDUTY_KEY"
  retry:
    max_attempts: 5
    backoff: exponential
  description: "PagerDuty page — critical SLO or deployment risk"
```

### log_only — audit trail only, no external delivery

Use this for low-priority monitoring tasks, observation mode, or any situation where you want the decision recorded but do not need anyone notified immediately.

```yaml
- id: audit_only
  type: log_only
  description: "Records decision to audit log — no notification sent"
```

---

## The id Field

Every action needs a unique identifier. This is what your team members see when they choose which action to attach to a monitoring task. Make it readable and descriptive.

```yaml
id: slack_sales_ops          # clear — who gets it and via what channel
id: compliance_slack         # clear
id: credit_risk_email        # clear
id: pagerduty_critical       # clear
id: aml_audit_log            # clear

id: action1                  # bad — not descriptive
id: webhook                  # bad — not unique if you have multiple webhooks
```

---

## Using Environment Variables for Secrets

Never put real API keys, webhook secrets, or passwords directly in the config file. Use `$VARIABLE_NAME` references instead — the server resolves these from environment variables at startup.

```yaml
# Wrong — secret hardcoded in config file
endpoint: https://hooks.slack.com/services/T00000/B00000/XXXX

# Right — secret stored as environment variable
endpoint: https://hooks.slack.com/services/$SLACK_SALES_OPS_WEBHOOK
```

Ask your data engineer to set the corresponding environment variables on the server. They will never appear in the config file itself.

---

## Retry Configuration

For webhook actions, configure retries so that a temporary network failure does not cause a missed alert:

```yaml
- id: crm_webhook
  type: webhook
  endpoint: https://myapp.com/hooks/alert
  retry:
    max_attempts: 3         # try up to 3 times total
    backoff: exponential    # wait longer between each retry
  timeout_seconds: 10       # give up on a single attempt after 10 seconds
  description: "CRM alert webhook"
```

**Backoff options:**
- `exponential` — waits longer after each failed attempt (recommended for most cases)
- `linear` — same wait time between each retry
- `none` — retries immediately (only for very fast, reliable endpoints)

---

## Complete Examples

### SaaS Platform — Multiple Alert Channels

```yaml
actions:

  # Customer success team — Slack
  - id: slack_customer_success
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_CS_WEBHOOK
    description: "Churn risk alert to #customer-success Slack channel"

  # Account executives — email
  - id: ae_email_alert
    type: notification
    channel: email
    endpoint: account-executives@mycompany.com
    description: "Deal risk alert email to account executive team"

  # CRM workflow system — webhook
  - id: crm_churn_webhook
    type: webhook
    endpoint: https://crm.myapp.com/hooks/churn-alert
    headers:
      Authorization: "Bearer $CRM_WEBHOOK_SECRET"
    retry:
      max_attempts: 3
      backoff: exponential
    timeout_seconds: 10
    description: "Churn alert posted to CRM workflow — triggers outreach sequence"

  # Observation mode — log only
  - id: log_only
    type: log_only
    description: "Decision recorded to audit log only — no notification"
```

### Compliance / Financial Services

```yaml
actions:

  # Compliance team — immediate Slack alert
  - id: compliance_slack_alert
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_COMPLIANCE_WEBHOOK
    description: "AML and compliance alerts to #compliance Slack channel"

  # Compliance system — webhook with high retry count
  - id: compliance_system_webhook
    type: webhook
    endpoint: https://compliance.myapp.com/hooks/aml-alert
    headers:
      Authorization: "Bearer $COMPLIANCE_WEBHOOK_SECRET"
      X-Alert-Priority: "high"
    retry:
      max_attempts: 5
      backoff: exponential
    timeout_seconds: 15
    description: "AML alert to compliance management system — high priority, 5 retries"

  # Credit risk team — email
  - id: credit_risk_email
    type: notification
    channel: email
    endpoint: credit-risk@mycompany.com
    description: "Credit early warning email to credit risk team"

  # Audit trail only
  - id: audit_log
    type: log_only
    description: "Decision recorded to audit log — no notification"
```

### Clinical Trials / Healthcare

```yaml
actions:

  # Medical monitor — immediate Slack alert
  - id: safety_slack_alert
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_SAFETY_WEBHOOK
    description: "Safety signal alert to medical monitor via Slack"

  # Safety database system — webhook
  - id: safety_system_webhook
    type: webhook
    endpoint: https://safety.myapp.com/hooks/ae-alert
    headers:
      Authorization: "Bearer $SAFETY_SYSTEM_SECRET"
    retry:
      max_attempts: 5
      backoff: exponential
    timeout_seconds: 10
    description: "Adverse event alert posted to pharmacovigilance system"

  # Medical monitor — email
  - id: medical_monitor_email
    type: notification
    channel: email
    endpoint: medical-monitor@mycompany.com
    description: "Safety alert email to medical monitor"

  # Regulatory audit trail
  - id: regulatory_audit_log
    type: log_only
    description: "Decision recorded to audit log for regulatory inspection readiness"
```

### DevOps / SRE

```yaml
actions:

  # Critical — PagerDuty page
  - id: pagerduty_page
    type: webhook
    endpoint: https://events.pagerduty.com/v2/enqueue
    headers:
      Authorization: "Token token=$PAGERDUTY_INTEGRATION_KEY"
    retry:
      max_attempts: 5
      backoff: exponential
    timeout_seconds: 5
    description: "PagerDuty page — critical SLO breach risk or deployment block"

  # Medium priority — SRE Slack
  - id: sre_slack_alert
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_SRE_WEBHOOK
    description: "SRE Slack alert — early warning conditions"

  # Deployment hold — CI/CD webhook
  - id: deployment_block
    type: webhook
    endpoint: https://ci.myapp.com/hooks/deployment-block
    headers:
      Authorization: "Bearer $CI_WEBHOOK_SECRET"
    retry:
      max_attempts: 3
      backoff: linear
    timeout_seconds: 5
    description: "Deployment hold signal sent to CI/CD pipeline"
```

---

## Common Mistakes

**Using one action for everything.** Different alerts warrant different delivery mechanisms. A critical safety signal and a low-priority early warning should not go to the same channel with the same urgency. Register separate actions for different priority levels.

**Not configuring retries for webhooks.** Network failures happen. Without retries, a transient failure means a missed alert. Always configure retries for webhook actions in production.

**Hardcoding secrets in the endpoint URL.** If your Slack webhook URL contains your actual token, it is visible to anyone who can read the config file. Use `$ENV_VAR` references.

**Not testing actions before go-live.** Ask your data engineer to run `POST /actions/&#123;action_id&#125;/test` for each action before users start creating tasks. This confirms the endpoint is reachable and responding correctly.

---

## Setup Complete

You have now configured all four sections of `memintel_config.yaml`:

1. ✓ **Application Context** — domain understanding submitted via API
2. ✓ **Primitives** — the signal vocabulary defined
3. ✓ **Guardrails** — the compiler policy and thresholds defined
4. ✓ **Actions** — alert delivery configured

Ask your data engineer to **restart the server** to load the updated config, and to verify that each action endpoint is reachable.

After the restart, your team can begin creating monitoring tasks. Point them to the [Quickstart](/docs/intro/quickstart) to get started.

---

## The Full Config File

Here is the complete structure of `memintel_config.yaml` with all four sections — use this as your reference:

```yaml
# memintel_config.yaml

# Application context is submitted separately via POST /context
# It does not live in this file

# ─────────────────────────────
# SECTION 1: PRIMITIVES
# ─────────────────────────────
primitives:
  - id: account.active_user_rate_30d
    type: float
    source: activity_pipeline
    entity: account_id
    description: "Ratio of active users to total licensed seats, 0-1"

  # add more primitives here...

# ─────────────────────────────
# SECTION 2: GUARDRAILS
# ─────────────────────────────
guardrails:
  type_strategy_map:
    float:      [threshold, percentile, z_score, change]
    int:        [threshold, percentile, change]
    boolean:    [equals]
    categorical: [equals]
    time_series<float>: [z_score, change, percentile]
    time_series<int>:   [z_score, change, percentile]
    float?: [threshold]

  parameter_priors:
    account.active_user_rate_30d:
      low_severity:    { value: 0.60 }
      medium_severity: { value: 0.45 }
      high_severity:   { value: 0.30 }
    # add more priors here...

  bias_rules:
    urgent:      high_severity
    significant: medium_severity
    early:       low_severity
    # add more bias rules here...

  threshold_directions:
    account.active_user_rate_30d: below
    # add more direction overrides here...

  global_default_strategy:   threshold
  global_preferred_strategy: percentile

# ─────────────────────────────
# SECTION 3: ACTIONS
# ─────────────────────────────
actions:
  - id: slack_alert
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_WEBHOOK
    description: "Alert to Slack channel"

  # add more actions here...
```
