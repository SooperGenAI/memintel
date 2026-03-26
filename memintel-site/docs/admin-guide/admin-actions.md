---
id: admin-actions
title: Configuring Actions
sidebar_label: Actions
---

# Configuring Actions

Actions define what happens when a condition fires. An action is the output of the evaluation pipeline — the moment a deterministic decision produces a real-world effect. The admin configures the available action types and their delivery endpoints. Users then bind tasks to actions when they create monitoring tasks.

---

## What Actions Do

When a task evaluates a condition and the condition is met, one or more actions are triggered. Memintel supports three action types:

| Type | What it does | Best for |
|---|---|---|
| `webhook` | HTTP POST to a configured endpoint with the decision payload | Integrating with any downstream system |
| `notification` | Delivers a formatted alert to a configured channel | Slack, email, in-app notifications |
| `log_only` | Records the decision to the audit log without external delivery | Audit trail, dry-run mode, debugging |

---

## The Actions Config File

Actions are defined in `memintel_actions.yaml`. Like primitives and guardrails, this file is loaded at startup and changes require a restart.

```yaml
# memintel_actions.yaml

actions:

  - id: <string>              # required — unique identifier
    type: webhook             # webhook | notification | log_only
    endpoint: <url>           # required for webhook
    method: POST              # optional — default POST
    headers:                  # optional — static headers
      <key>: <value>
    retry:                    # optional — retry configuration
      max_attempts: 3
      backoff: exponential
    timeout_seconds: 10       # optional — default 10
    description: <string>     # required — plain English

  - id: <string>
    type: notification
    channel: <string>         # slack | email | in_app
    endpoint: <url>           # Slack webhook URL, email address, etc.
    template: <string>        # optional — named template ID
    description: <string>

  - id: <string>
    type: log_only
    description: <string>
```

---

## Webhook Actions

Webhooks are the most flexible action type. When a condition fires, Memintel sends an HTTP POST to your configured endpoint with the full decision payload. Your application receives the webhook and does whatever it needs to do — create a ticket, trigger a workflow, send a notification, update a dashboard.

### Basic Webhook

```yaml
actions:

  - id: deal_risk_alert
    type: webhook
    endpoint: https://myapp.com/hooks/deal-risk
    method: POST
    timeout_seconds: 10
    retry:
      max_attempts: 3
      backoff: exponential
    description: Fires when a deal risk condition is met — routes to sales ops workflow
```

### Webhook with Authentication Headers

For endpoints that require authentication, configure static headers. Use environment variable references (prefixed with `$`) for secrets — never hardcode credentials in the config file.

```yaml
actions:

  - id: compliance_alert
    type: webhook
    endpoint: https://compliance.myapp.com/hooks/aml-alert
    method: POST
    headers:
      Authorization: "Bearer $COMPLIANCE_WEBHOOK_SECRET"
      X-Source: "memintel"
      Content-Type: "application/json"
    timeout_seconds: 15
    retry:
      max_attempts: 5
      backoff: exponential
    description: AML alert delivery to compliance system — authenticated endpoint

  - id: credit_risk_webhook
    type: webhook
    endpoint: https://risk.myapp.com/hooks/credit-early-warning
    headers:
      X-API-Key: "$RISK_SYSTEM_API_KEY"
    timeout_seconds: 10
    retry:
      max_attempts: 3
      backoff: linear
    description: Credit early warning delivery to risk management platform
```

### Multiple Webhooks for Priority Routing

Register separate action IDs for different priority levels. Users bind tasks to the appropriate action when creating them.

```yaml
actions:

  - id: sre_page
    type: webhook
    endpoint: https://pagerduty.com/integration/your-key/enqueue
    headers:
      Authorization: "Token token=$PAGERDUTY_TOKEN"
    timeout_seconds: 5
    retry:
      max_attempts: 5
      backoff: exponential
    description: High-priority page — SLO breach risk, active incident, deployment block

  - id: sre_slack_alert
    type: webhook
    endpoint: https://hooks.slack.com/services/$SLACK_SRE_WEBHOOK
    timeout_seconds: 10
    retry:
      max_attempts: 3
      backoff: exponential
    description: Medium-priority Slack alert — SRE channel for early warnings

  - id: sre_log_only
    type: log_only
    description: Low-priority — decision logged only, no external delivery. Used for monitoring tasks in observation mode.
```

### Webhook Payload

When a webhook fires, Memintel sends a JSON payload to the configured endpoint. The payload always includes:

```json
{
  "decision_id":        "dec_9x2k1m",
  "task_id":            "tsk_8f3k2",
  "entity":             "deal_acme_corp_q2",
  "timestamp":          "2024-03-15T09:00:00Z",
  "concept_id":         "deal.stall_risk",
  "concept_version":    "1.0",
  "condition_id":       "deal.at_risk_of_stalling",
  "condition_version":  "1.0",
  "result_value":       0.81,
  "decision_value":     true,
  "action_triggered":   true,
  "action_id":          "deal_risk_alert",
  "contributions": {
    "thread_pressure":   0.35,
    "stage_pressure":    0.29,
    "sentiment_score":   0.17
  },
  "context_version":    "v1"
}
```

Your application should return HTTP `2xx` within the configured `timeout_seconds`. Any non-2xx response triggers the retry logic.

---

## Notification Actions

Notification actions deliver formatted alerts to messaging channels. Memintel handles the formatting — your team receives a readable, actionable message rather than a raw JSON payload.

### Slack Notification

```yaml
actions:

  - id: sales_ops_slack
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_SALES_OPS_WEBHOOK
    template: deal_risk_standard
    description: Deal risk alert delivered to #sales-ops Slack channel

  - id: compliance_slack
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_COMPLIANCE_WEBHOOK
    template: aml_alert_standard
    description: AML alert delivered to #compliance Slack channel — includes SAR workflow link

  - id: sre_slack
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_SRE_WEBHOOK
    template: slo_alert_standard
    description: SLO early warning alert delivered to #sre Slack channel
```

### Email Notification

```yaml
actions:

  - id: credit_risk_email
    type: notification
    channel: email
    endpoint: credit-risk-team@mycompany.com
    template: credit_early_warning_email
    description: Credit risk early warning email to credit risk team

  - id: network_compliance_email
    type: notification
    channel: email
    endpoint: provider-relations@mycompany.com
    template: provider_credentialing_alert
    description: Provider credentialing alert to network management team
```

### Notification Templates

Templates control how the decision payload is formatted for human consumption. Reference a named template in the action config — templates are defined separately in `memintel_templates.yaml`.

A well-designed notification template includes:
- What was detected and for which entity
- The key signal values that drove the condition
- The recommended action
- A link to the relevant workflow or dashboard

```yaml
# memintel_templates.yaml

templates:

  - id: deal_risk_standard
    title: "Deal Risk Alert — {{entity}}"
    body: |
      Deal {{entity}} has reached a stall risk score of {{result_value | percent}}.

      Top signals:
      {{#contributions}}
      • {{key}}: {{value | percent}}
      {{/contributions}}

      Threshold: {{condition.params.value | percent}}
      Condition version: {{condition_version}}

    actions:
      - label: "View deal"
        url: "https://crm.myapp.com/deals/{{entity}}"
      - label: "Mark reviewed"
        url: "https://myapp.com/hooks/mark-reviewed?decision_id={{decision_id}}"

  - id: aml_alert_standard
    title: "AML Alert — {{entity}}"
    body: |
      Customer {{entity}} — transaction risk score: {{result_value | percent}}

      Risk signals:
      • Value vs baseline: {{contributions.value_vs_baseline | x_multiple}}
      • Structuring signal: {{contributions.structuring_signal | percent}}
      • Jurisdiction risk: {{contributions.jurisdiction_risk | risk_level}}

    actions:
      - label: "Open SAR workflow"
        url: "https://compliance.myapp.com/sar/new?customer={{entity}}"
      - label: "View transaction history"
        url: "https://myapp.com/customers/{{entity}}/transactions"
```

---

## Retry Configuration

Configure retry behaviour separately for each action. Production systems should always configure retries for webhooks that deliver to external services.

```yaml
retry:
  max_attempts: 3        # total attempts including the first
  backoff: exponential   # exponential | linear | none
  initial_delay_ms: 500  # delay before first retry (exponential/linear only)
  max_delay_ms: 30000    # cap on retry delay
```

**Backoff strategies:**

| Strategy | Behaviour | Best for |
|---|---|---|
| `exponential` | Delay doubles on each retry (500ms → 1s → 2s...) | Downstream service under load — gives it time to recover |
| `linear` | Delay is constant on each retry | Simple retry without back-pressure concern |
| `none` | Immediate retry | Only for very low-latency, idempotent endpoints |

**Retry failure handling:**

If all retry attempts are exhausted, the delivery failure is logged with the full decision payload. The decision itself is permanently recorded in the audit log regardless of delivery outcome. A failed webhook does not affect the determinism or auditability of the decision.

---

## Complete Example — Multi-Domain Deployment

```yaml
# memintel_actions.yaml

actions:

  # ── Sales / CRM ──────────────────────────────────────────────────
  - id: deal_risk_webhook
    type: webhook
    endpoint: https://myapp.com/hooks/deal-risk
    headers:
      Authorization: "Bearer $SALES_WEBHOOK_SECRET"
    retry:
      max_attempts: 3
      backoff: exponential
    timeout_seconds: 10
    description: Deal stall risk alert — routes to sales ops workflow

  - id: deal_risk_slack
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_SALES_WEBHOOK
    template: deal_risk_standard
    description: Deal risk notification to #sales-ops Slack channel

  # ── Compliance / AML ─────────────────────────────────────────────
  - id: aml_alert_webhook
    type: webhook
    endpoint: https://compliance.myapp.com/hooks/aml-alert
    headers:
      Authorization: "Bearer $COMPLIANCE_WEBHOOK_SECRET"
      X-Priority: "high"
    retry:
      max_attempts: 5
      backoff: exponential
      initial_delay_ms: 200
    timeout_seconds: 15
    description: AML alert delivery — high-priority, 5 retries

  - id: aml_log_only
    type: log_only
    description: AML decision logging only — used for low-severity monitoring tasks in observation mode

  # ── Credit Risk ───────────────────────────────────────────────────
  - id: credit_early_warning_webhook
    type: webhook
    endpoint: https://risk.myapp.com/hooks/credit-warning
    headers:
      X-API-Key: "$RISK_SYSTEM_API_KEY"
    retry:
      max_attempts: 3
      backoff: exponential
    timeout_seconds: 10
    description: Credit early warning delivery to risk management platform

  - id: credit_risk_email
    type: notification
    channel: email
    endpoint: credit-risk@mycompany.com
    template: credit_early_warning_email
    description: Credit risk email alert to credit risk team

  # ── SRE / DevOps ──────────────────────────────────────────────────
  - id: sre_page
    type: webhook
    endpoint: https://events.pagerduty.com/v2/enqueue
    headers:
      Authorization: "Token token=$PAGERDUTY_INTEGRATION_KEY"
    retry:
      max_attempts: 5
      backoff: exponential
      initial_delay_ms: 100
    timeout_seconds: 5
    description: PagerDuty page — critical SLO breach risk or active incident

  - id: sre_slack_alert
    type: notification
    channel: slack
    endpoint: https://hooks.slack.com/services/$SLACK_SRE_WEBHOOK
    template: slo_alert_standard
    description: SRE Slack alert — early warning and non-critical conditions

  - id: deployment_block_webhook
    type: webhook
    endpoint: https://ci.myapp.com/hooks/deployment-block
    headers:
      Authorization: "Bearer $CI_WEBHOOK_SECRET"
    retry:
      max_attempts: 3
      backoff: linear
    timeout_seconds: 5
    description: Deployment block signal — sent to CI/CD pipeline to hold deployment

  # ── Healthcare ────────────────────────────────────────────────────
  - id: safety_alert_webhook
    type: webhook
    endpoint: https://safety.myapp.com/hooks/ae-alert
    headers:
      Authorization: "Bearer $SAFETY_SYSTEM_SECRET"
      X-Trial-ID: "$TRIAL_ID"
    retry:
      max_attempts: 5
      backoff: exponential
    timeout_seconds: 10
    description: Safety signal alert delivery to pharmacovigilance system

  - id: network_compliance_email
    type: notification
    channel: email
    endpoint: network-management@mycompany.com
    template: provider_credentialing_alert
    description: Provider network compliance alert — credentialing and OIG exclusion events
```

---

## Validating Actions

At startup, the server validates all action configurations:

```bash
# Check all actions loaded
curl http://localhost:8000/actions

# Verify a specific action is reachable
curl -X POST http://localhost:8000/actions/deal_risk_webhook/test
# Sends a test payload to the configured endpoint
```

### Startup Validation Checks

| Check | What it verifies |
|---|---|
| Unique IDs | No duplicate action IDs |
| Valid type | `webhook`, `notification`, or `log_only` only |
| Endpoint present | Webhook and notification actions have an endpoint |
| Channel valid | Notification channel is `slack`, `email`, or `in_app` |
| No credentials in config | Endpoint strings do not contain raw API keys |
| Environment variable references resolve | All `$VAR` references are set in the environment |

---

## Security

**Never hardcode credentials in the actions config.** Use `$ENV_VAR` references for all secrets. The config loader resolves these from the environment at startup and never logs the resolved values.

```yaml
# Wrong — credential hardcoded in config file
headers:
  Authorization: "Bearer sk-abc123xyz..."

# Right — resolved from environment variable
headers:
  Authorization: "Bearer $MY_WEBHOOK_SECRET"
```

**Validate webhook endpoints before go-live.** Use the test endpoint (`POST /actions/&#123;id&#125;/test`) to verify each webhook is reachable and returns a 2xx response before onboarding users.

**Configure appropriate timeouts.** If a downstream system is slow, a long timeout blocks the evaluation thread. For most webhooks, 10 seconds is sufficient. For critical safety systems where retries matter more than latency, configure shorter timeouts with more retry attempts.

---

## Common Mistakes

**Registering only one action for all use cases.** Different severity levels and use cases warrant different delivery mechanisms. Register a paging action for critical conditions, a Slack action for early warnings, and a log-only action for observation mode tasks.

**Not configuring retries.** Network failures are inevitable. Without retries, a transient failure means a missed alert. Always configure retries for production webhook actions.

**Using the same endpoint for all alert types.** If your downstream system cannot distinguish between an AML alert and a credit risk alert, it cannot route them appropriately. Use separate action IDs for different alert domains — even if they go to the same service, include an action-specific header or path parameter that the downstream system can use for routing.

**Forgetting to set environment variables for `$` references.** The server validates that `$ENV_VAR` references resolve at startup. If a variable is not set, the server will refuse to start. Verify all referenced environment variables are set in your deployment environment.

---

## Admin Setup Complete

You have now configured all four layers of the Memintel admin setup:

1. ✓ **Application Context** — domain understanding for the compiler
2. ✓ **Primitives** — the signal vocabulary
3. ✓ **Guardrails** — compiler policy and constraints
4. ✓ **Actions** — delivery configuration

The system is ready for users to create tasks and begin monitoring.

**Recommended next steps:**

- Run the full smoke test sequence from the [Self-Hosting guide](/docs/intro/self-hosting#step-6----smoke-test) to verify the end-to-end pipeline
- Create the first real task using the [Quickstart](/docs/intro/quickstart)
- Review [Common Mistakes](/docs/intro/common-mistakes) before onboarding users
