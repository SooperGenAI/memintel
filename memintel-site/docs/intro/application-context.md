---
id: application-context
title: Application Context
sidebar_label: Application Context
---

# Application Context

Application context is an optional but strongly recommended configuration layer that lets the admin describe the application's domain in natural language before defining primitives or creating tasks. This context is used by the LLM during task creation to produce more accurate, domain-aware concept and condition definitions.

:::tip
Defining application context before creating primitives gives the LLM the domain knowledge it needs to generate more accurate concept and condition definitions. Without context, task creation still works but produces generic definitions that may need more calibration cycles to reach production accuracy.
:::

---

## How It Works

Without context, the LLM compiles user intent using only the registered primitive vocabulary and guardrails. It has no knowledge of what the application does, who the users are, what regulatory environment applies, or what domain-specific terms mean.

With context, the LLM understands:
- What the application does and what domain it operates in
- What entities are being monitored and what decisions are being made
- What domain-specific terms mean — for example, that "active user" means "logged in AND performed core action in last 14 days", not just "logged in"
- What the cost asymmetry is between false positives and false negatives — which directly influences how it resolves severity language
- What regulatory frameworks apply — which informs what "significant" or "elevated" means in that context

The result is that a user saying *"alert me when a high-value account shows churn risk"* compiles to a much more precise condition when the LLM knows what "high-value account" means and what the meaningful time windows are for this domain.

---

## Key Behaviours

- **Context is optional** — the system works without it, but task creation is less accurate
- **Context is versioned** — each POST to `/context` creates a new version (v1, v2, v3...) and deactivates the previous one
- **Only one version is active at a time** — the active version is used for all new task compilations
- **Context is never deleted** — old versions are retained for audit purposes and can be retrieved by version string
- **Tasks record their context version** — every task carries a `context_version` field showing which context was active when it was compiled
- **Missing context is surfaced** — if no context is defined at task creation time, the response includes a `context_warning` field

---

## API Endpoints

### POST /context

Create or update application context. Creates a new version and deactivates the previous active version.

**Request schema:**

| Field | Type | Required | Description |
|---|---|---|---|
| `domain.description` | string | Yes | Natural language description of what the application does and what domain it operates in |
| `domain.entities` | array | No | List of entity declarations. Each has `name` (string) and `description` (string) |
| `domain.decisions` | array | No | List of decision type names relevant to this domain (e.g. `churn_risk`, `fraud_alert`) |
| `behavioural.data_cadence` | enum | No | `batch` \| `streaming` \| `mixed`. Default: `batch` |
| `behavioural.meaningful_windows` | object | No | `min` and `max` duration strings (e.g. `30d`, `90d`). Informs calibration window clamping |
| `behavioural.regulatory` | array | No | List of regulatory frameworks in scope (e.g. `GDPR`, `SOC2`, `HIPAA`) |
| `semantic_hints` | array | No | Domain-specific term definitions. Each has `term` (string) and `definition` (string) |
| `calibration_bias.false_negative_cost` | enum | No | `high` \| `medium` \| `low`. Cost of missing a true positive |
| `calibration_bias.false_positive_cost` | enum | No | `high` \| `medium` \| `low`. Cost of a false alarm |

:::note
`bias_direction` is auto-derived from `false_negative_cost` vs `false_positive_cost` and cannot be set manually:
- `false_negative_cost > false_positive_cost` → `recall` bias
- `false_positive_cost > false_negative_cost` → `precision` bias
- Equal → `balanced`
:::

**Example request — SaaS churn detection:**

```json
{
  "domain": {
    "description": "B2B SaaS churn detection for mid-market software companies.",
    "entities": [
      { "name": "user", "description": "individual platform user" },
      { "name": "account", "description": "company-level subscription" }
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
      "definition": "logged in AND performed core action in last 14 days"
    },
    {
      "term": "high value account",
      "definition": "ARR above $50,000"
    }
  ],
  "calibration_bias": {
    "false_negative_cost": "high",
    "false_positive_cost": "medium"
  }
}
```

**Response schema — ApplicationContext:**

| Field | Type | Description |
|---|---|---|
| `context_id` | string (UUID) | Auto-generated unique identifier |
| `version` | string | Auto-assigned version string: `v1`, `v2`, `v3`... |
| `domain` | object | The domain context as submitted |
| `behavioural` | object | The behavioural context as submitted |
| `semantic_hints` | array | The semantic hints as submitted |
| `calibration_bias` | object \| null | The calibration bias including auto-derived `bias_direction` |
| `created_at` | datetime | ISO 8601 UTC timestamp of creation |
| `is_active` | boolean | `true` if this is the currently active version |

**Response codes:**

| Code | Description |
|---|---|
| `201` | Context created successfully. New version activated. |
| `400` | Invalid request — missing required fields or invalid enum values |
| `401` | Unauthorised — invalid or missing API key |

---

### GET /context

Get the currently active context version.

:::note
When no context has been defined, this endpoint returns HTTP `404` with `error.type: not_found` and message `"No active application context exists."` This is expected behaviour — not an error condition. It means context has not been configured yet.
:::

**Response codes:**

| Code | Description |
|---|---|
| `200` | Active context returned |
| `404` | No active context defined. `error.type: not_found` |

---

### GET /context/versions

List all context versions, newest first.

**Response:** Array of ApplicationContext objects, ordered by `created_at` descending.

---

### GET /context/versions/{version}

Get a specific context version by version string (e.g. `v1`, `v2`).

**Path parameters:**

| Parameter | Type | Description |
|---|---|---|
| `version` | string | Version string — e.g. `v1`, `v2`, `v3` |

**Response codes:**

| Code | Description |
|---|---|
| `200` | Context version returned |
| `404` | Version not found |

---

### GET /context/impact

Shows how many tasks were compiled under older context versions — useful for identifying tasks that may benefit from recompilation under the current context.

**Response schema:**

| Field | Type | Description |
|---|---|---|
| `current_version` | string | The currently active context version |
| `tasks_on_current_version` | int | Tasks compiled under the current context version |
| `tasks_on_older_versions` | array | List of objects: `{ version, task_count }` for each older version that still has active tasks |
| `total_stale_tasks` | int | Total tasks not on the current context version |

---

## Effect on Task Creation

When context is active, the POST /tasks response includes two additional fields:

| Field | Type | Description |
|---|---|---|
| `context_version` | string \| null | The context version active when this task was created. `null` if no context was defined |
| `context_warning` | string \| null | Warning message if no context was defined at task creation time. `null` if context was present |

When no context is defined, the task is still created successfully — but the response will include:

```json
{
  "context_version": null,
  "context_warning": "No active application context exists. Task compiled without domain context — definitions may be less accurate. Define context via POST /context and consider recompiling this task."
}
```

---

## Effect on Calibration

If `calibration_bias` is defined in the active context, the calibration service uses it to adjust the statistically optimal threshold toward recall or precision before returning a recommendation.

The POST /conditions/calibrate response includes these fields when context is active:

| Field | Type | Description |
|---|---|---|
| `statistically_optimal` | float | The raw optimal value from feedback data alone |
| `context_adjusted` | float \| null | The bias-adjusted value after applying `calibration_bias`. `null` if no context or no bias defined |
| `recommended` | float | The final recommended value — equals `context_adjusted` if bias was applied, otherwise `statistically_optimal` |
| `adjustment_explanation` | string \| null | Human-readable explanation of the adjustment. `null` if no adjustment was made |

**Bias adjustment logic:**

- `bias_direction = recall` → threshold lowered. Adjustment: `high` = 10%, `medium` = 5%, `low` = 2%
- `bias_direction = precision` → threshold raised. Same adjustment factors apply
- `bias_direction = balanced` → no adjustment. `recommended = statistically_optimal`
- Adjusted values are always clamped to `[0.0, 1.0]`
- If `behavioural.meaningful_windows` is defined, any window parameter in the recommendation is clamped to the declared min/max range

**Example calibration response with context-adjusted recommendation:**

```json
{
  "condition_id": "churn.high_risk",
  "condition_version": "v3",
  "status": "recommendation_available",
  "statistically_optimal": 0.78,
  "context_adjusted": 0.702,
  "recommended": 0.702,
  "adjustment_explanation": "Threshold adjusted from 0.78 to 0.702 toward recall based on application context (false_negative_cost=high)",
  "calibration_token": "cal_abc123...",
  "feedback_count": 47,
  "false_positive_rate": 0.12,
  "false_negative_rate": 0.08
}
```

---

## Recommended Setup Order

Before creating your first task, follow this setup sequence:

1. **Define application context** — `POST /context`
2. **Register primitives** — `POST /definitions` (primitives)
3. **Create tasks** — `POST /tasks`
4. **Execute** — `POST /execute/full`

---

## Updating Context

When the domain understanding changes — a new regulatory framework applies, a new entity type is introduced, or the cost balance between false positives and false negatives shifts — post a new context. Each POST creates an immutable new version.

Existing tasks retain the `context_version` they were compiled under. Use `GET /context/impact` to identify tasks compiled under older versions. Recompile affected tasks to benefit from the updated context.

```bash
# Check how many tasks are on older context versions
GET /context/impact

# Response shows:
# { "current_version": "v2", "tasks_on_current_version": 14,
#   "tasks_on_older_versions": [{ "version": "v1", "task_count": 8 }],
#   "total_stale_tasks": 8 }
```

---

## Further Reading

- [Guardrails System](/docs/intro/guardrails) — how guardrails and context work together
- [Application Context Tutorial](/docs/tutorials/application-context) — worked examples across SaaS and fintech domains
- [Core Concepts](/docs/intro/core-concepts) — the ψ → φ → α model
