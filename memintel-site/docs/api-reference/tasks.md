---
id: tasks
title: Tasks
sidebar_label: Tasks
---

# Tasks

A Task is a version-pinned, immutable reference to a concept, condition, and action, created from natural language intent. Task logic cannot be mutated after creation — changes require the calibration flow and explicit rebinding.

:::warning Key Constraint
Tasks cannot be created without a valid action binding. If no action can be resolved, task creation fails with `action_binding_failed`.
:::

---

## Create a Task

```
POST /tasks
```

Submits a natural language intent through the full LLM + guardrails pipeline. The system classifies intent, resolves primitives, selects a strategy via the guardrails priority order, fills parameters, generates a concept and condition, binds an action, validates, compiles, and persists a version-pinned Task.

:::note Parameter Determinism
Parameter values are not inferred heuristically. They are derived deterministically from guardrails-defined priors and application instructions via the severity vocabulary and parameter bias rules. The **same intent + same guardrails always produces the same strategy and parameters.**
:::

:::tip dry_run
Pass `dry_run: true` to preview the generated concept, condition, action binding, and validation result without persisting. Use in development to verify intent resolution before committing.
:::

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `intent` | string | **Required** | Natural language monitoring intent. Example: `"Alert me when AAPL price rises significantly"`. |
| `entity_scope` | string | **Required** | Entity or group to evaluate. Examples: `"AAPL"`, `"user_abc123"`. |
| `delivery` | DeliveryConfig | **Required** | How to deliver alerts when the condition fires. |
| `constraints` | ConstraintsConfig | Optional | LLM hints: `sensitivity` (`low`\|`medium`\|`high`) and `namespace`. |
| `dry_run` | boolean | Optional | Default `false`. Returns `DryRunResult` without persisting. |

### Response — Task

| Parameter | Type | Required | Description |
|---|---|---|---|
| `task_id` | string | Always | Unique task identifier. |
| `intent` | string | Always | Original natural language intent. |
| `concept_id` / `concept_version` | string | Always | Pinned concept reference. |
| `condition_id` / `condition_version` | string | Always | Pinned condition reference. |
| `action_id` / `action_version` | string | Always | Pinned action binding. |
| `entity_scope` | string | Always | Entity scope. |
| `status` | enum | Always | `active` \| `paused` \| `deleted` \| `preview`. |
| `created_at` / `last_triggered_at` | datetime | Always | ISO 8601. `last_triggered_at` is `null` if never triggered. |
| `context_version` | string \| null | Always | The application context version active when this task was compiled. `null` if no context was defined at creation time. |
| `context_warning` | string \| null | Always | Warning message if no application context was defined at task creation time. `null` if context was present. When present, value is: `"No active application context exists. Task compiled without domain context — definitions may be less accurate. Define context via POST /context and consider recompiling this task."` |

### Response Codes

| Status | Description |
|---|---|
| **200** | Task created, or `DryRunResult` when `dry_run: true`. |
| **400** | Schema or constraint validation failed. |
| **401** | Missing or invalid API key. |
| **422** | Intent could not be resolved — no primitive, no valid strategy, or action binding failed. |
| **429** | Rate limit exceeded. |

### TypeScript Example

```typescript
// Dry run: verify guardrails-resolved strategy and params
const preview = await client.tasks.create({
  intent: "Alert me when AAPL price rises significantly",
  entityScope: "AAPL",
  delivery: { type: "notification" },
  dryRun: true,
});

// "significantly" → high severity → change.high prior → value: 0.10
console.log(preview.condition.strategy);
// { type: "change", params: { direction: "increase", value: 0.10, window: "1d" } }

const task = await client.tasks.create({
  intent: "Alert me when AAPL price rises significantly",
  entityScope: "AAPL",
  delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/alert" },
});
```

---

## List Tasks

```
GET /tasks
```

Returns a paginated list of tasks in the current workspace.

### Query Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `status` | enum | Optional | Filter: `active` \| `paused` \| `deleted`. |
| `limit` | integer | Optional | Default `20`, max `100`. |
| `cursor` | string | Optional | Pagination cursor. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Paginated `TaskList`: `items[]`, `has_more`, `next_cursor`, `total_count`. |
| **401** | Unauthorised. |

### TypeScript Example

```typescript
const { items } = await client.tasks.list({ status: "active", limit: 50 });
```

---

## Get Task

```
GET /tasks/{id}
```

Retrieves the full task definition including pinned concept, condition, and action references, plus execution metadata.

### Path Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `id` | string | **Required** | The `task_id`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Full Task object. |
| **401** | Unauthorised. |
| **404** | Task not found. |

### TypeScript Example

```typescript
const task = await client.tasks.get(taskId);
console.log(task.condition_id, task.condition_version);
```

---

## Update Task

```
PATCH /tasks/{id}
```

Updates operational settings. Cannot modify concept logic, condition logic, strategy, or action logic — those require the calibration flow.

:::note Rebinding
Passing `conditionVersion` rebinds the task to a new condition version — the final step of the calibration flow. Verify the new version with `GET /conditions/{id}` and `POST /conditions/explain` before rebinding.
:::

### Request Body (all optional, at least one required)

| Parameter | Type | Required | Description |
|---|---|---|---|
| `condition_version` | string | Optional | Rebind to this condition version after `apply-calibration`. |
| `delivery` | DeliveryConfig | Optional | Update alert delivery channel. |
| `entity_scope` | string | Optional | Update entity or group evaluated. |
| `status` | enum | Optional | `active` \| `paused`. |

**Disallowed (HTTP 400):** `concept_id`, `concept_version`, condition logic, strategy, parameters, `action_id`, `action_version`.

### Response Codes

| Status | Description |
|---|---|
| **200** | Updated Task. |
| **400** | Validation error or disallowed update. |
| **401** | Unauthorised. |
| **404** | Not found. |

### TypeScript Example

```typescript
await client.tasks.update(taskId, { status: "paused" });
await client.tasks.update(taskId, { conditionVersion: "1.3" });
await client.tasks.update(taskId, { entityScope: "premium_users" });
```

---

## Delete Task

```
DELETE /tasks/{id}
```

Soft-deletes a task (`status = deleted`). Retained for audit and replay. Hard deletion is not supported.

### Response Codes

| Status | Description |
|---|---|
| **200** | Soft-deleted Task. |
| **401** | Unauthorised. |
| **404** | Not found. |

### TypeScript Example

```typescript
const deleted = await client.tasks.delete(taskId);
console.log(deleted.status); // "deleted"
```
