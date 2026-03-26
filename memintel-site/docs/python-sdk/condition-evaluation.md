---
id: condition-evaluation
title: Condition Evaluation
sidebar_label: Condition Evaluation
---

# Condition Evaluation

Endpoints that evaluate a Condition (φ) against a concept's Result (Rₜ) and produce a DecisionResult (Aₜ). All are deterministic when `timestamp` is provided.

:::info Implicit Concept Execution
`POST /evaluate/condition` and `POST /evaluate/full` execute the bound concept internally if its Result is not already cached. This is transparent but incurs latency on first call. Pre-warm with `POST /execute` if latency matters.
:::

---

## Evaluate Condition

```
POST /evaluate/condition
```

Evaluates a single Condition against its bound concept for a given entity. Returns a `DecisionResult` (Aₜ) that includes the decision value and any actions triggered.

### Request

| Field | Type | Required | Description |
|---|---|---|---|
| `condition_id` | str | **Required** | Fully qualified condition id (`namespace.id`). |
| `condition_version` | str | **Required** | Explicit version. |
| `entity` | str | **Required** | Entity to evaluate. |
| `timestamp` | str | Optional | ISO 8601 UTC. Deterministic when provided. |
| `dry_run` | bool | Optional | Default `False`. Evaluates but does not fire actions. |

### Response — DecisionResult (Aₜ)

| Field | Type | Required | Description |
|---|---|---|---|
| `value` | bool\|str | Always | Decision value. `True`/`False` for boolean strategies; matched label string for `equals`. |
| `type` | str | Always | `'boolean'` \| `'categorical'`. |
| `entity` | str | Always | Entity evaluated. |
| `condition_id` | str | Always | Condition identifier. |
| `condition_version` | str | Always | Condition version used. |
| `timestamp` | str\|None | Always | Evaluation timestamp. |
| `actions_triggered` | list[ActionTriggered] | Always | Actions fired. Each has `action_id`, `action_version`, `status`, `payload_sent`, `error?`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | DecisionResult (Aₜ). Action failures return 200 — inspect `actions_triggered[].status`. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **404** | Condition not found. |
| **408** | Timeout. |
| **422** | Execution error. |
| **429** | Rate limit. |

---

## Evaluate Condition Batch

```
POST /evaluate/condition/batch
```

Evaluates a single condition across multiple entities in parallel. Returns a list of `DecisionResult` objects, one per entity. Always HTTP 200 — inspect each item's `status`.

### Request

| Field | Type | Required | Description |
|---|---|---|---|
| `condition_id` | str | **Required** | Fully qualified condition id. |
| `condition_version` | str | **Required** | Explicit version. |
| `entities` | list[str] | **Required** | Entity IDs to evaluate. |
| `timestamp` | str | Optional | Shared evaluation timestamp. |

### Response Codes

| Status | Description |
|---|---|
| **200** | `list[DecisionResult]`. Always 200 — inspect each item. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **429** | Rate limit. |

---

## Execute Full Pipeline

```
POST /evaluate/full
```

Runs concept execution (ψ), condition evaluation (φ), and action triggering (α) in one atomic call. The default SDK path for production use cases.

### Request — EvaluateFullRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `concept_id` | str | **Required** | Fully qualified concept id. |
| `concept_version` | str | **Required** | Explicit concept version. |
| `condition_id` | str | **Required** | Fully qualified condition id. |
| `condition_version` | str | **Required** | Explicit condition version. |
| `entity` | str | **Required** | Entity to evaluate. |
| `timestamp` | str | Optional | ISO 8601 UTC. Deterministic when provided. |
| `explain` | bool | Optional | Default `False`. Include `Explanation` in result. |
| `dry_run` | bool | Optional | Default `False`. Simulates without firing actions. |
| `missing_data_policy` | str | Optional | `'null'` \| `'zero'` \| `'forward_fill'` \| `'backward_fill'`. |

### Response — FullPipelineResult

| Field | Type | Required | Description |
|---|---|---|---|
| `result` | Result (Rₜ) | Always | Concept execution output. |
| `decision` | DecisionResult (Aₜ) | Always | Condition evaluation output. Contains `actions_triggered[]`. |
| `dry_run` | bool | Optional | Reflects `dry_run` from request. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Pipeline executed. Action failures return 200 — inspect `decision.actions_triggered[]`. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **404** | Not found. |
| **408** | Timeout. |
| **422** | Execution error. |
| **429** | Rate limit. |
