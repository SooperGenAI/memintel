---
id: concept-execution
title: Concept Execution
sidebar_label: Concept Execution
---

# Concept Execution

Endpoints that evaluate a concept against an entity and return a Result (Rₜ). All are deterministic when `timestamp` is provided. The LLM is never involved.

---

## Execute Concept

```
POST /execute
```

Evaluates a compiled concept for a single entity. The primary execution endpoint for the SDK hot path. Implicitly executes the concept graph, fetches primitives via the data resolution layer, and returns a typed Result (Rₜ).

### Request — ExecuteRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | **Required** | Fully qualified concept id (`namespace.id`). |
| `version` | str | **Required** | Explicit version. HTTP 400 if missing or `'latest'`. |
| `entity` | str | **Required** | Entity to evaluate. |
| `timestamp` | str | Optional | ISO 8601 UTC. Deterministic execution when set. Snapshot mode when absent. |
| `explain` | bool | Optional | Default `False`. Include `Explanation` in response. |
| `explain_mode` | str | Optional | `'summary'` \| `'full'` \| `'debug'`. Default `'full'`. |
| `cache` | bool | Optional | Default `True`. Set `False` to force recomputation. |
| `missing_data_policy` | str | Optional | `'null'` \| `'zero'` \| `'forward_fill'` \| `'backward_fill'`. |

### Response — Result (Rₜ)

| Field | Type | Required | Description |
|---|---|---|---|
| `value` | float\|bool\|str | Always | Computed output. Type matches the concept's declared output type. |
| `type` | str | Always | `'float'` \| `'boolean'` \| `'categorical'`. |
| `entity` | str | Always | The entity evaluated. |
| `version` | str | Always | Concept version used. |
| `deterministic` | bool | Always | `True` when `timestamp` was provided. |
| `timestamp` | str\|None | Always | Echo of the request timestamp, or `None` in snapshot mode. |
| `explanation` | dict\|None | Optional | Populated when `explain=True`. Contains `contributions`, `nodes[]`, `trace[]`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Result (Rₜ) produced. |
| **400** | Validation error — missing required field, or `version='latest'` rejected. |
| **401** | Unauthorised — missing or invalid `X-API-Key`. |
| **404** | Concept not found at this id and version. |
| **408** | Execution timed out (30s). Switch to `POST /execute/async` for heavy workloads. |
| **422** | Execution error — primitive fetch failed, null propagation, or data source unavailable. |
| **429** | Rate limit exceeded. Respect the `Retry-After` header. |

### Python Example

```python
import memintel

client = memintel.AsyncClient()

result = await client.execute(
    id='org.churn_risk',
    version='1.2',
    entity='user_abc123',
    timestamp='2024-03-15T09:00:00Z',
    explain=True,
)

print(result.value)         # 0.87
print(result.deterministic) # True
```

---

## Execute via Precompiled Graph

```
POST /execute/graph
```

Executes a previously compiled `ExecutionGraph` directly by `graph_id`, bypassing the compilation step. This is the highest-throughput execution path — compile once at startup, call this in the hot path. Optionally supply `ir_hash` to verify the graph has not changed since compilation (audit trail mechanism).

### Request — ExecuteGraphRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `graph_id` | str | **Required** | ID returned by `POST /compile`. |
| `entity` | str | **Required** | Entity to evaluate. |
| `ir_hash` | str | Optional | If provided, server verifies stored graph's `ir_hash` matches. Returns HTTP 409 on mismatch. |
| `timestamp` | str | Optional | ISO 8601 UTC. Deterministic when provided. |
| `explain` | bool | Optional | Default `False`. |
| `cache` | bool | Optional | Default `True`. |
| `missing_data_policy` | str | Optional | `'null'` \| `'zero'` \| `'forward_fill'` \| `'backward_fill'`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Result (Rₜ) — same schema as `POST /execute`. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **404** | Graph not found. |
| **408** | Execution timed out. |
| **409** | `ir_hash` mismatch — stored graph has changed. Re-compile and update your `graph_id`. |
| **422** | Execution error. |
| **429** | Rate limit. |

---

## Execute Batch

```
POST /execute/batch
```

Evaluates a concept for a list of entities in parallel. Independent entities are evaluated concurrently. Shared subgraph computations are deduplicated. Partial failure is possible — always inspect each item's `status` field. The response is always HTTP 200.

### Request — BatchExecuteRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | **Required** | Fully qualified concept id. |
| `version` | str | **Required** | Explicit version. |
| `entities` | list[str] | **Required** | Entity IDs to evaluate in parallel. Max 500 per batch. |
| `timestamp` | str | Optional | Shared evaluation timestamp. All results deterministic when set. |
| `explain` | bool | Optional | Default `False`. |
| `missing_data_policy` | str | Optional | `'null'` \| `'zero'` \| `'forward_fill'` \| `'backward_fill'`. |

### Response — BatchExecuteResult

| Field | Type | Required | Description |
|---|---|---|---|
| `results` | list[BatchResultItem] | Always | One item per entity in request order. Each has `entity`, `value`, `status` (`'ok'`\|`'error'`), `error?`. |
| `succeeded` | int | Always | Count of successfully evaluated entities. |
| `failed` | int | Always | Count of failed entities. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Always 200. Inspect `results[].status` per entity. |
| **400** | Validation error on the batch request itself. |
| **401** | Unauthorised. |
| **404** | Concept not found. |
| **429** | Rate limit. |

---

## Execute Time Range

```
POST /execute/range
```

Evaluates a concept at each point in a time range for a single entity. Produces a time-ordered array of Result objects. All results use deterministic (timestamped) execution — `result.deterministic` is always `True`.

### Request — ExecuteRangeRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | **Required** | Fully qualified concept id. |
| `version` | str | **Required** | Explicit version. |
| `entity` | str | **Required** | Entity to evaluate across the time range. |
| `start` | str | **Required** | Range start timestamp (ISO 8601 UTC, inclusive). |
| `end` | str | **Required** | Range end timestamp (ISO 8601 UTC, inclusive). |
| `interval` | str | Optional | Evaluation interval. Examples: `'1d'`, `'1h'`, `'7d'`. Defaults to daily. |
| `missing_data_policy` | str | Optional | `'null'` \| `'zero'` \| `'forward_fill'` \| `'backward_fill'`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Time-ordered `list[Result]`. All have `deterministic=True`. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **404** | Not found. |
| **408** | Timeout — use `POST /execute/async` for large ranges. |
| **422** | Execution error. |
| **429** | Rate limit. |

---

## Execute Async

```
POST /execute/async
```

Submits a concept execution as an async job. Returns HTTP 202 immediately with a Job object. Poll `GET /jobs/{job_id}` for status and result. Cancel with `DELETE /jobs/{job_id}`. Use for workloads expected to exceed 30 seconds.

**Request:** Same schema as `POST /execute` (ExecuteRequest).

### Response — Job

| Field | Type | Required | Description |
|---|---|---|---|
| `job_id` | str | Always | Unique job identifier. |
| `status` | str | Always | `'queued'` \| `'running'` \| `'completed'` \| `'failed'` \| `'cancelled'`. |
| `poll_interval_seconds` | int | Optional | Suggested polling interval in seconds. |

### Response Codes

| Status | Description |
|---|---|
| **202** | Job accepted. Poll `/jobs/{job_id}` for result. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **404** | Concept not found. |
| **429** | Rate limit. |

---

## Get Async Job

```
GET /jobs/{job_id}
```

Polls a job for its current status and result.

### Path Parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `job_id` | str | **Required** | `job_id` returned by `POST /execute/async`. |

### Response — JobResult

| Field | Type | Required | Description |
|---|---|---|---|
| `job_id` | str | Always | Job identifier. |
| `status` | str | Always | `'queued'` \| `'running'` \| `'completed'` \| `'failed'` \| `'cancelled'`. |
| `result` | Result | Optional | Populated when `status='completed'`. |
| `error` | dict | Optional | Populated when `status='failed'`. Has `type`, `message`, `location`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | JobResult. |
| **401** | Unauthorised. |
| **404** | Job not found. |

---

## Cancel Async Job

```
DELETE /jobs/{job_id}
```

Cancels a job in `'queued'` or `'running'` state. Returns HTTP 409 if the job has already completed or failed.

### Response Codes

| Status | Description |
|---|---|
| **200** | Job cancelled. Returns final JobResult with `status='cancelled'`. |
| **401** | Unauthorised. |
| **404** | Job not found. |
| **409** | Cannot cancel — job already completed or failed. |
