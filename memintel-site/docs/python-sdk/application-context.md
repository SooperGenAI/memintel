---
id: application-context
title: Application Context
sidebar_label: Application Context
---

# Application Context

Endpoints for managing application context — the domain briefing that guides the LLM compiler when resolving user intent into concept and condition definitions. Context is versioned and takes effect immediately on creation. No server restart required.

---

## Create Context

```
POST /context
```

Creates a new application context version and immediately activates it. Deactivates the previous version atomically. The LLM compiler uses the active context for all subsequent task compilations.

### Request — CreateContextRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `domain` | dict | **Required** | Domain description object. Must include `description` (str). Optional: `entities` (list), `decisions` (list[str]). |
| `domain.description` | str | **Required** | Natural language description of what the application does and what domain it operates in. |
| `domain.entities` | list[dict] | Optional | List of entity declarations. Each has `name` (str) and `description` (str). |
| `domain.decisions` | list[str] | Optional | Decision type names relevant to this domain (e.g. `churn_risk`, `fraud_alert`). |
| `behavioural` | dict | Optional | Behavioural settings. See fields below. |
| `behavioural.data_cadence` | str | Optional | `'batch'` \| `'streaming'` \| `'mixed'`. Default `'batch'`. |
| `behavioural.meaningful_windows` | dict | Optional | `{ 'min': str, 'max': str }` — e.g. `{ 'min': '30d', 'max': '90d' }`. Informs calibration window clamping. |
| `behavioural.regulatory` | list[str] | Optional | Regulatory frameworks in scope (e.g. `['GDPR', 'SOC2', 'HIPAA']`). |
| `semantic_hints` | list[dict] | Optional | Domain-specific term definitions. Each has `term` (str) and `definition` (str). |
| `calibration_bias` | dict | Optional | Calibration cost asymmetry. See fields below. |
| `calibration_bias.false_negative_cost` | str | Optional | `'high'` \| `'medium'` \| `'low'`. Cost of missing a true positive. |
| `calibration_bias.false_positive_cost` | str | Optional | `'high'` \| `'medium'` \| `'low'`. Cost of a false alarm. |

`bias_direction` is auto-derived and cannot be set manually. `false_negative_cost > false_positive_cost` → `recall`. Reverse → `precision`. Equal → `balanced`.

### Response — ApplicationContext

| Field | Type | Description |
|---|---|---|
| `context_id` | str (UUID) | Auto-generated unique identifier. |
| `version` | str | Auto-assigned version string: `v1`, `v2`, `v3`... |
| `domain` | dict | The domain context as submitted. |
| `behavioural` | dict \| None | The behavioural context as submitted. |
| `semantic_hints` | list | The semantic hints as submitted. |
| `calibration_bias` | dict \| None | The calibration bias including auto-derived `bias_direction`. |
| `created_at` | str | ISO 8601 UTC timestamp. |
| `is_active` | bool | `True` — newly created versions are always immediately active. |

### Response Codes

| Status | Description |
|---|---|
| **201** | Context created and active. |
| **400** | Validation error — missing required field or invalid enum value. |
| **401** | Unauthorised — missing or invalid `X-API-Key`. |

### Python Example

```python
import memintel

client = memintel.AsyncClient()

context = await client.context.create(
    domain={
        "description": "B2B SaaS churn detection for mid-market software companies.",
        "entities": [
            {"name": "account", "description": "company-level subscription"},
            {"name": "user",    "description": "individual platform user"},
        ],
        "decisions": ["churn_risk", "expansion_opportunity"],
    },
    behavioural={
        "data_cadence": "batch",
        "meaningful_windows": {"min": "30d", "max": "90d"},
        "regulatory": ["GDPR", "SOC2"],
    },
    semantic_hints=[
        {"term": "active user",        "definition": "logged in AND performed core action in last 14 days"},
        {"term": "high value account", "definition": "ARR above $50,000"},
    ],
    calibration_bias={
        "false_negative_cost": "high",
        "false_positive_cost": "medium",
    },
)

print(context.version)    # v1
print(context.is_active)  # True
# context.calibration_bias.bias_direction → 'recall' (auto-derived)
```

---

## Get Active Context

```
GET /context
```

Returns the currently active context version. Returns HTTP 404 if no context has been defined.

### Response Codes

| Status | Description |
|---|---|
| **200** | Active ApplicationContext returned. |
| **401** | Unauthorised. |
| **404** | No active context defined. `error.type: not_found`. |

### Python Example

```python
try:
    context = await client.context.get_active()
    print(context.version)      # v2
    print(context.is_active)    # True
except memintel.NotFoundError:
    print("No context defined yet")
```

---

## List Context Versions

```
GET /context/versions
```

Returns all context versions, newest first. Useful for auditing the history of domain definition changes.

### Response

`list[ApplicationContext]` ordered by `created_at` descending.

### Python Example

```python
versions = await client.context.list_versions()

for v in versions:
    print(f"{v.version} — {'active' if v.is_active else 'inactive'} — {v.created_at}")
# v2 — active   — 2026-03-27T10:00:00Z
# v1 — inactive — 2026-03-01T09:00:00Z
```

---

## Get Context Version

```
GET /context/versions/{version}
```

Returns a specific context version by version string.

### Path Parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `version` | str | **Required** | Version string — e.g. `'v1'`, `'v2'`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | ApplicationContext for the requested version. |
| **401** | Unauthorised. |
| **404** | Version not found. |

### Python Example

```python
v1 = await client.context.get_version("v1")
print(v1.domain["description"])
print(v1.is_active)  # False — superseded by v2
```

---

## Get Context Impact

```
GET /context/impact
```

Returns a summary of how many tasks were compiled under older context versions. Use after updating context to identify tasks that would benefit from recompilation.

### Response — ContextImpactResult

| Field | Type | Description |
|---|---|---|
| `current_version` | str | The currently active context version. |
| `tasks_on_current_version` | int | Tasks compiled under the current version. |
| `tasks_on_older_versions` | list[dict] | List of `{ version, task_count }` for each older version with active tasks. |
| `total_stale_tasks` | int | Total tasks not on the current context version. |

### Response Codes

| Status | Description |
|---|---|
| **200** | ContextImpactResult. |
| **401** | Unauthorised. |

### Python Example

```python
impact = await client.context.get_impact()

print(f"Current version: {impact.current_version}")
print(f"Tasks on current: {impact.tasks_on_current_version}")
print(f"Stale tasks: {impact.total_stale_tasks}")

for entry in impact.tasks_on_older_versions:
    print(f"  {entry['version']}: {entry['task_count']} tasks need recompilation")
```
