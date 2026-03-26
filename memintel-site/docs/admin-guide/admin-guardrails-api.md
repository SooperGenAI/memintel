---
id: admin-guardrails-api
title: Step 2A — Guardrails via API
sidebar_label: Step 2A — Guardrails (API)
---

# Step 2A — Managing Guardrails via API

This is the recommended way to manage guardrails after your first deployment. Changes take effect immediately — no server restart required — and every change creates a versioned, auditable record.

:::tip Why use the API instead of the file?
- Takes effect **immediately** — no server restart needed
- Full **version history** — every change is recorded with a timestamp and optional change note
- Works from **anywhere** — no need to access the server filesystem
- Enables **UI-driven management** — can be integrated into an admin dashboard
:::

---

## What You Need

Before posting guardrails via API you need:

1. Your Memintel domain URL (e.g. `https://your-memintel-domain`)
2. Your standard API key
3. Your **elevated key** (`MEMINTEL_ELEVATED_KEY`) — required for `POST /guardrails` only. Ask your data engineer for this value.

---

## Step 1 — Write Your Guardrails

Create a file called `guardrails.json` on your local machine. This contains the same policy you would otherwise put in `memintel_guardrails.yaml` — but in JSON format.

The full schema is documented in the [Guardrails API Reference](/docs/api-reference/guardrails). A working example:

```json
{
  "guardrails": {
    "strategy_registry": [
      "threshold", "percentile", "z_score", "change", "equals", "composite"
    ],
    "type_strategy_map": {
      "float":              ["threshold", "percentile", "z_score", "change"],
      "int":                ["threshold", "percentile", "change"],
      "boolean":            ["equals"],
      "categorical":        ["equals"],
      "time_series<float>": ["z_score", "change", "percentile"],
      "float?":             ["threshold"]
    },
    "parameter_priors": {
      "account.active_user_rate_30d": {
        "low_severity":    { "value": 0.60 },
        "medium_severity": { "value": 0.45 },
        "high_severity":   { "value": 0.30 }
      },
      "account.days_to_renewal": {
        "low_severity":    { "value": 90 },
        "medium_severity": { "value": 60 },
        "high_severity":   { "value": 30 }
      }
    },
    "bias_rules": {
      "urgent":       "high_severity",
      "critical":     "high_severity",
      "significant":  "medium_severity",
      "material":     "medium_severity",
      "early":        "low_severity",
      "approaching":  "low_severity"
    },
    "threshold_directions": {
      "account.active_user_rate_30d": "below",
      "account.days_to_renewal":      "below"
    },
    "global_preferred_strategy": "percentile",
    "global_default_strategy":   "threshold"
  },
  "change_note": "Initial guardrails — SaaS churn detection"
}
```

See [Step 2B — Guardrails via File](/docs/admin-guide/admin-guardrails) for the full guide on what each field means and how to set the right values for your domain.

---

## Step 2 — Post the Guardrails

```bash
curl -X POST https://your-memintel-domain/guardrails \
  -H "Content-Type: application/json" \
  -H "X-Elevated-Key: your-elevated-key" \
  -d @guardrails.json
```

Expected response:

```json
{
  "guardrails_id": "grls_8f3k2m...",
  "version": "v1",
  "is_active": true,
  "source": "api",
  "change_note": "Initial guardrails — SaaS churn detection",
  "created_at": "2026-03-27T10:00:00Z"
}
```

`"version": "v1"` and `"is_active": true` confirm the guardrails are live. The server has already reloaded them into memory — they are in effect right now.

---

## Step 3 — Verify the Active Version

```bash
curl https://your-memintel-domain/guardrails
```

This returns the currently active guardrails version. If you see your `guardrails_id` and `"is_active": true`, you are done.

---

## Updating Guardrails

Every time you post new guardrails, a new immutable version is created and becomes active. The previous version is deactivated but never deleted.

```bash
# Edit guardrails.json locally, then post again
curl -X POST https://your-memintel-domain/guardrails \
  -H "Content-Type: application/json" \
  -H "X-Elevated-Key: your-elevated-key" \
  -d @guardrails.json
```

The response will show `"version": "v2"`.

---

## Viewing Version History

```bash
curl https://your-memintel-domain/guardrails/versions
```

Returns all versions, newest first. Each entry shows when it was created, who created it (via the `change_note`), and whether it is currently active.

---

## Checking Which Tasks Are on Older Versions

After updating guardrails, use `GET /guardrails/impact` to see how many tasks were compiled under older versions:

```bash
curl https://your-memintel-domain/guardrails/impact
```

```json
{
  "current_version": "v2",
  "tasks_on_current_version": 14,
  "tasks_on_older_versions": [{ "version": "v1", "task_count": 8 }],
  "total_stale_tasks": 8
}
```

Tasks on older versions continue running correctly — they are pinned to their compiled guardrails version. But they will not benefit from any new `parameter_priors` or `bias_rules` you added. Ask your data engineer to recompile affected tasks to use the new version.

---

## Task Provenance

Every task records which guardrails version was active when it was compiled:

| Field | What it records |
|---|---|
| `context_version` | Which application context was active at task creation |
| `guardrails_version` | Which guardrails version was active at task creation. `null` if file-based guardrails were in use. |
| `context_warning` | Warning if no application context was defined |

---

## File vs API Precedence

| Scenario | Guardrails in use |
|---|---|
| Fresh deployment, no `POST /guardrails` called yet | `memintel_guardrails.yaml` loaded from file |
| `POST /guardrails` called at least once | Most recent API version — file is ignored |
| Server restarted after API post | API version reloaded from database — file still ignored |

The file is the seed and fallback. The API is the override.

---

## Next Step

→ [Step 3: Validate with your data engineer](/docs/intro/self-hosting#step-6----smoke-test)
