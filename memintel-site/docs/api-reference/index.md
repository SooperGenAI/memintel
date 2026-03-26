---
id: overview
title: App Developer API Reference
sidebar_label: Overview
slug: /api-reference/overview
---

# App Developer API Reference

Complete reference for the Memintel REST API — Tasks, Execution, Conditions, Decisions, Feedback.

| | |
|---|---|
| **Base URL** | `https://api.memsdl.ai/v1` |
| **Auth** | `X-API-Key: <your-key>` (header) |
| **SDK** | `@memintel/sdk` (TypeScript) |
| **Version** | 2.1 |

---

## Design Philosophy

Three principles inform every endpoint:

- **Determinism first.** Every execution is reproducible. Provide a `timestamp` and the same input always returns the same output.
- **Explicit versioning.** No endpoint resolves to `"latest"`. Every request pins to a specific definition version.
- **Separation of concerns.** The LLM operates only at task creation. Everything after compilation is deterministic — no LLM involvement.

---

## The Pipeline

Every decision in Memintel flows through this sequence:

```
Intent        →  Guardrails           →  Concept (ψ)       →  Condition (φ)          →  Action (α)
POST /tasks      strategy + params       compute meaning       evaluate via strategy      trigger
```

**Feedback loop:**
```
POST /feedback/decision
  → POST /conditions/calibrate
  → POST /conditions/apply-calibration
  → PATCH /tasks/{id}
```

:::info LLM Boundary
The LLM is active **only** during the Intent → Task step. Concept execution, condition evaluation, action triggering, calibration, and feedback are all deterministic. The LLM is never invoked after a task is compiled.
:::

---

## Guardrails System

All strategy selection, parameter resolution, and constraint enforcement is governed by a guardrails configuration (`memintel.guardrails.md`).

:::note Core Principle
The LLM does not freely choose strategies or thresholds. It resolves strategy and parameters within deterministic guardrails that define every valid strategy, every valid type-strategy pairing, every parameter prior, and every resolution rule. The **same intent + same guardrails always produces the same strategy and parameters.**
:::

Guardrails define five things:

- **Strategy registry.** Every strategy (`threshold`, `percentile`, `change`, `z_score`, `equals`, `composite`) is a versioned object with a declared input type, parameter schema, and output type. The LLM can only select strategies from this registry.
- **Type-strategy compatibility.** A compatibility map enforces which strategies are valid for each primitive type. `float` accepts `threshold` and `percentile` but not `change` or `z_score` (which require `time_series` input). Incompatible pairings are rejected at compile time.
- **Parameter priors and bounds.** Default threshold values are declared per strategy and per severity level (`low`, `medium`, `high`). Hard bounds define the maximum range calibration can reach. Priors can be crossed; bounds cannot.
- **Deterministic parameter bias rules.** Application context instructions are mapped to severity shifts via explicit lookup rules — not LLM interpretation. For example, `"early detection"` maps to `severity_shift: -1`, selecting the prior one tier lower. This is deterministic.
- **Strategy selection priority.** When multiple strategies are valid, resolution follows a strict priority order.

**Guardrails scope:** Guardrails influence task generation only (`POST /tasks`). Zero influence on execution, calibration, feedback, or explanation values.

### Strategy Selection Priority

| Priority | Source | Description |
|---|---|---|
| 1 (highest) | `user_explicit` | Threshold or strategy explicitly provided by the user. Always wins. Bias rules never applied. |
| 2 | `primitive_hint` | Strategy hints declared on the primitive in guardrails. Bias rules applied to prior lookup. |
| 3 | `mapping_rule` | Intent pattern matched to a strategy (e.g. `"rises"` → `change`, `"unusual"` → `z_score`). |
| 4 | `application_context` | Strategy bias from domain instructions in the application context block. |
| 5 | `global_preferred` | Globally preferred strategies declared in guardrails. |
| 6 (fallback) | `global_default` | Global threshold priors. Parameter bias rules applied at this step. |

---

## Terminology

| Term | Description |
|---|---|
| **Result (Rₜ)** | The typed output of concept execution (ψ) — the meaning layer. |
| **Decision (Aₜ)** | The output of condition evaluation (φ) — `true`/`false` or a label, plus `actions_triggered[]`. |
| **Concept (ψ)** | Defines what is computed from state. |
| **Condition (φ)** | Defines when that meaning is significant, using a strategy and parameters. |
| **Action (α)** | Defines what happens when the condition fires. |
| **Strategy** | The evaluation logic template applied by a condition (`threshold`, `percentile`, `change`, `z_score`, `equals`, `composite`). |

---

## Authentication

```typescript
const headers = {
  "X-API-Key": process.env.MEMINTEL_API_KEY,
  "Content-Type": "application/json"
};
```

---

## Idempotency

`POST` requests to `/tasks`, `/evaluate/full`, `/conditions/calibrate`, and `/conditions/apply-calibration` accept an optional `Idempotency-Key` header. The same key within 24 hours returns the cached response.

```typescript
headers["Idempotency-Key"] = crypto.randomUUID();
```

---

## Rate Limiting

Responses include `X-RateLimit-Limit` and `X-RateLimit-Remaining`. HTTP `429` includes a `Retry-After` header.
