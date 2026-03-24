---
id: python-overview
title: Python Backend API Reference
sidebar_label: Overview
slug: /python-sdk/python-overview
---

# Python Backend API Reference

FastAPI · Pydantic v2 · asyncio — Server-side endpoint reference.

| | |
|---|---|
| **Framework** | FastAPI + Pydantic v2 |
| **Field naming** | `snake_case` (all models and API fields) |
| **Base URL** | `https://api.memsdl.ai/v1` |
| **Auth** | `X-API-Key: <your-key>` |

---

## Overview

The Memintel Python backend is the intelligence engine — a FastAPI application that owns concept execution, condition evaluation, compilation, the registry, the async job queue, LLM-assisted authoring, and the full Interaction API. All callers are services, never humans. All field names are `snake_case`.

**Two endpoint groups:**

- **Execution & Engine** — concept execution, condition evaluation, compilation, registry, async jobs, agents. Called by the TypeScript SDK and internal services.
- **Interaction API** — tasks, conditions, calibration, feedback, decisions. Called by the TypeScript SDK on behalf of app developers.

:::note Determinism Rule
All execution endpoints are fully deterministic when `timestamp` is provided. The same `(id, version, entity, timestamp)` always returns the same result. The LLM is invoked **only** in `POST /tasks` and agent endpoints — never in the execution path.
:::

---

## Condition Strategies

Python is the **only** place condition strategies are implemented. All six strategies are required. TypeScript must never contain strategy logic.

| Strategy | Key Param | Direction Values | Output Type |
|---|---|---|---|
| `threshold` | `'value': float` | `'above'` \| `'below'` | `decision<boolean>` |
| `percentile` | `'value': float` 0–100 | `'top'` \| `'bottom'` | `decision<boolean>` |
| `z_score` | `'threshold': float` | `'above'` \| `'below'` \| `'any'` | `decision<boolean>` |
| `change` | `'value': float` | `'increase'` \| `'decrease'` \| `'any'` | `decision<boolean>` |
| `equals` | `'value': string` | N/A — categorical match | `decision<categorical>` |
| `composite` | `operands: list` | `'AND'` \| `'OR'` | `decision<boolean>` |

:::note Equals Calibration
Conditions using the `equals` strategy always return `status='no_recommendation'` with `reason='not_applicable_strategy'` from `POST /conditions/calibrate`. There is no numeric parameter to adjust for categorical conditions.
:::

---

## Pydantic Model Conventions

Every request and response must be a Pydantic v2 model. Required fields have no default. Optional fields declare a default of `None` or a sensible value.

```python
from pydantic import BaseModel

class ExampleRequest(BaseModel):
    id: str               # required — no default
    version: str          # required — no default
    entity: str           # required — no default
    timestamp: str | None = None  # optional — always has a default
    explain: bool = False         # optional with sensible default
    cache: bool = True
```
