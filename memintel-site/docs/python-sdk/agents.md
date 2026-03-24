---
id: agents
title: Agents
sidebar_label: Agents
---

# Agents

:::warning Non-Determinism Notice
Agent endpoints use the LLM to generate definitions. Output varies between calls. All generated output must pass the deterministic compiler before registration. Agents are **never** on the critical execution path.
:::

---

## Generate Definition

```
POST /agents/define
```

Converts a natural language description into a validated MemSDL definition. The returned definition is compiler-validated but **not** registered or compiled. Call `POST /definitions/validate`, then `POST /registry/definitions`, then `POST /compile` before executing.

### Request — AgentDefineRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `description` | str | **Required** | Natural language description of the concept or primitive. |
| `context.domain` | str | Optional | Domain hint. Examples: `finance`, `saas`, `fraud_detection`. |
| `context.preferred_primitives` | list | Optional | Existing primitive IDs to prefer as inputs. |

### Response — AgentDefineResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `definition` | Concept\|Primitive | Always | Generated and compiler-validated definition. Not yet registered. |
| `validation` | ValidationResult | Always | Compiler validation result. Check `valid=True` before proceeding. |
| `warnings` | list | Optional | Non-blocking LLM notes — ambiguities resolved, default mappings used. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Definition generated. |
| **400** | Validation error. |
| **401** | Unauthorised. |

---

## Semantic Refinement

```
POST /agents/semantic-refine
```

Closes the LLM generation loop. Takes an existing Concept definition, its `SemanticGraph` from `POST /compile/semantic`, and a refinement instruction. Returns a revised definition with updated semantic view and a diff showing meaning changes. Repeat until `convergence_delta` reaches `0.0`.

### Request — SemanticRefineRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `definition` | dict | **Required** | Current Concept definition to refine. |
| `semantic_view` | dict | **Required** | Current SemanticGraph from `POST /compile/semantic`. |
| `instruction` | str | **Required** | Refinement instruction. Example: `'Add volatility feature weighted at 30%'`. |
| `target_meaning_hash` | str | Optional | `semantic_hash` to converge toward. `convergence_delta` tracks distance. |
| `iteration` | int | Optional | Current iteration number (1-indexed). |

### Response — SemanticRefineResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `definition` | Concept | Always | The refined Concept. |
| `semantic_view` | SemanticGraph | Always | Updated semantic view. Input for the next iteration. |
| `validation` | ValidationResult | Always | Check `valid=True` before proceeding to next iteration. |
| `convergence_delta` | float | Optional | `0.0` = converged to `target_meaning_hash`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Refined definition. |
| **400** | Validation error. |
| **401** | Unauthorised. |
