---
id: errors
title: Error Handling
sidebar_label: Error Handling
---

# Error Handling

All errors return an `ErrorResponse` with a typed `error.type` field. Always branch on `error.type` — never on `error.message`, which may change between versions.

---

## Error Response Shape

```typescript
interface ErrorResponse {
  error: {
    type: ErrorType;      // machine-readable, stable
    message: string;      // human-readable
    location?: string;    // where the error occurred
    suggestion?: string;  // actionable fix hint
  };
}
```

---

## Error Types

| Type | Category | Description |
|---|---|---|
| `type_error` | Type system | Invalid input type or incompatible type-strategy pairing. |
| `semantic_error` | Compiler | Definition fails semantic validation. |
| `reference_error` | Compiler | Unknown primitive, feature, or concept reference. |
| `parameter_error` | Validation | Invalid parameter value. |
| `graph_error` | Compiler | Circular dependency in DAG. |
| `execution_error` | Runtime | Concept compiled but runtime failed. |
| `execution_timeout` | Runtime | Exceeded 30-second synchronous timeout. |
| `auth_error` | Auth | Missing or invalid API key. |
| `not_found` | Registry | Resource not found at given id and version. |
| `conflict` | Registry | Definition already exists at this id + version. |
| `rate_limit_exceeded` | Rate limiting | Too many requests. Respect `Retry-After`. |
| `bounds_exceeded` | Calibration | Recommendation would exceed guardrails `threshold_bounds`. |
| `action_binding_failed` | Task creation | No valid action resolved during task creation. |

---

## TypeScript Error Handling

```typescript
try {
  const task = await client.tasks.create({ intent: "...", ... });
} catch (err) {
  if (err instanceof MemintelError) {
    switch (err.type) {
      case "not_found":
        console.error("Check id/version:", err.message);
        break;
      case "execution_error":
        console.error("Data issue at", err.location);
        break;
      case "rate_limit_exceeded":
        await sleep(err.retryAfterSeconds * 1000);
        break;
      case "action_binding_failed":
        console.error(err.suggestion);
        break;
      case "bounds_exceeded":
        console.error("Hit guardrail bound:", err.message);
        break;
      default:
        throw err;
    }
  }
}
```
