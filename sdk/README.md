# @memintel/sdk

TypeScript SDK for the [Memintel API](https://www.memintel.io) — deterministic semantic compiler and runtime for agentic AI systems.

[![npm version](https://badge.fury.io/js/%40memintel%2Fsdk.svg)](https://www.npmjs.com/package/@memintel/sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## What is Memintel?

Memintel solves the indeterminacy problem in agentic AI. It compiles natural language intent into a deterministic execution graph — so the same input always produces the same decision, with a full audit trail and no LLM on the critical path.

```
Intent → Concept (ψ) → Condition (φ) → Action (α)
```

## Installation

```bash
npm install @memintel/sdk
```

## Quick Start

```typescript
import Memintel from '@memintel/sdk';

const client = new Memintel({
  apiKey: process.env.MEMINTEL_API_KEY,
});

// Execute the full ψ → φ → α pipeline
const result = await client.evaluateFull({
  concept_id: 'org.churn_risk',
  concept_version: '1.2',
  condition_id: 'org.high_churn',
  condition_version: '1.0',
  entity: 'user_abc123',
  timestamp: new Date().toISOString(), // deterministic execution
});

console.log(result.result.value);   // 0.87 — concept output (Rₜ)
console.log(result.decision.value); // true — condition fired (Aₜ)
console.log(result.decision.actions_triggered); // actions executed
```

## Create a Task from Natural Language

```typescript
// Describe what you want in plain English
const task = await client.tasks.create({
  intent: 'Alert me when a customer\'s churn risk rises significantly',
  entityScope: 'user_abc123',
  delivery: {
    type: 'webhook',
    endpoint: 'https://myapp.com/hooks/alert',
  },
});

// Memintel compiles it into a deterministic execution graph
console.log(task.concept_id);    // resolved concept
console.log(task.condition_id);  // resolved condition with strategy + params
```

## Dry Run — Preview Before Committing

```typescript
const preview = await client.tasks.create({
  intent: 'Alert me when AAPL price rises significantly',
  entityScope: 'AAPL',
  delivery: { type: 'notification' },
  dryRun: true, // preview without persisting
});

// See exactly how "significantly" was resolved
console.log(preview.condition.strategy);
// { type: 'change', params: { direction: 'increase', value: 0.10, window: '1d' } }
```

## Explain a Decision

```typescript
// Why did this condition fire for this entity at this time?
const explanation = await client.decisions.explain({
  conditionId: 'org.high_churn',
  conditionVersion: '1.0',
  entity: 'user_abc123',
  timestamp: '2024-03-15T09:00:00Z',
});

console.log(explanation.decision);         // true
console.log(explanation.concept_value);    // 0.87
console.log(explanation.threshold_applied); // 0.8
console.log(explanation.drivers);          // signal contributions
```

## Submit Feedback & Calibrate

```typescript
// Mark a decision as incorrect
await client.feedback.decision({
  conditionId: 'org.high_churn',
  conditionVersion: '1.0',
  entity: 'user_abc123',
  timestamp: '2024-03-15T09:00:00Z',
  feedback: 'false_positive',
  note: 'One-off spike — user was on holiday',
});

// Get a calibration recommendation
const cal = await client.conditions.calibrate({
  conditionId: 'org.high_churn',
  conditionVersion: '1.0',
});

if (cal.status === 'recommendation_available') {
  console.log(cal.recommended_params); // { value: 0.85 }
  console.log(cal.impact.delta_alerts); // -12 per day

  // Apply — creates a new immutable version, never mutates existing
  const applied = await client.conditions.applyCalibration({
    calibrationToken: cal.calibration_token,
  });

  // Explicitly rebind tasks to the new version
  for (const t of applied.tasks_pending_rebind) {
    await client.tasks.update(t.task_id, {
      conditionVersion: applied.new_version,
    });
  }
}
```

## Authentication

```typescript
const client = new Memintel({
  apiKey: process.env.MEMINTEL_API_KEY,
});
```

Or set the header directly:

```typescript
const headers = {
  'X-API-Key': process.env.MEMINTEL_API_KEY,
  'Content-Type': 'application/json',
};
```

## Error Handling

```typescript
try {
  const task = await client.tasks.create({ intent: '...', ... });
} catch (err) {
  if (err instanceof MemintelError) {
    switch (err.type) {
      case 'not_found':
        console.error('Check id/version:', err.message);
        break;
      case 'execution_timeout':
        // Switch to async execution for heavy workloads
        break;
      case 'action_binding_failed':
        console.error(err.suggestion);
        break;
    }
  }
}
```

## Documentation

Full API reference, guides, and code examples at **[www.memintel.io](https://www.memintel.io)**

- [Introduction & Core Concepts](https://www.memintel.io/docs/intro/overview)
- [App Developer API Reference](https://www.memintel.io/docs/api-reference/overview)
- [Python Backend SDK](https://www.memintel.io/docs/python-sdk/python-overview)
- [End-to-End Workflow](https://www.memintel.io/docs/api-reference/end-to-end-workflow)
- [Code Snippets (20 examples)](https://www.memintel.io/docs/intro/snippets)

## Base URL

```
https://api.memsdl.ai/v1
```

## License

MIT
