# Memintel — TypeScript Instruction Layer
### External API + SDK + Developer Experience · Feed to TypeScript codebase ONLY

> **Prerequisite:** You must have read `core-spec.md` before this file. The Core Spec defines
> the execution model, semantics, schemas, and Interaction API that this layer implements.
> Do not redefine anything from the Core Spec here.

---

## What TypeScript Is Responsible For

TypeScript is the **interface layer**. Its job is to provide a clean, ergonomic, strongly-typed
surface to app developers. It must not implement execution logic, condition strategies, or
business rules. All intelligence lives in Python.

| Responsibility | Description | Key constraint |
|---|---|---|
| External API layer | HTTP client for all App Developer API + Interaction API endpoints | Must not call Internal Platform endpoints |
| SDK ergonomics | Class-based client with Stripe-like method naming | `new Memintel({ apiKey })` pattern |
| Strong typing | Full TypeScript types for all request/response shapes | No `any` types in public API surface |
| Request validation | Validate required fields before making HTTP calls | Fail fast with clear error messages |
| Response shaping | Map `snake_case` API fields to `camelCase` SDK fields | `actionsTriggered` not `actions_triggered` |
| Developer experience | Sensible defaults, optional parameters, `dryRun` helper | Hide internal complexity from app developers |
| Error handling | Typed `MemintelError` with `.type`, `.message`, `.location`, `.suggestion` | Branch on `.type`, never on `.message` |
| Async by default | All methods return `Promise<T>` | No synchronous variants in public API |

---

## Client Structure

```typescript
import Memintel from '@memintel/sdk';

const client = new Memintel({
  apiKey: process.env.MEMINTEL_API_KEY!,
  baseUrl: 'https://api.memsdl.ai/v1',   // optional override
  timeout: 30_000,                        // ms, default 30s
});

// ── Execution (top-level methods) ─────────────────────────────────────────
client.evaluateFull(...)           // POST /evaluate/full
client.execute(...)                // POST /execute
client.evaluateCondition(...)      // POST /evaluate/condition
client.evaluateConditionBatch(...) // POST /evaluate/condition/batch
client.executeBatch(...)           // POST /execute/batch
client.executeRange(...)           // POST /execute/range
client.executeAsync(...)           // POST /execute/async
client.getJob(...)                 // GET  /jobs/{jobId}
client.cancelJob(...)              // DELETE /jobs/{jobId}
client.explain(...)                // POST /explain
client.validate(...)               // POST /definitions/validate
client.conditionImpact(...)        // POST /intelligence/condition-impact

// ── Interaction API sub-clients ────────────────────────────────────────────
client.tasks.create(...)           // POST /tasks
client.tasks.list(...)             // GET  /tasks
client.tasks.get(...)              // GET  /tasks/{id}
client.tasks.update(...)           // PATCH /tasks/{id}
client.tasks.delete(...)           // DELETE /tasks/{id}

client.conditions.get(...)         // GET  /conditions/{id}
client.conditions.explain(...)     // POST /conditions/explain
client.conditions.calibrate(...)   // POST /conditions/calibrate
client.conditions.applyCalibration(...) // POST /conditions/apply-calibration

client.decisions.explain(...)      // POST /decisions/explain

client.feedback.submit(...)        // POST /feedback/decision

// ── Registry sub-clients ───────────────────────────────────────────────────
client.actions.list(...)           // GET  /actions
client.actions.get(...)            // GET  /actions/{id}
client.actions.trigger(...)        // POST /actions/{id}/trigger

client.registry.list(...)          // GET  /registry/definitions
client.registry.search(...)        // GET  /registry/search
client.registry.versions(...)      // GET  /registry/definitions/{id}/versions

client.features.search(...)        // GET  /registry/features
client.features.get(...)           // GET  /registry/features/{id}

client.agents.query(...)           // POST /agents/query
client.agents.define(...)          // POST /agents/define
client.agents.defineCondition(...) // POST /agents/define-condition
```

---

## Task-Centric Execution Rule

```typescript
// TASK-CENTRIC EXECUTION RULE:
// In production, all monitoring and decisioning MUST go through Tasks.
// client.evaluateFull() and client.evaluateCondition() are intended ONLY for:
//   - debugging a specific concept or condition in isolation
//   - testing parameter changes before committing a new condition version
//   - backfilling historical evaluations with a known entity + timestamp
//   - CI/CD validation before deploying new task definitions
//
// They are NOT a substitute for Tasks in production use cases.
//
// Production lifecycle (what developers should be doing):
//   1. tasks.create({ intent })            → LLM generates + compiles + persists Task
//   2. evaluateFull({ ...task refs })      → deterministic execution
//   3. feedback.submit({ feedbackType })   → signal quality
//   4. conditions.calibrate()              → get recommendation
//   5. conditions.applyCalibration()       → new condition version
//   6. tasks.update({ conditionVersion })  → explicit rebind
//
// If you are calling evaluateFull() in a production loop without a Task,
// you should be creating a Task instead.
```

## TypeScript-Specific Constraints

### What TypeScript MUST do

```typescript
// ✅ Strong typing on all public methods
async evaluateFull(params: EvaluateFullParams): Promise<FullPipelineResult>

// ✅ camelCase field names on SDK objects (map from snake_case API)
result.actionsTriggered        // not result.actions_triggered
result.conceptId               // not result.concept_id
params.dryRun                  // not params.dry_run
params.missingDataPolicy       // not params.missing_data_policy
result.conditionVersion        // not result.condition_version
result.tasksPendingRebind      // not result.tasks_pending_rebind
result.paramsApplied           // not result.params_applied
result.recommendedParams       // not result.recommended_params
result.calibrationToken        // not result.calibration_token
result.noRecommendationReason  // not result.no_recommendation_reason

// ✅ Required fields enforced at the type level
type EvaluateFullParams = {
  conceptId: string;           // required
  conceptVersion: string;      // required
  conditionId: string;         // required
  conditionVersion: string;    // required
  entity: string;              // required
  timestamp?: string;
  dryRun?: boolean;
  explain?: boolean;
  explainMode?: ExplainMode;
  missingDataPolicy?: MissingDataPolicy;
}

// ✅ Typed error class
class MemintelError extends Error {
  type: ErrorType;             // machine-readable — branch on this
  message: string;             // human-readable — do not branch on this
  location?: string;
  suggestion?: string;
  retryAfterSeconds?: number;  // populated on rate_limit_exceeded
}

// ✅ Idiomatic async/await everywhere
const result = await client.evaluateFull({ ... });
```

### What TypeScript Must NOT do

```typescript
// ❌ DO NOT implement execution logic or condition strategies
// No threshold, z_score, percentile, change, equals, or composite logic

// ❌ DO NOT call Internal Platform API endpoints
// No calls to: /compile, /execute/graph, /registry/definitions (POST),
// /definitions/batch, /agents/semantic-refine, /agents/workflows/compile,
// /registry/definitions/{id}/deprecate, /registry/definitions/{id}/promote,
// /registry/features (POST), /actions (POST)

// ❌ DO NOT silently swallow errors
// Every non-2xx response must surface as a typed MemintelError

// ❌ DO NOT use 'any' in the public API surface
// Open payload fields may use Record<string, unknown>

// ❌ DO NOT implement synchronous variants of async methods

// ❌ DO NOT accept 'useful' or 'not_useful' as feedback values
// Valid FeedbackValue: 'false_positive' | 'false_negative' | 'correct'
// Throw parameter_error before making the HTTP call if invalid value received

// ❌ DO NOT add rebindTasks to ApplyCalibrationParams
// The calibration_token is the only input. There is no rebind_tasks parameter.

// ❌ DO NOT add a 'status' field expectation to ApplyCalibrationResult
// ApplyCalibrationResult has no top-level status field.
```

---

## Naming Conventions

All REST API fields use `snake_case`. All TypeScript SDK fields use `camelCase`. Apply
consistently on both request serialisation and response deserialisation.

| REST API field | TypeScript SDK field |
|---|---|
| `concept_id` | `conceptId` |
| `concept_version` | `conceptVersion` |
| `condition_id` | `conditionId` |
| `condition_version` | `conditionVersion` |
| `action_id` | `actionId` |
| `action_version` | `actionVersion` |
| `task_id` | `taskId` |
| `entity_scope` | `entityScope` |
| `dry_run` | `dryRun` |
| `actions_triggered` | `actionsTriggered` |
| `missing_data_policy` | `missingDataPolicy` |
| `explain_mode` | `explainMode` |
| `payload_sent` | `payloadSent` |
| `version_policy` | `versionPolicy` |
| `last_triggered_at` | `lastTriggeredAt` |
| `created_at` | `createdAt` |
| `recommended_params` | `recommendedParams` |
| `calibration_token` | `calibrationToken` |
| `current_params` | `currentParams` |
| `params_applied` | `paramsApplied` |
| `tasks_pending_rebind` | `tasksPendingRebind` |
| `no_recommendation_reason` | `noRecommendationReason` |
| `delta_alerts` | `deltaAlerts` |
| `feedback_direction` | `feedbackDirection` |
| `feedback_type` | `feedbackType` |
| `previous_version` | `previousVersion` |
| `new_version` | `newVersion` |
| `decision_type` | `decisionType` |
| `concept_value` | `conceptValue` |
| `strategy_type` | `strategyType` |
| `threshold_applied` | `thresholdApplied` |
| `label_matched` | `labelMatched` |

---

## Type Definitions

```typescript
export type MissingDataPolicy = 'null' | 'zero' | 'forward_fill' | 'backward_fill';
export type ExplainMode = 'summary' | 'full' | 'debug';
export type TaskStatus = 'active' | 'paused' | 'deleted' | 'preview';
export type FeedbackValue = 'false_positive' | 'false_negative' | 'correct';
// NOTE: 'useful' and 'not_useful' are NOT valid — throw parameter_error

// IMPORTANT: Strategies are generated by the LLM at task creation time — they are NOT
// predefined by the admin. Every condition from POST /tasks includes explicit strategy.type
// and strategy.params. The compiler rejects any condition without a fully specified strategy.
export type ConditionStrategyType =
  | 'threshold'   // float/int input → decision<boolean>
  | 'percentile'  // float/int input → decision<boolean>
  | 'z_score'     // float/int input → decision<boolean> — NOTE: underscore, not 'zscore'
  | 'change'      // float/int input → decision<boolean>
  | 'equals'      // categorical/string input → decision<categorical>
  | 'composite';  // decision<boolean> inputs ONLY → decision<boolean>
  // COMPOSITE CONSTRAINTS (compiler-enforced):
  //   operands must ALL be decision<boolean> conditions
  //   equals conditions (decision<categorical>) CANNOT be composite operands → type_error
  //   operands list must contain at least 2 condition_ids
  //   nested composites (composite inside composite) are not allowed → semantic_error

export type CalibrationStatus = 'recommendation_available' | 'no_recommendation';
export type NoRecommendationReason =
  | 'bounds_exceeded'
  | 'not_applicable_strategy'
  | 'insufficient_data';

export type ErrorType =
  | 'syntax_error'
  | 'type_error'
  | 'semantic_error'
  | 'reference_error'
  | 'parameter_error'
  | 'graph_error'
  | 'execution_error'
  | 'execution_timeout'
  | 'auth_error'
  | 'not_found'
  | 'conflict'
  | 'rate_limit_exceeded'
  | 'bounds_exceeded'        // calibration hit a guardrail bound
  | 'action_binding_failed'; // no action could be resolved at task creation
```

---

## Interaction API — Task Params

```typescript
// POST /tasks
export type CreateTaskParams = {
  intent: string;                   // required — natural language description
  entityScope: string;              // required
  delivery: DeliveryConfig;         // required
  constraints?: ConstraintsConfig;
  dryRun?: boolean;                 // default false — returns preview Task, nothing persisted
};

export type DeliveryConfig = {
  type: 'webhook' | 'notification' | 'email' | 'workflow';
  endpoint?: string;
  channel?: string;
};

export type ConstraintsConfig = {
  sensitivity?: 'low' | 'medium' | 'high';
  namespace?: 'personal' | 'team' | 'org' | 'global';
};

// PATCH /tasks/{id} — at least one field required
export type UpdateTaskParams = {
  conditionVersion?: string;        // rebind to new calibrated version
  delivery?: DeliveryConfig;
  entityScope?: string;
  status?: 'active' | 'paused';     // use tasks.delete() to soft-delete
  // concept_id, condition_id, action_id, strategy, parameters — CANNOT be updated
};

// POST /conditions/calibrate
export type CalibrateParams = {
  conditionId: string;              // required
  conditionVersion: string;         // required
  feedbackType?: FeedbackValue;     // optional if feedbackDirection provided
  feedbackDirection?: 'tighten' | 'relax';  // explicit override — bypasses feedback aggregation
  target?: { alertsPerDay: number }; // optional target alert frequency
  context?: { entity?: string; timestamp?: string };
};

// POST /conditions/apply-calibration
export type ApplyCalibrationParams = {
  calibrationToken: string;         // required — ONLY path, no condition_id fallback
  newVersion?: string;              // optional — auto-incremented if omitted
  // NO rebindTasks — rebinding is always explicit via tasks.update()
};

// POST /feedback/decision
export type FeedbackParams = {
  conditionId: string;              // required
  conditionVersion: string;         // required
  entity: string;                   // required
  timestamp: string;                // required — ISO 8601 UTC of the original decision
  feedbackType: FeedbackValue;      // required — 'false_positive' | 'false_negative' | 'correct'
  note?: string;
};

// POST /decisions/explain
export type DecisionExplainParams = {
  conditionId: string;              // required
  conditionVersion: string;         // required
  entity: string;                   // required
  timestamp: string;                // required — must match original decision timestamp exactly
};
```

---

## Implementation Skeleton — evaluateFull

```typescript
export class Memintel {
  private baseUrl: string;
  private apiKey: string;
  private timeout: number;

  constructor(config: MemintelConfig) {
    if (!config.apiKey) throw new MemintelError('auth_error', 'apiKey is required');
    this.apiKey  = config.apiKey;
    this.baseUrl = config.baseUrl ?? 'https://api.memsdl.ai/v1';
    this.timeout = config.timeout ?? 30_000;
  }

  // evaluateFull() calls the deterministic runtime — NO LLM involvement.
  // Execution follows deterministic topological order of the compiled DAG.
  // Independent nodes may execute in parallel — order is stable across runs.
  // Same inputs (concept version + condition version + entity + timestamp) always
  // produce the same result in the same node execution sequence.
  // With timestamp: same inputs always produce same result (deterministic).
  // Without timestamp: reflects current data state (snapshot, not reproducible).
  // The LLM was used ONLY when tasks.create() was called — never during execution.
  // Actions are pre-bound at task creation time — evaluateFull() only triggers them.
  async evaluateFull(params: EvaluateFullParams): Promise<FullPipelineResult> {
    // 1. Validate required fields
    const { conceptId, conceptVersion, conditionId, conditionVersion, entity } = params;
    if (!conceptId || !conceptVersion || !conditionId || !conditionVersion || !entity) {
      throw new MemintelError('parameter_error',
        'conceptId, conceptVersion, conditionId, conditionVersion, and entity are all required');
    }

    // 2. Map camelCase → snake_case
    const body: Record<string, unknown> = {
      concept_id:        params.conceptId,
      concept_version:   params.conceptVersion,
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
      entity:            params.entity,
    };
    if (params.timestamp)           body.timestamp           = params.timestamp;
    if (params.dryRun !== undefined) body.dry_run             = params.dryRun;
    if (params.explain !== undefined) body.explain            = params.explain;
    if (params.explainMode)         body.explain_mode         = params.explainMode;
    if (params.missingDataPolicy)   body.missing_data_policy  = params.missingDataPolicy;

    // 3. HTTP call
    const raw = await this.post('/evaluate/full', body);

    // 4. Map snake_case response → camelCase
    return mapFullPipelineResult(raw);
  }

  private async post(path: string, body: unknown): Promise<unknown> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout),
    });
    if (!res.ok) {
      const err = await res.json()
        .catch(() => ({ error: { type: 'unknown', message: res.statusText } }));
      throw new MemintelError(
        err.error?.type ?? 'unknown',
        err.error?.message ?? 'Unknown error',
        {
          location: err.error?.location,
          suggestion: err.error?.suggestion,
          retryAfterSeconds: res.headers.get('Retry-After')
            ? Number(res.headers.get('Retry-After'))
            : undefined,
        }
      );
    }
    return res.json();
  }
}
```

---

## Idempotency Key Support

Task creation should support an idempotency key to prevent duplicate tasks in retry scenarios.

```typescript
// Pattern: pass an options object as the second argument to tasks.create()
const task = await client.tasks.create(
  {
    intent: 'Alert me when churn risk exceeds 0.8',
    entityScope: 'user',
    delivery: { type: 'webhook', endpoint: 'https://my-app.com/alerts' },
  },
  { idempotencyKey: crypto.randomUUID() }  // prevents duplicate task on retry
);

// Why this matters:
// Network failures can cause task creation requests to be retried.
// Without idempotency keys, retries create duplicate tasks — each with its own
// compiled concept, condition, and action.
// With an idempotency key, the same key within 24h returns the original response.

// Implementation:
// The idempotency key is sent as the Idempotency-Key HTTP header.
// The second options argument is optional — omitting it skips the header.

// TasksClient.create() signature:
async create(
  params: CreateTaskParams,
  options?: { idempotencyKey?: string }
): Promise<Task>

// All other mutating methods (update, delete) also accept the options argument
// but idempotency keys are most critical for create() to avoid duplicate tasks.
```

## Interaction API Sub-Client Skeletons

```typescript
export class TasksClient {
  constructor(private http: HttpClient) {}

  async create(params: CreateTaskParams): Promise<Task> {
    const body: Record<string, unknown> = {
      intent:       params.intent,
      entity_scope: params.entityScope,
      delivery:     mapDeliveryToSnake(params.delivery),
    };
    if (params.constraints) body.constraints = mapConstraintsToSnake(params.constraints);
    if (params.dryRun !== undefined) body.dry_run = params.dryRun;
    const raw = await this.http.post('/tasks', body);
    return mapTask(raw);
  }

  async list(params?: { status?: TaskStatus; limit?: number; cursor?: string }): Promise<TaskList> {
    const raw = await this.http.get('/tasks', params);
    return mapTaskList(raw);
  }

  async get(id: string): Promise<Task> {
    const raw = await this.http.get(`/tasks/${id}`);
    return mapTask(raw);
  }

  async update(id: string, params: UpdateTaskParams): Promise<Task> {
    if (Object.keys(params).length === 0) {
      throw new MemintelError('parameter_error', 'At least one field must be provided for update');
    }
    const body: Record<string, unknown> = {};
    if (params.conditionVersion) body.condition_version = params.conditionVersion;
    if (params.delivery) body.delivery = mapDeliveryToSnake(params.delivery);
    if (params.entityScope) body.entity_scope = params.entityScope;
    if (params.status) body.status = params.status;
    const raw = await this.http.patch(`/tasks/${id}`, body);
    return mapTask(raw);
  }

  async delete(id: string): Promise<Task> {
    const raw = await this.http.delete(`/tasks/${id}`);
    return mapTask(raw);
  }
}

export class ConditionsClient {
  constructor(private http: HttpClient) {}

  async get(id: string, version: string): Promise<ConditionDefinition> {
    const raw = await this.http.get(`/conditions/${id}`, { version });
    return mapConditionDefinition(raw);
  }

  async explain(params: {
    conditionId: string;
    conditionVersion: string;
    timestamp?: string;
  }): Promise<ConditionExplanation> {
    const body = {
      condition_id: params.conditionId,
      condition_version: params.conditionVersion,
      ...(params.timestamp && { timestamp: params.timestamp }),
    };
    const raw = await this.http.post('/conditions/explain', body);
    return mapConditionExplanation(raw);
  }

  async calibrate(params: CalibrateParams): Promise<CalibrationResult> {
    const body: Record<string, unknown> = {
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
    };
    if (params.feedbackType)      body.feedback_type      = params.feedbackType;
    if (params.feedbackDirection) body.feedback_direction = params.feedbackDirection;
    if (params.target)            body.target             = { alerts_per_day: params.target.alertsPerDay };
    if (params.context)           body.context            = params.context;
    const raw = await this.http.post('/conditions/calibrate', body);
    return mapCalibrationResult(raw);
  }

  async applyCalibration(params: ApplyCalibrationParams): Promise<ApplyCalibrationResult> {
    // calibration_token is the ONLY path — no condition_id, no threshold, no rebind_tasks
    const body: Record<string, unknown> = {
      calibration_token: params.calibrationToken,
    };
    if (params.newVersion) body.new_version = params.newVersion;
    const raw = await this.http.post('/conditions/apply-calibration', body);
    return mapApplyCalibrationResult(raw);
  }
}

export class DecisionsClient {
  constructor(private http: HttpClient) {}

  async explain(params: DecisionExplainParams): Promise<DecisionExplanation> {
    const body = {
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
      entity:            params.entity,
      timestamp:         params.timestamp,
    };
    const raw = await this.http.post('/decisions/explain', body);
    return mapDecisionExplanation(raw);
  }
}

export class FeedbackClient {
  constructor(private http: HttpClient) {}

  async submit(params: FeedbackParams): Promise<FeedbackResponse> {
    // Validate feedback value before making HTTP call
    const validValues: FeedbackValue[] = ['false_positive', 'false_negative', 'correct'];
    if (!validValues.includes(params.feedbackType)) {
      throw new MemintelError('parameter_error',
        `Invalid feedback value '${params.feedbackType}'. Valid values: false_positive, false_negative, correct`);
    }
    const body = {
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
      entity:            params.entity,
      timestamp:         params.timestamp,
      feedback:          params.feedbackType,
      ...(params.note && { note: params.note }),
    };
    const raw = await this.http.post('/feedback/decision', body);
    return mapFeedbackResponse(raw);
  }
}
```

---

## Response Mapping — Key Patterns

```typescript
// These mapper functions convert snake_case API responses to camelCase SDK types.
// Implement one mapper per schema. Never spread raw API responses directly.

function mapTask(raw: unknown): Task {
  const r = raw as Record<string, unknown>;
  return {
    taskId:            r.task_id as string | undefined,
    intent:            r.intent as string,
    conceptId:         r.concept_id as string,
    conceptVersion:    r.concept_version as string,
    conditionId:       r.condition_id as string,
    conditionVersion:  r.condition_version as string,
    actionId:          r.action_id as string,
    actionVersion:     r.action_version as string,
    entityScope:       r.entity_scope as string,
    delivery:          mapDeliveryFromSnake(r.delivery),
    status:            r.status as TaskStatus,
    createdAt:         r.created_at as string | undefined,
    lastTriggeredAt:   r.last_triggered_at as string | null | undefined,
  };
}

function mapCalibrationResult(raw: unknown): CalibrationResult {
  const r = raw as Record<string, unknown>;
  return {
    status:                  r.status as CalibrationStatus,
    recommendedParams:       r.recommended_params as Record<string, unknown> | undefined,
    calibrationToken:        r.calibration_token as string | undefined,
    currentParams:           r.current_params as Record<string, unknown>,
    impact:                  r.impact ? mapCalibrationImpact(r.impact) : undefined,
    noRecommendationReason:  r.no_recommendation_reason as NoRecommendationReason | undefined,
  };
}

function mapApplyCalibrationResult(raw: unknown): ApplyCalibrationResult {
  const r = raw as Record<string, unknown>;
  return {
    conditionId:         r.condition_id as string,
    previousVersion:     r.previous_version as string,
    newVersion:          r.new_version as string,
    paramsApplied:       r.params_applied as Record<string, unknown>,
    tasksPendingRebind:  (r.tasks_pending_rebind as Array<Record<string, unknown>> || [])
                           .map(t => ({ taskId: t.task_id as string, intent: t.intent as string })),
  };
}

function mapDecisionExplanation(raw: unknown): DecisionExplanation {
  const r = raw as Record<string, unknown>;
  return {
    conditionId:       r.condition_id as string,
    conditionVersion:  r.condition_version as string,
    entity:            r.entity as string,
    timestamp:         r.timestamp as string,
    decision:          r.decision as boolean | string,
    decisionType:      r.decision_type as 'boolean' | 'categorical',
    conceptValue:      r.concept_value as number | string,
    strategyType:      r.strategy_type as ConditionStrategyType,
    thresholdApplied:  r.threshold_applied as number | undefined,
    labelMatched:      r.label_matched as string | undefined,
    drivers:           (r.drivers as Array<Record<string, unknown>> || [])
                         .map(d => ({
                           signal:       d.signal as string,
                           contribution: d.contribution as number,
                           value:        d.value as number | string,
                         })),
  };
}
```

---

## Error Handling Pattern

```typescript
try {
  const result = await client.evaluateFull({ ... });
} catch (err) {
  if (err instanceof MemintelError) {
    switch (err.type) {
      case 'not_found':
        console.error('Check id and version:', err.message);
        break;
      case 'execution_error':
        console.error('Data issue:', err.message, 'at', err.location);
        break;
      case 'execution_timeout':
        const job = await client.executeAsync({ ... });
        break;
      case 'rate_limit_exceeded':
        await sleep(err.retryAfterSeconds! * 1000);
        // retry
        break;
      case 'bounds_exceeded':
        console.error('Calibration hit guardrail bound:', err.message);
        break;
      case 'action_binding_failed':
        console.error('No action could be resolved:', err.suggestion);
        break;
      case 'parameter_error':
        console.error('Invalid parameter:', err.message);
        break;
      // LLM failure at task creation — POST /tasks only
      // Returned when the LLM fails to produce a valid definition after MAX_RETRIES
      // err.message contains the last compiler validation error
      // err.location identifies where in the definition the error occurred
      // No partial task was created — safe to retry with a revised intent
      case 'semantic_error':
        if (err.location) {
          console.error(`Definition error at ${err.location}: ${err.message}`);
          if (err.suggestion) console.error('Fix:', err.suggestion);
        }
        break;
      default:
        throw err;
    }
  }
}
```

---

## Interaction API — Usage Examples

```typescript
// ── CORE MENTAL MODEL ─────────────────────────────────────────────────────
// Task = ATOMIC UNIT of execution: Concept (psi) + Condition (phi) + Action (alpha)
// The LLM generates all three at tasks.create() time. After that — pure determinism.
// Actions are pre-bound at task creation. evaluateFull() only TRIGGERS them, never selects.
//   task.actionId + task.actionVersion  →  pre-bound at creation, never changes at runtime
//   result.decision.actionsTriggered    →  actions nested here, never at result top-level
// ─────────────────────────────────────────────────────────────────────────

// Create a task
const task = await client.tasks.create({
  intent: 'Alert me when user engagement drops significantly over the last week',
  entityScope: 'user',
  delivery: { type: 'webhook', endpoint: 'https://my-app.com/alerts' },
  constraints: { sensitivity: 'medium', namespace: 'org' },
});

// Dry run — preview without persisting
// dry_run propagates: concept executes, condition evaluates, actions DO NOT fire
// All actionsTriggered[].status will be 'would_trigger' in dry_run mode
const preview = await client.tasks.create({
  intent: 'Alert me when churn risk exceeds high threshold',
  entityScope: 'user',
  delivery: { type: 'notification', channel: 'ops-team' },
  dryRun: true,  // returns Task with status='preview', no task_id
});

// Full pipeline evaluation
const result = await client.evaluateFull({
  conceptId: 'org.churn_risk',
  conceptVersion: '1.2',
  conditionId: 'org.high_churn',
  conditionVersion: '1.0',
  entity: 'user_abc',
  timestamp: '2024-03-15T09:00:00Z',
});
// result.decision.actionsTriggered — actions nested inside decision

// Submit feedback
await client.feedback.submit({
  conditionId: 'org.high_churn',
  conditionVersion: '1.0',
  entity: 'user_abc',
  timestamp: '2024-03-15T09:00:00Z',
  feedbackType: 'false_positive',
  note: 'User was on planned leave, not genuine churn risk',
});

// Calibrate condition
const calibration = await client.conditions.calibrate({
  conditionId: 'org.high_churn',
  conditionVersion: '1.0',
  // Optional: target alert volume
  target: { alertsPerDay: 3 },
});

if (calibration.status === 'recommendation_available') {
  // Apply calibration — token is the ONLY input
  const applied = await client.conditions.applyCalibration({
    calibrationToken: calibration.calibrationToken!,
    newVersion: '1.1',
  });

  // Rebind task explicitly — never happens automatically
  await client.tasks.update(task.taskId!, {
    conditionVersion: applied.newVersion,
  });

  // applied.tasksPendingRebind lists other tasks still on old version
  console.log('Tasks still on old version:', applied.tasksPendingRebind);
}

// Explain a decision
const explanation = await client.decisions.explain({
  conditionId: 'org.high_churn',
  conditionVersion: '1.0',
  entity: 'user_abc',
  timestamp: '2024-03-15T09:00:00Z',
});
// explanation.thresholdApplied — numeric threshold (null for equals/composite)
// explanation.labelMatched     — matched label (defined for equals only)
// explanation.drivers          — contributions sum to 1.0
```

---

## Installation and Package

```bash
npm install @memintel/sdk
# or
yarn add @memintel/sdk
```

Package must export:
- `Memintel` — default export, main client class
- `MemintelError` — named export, typed error class
- All public TypeScript types as named exports:
  `Task`, `TaskStatus`, `DeliveryConfig`, `ConstraintsConfig`,
  `CalibrationResult`, `ApplyCalibrationResult`, `CalibrationStatus`,
  `DecisionExplanation`, `ConditionExplanation`, `FeedbackValue`, `FeedbackResponse`,
  `FullPipelineResult`, `Result`, `DecisionResult`, `ActionTriggered`,
  `Explanation`, `ErrorResponse`, `ErrorType`, `ConditionStrategyType`,
  `MissingDataPolicy`, `ExplainMode`, `Job`, `JobResult`
