# Memintel — Core Spec
### Single Source of Truth · Feed to BOTH TypeScript and Python before any language-specific instructions

> This document defines the semantics, contracts, and interfaces that neither implementation may
> diverge from. If TypeScript and Python ever produce different results for the same inputs, this
> spec is the arbiter.

---

## 1A — API Contracts

### Base URL and Authentication

```
Base URL:  https://api.memsdl.ai/v1
Auth:      X-API-Key: <your-key>   (header on every request)
           MEMINTEL_API_KEY env var is the recommended source

Idempotency: POST requests accept Idempotency-Key header
             Same key within 24h returns cached response
Rate limits: X-RateLimit-Limit / X-RateLimit-Remaining on every response
             429 → Retry-After header (seconds to wait before retrying)
```

### Section 1 — App Developer API (both SDKs implement these)

| Method + Path | TS SDK method | Python SDK method | Returns |
|---|---|---|---|
| POST /evaluate/full | `evaluateFull()` | `evaluate_full()` | FullPipelineResult |
| POST /execute | `execute()` | `execute()` | Result |
| POST /evaluate/condition | `evaluateCondition()` | `evaluate_condition()` | DecisionResult |
| POST /evaluate/condition/batch | `evaluateConditionBatch()` | `evaluate_condition_batch()` | DecisionResult[] |
| POST /execute/batch | `executeBatch()` | `execute_batch()` | BatchExecuteResult |
| POST /execute/range | `executeRange()` | `execute_range()` | Result[] |
| POST /execute/async | `executeAsync()` | `execute_async()` | Job |
| GET /jobs/{jobId} | `getJob()` | `get_job()` | JobResult |
| DELETE /jobs/{jobId} | `cancelJob()` | `cancel_job()` | JobResult |
| POST /explain | `explain()` | `explain()` | Explanation |
| GET /actions | `actions.list()` | `actions.list()` | Action[] |
| GET /actions/{id} | `actions.get()` | `actions.get()` | Action |
| POST /actions/{id}/trigger | `actions.trigger()` | `actions.trigger()` | ActionResult |
| POST /definitions/validate | `validate()` | `validate()` | ValidationResult |
| GET /definitions/{id} | `definitions.get()` | `definitions.get()` | Definition |
| GET /registry/definitions | `registry.list()` | `registry.list()` | SearchResult |
| GET /registry/search | `registry.search()` | `registry.search()` | SearchResult |
| GET /registry/features | `features.search()` | `features.search()` | FeatureSearchResult |
| POST /agents/query | `agents.query()` | `agents.query()` | AgentQueryResponse |
| POST /agents/define | `agents.define()` | `agents.define()` | AgentDefineResponse |
| POST /agents/define-condition | `agents.defineCondition()` | `agents.define_condition()` | AgentDefineResponse |
| POST /intelligence/condition-impact | `conditionImpact()` | `condition_impact()` | dict |

### Section 1 — Interaction API (both SDKs implement these)

| Method + Path | TS SDK method | Python SDK method | Returns |
|---|---|---|---|
| POST /tasks | `tasks.create()` | `tasks.create()` | Task |
| GET /tasks | `tasks.list()` | `tasks.list()` | TaskList |
| GET /tasks/{id} | `tasks.get()` | `tasks.get()` | Task |
| PATCH /tasks/{id} | `tasks.update()` | `tasks.update()` | Task |
| DELETE /tasks/{id} | `tasks.delete()` | `tasks.delete()` | Task |
| GET /conditions/{id} | `conditions.get()` | `conditions.get()` | ConditionDefinition |
| POST /conditions/explain | `conditions.explain()` | `conditions.explain()` | ConditionExplanation |
| POST /conditions/calibrate | `conditions.calibrate()` | `conditions.calibrate()` | CalibrationResult |
| POST /conditions/apply-calibration | `conditions.applyCalibration()` | `conditions.apply_calibration()` | ApplyCalibrationResult |
| POST /decisions/explain | `decisions.explain()` | `decisions.explain()` | DecisionExplanation |
| POST /feedback/decision | `feedback.submit()` | `feedback.submit()` | FeedbackResponse |

### Section 2 — Internal Platform API (Python only — TypeScript must NOT implement these)

| Method + Path | Python SDK method | Returns |
|---|---|---|
| POST /compile | `client.compile()` | ExecutionGraph |
| POST /compile/semantic | `client.compile_semantic()` | SemanticGraph |
| POST /compile/explain-plan | `client.compile_explain_plan()` | ExecutionPlan |
| GET /graphs/{graphId} | `client.get_graph()` | ExecutionGraph |
| POST /execute/graph | `client.execute_graph()` | Result |
| POST /definitions/batch | `client.definitions.batch_create()` | BatchDefinitionResult |
| POST /definitions/concepts | `client.definitions.create_concept()` | DefinitionResponse |
| POST /definitions/conditions | `client.definitions.create_condition()` | DefinitionResponse |
| POST /definitions/primitives | `client.definitions.create_primitive()` | DefinitionResponse |
| POST /registry/definitions | `client.registry.register()` | DefinitionResponse |
| GET /registry/definitions/{id}/versions | `client.registry.versions()` | VersionListResult |
| GET /registry/definitions/{id}/lineage | `client.registry.lineage()` | LineageResult |
| GET /registry/definitions/{id}/semantic-diff | `client.registry.semantic_diff()` | SemanticDiffResult |
| POST /registry/definitions/{id}/deprecate | `client.registry.deprecate()` | DefinitionResponse |
| POST /registry/definitions/{id}/promote | `client.registry.promote()` | DefinitionResponse |
| POST /registry/definitions/similar | `client.registry.find_similar()` | SimilarityResult |
| POST /registry/features | `client.features.register()` | FeatureRegistrationResult |
| GET /registry/features/{id} | `client.features.get()` | RegisteredFeature |
| GET /registry/features/{id}/usages | `client.features.usages()` | UsageResult |
| POST /actions | `client.actions.create()` | DefinitionResponse |
| POST /agents/semantic-refine | `client.agents.semantic_refine()` | SemanticRefineResponse |
| POST /agents/workflows/compile | `client.agents.compile_workflow()` | ExecutionPlan |

---

## 1B — Execution Model

The three-layer pipeline is the fundamental data flow. `evaluateFull` collapses all three layers
into one atomic call.

```
┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION PIPELINE                         │
│                                                                 │
│  Concept (ψ)          Condition (φ)          Action (α)         │
│  POST /execute   ──►  POST /evaluate    ──►  auto-triggered     │
│       │               /condition              on decision        │
│       ▼                    │                                    │
│  Result (Rₜ)              ▼                                    │
│  value / type       DecisionResult (Aₜ)                        │
│  deterministic      value / actions_triggered[]                 │
│                                                                 │
│  ── or in one call: POST /evaluate/full ──────────────────────► │
│                     FullPipelineResult                          │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Interface | Produces |
|---|---|---|
| ψ Concept | `ConceptExecutor.execute(id, version, entity, timestamp?, ...)` | `Result: { value, type, entity, deterministic, timestamp?, explanation? }` |
| φ Condition | `ConditionEvaluator.evaluate(condition_id, version, entity, timestamp?, ...)` | `DecisionResult: { value, type, entity, condition_id, condition_version, timestamp, actions_triggered[] }` |
| α Action | `ActionExecutor.trigger(action_id, version, entity, timestamp?, dry_run?)` | `ActionResult: { status, payload_sent?, error? }` |

### Task Execution Trigger Model

```
TASK EXECUTION TRIGGER MODEL:

Tasks define WHAT to evaluate and WHEN TO ALERT — not when to run.

Execution triggering is owned by the APPLICATION LAYER, not Memintel:
  Scheduled     → the application calls evaluateFull on a cron schedule
  Event-driven  → the application calls evaluateFull on a data change event
  Stream-based  → the application calls evaluateFull on each message arrival
  Manual        → the developer calls evaluateFull directly for a given entity

Memintel's job at evaluation time:
  → accept the call
  → execute the compiled task deterministically
  → return the Decision (Aₜ)
  → fire pre-bound actions if the condition is met
  → return HTTP 200

Memintel does NOT:
  → schedule tasks automatically
  → poll data sources
  → push evaluations without being called
  → manage trigger frequency or rate limiting

This is a deliberate architectural choice: scheduling semantics vary
widely by use case (once/day, per-event, per-second). Memintel provides
the deterministic evaluation engine. The application provides the trigger.

IMPLICATION FOR SDK USAGE:
  A production integration always has an outer loop:
    for entity in entities:
      result = await client.evaluateFull({
        conceptId: task.conceptId, conceptVersion: task.conceptVersion,
        conditionId: task.conditionId, conditionVersion: task.conditionVersion,
        entity: entity, timestamp: currentTimestamp
      })
  The application owns this loop. Memintel owns what happens inside it.
```

### Task-Centric Execution Rule

```
TASK-CENTRIC EXECUTION RULE:

All production use cases MUST use Tasks as the execution unit.
A Task is the only persistent, version-pinned, auditable unit of execution.

Direct calls to:
  - POST /evaluate/full
  - POST /evaluate/condition
  - POST /evaluate/condition/batch
  - POST /execute

are intended ONLY for:
  - debugging a specific condition or concept in isolation
  - testing parameter changes before committing a new version
  - backfilling historical evaluations for a known entity + timestamp
  - CI/CD pipeline validation

They are NOT a substitute for Tasks in production monitoring workflows.
Using direct evaluation endpoints in production means:
  - no version pinning across concept + condition + action
  - no task lifecycle management (pause, resume, rebind)
  - no calibration or feedback association
  - no audit trail linking decisions to a named task

If you find yourself calling evaluateFull() in production without a task,
you should be creating a task instead.
```

### Interaction Layer Ownership

The Interaction Layer is the authoritative owner of the following responsibilities.
No other layer may perform these operations.

```
TASK LIFECYCLE STATE MACHINE:
  create → active → pause ↔ resume → delete (soft)
  preview is a transient non-persisted state (dry_run result only)
  State transitions are owned and validated by the Interaction Layer.
  Deleted tasks cannot be updated (HTTP 409).

CONDITION VERSION REBINDING:
  Validation rules enforced by the Interaction Layer:
  - new condition_version must be a registered version of the same condition_id
  - concept_id, condition_id (identifier), action_id, strategy, parameters
    MUST NOT be changed via PATCH /tasks/{id} — return HTTP 400 if present

DELIVERY CONFIG ENFORCEMENT:
  DeliveryConfig.type must be one of: webhook | notification | email | workflow
  endpoint is required when type=webhook
  channel is required when type=notification or type=email
  workflow_id is required when type=workflow

FEEDBACK INGESTION AND ROUTING:
  feedback.submit() stores feedback records against condition_id + condition_version
  Feedback is NOT applied immediately — it feeds into POST /conditions/calibrate
  Invalid feedback values (anything other than false_positive | false_negative | correct)
  must be rejected at the Interaction Layer before storage
```

### DAG Execution Order Guarantee

```
DETERMINISTIC TOPOLOGICAL EXECUTION ORDER:

The execution graph (DAG) is executed in a deterministic topological order
on every run. This is a hard guarantee, not a best-effort.

Rules:
  1. Topological sort — nodes are executed in topological order derived from
     the DAG edges. Nodes with no dependencies execute first.

  2. Parallel execution — nodes that have no dependency relationship with each
     other (independent nodes) MAY be executed concurrently by the runtime.
     The set of parallelizable groups is determined at compile time and
     does not change between executions.

  3. Order stability — the topological sort is deterministic. The same DAG
     always produces the same execution order. Two runs of the same graph
     on the same data must visit nodes in the same sequence.

  4. No runtime reordering — the runtime must not reorder nodes based on
     data availability, latency, or any dynamic factor.

This guarantee is what makes ir_hash meaningful: same definition → same graph
→ same execution order → same result.

Use POST /compile/explain-plan to inspect the execution order and
parallelizable groups before executing.
```

### Task Lifecycle (Interaction API)

A **Task** is the atomic unit of execution — a version-pinned, immutable bundle of
Concept + Condition + Action. The Interaction API manages the full task lifecycle.

**Action binding — critical rule:**
Actions are NOT dynamically selected or resolved at execution time. They are pre-bound to the
condition at task creation and compilation time. When `client.evaluateFull()` or
`client.evaluateCondition()` executes, it triggers only the pre-bound actions that are already
part of the compiled task definition. There is no runtime action resolution.

```
Task = Concept (ψ) + Condition (φ) + Action (α)  [all pre-bound at compile time]

# CORRECT mental model:
task = await client.tasks.create({ intent: '...' })
# task.actionId and task.actionVersion are set here — never changes at execution

# INCORRECT assumption:
result = await client.evaluateFull({ ... })  // does NOT dynamically select actions
```

```
POST /tasks         → LLM generates Concept + Condition + Action → compiled → persisted as Task
GET  /tasks/{id}    → retrieve task and its version-pinned references
PATCH /tasks/{id}   → update condition_version (rebind), delivery, entity_scope, or status
DELETE /tasks/{id}  → soft-delete (logical deletion, history retained)

POST /conditions/calibrate          → analyse feedback → return recommendation + calibration_token
POST /conditions/apply-calibration  → apply token → create new condition version (never mutate)
PATCH /tasks/{id}                   → explicit rebind to new condition version (user-initiated only)
POST /feedback/decision             → record false_positive | false_negative | correct
```

---

## 1C — Semantics

### Determinism Contract

**All execution endpoints are bound by this contract:**

```
DETERMINISM INVARIANT:
  execute(id, version, entity, timestamp) MUST return identical results
  on every call, regardless of when, how often, or from where it is called.

CACHE KEY CONTRACT:
  Cache key MUST be: (concept_id, version, entity, timestamp)
  A None/absent timestamp is a DIFFERENT key from any specific timestamp.
  Snapshot results (no timestamp) MUST NOT be cached across requests.

NO LLM IN RUNTIME — ABSOLUTE RULE:
  The execution engine MUST NOT call the LLM under any circumstance.
  All runtime paths (execute, evaluate/condition, evaluate/full) are
  fully deterministic. LLM is used ONLY in POST /tasks and agent endpoints.
  Any code path in the execution engine that calls an LLM is a critical bug.
```

### Determinism and Timestamps

Timestamp is the master switch between two execution modes.

```python
# DETERMINISTIC MODE — timestamp provided
# → Same entity + same timestamp = same result, forever
# → result.deterministic == True
result = client.execute(id='org.churn_risk', version='1.2',
                        entity='user_abc', timestamp='2024-03-15T09:00:00Z')

# SNAPSHOT MODE — timestamp omitted
# → Reflects current data state
# → result.deterministic == False
# → NOT reproducible across separate calls
result = client.execute(id='org.churn_risk', version='1.2', entity='user_abc')
```

### Caching Expectations

```
- Cache is ON by default. Both SDKs must pass cache=true unless explicitly bypassed.
- Cache key: (concept_id, version, entity, timestamp)
- A None/absent timestamp is a DIFFERENT key from any specific timestamp.
- Never cache snapshot (no-timestamp) results across requests.
- evaluateCondition() implicitly executes the concept on first call (cache miss).
  Pre-warm: callers may call execute() before evaluateCondition() if latency matters.
- Python only: compile() produces a graph_id; execute_graph() bypasses re-compile.
  Cache graph_id at startup: GRAPH_CACHE['org.churn_risk:1.2'] = graph.graph_id
```

### Action Idempotency Contract

```
ACTION IDEMPOTENCY CONTRACT:

Memintel executes each bound action AT MOST ONCE per evaluation call.
There is no automatic retry on action failure.

Idempotency guarantee:
  Within a single evaluateFull() or evaluateCondition() call, each
  bound action fires at most once, regardless of execution mode or
  pipeline retries.

What Memintel guarantees:
  → at-most-once delivery per evaluation call
  → failure is captured in actions_triggered[].status = 'failed'
  → the pipeline continues and returns HTTP 200 regardless of action outcome
  → the same action does NOT retry automatically

What Memintel does NOT guarantee:
  → exactly-once delivery across multiple evaluation calls
  → deduplication if the application calls evaluateFull() twice for the
    same entity + timestamp (the action will fire twice)

Downstream system responsibility:
  Downstream action receivers (webhooks, notification systems, workflows)
  MUST implement their own idempotency if required. The recommended
  deduplication key for downstream systems is:
    (condition_id, condition_version, entity, timestamp)
  This tuple uniquely identifies a specific decision. Any action fired
  for the same tuple is a duplicate and can be safely deduplicated.

SDK guidance:
  If you need exactly-once semantics, use the Idempotency-Key header on
  evaluateFull() to prevent duplicate pipeline executions from your
  application's own retry logic.
```

### Action Execution — Best Effort

```
dry_run PROPAGATION RULES:
  dry_run: true passed to evaluateFull, evaluateCondition, or actions.trigger()
  → concept EXECUTES (real computation)
  → condition EVALUATES (real decision)
  → actions DO NOT fire
  → actions_triggered[].status === 'would_trigger' for all bound actions
  → No HTTP calls are made to action endpoints
  → No side effects outside Memintel

dry_run is useful for:
  - Verifying action bindings before go-live
  - Staging environment testing
  - Confirming a condition would fire for a given entity + timestamp
```

```
Actions fire automatically when a Condition evaluates to its fire_on value.
Pipeline returns HTTP 200 even if an individual action fails.
Inspect actions_triggered[].status per action:
  'triggered'     → fired and delivered successfully
  'skipped'       → condition did not meet fire_on rule
  'failed'        → fired but delivery failed (see .error)
  'would_trigger' → dry_run=True, not actually fired

dry_run mode: concept executes, condition evaluates, NO actions fire.
Both SDKs MUST expose dry_run on: evaluateFull, evaluateCondition, actions.trigger()
```

### Condition Strategies

Python is the ONLY place strategy logic is implemented. TypeScript must never contain strategy logic.

**Strategy ownership — critical rule:**
Strategies are NOT predefined by the admin or hardcoded. Every condition generated by the LLM
during `POST /tasks` MUST include an explicit `strategy.type` and `strategy.params`. These are
generated by the LLM and validated by the compiler — the compiler will reject any condition that
lacks a fully specified strategy. There is no default or fallback strategy. The LLM must always
resolve strategy explicitly using the guardrails priority order.

```
Every condition MUST include:
  strategy.type   — one of: threshold | percentile | z_score | change | equals | composite
  strategy.params — fully specified parameter object matching the strategy schema
These are LLM-generated at task creation time and compiler-validated before persisting.
```

| Strategy | Input Types | Key Param | Direction Values | Output Type |
|---|---|---|---|---|
| `threshold` | float, int | `value` | `above` \| `below` | decision\<boolean\> |
| `percentile` | float, int | `value` (0–100) | `top` \| `bottom` | decision\<boolean\> |
| `z_score` | float, int | `threshold` | `above` \| `below` \| `any` | decision\<boolean\> |
| `change` | float, int | `value` | `increase` \| `decrease` \| `any` | decision\<boolean\> |
| `equals` | categorical, string | `value` | N/A | decision\<categorical\> |
| `composite` | decision\<boolean\> | `operands` | `AND` \| `OR` | decision\<boolean\> |

**Critical notes:**
- `threshold` and `change` use param key `'value'` — NOT `'cutoff'`
- `z_score` uses param key `'threshold'` — NOT `'value'`
- `equals` calibration always returns `no_recommendation` (not_applicable_strategy)
- `composite` calibration always returns `no_recommendation` — calibrate the operand conditions individually

**Composite strategy constraints (compiler-enforced):**
```
composite operands MUST be decision<boolean> — this is strictly enforced:
  - equals conditions produce decision<categorical> → CANNOT be composite operands
  - attempting to use an equals condition as a composite operand → type_error
  - composite cannot be nested inside another composite's operands
  - operands list must contain at least 2 condition_ids
  - all operand conditions must exist in the registry at compile time → reference_error if not

Execution order for composite:
  1. Each operand condition is evaluated independently
  2. operator (AND | OR) is applied across the boolean results
  3. Result is a single decision<boolean>
```

### LLM Failure Handling at Task Creation

When `POST /tasks` invokes the LLM pipeline, failures must be handled explicitly:

```
LLM FAILURE CONTRACT (POST /tasks):

If the LLM produces an invalid definition (type_error, semantic_error, etc.):
  → Feed structured validation errors back to the LLM
  → Retry up to MAX_RETRIES (implementation-defined, recommended: 3)
  → On each retry: fix only the invalid parts, preserve original intent

If MAX_RETRIES is exceeded without a valid definition:
  → Return HTTP 422 with error.type = 'semantic_error'
  → Include the last validation error in error.message and error.location
  → Include a suggestion hint if available from the compiler
  → Do NOT persist any partial definition — no task_id is assigned
  → The system remains in a clean state as if the request was never made

If the LLM endpoint is unavailable:
  → Return HTTP 422 with error.type = 'execution_error'
  → Message: 'LLM endpoint unavailable — task creation requires LLM'
  → Do NOT attempt to create the task without LLM generation

IMPORTANT: This failure contract applies ONLY to POST /tasks.
All other endpoints (execution, calibration, feedback) do not invoke the LLM
and therefore cannot fail with an LLM-related error.
```

### Feedback Idempotency and Deduplication

```
FEEDBACK IDEMPOTENCY CONTRACT:

Feedback uniqueness is defined by the composite key:
  (condition_id, condition_version, entity, timestamp)

This key uniquely identifies a specific Decision (Aₜ). There is
exactly one correct feedback signal per decision — submitting
false_positive twice for the same decision is not valid.

Duplicate submission behaviour:
  Option A (recommended): reject with HTTP 409, error.type = 'conflict'
    → the caller must deduplicate before submitting
  Option B: accept but use only the most recent submission in aggregation
    → simpler for callers, slightly more complex in CalibrationService

The system MUST be consistent — choose one option and document it.
Mixing behaviours across deployments will corrupt calibration direction.

Why this matters for calibration:
  CalibrationService.derive_direction() counts false_positive vs
  false_negative votes to derive 'tighten' or 'relax'. Duplicate
  submissions inflate one side of the count and bias calibration
  toward more tightening or relaxing than the signal warrants.
  A condition with 3 real false_positives and 6 duplicate submissions
  of the same false_positive will calibrate far more aggressively
  than intended.

Implementation requirement:
  FeedbackStore must enforce uniqueness on
  (condition_id, condition_version, entity, timestamp) at the
  database level — unique constraint, not just application-level check.
```

### Calibration Analytics and System-Level Learning

The feedback → calibration loop operates at two levels:

**Local calibration** (per-condition, available now):
Each condition accumulates its own feedback and produces its own parameter adjustments.
This is what `POST /conditions/calibrate` implements.

**Platform-level learning** (aggregated, premium feature — Analytics Module):
Across all opt-in deployments, anonymised calibration patterns are aggregated:
- Parameter shift distributions per strategy type and domain
- Calibration convergence rates (how many cycles to stable)
- Benchmark priors — the median stable parameter value across similar conditions
- Alert volume benchmarks — typical alerts/entity/month for comparable conditions

This platform-level signal feeds back into guardrails default priors over time,
making the system's initial parameter selections better for everyone. It is the
foundation of the Analytics Module (Benchmark Priors, Domain Guardrails Packs,
Condition Templates with validated ranges). No customer data is exposed —
only anonymised structural metadata (strategy type, param ranges, calibration cycles).

### Calibration Flow

```
1. Accumulate feedback via POST /feedback/decision
   Valid values: false_positive | false_negative | correct
   Invalid: 'useful', 'not_useful' — these must be rejected

2. POST /conditions/calibrate
   → Analyses feedback + optional feedback_direction + optional target.alerts_per_day
   → Returns CalibrationResult with recommended_params + calibration_token (24h, single-use)
   → No recommendation for equals and composite strategies

3. POST /conditions/apply-calibration
   → Input: calibration_token ONLY (no condition_id, no threshold, no rebind_tasks)
   → Creates new condition version — NEVER mutates existing version
   → Returns ApplyCalibrationResult with tasks_pending_rebind (informational only)
   → Invalidates token after use

4. PATCH /tasks/{id}
   → User explicitly rebinds task to new condition_version
   → apply-calibration NEVER auto-rebinds tasks
```

### Namespace and Identifier Rules

```
- All definition IDs follow namespace.id format.
  namespace must match the prefix in id — mismatch returns HTTP 400.
  Examples:
    org.churn_risk          → namespace='org'
    team_alpha.event_spike  → namespace='team_alpha'

- Namespace hierarchy: personal → team → org → global
  Promotion follows this order (internal Python SDK only).

- Versions are ALWAYS explicit. No implicit 'latest' resolution.
  Every SDK call that accepts an id MUST also accept a version.
  HTTP 400 if version is missing or 'latest'.
```

### Error Handling Contract

Always branch on `error.type`. Never rely on `error.message` — it may change between versions.

| error.type | When it occurs | SDK behaviour |
|---|---|---|
| `syntax_error` | Invalid definition structure | Raise typed error |
| `type_error` | Type mismatch in operators or strategies | Raise typed error |
| `semantic_error` | Invalid parameters or semantic rules | Expose location + suggestion |
| `reference_error` | Missing operator/strategy/action/primitive | Raise typed error |
| `parameter_error` | Invalid parameter value (e.g. bad feedback value) | Raise typed error |
| `graph_error` | Circular dependency in concept DAG | Python only — raise GraphError |
| `execution_error` | Runtime failed: missing data, null propagation | Raise ExecutionError |
| `execution_timeout` | Exceeded 30s synchronous limit | Suggest switching to executeAsync() |
| `auth_error` | Missing or invalid API key | Raise before any network call |
| `not_found` | Concept/condition not at given id+version | Raise NotFoundError |
| `conflict` | Definition already exists at id+version | Raise ConflictError |
| `rate_limit_exceeded` | Too many requests | Expose retryAfterSeconds |
| `bounds_exceeded` | Calibration recommendation violates guardrail bounds | Raise typed error |
| `action_binding_failed` | No action could be resolved at task creation | Raise typed error |

---

## 1D — Engine Interfaces

Canonical contracts between the TypeScript API layer and the Python execution engine.

### ConceptExecutor

```python
# Python (authoritative)
class ConceptExecutor:
    async def execute(
        self,
        id: str,                          # 'org.churn_risk'
        version: str,                     # '1.2'
        entity: str,                      # 'user_abc123'
        timestamp: str | None = None,     # ISO 8601 UTC; None = snapshot mode
        explain: bool = False,
        explain_mode: str = 'full',       # summary | full | debug
        cache: bool = True,
        missing_data_policy: str | None = None,
    ) -> Result: ...
```

```typescript
// TypeScript (delegates to HTTP)
interface ConceptExecutor {
  execute(params: {
    id: string;
    version: string;
    entity: string;
    timestamp?: string;
    explain?: boolean;
    explainMode?: 'summary' | 'full' | 'debug';
    cache?: boolean;
    missingDataPolicy?: MissingDataPolicy;
  }): Promise<Result>;
}
```

### ConditionEvaluator

```python
# Python
class ConditionEvaluator:
    async def evaluate(
        self,
        condition_id: str,
        condition_version: str,
        entity: str,
        timestamp: str | None = None,
        dry_run: bool = False,
    ) -> DecisionResult: ...
```

```typescript
// TypeScript
interface ConditionEvaluator {
  evaluate(params: {
    conditionId: string;
    conditionVersion: string;
    entity: string;
    timestamp?: string;
    dryRun?: boolean;
  }): Promise<DecisionResult>;
}
```

### TaskAuthoringService (Python only)

```python
# Python — Interaction API backend
class TaskAuthoringService:
    async def create_task(self, req: CreateTaskRequest) -> Task:
        # 1. Load guardrails + application_context BEFORE any LLM call
        # 2. Build LLM context in order: type_system → guardrails → app_context
        #    → bias_rules → primitive_registry
        # 3. LLM generates Concept + Condition + Action within guardrail constraints
        # 4. Validate through compiler (retry loop on failure, max N retries)
        # 5. Register all three definitions
        # 6. Compile concept to execution graph
        # 7. Persist version-pinned Task
        # dry_run=True: return preview Task, nothing registered or persisted
        ...
```

---

## 1E — Canonical Schemas

Both SDKs must implement types structurally equivalent to these. Field names follow language
conventions (camelCase in TS, snake_case in Python) but all fields must be present.

### Result — output of ConceptExecutor.execute

```python
# Python
class Result(BaseModel):
    value: float | bool | str          # computed concept output
    type: str                          # 'float' | 'boolean' | 'categorical'
    entity: str
    version: str
    deterministic: bool                # True when timestamp was provided
    timestamp: str | None = None       # ISO 8601, present when deterministic
    explanation: Explanation | None = None  # populated when explain=True
    metadata: dict | None = None       # compute_time_ms, cache_hit, nodes_executed
```

```typescript
// TypeScript
interface Result {
  value: number | boolean | string;
  type: 'float' | 'boolean' | 'categorical';
  entity: string;
  version: string;
  deterministic: boolean;
  timestamp?: string;
  explanation?: Explanation;
  metadata?: ResultMetadata;
}
```

### DecisionResult — output of ConditionEvaluator.evaluate

```python
# Python
class DecisionResult(BaseModel):
    value: bool | str                  # bool for boolean strategies; label for equals
    type: str                          # 'boolean' | 'categorical'
    entity: str
    condition_id: str
    condition_version: str
    timestamp: str | None = None
    actions_triggered: list[ActionTriggered] = []  # ALWAYS nested here, never top-level

class ActionTriggered(BaseModel):
    action_id: str
    action_version: str
    status: str  # 'triggered' | 'skipped' | 'failed' | 'would_trigger'
    payload_sent: dict | None = None
    error: ErrorResponse | None = None
```

```typescript
// TypeScript
interface DecisionResult {
  value: boolean | string;
  type: 'boolean' | 'categorical';
  entity: string;
  conditionId: string;
  conditionVersion: string;
  timestamp?: string;
  actionsTriggered: ActionTriggered[];  // ALWAYS nested here, never top-level
}

interface ActionTriggered {
  actionId: string;
  actionVersion: string;
  status: 'triggered' | 'skipped' | 'failed' | 'would_trigger';
  payloadSent?: object;
  error?: ErrorResponse;
}
```

### FullPipelineResult

```python
# Python
class FullPipelineResult(BaseModel):
    result: Result                     # Rₜ — concept output
    decision: DecisionResult           # Aₜ — condition output (contains actions_triggered)
    dry_run: bool | None = None
    # IMPORTANT: there is no top-level actions_triggered on FullPipelineResult
    # actions_triggered lives ONLY inside decision:
    #   result.decision.actions_triggered   ✅
    #   result.actions_triggered            ❌ does not exist
```

```typescript
// TypeScript
interface FullPipelineResult {
  result: Result;
  decision: DecisionResult;           // actionsTriggered nested here — NEVER at top level
  dryRun?: boolean;
  // WARNING: result.actionsTriggered does not exist
  // CORRECT:  result.decision.actionsTriggered
  // INCORRECT: result.actionsTriggered
}
```

### Task — primary Interaction API object

```python
# Python
class Task(BaseModel):
    task_id: str | None = None         # None for preview (dry_run) tasks
    intent: str
    concept_id: str
    concept_version: str
    condition_id: str
    condition_version: str             # version-pinned — never auto-updates
    action_id: str
    action_version: str
    entity_scope: str
    delivery: DeliveryConfig
    status: str                        # 'active' | 'paused' | 'deleted' | 'preview'
    created_at: str | None = None      # None for preview tasks
    last_triggered_at: str | None = None

class DeliveryConfig(BaseModel):
    type: str                          # 'webhook' | 'notification' | 'email' | 'workflow'
    endpoint: str | None = None
    channel: str | None = None

class ConstraintsConfig(BaseModel):
    sensitivity: str | None = None    # 'low' | 'medium' | 'high'
    namespace: str | None = None      # 'personal' | 'team' | 'org' | 'global'
```

```typescript
// TypeScript
type TaskStatus = 'active' | 'paused' | 'deleted' | 'preview';

interface Task {
  taskId?: string;                    // undefined for preview tasks
  intent: string;
  conceptId: string;
  conceptVersion: string;
  conditionId: string;
  conditionVersion: string;
  actionId: string;
  actionVersion: string;
  entityScope: string;
  delivery: DeliveryConfig;
  status: TaskStatus;
  createdAt?: string;
  lastTriggeredAt?: string | null;
}
```

### CalibrationResult — output of POST /conditions/calibrate

```python
# Python
class CalibrationResult(BaseModel):
    status: str                        # 'recommendation_available' | 'no_recommendation'
    recommended_params: dict | None = None        # generic dict — strategy-aware keys
    calibration_token: str | None = None          # single-use, 24h expiry
    current_params: dict                          # current strategy parameters
    impact: CalibrationImpact | None = None
    no_recommendation_reason: str | None = None  # 'bounds_exceeded' |
                                                 # 'not_applicable_strategy' |
                                                 # 'insufficient_data'

class CalibrationImpact(BaseModel):
    delta_alerts: float                # estimated change in daily alert volume
    direction: str                     # 'increase' | 'decrease'
```

```typescript
// TypeScript
type CalibrationStatus = 'recommendation_available' | 'no_recommendation';
type NoRecommendationReason = 'bounds_exceeded' | 'not_applicable_strategy' | 'insufficient_data';

interface CalibrationResult {
  status: CalibrationStatus;
  recommendedParams?: Record<string, unknown>;
  calibrationToken?: string;          // single-use, expires 24h
  currentParams: Record<string, unknown>;
  impact?: CalibrationImpact;
  noRecommendationReason?: NoRecommendationReason;
}

interface CalibrationImpact {
  deltaAlerts: number;
  direction: 'increase' | 'decrease';
}
```

### ApplyCalibrationResult — output of POST /conditions/apply-calibration

```python
# Python
class ApplyCalibrationResult(BaseModel):
    condition_id: str
    previous_version: str
    new_version: str
    params_applied: dict               # NOT 'applied_params' — params_applied
    tasks_pending_rebind: list[dict]   # [{task_id, intent}] — informational only
    # IMPORTANT: no 'status' field on this model
    # IMPORTANT: no rebind_tasks parameter — rebinding is always explicit via PATCH /tasks/{id}
```

```typescript
// TypeScript
interface ApplyCalibrationResult {
  conditionId: string;
  previousVersion: string;
  newVersion: string;
  paramsApplied: Record<string, unknown>;
  tasksPendingRebind: Array<{ taskId: string; intent: string }>;
}
```

### FeedbackValue and FeedbackResponse

```python
# Python
class FeedbackValue(str, Enum):
    FALSE_POSITIVE = 'false_positive'  # condition fired but should not have → tighten
    FALSE_NEGATIVE = 'false_negative'  # condition did not fire but should have → relax
    CORRECT        = 'correct'         # expected behaviour → no calibration effect
# 'useful' and 'not_useful' are NOT valid values — raise parameter_error if received

class FeedbackResponse(BaseModel):
    status: str   # 'recorded'
    feedback_id: str
```

```typescript
// TypeScript
type FeedbackValue = 'false_positive' | 'false_negative' | 'correct';

interface FeedbackResponse {
  status: 'recorded';
  feedbackId: string;
}
```

### DecisionExplanation — output of POST /decisions/explain

```python
# Python
class DecisionExplanation(BaseModel):
    condition_id: str
    condition_version: str
    entity: str
    timestamp: str
    decision: bool | str               # bool or matched label
    decision_type: str                 # 'boolean' | 'categorical'
    concept_value: float | str
    strategy_type: str
    threshold_applied: float | None    # None for equals and composite
    label_matched: str | None          # None for non-equals strategies
    drivers: list[Driver]              # contributions must sum to 1.0

class Driver(BaseModel):
    signal: str
    contribution: float                # must sum to 1.0 across all drivers
    value: float | str
```

```typescript
// TypeScript
interface DecisionExplanation {
  conditionId: string;
  conditionVersion: string;
  entity: string;
  timestamp: string;
  decision: boolean | string;
  decisionType: 'boolean' | 'categorical';
  conceptValue: number | string;
  strategyType: ConditionStrategyType;
  thresholdApplied?: number;          // undefined for equals and composite
  labelMatched?: string;              // defined for equals only
  drivers: Driver[];
}

interface Driver {
  signal: string;
  contribution: number;               // all contributions must sum to 1.0
  value: number | string;
}
```

### Explanation — output of POST /explain

```python
# Python
class Explanation(BaseModel):
    output: float | bool | str
    nodes: list[ExplanationNode]
    contributions: dict[str, float]   # signal → contribution
    trace: list[TraceStep]
    metadata: dict | None = None
```

```typescript
// TypeScript
interface Explanation {
  output: number | boolean | string;
  nodes: ExplanationNode[];
  contributions: Record<string, number>;
  trace: TraceStep[];
  metadata?: Record<string, unknown>;
}
```

### ErrorResponse

```python
# Python
class ErrorResponse(BaseModel):
    type: str         # machine-readable — always branch on this
    message: str      # human-readable — never branch on this
    location: str | None = None
    suggestion: str | None = None    # actionable remediation hint
```

```typescript
// TypeScript
type ErrorType =
  | 'syntax_error' | 'type_error' | 'semantic_error' | 'reference_error'
  | 'parameter_error' | 'graph_error' | 'execution_error' | 'execution_timeout'
  | 'auth_error' | 'not_found' | 'conflict' | 'rate_limit_exceeded'
  | 'bounds_exceeded' | 'action_binding_failed';

interface ErrorResponse {
  type: ErrorType;      // branch on this
  message: string;      // do not branch on this
  location?: string;
  suggestion?: string;  // actionable remediation hint
}
```

---

## 1F — Guardrails System

The guardrails system is the administrative policy layer that constrains all LLM output during
task creation. It is loaded by Python at startup via `memintel apply`. TypeScript never accesses
it directly.

```
Guardrails govern:
  - strategy_registry        Available strategies and their parameter schemas
  - type_strategy_map        Valid strategies per input type
  - threshold_priors         Default parameter values by severity level
  - threshold_bounds         Hard limits calibration can never cross
  - parameter_bias_rules     Deterministic NL vocabulary → severity shift mappings
  - strategy_selection_priority:
      1. user_explicit       (always wins)
      2. primitive_hint
      3. mapping_rule
      4. application_context
      5. global_preferred
      6. global_default      (always fallback)

LLM context injection order (Python — strict, do not reorder):
  [1] Type system (hard rules)
  [2] Guardrails (strategy registry, bounds, priors, bias rules)
  [3] Application context (semantic guidance)
  [4] Parameter bias rules (severity shift mappings)
  [5] Primitive registry (available data signals)

The LLM resolves within these constraints. It does not freely choose strategies
or parameters. The compiler validates all LLM output before any task is persisted.
```

---

## 1H — Observability Contract

Every production execution path must emit structured logs. These are the audit trail for debugging and compliance.

```
REQUIRED LOG EVENTS:

concept_executed:
  fields: concept_id, version, entity, timestamp, deterministic,
          cache_hit, compute_time_ms, result_type

condition_evaluated:
  fields: condition_id, condition_version, entity, timestamp,
          decision_value, decision_type, strategy_type,
          params_applied, actions_triggered_count

calibration_recommended:
  fields: condition_id, condition_version, strategy_type,
          old_params, recommended_params, delta_alerts,
          feedback_direction

calibration_applied:
  fields: condition_id, previous_version, new_version,
          params_applied, tasks_pending_rebind count

memintel_error:
  fields: error_type, location, entity (if available),
          concept_id (if available), condition_id (if available)

NEVER LOG:
  - Resolved credential values (${ENV_VAR} values)
  - Raw primitive data or entity attribute values
  - Feedback note fields (may contain PII)
  - Strategy params that include raw threshold values from config
    (params_applied is acceptable — it's a dict of final values, not credentials)
```

## 1G — Alignment Test

Verify every row passes before considering the implementation complete.

| Test | TypeScript | Python |
|---|---|---|
| evaluateFull with timestamp | `result.deterministic === true` | `result.deterministic == True` |
| evaluateFull without timestamp | `result.deterministic === false` | `result.deterministic == False` |
| dryRun: true | `actionsTriggered[].status === 'would_trigger'`, no action HTTP calls | Same |
| threshold condition > 0.8 | `decision.value === true` when score = 0.87 | Same |
| z_score condition | `decision.value === true` when z > threshold | Python implements z_score strategy |
| change strategy | fires on transition (increase/decrease/any), not every period | Python implements this correctly |
| percentile condition | `decision.value === true` when value in top/bottom X% | Python implements relative ranking |
| equals condition | `decision.value === 'high_risk'` when categorical matches | Python implements categorical match |
| composite condition | `decision.value === true` when AND/OR of operands is true | Python implements logical combination |
| actions_triggered | Always nested inside decision, never at top level | Same |
| evaluateConditionBatch | Returns `DecisionResult[]`, one per entity | Same |
| executeRange with interval | Returns time-ordered `Result[]` | Same |
| executeAsync + getJob | Job accepted, polling returns completed result | Same |
| explain: true | `result.explanation.contributions` is populated | Same |
| tasks.create() dry_run | Returns Task with status='preview', task_id=undefined | Returns Task with status='preview', task_id=None |
| conditions.calibrate() on equals | `status === 'no_recommendation'` | `status == 'no_recommendation'` |
| conditions.calibrate() on composite | `status === 'no_recommendation'` | `status == 'no_recommendation'` |
| conditions.applyCalibration() | Returns paramsApplied + tasksPendingRebind | Returns params_applied + tasks_pending_rebind |
| feedback with 'useful' | Throws/raises parameter_error | Raises parameter_error |
| ir_hash mismatch | N/A (TS does not call /execute/graph) | Returns HTTP 409 |
| Error: not_found | Throws `MemintelError` with `type === 'not_found'` | Raises `MemintelError` with `type == 'not_found'` |
| Error: rate_limit_exceeded | Exposes `retryAfterSeconds` on error | Same |
| Error: bounds_exceeded | Throws typed error | Raises typed error |
| Error: action_binding_failed | Throws typed error | Raises typed error |
