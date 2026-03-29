# Memintel — Claude Code Implementation Process

### Revised and Complete · All feedback incorporated

\---

## The Guiding Principle

What makes this process work is not just sequencing — it is **constraint injection**. Every prompt reduces Claude Code's degrees of freedom: it cites the source document, states the invariants, defines what must not happen, and scopes to exactly one component. This turns vibe coding into something closer to deterministic compilation. The narrower the prompt, the more reliable the output.

\---

## Pre-Flight: Context Package

Before writing a single prompt, assemble the complete context folder. Claude Code's output quality is directly proportional to its input quality. Every document must be present before session 1 begins.

```
/memintel-context
  core-spec.md                ← execution model, all schemas, contracts
  py-instructions.md          ← Python backend implementation guide
  ts-instructions.md          ← TypeScript SDK implementation guide
  memintel\_type\_system.md     ← type system v1.1 — all types and rules
  memintel.guardrails.md      ← guardrails v1.4 — strategy registry, priors
  memintel.config.md          ← admin config file format + example
  llm-integration.md          ← LLM prompt templates + output schemas
  developer\_api.yaml          ← App Developer API OpenAPI spec
  Memintel\_InternalPlatform\_API\_Reference.docx  ← Internal Platform API reference
  persistence-schema.md       ← DB schema, table definitions, indexes
  llm\_fixtures/               ← fixed JSON fixtures for LLM output stubs
    threshold\_task.json
    z\_score\_task.json
    composite\_task.json
    equals\_task.json
```

**Never ask Claude Code to remember something** — always tell it exactly which document and section to read. If a document is missing, create it before that session, not during.

\---

## Layer Map

Every session implements exactly one layer. Each layer depends only on layers below it. This prevents the "imports that don't exist yet" problem that breaks long sessions.

```
Layer 0:  Repository structure           (Session 1)
Layer 1:  Pydantic models + schemas      (Session 2)
Layer 1.5 Persistence layer              (Session 3)   ← NEW — non-optional
Layer 2:  Type system + compiler         (Sessions 4–5)
Layer 3:  Condition strategies           (Sessions 6a–6f)
Layer 3.5 LLM output fixtures            (Session 7)   ← NEW — stabilizes everything above
Layer 4:  Runtime + execution engine     (Session 8)
Layer 5:  Registry                       (Session 9)
Layer 6:  Interaction API services       (Sessions 10a–10c)
Layer 7:  FastAPI routes                 (Session 11)
Layer 8:  TypeScript SDK                 (Session 12)
Layer 9:  Integration tests              (Sessions 13–14) ← NEW — full pipeline
```

\---

## Session 1 — Repository Structure

One job only: create the folder structure and empty module files. Do not implement any logic.

**Prompt:**

```
Read all files in /memintel-context.

Based on py-instructions.md, developer\_api.yaml, and
persistence-schema.md, create the repository structure for
the Memintel Python backend.

Do not implement any logic yet — only create the folder
structure, empty module files, and a requirements.txt with
the correct dependencies.

Stack: FastAPI, Pydantic v2, asyncio, asyncpg, aioredis,
       httpx, pytest, pytest-asyncio.

Structure:
/memintel-backend
  /app
    /api/routes/
      execute.py, compile.py, registry.py, agents.py
      tasks.py, conditions.py, decisions.py
      feedback.py, actions.py, jobs.py
    /compiler/
      type\_checker.py, dag\_builder.py
      ir\_generator.py, validator.py
    /runtime/
      executor.py, condition\_evaluator.py
      action\_trigger.py, data\_resolver.py, cache.py
    /strategies/
      base.py, threshold.py, percentile.py
      z\_score.py, change.py, equals.py, composite.py
    /registry/
      definitions.py, features.py
    /models/
      task.py, condition.py, concept.py
      result.py, errors.py, calibration.py
    /persistence/
      db.py, stores.py, migrations/
    /services/
      task\_authoring.py, calibration.py
      feedback.py, explanation.py
    /llm/
      client.py, prompts.py, fixtures.py
    /config/
      guardrails\_store.py, primitive\_registry.py
    main.py
  /tests/
    /unit/, /integration/, /fixtures/
  requirements.txt
  alembic.ini
  README.md
```

**Note on task execution:** Memintel does not include a scheduler. Tasks define what to evaluate — the application layer owns when. The repository structure reflects this: there is no `/scheduler` module. The application calls `evaluateFull()` on its own schedule.

**Review carefully** — structure is much harder to reorganise later than logic.

\---

## Session 2 — Layer 1: Pydantic Models

**Prompt:**

```
Read py-instructions.md and developer\_api.yaml.

Implement all Pydantic v2 models in /app/models/.

Rules:
- All field names snake\_case
- Required fields have no default
- Optional fields declare explicit defaults (None or value)
- Implement exactly the schemas defined in developer\_api.yaml
- Include ALL of:
    Task, DeliveryConfig, ConstraintsConfig, TaskUpdateRequest,
    StrategyDefinition, ConditionDefinition,
    Result, DecisionResult, FullPipelineResult, ActionTriggered,
    CalibrationResult, ApplyCalibrationResult,
    CalibrateRequest, ApplyCalibrationRequest,
    FeedbackRequest, FeedbackResponse,
    DecisionExplanation, ConditionExplanation,
    ErrorResponse, ValidationResult, ValidationError,
    DryRunResult, Job, JobResult, BatchExecuteResult

Do not implement any logic. Models only.

After implementing, add a models/\_\_init\_\_.py that exports
every model so they can be imported from a single location.
```

**After it runs:** check every model field name against `developer\_api.yaml`. This is the foundation — errors here propagate into every other layer.

\---

## Session 2.5 — Config \& Environment Bootstrapping ← NEW

This session wires the configuration and guardrails loading that everything else depends on. The system must refuse to start if configuration is invalid or incomplete — partial config is never acceptable.

**Prompt:**

```
Read py-instructions.md section "System Setup — memintel apply
Implementation" and memintel.config.md.

Implement the config and environment bootstrapping layer in
/app/config/:

config\_loader.py:
  class ConfigLoader:
    def load(self, config\_path: str) -> AppConfig:
      # 1. Parse memintel.config.md (Markdown with embedded YAML)
      # 2. Validate against ConfigSchema (Pydantic v2)
      #    → raise ConfigError if invalid — DO NOT proceed
      # 3. Resolve all ${ENV\_VAR} references
      #    → raise ConfigError if any referenced env var is missing
      #    → NEVER log resolved credential values
      # 4. Validate each connector is reachable (optional health check)
      # 5. Return validated AppConfig

guardrails\_store.py:
  class GuardrailsStore:
    async def load(self, guardrails\_path: str):
      # Parse memintel.guardrails.md
      # Validate against guardrails schema
      # Extract and store application\_context separately
      # Store in memory — immutable after load

    def get\_guardrails(self) -> Guardrails
    def get\_application\_context(self) -> ApplicationContext
    def get\_strategy\_registry(self) -> dict
    def get\_threshold\_bounds(self, strategy: str) -> dict

primitive\_registry.py:
  class PrimitiveRegistry:
    def load\_from\_config(self, config: AppConfig)
    def get(self, name: str) -> PrimitiveConfig | None
    def list\_all(self) -> list\[PrimitiveConfig]
    def get\_type(self, name: str) -> str  # returns Memintel type string

Startup invariant — enforce in /app/main.py startup event:
  SYSTEM MUST NOT START if any of these fail:
    - config file missing or invalid schema
    - any ${ENV\_VAR} reference unresolved
    - guardrails file missing or invalid
    - strategy registry empty
  On failure: log the specific error and exit with non-zero code
  DO NOT start with partial configuration under any circumstances
  DO NOT silently fall back to defaults

Write tests in /tests/unit/test\_config.py:
  - Valid config loads without error
  - Missing ${ENV\_VAR} raises ConfigError
  - Invalid config schema raises ConfigError (not a warning)
  - Guardrails loads strategy registry correctly
  - GuardrailsStore.get\_threshold\_bounds returns correct values
  - System refuses to start with empty strategy registry
```

**After this session:** verify that `guardrails\_store` and `primitive\_registry` are populated before any other session's code runs. The TaskAuthoringService (Session 10a) depends on both.

\---

## Session 3 — Layer 1.5: Persistence Layer ← CRITICAL, NON-OPTIONAL

This step was missing from the original process. Without an explicit persistence layer, Claude Code will invent schemas, mix mutable and immutable models, and break versioning guarantees. Do this before any business logic.

**Prompt:**

```
Read persistence-schema.md and py-instructions.md.

Implement the persistence layer in /app/persistence/.

db.py:
- Async connection pool using asyncpg (PostgreSQL)
- Redis connection using aioredis for cache
- get\_db\_pool() and get\_redis\_client() as FastAPI dependencies
- Connection health check

stores.py — implement these store classes:
  TaskStore:
    - create(task) → Task
    - get(task\_id) → Task | None
    - list(status, limit, cursor) → TaskList
    - update(task\_id, updates) → Task
    - find\_by\_condition\_version(condition\_id, version) → list\[Task]

  DefinitionStore:
    - register(definition, namespace) → DefinitionResponse
      RULE: 409 if id+version already exists (IMMUTABLE)
    - get(id, version) → definition | None
    - versions(id) → list\[VersionSummary] newest-first
    - list(type, namespace, limit, cursor) → paginated

  FeedbackStore:
    - create(record) → FeedbackRecord
    - get\_by\_condition(condition\_id, version) → list\[FeedbackRecord]
    - find(condition\_id, entity, timestamp) → FeedbackRecord | None

  CalibrationTokenStore:
    - create(token) → CalibrationToken
    - resolve(token\_string) → CalibrationToken | None
    - invalidate(token\_string)
    RULE: tokens are single-use with 24h expiry

  ResultCache:
    - Cache key MUST be: (concept\_id, version, entity, timestamp)
    - None timestamp is a DIFFERENT key from any specific timestamp
    - get(key) → Result | None
    - set(key, result, ttl) — deterministic: indefinite TTL;
      snapshot: do NOT cache at all
    - invalidate(key)

  GraphStore:
    - store(graph) → graph\_id
    - get(graph\_id) → ExecutionGraph | None

  JobStore:
    - enqueue(job) → Job
    - get(job\_id) → Job | None
    - update\_status(job\_id, status, result?, error?)
    - cancel(job\_id)

migrations/:
- Create Alembic migration for initial schema
- Tables: tasks, definitions, feedback\_records,
  calibration\_tokens, execution\_graphs, jobs
- All tables must have created\_at, updated\_at
- Definition table must enforce (id, version) uniqueness constraint

CONCURRENCY SAFETY RULES — enforce in all store implementations:

  Calibration tokens MUST be atomically invalidated:
    Use a database transaction or Redis SET NX (set-if-not-exists)
    Concurrent token redemption must guarantee exactly one succeeds
    Second concurrent attempt must fail with parameter\_error, not silently

  Task updates MUST be transactional:
    Read-modify-write must not allow concurrent updates to interleave
    Use SELECT FOR UPDATE or optimistic locking with version field
    Concurrent update of same task\_id must be serialised

  Cache writes MUST be idempotent:
    Writing the same key+value twice must produce the same state
    No race condition between cache miss check and cache write

  Job status transitions MUST be atomic:
    queued → running → completed/failed/cancelled
    No job may transition backwards (completed → running is invalid)
    Use a status transition table to enforce valid transitions

Write unit tests in /tests/unit/test\_stores.py:
- 409 on duplicate definition registration
- Token single-use enforcement (concurrent redemption: exactly one wins)
- Cache key distinction (None vs timestamp → different keys)
- Task immutability (concept\_id, condition\_id cannot be updated)
- Job status cannot transition backwards (completed → running raises error)
```

**Review:** check that the definition store enforces immutability at the DB level (unique constraint), not just in application code.

\---

## Session 4 — Layer 2a: Type Checker

**Prompt:**

```
Read memintel\_type\_system.md and py-instructions.md
section "Type System Enforcement".

Implement TypeChecker in /app/compiler/type\_checker.py.

Implement:
- is\_assignable(actual, expected) with ALL subtype rules:
    int → float (widening, implicit)
    T → T? (nullable widening)
    time\_series<int> → time\_series<float>
    list<int> → list<float>
    float → int requires to\_int() → raise type\_error otherwise
    T? → T requires null handler → raise type\_error otherwise
- check\_node(node, input\_types) → output type | raises type\_error
- All null propagation rules:
    if any input is T?, output is T? unless operator handles null
    coalesce, fill\_null, drop\_null explicitly consume T?

Write unit tests in /tests/unit/test\_type\_checker.py covering:
- int → float widening
- T → T? widening
- time\_series<int> → time\_series<float>
- Invalid: float → int without to\_int() raises type\_error
- Null propagation through operator chain
- decision<boolean> cannot be used as operator input → type\_error
- equals strategy on float input → type\_error
- composite strategy on decision<categorical> operand → type\_error
```

\---

## Session 5 — Layer 2b: Compiler Pipeline

**Prompt:**

```
Read py-instructions.md "Compiler Layer" section and
core-spec.md section 1B (Execution Model).

Implement the compilation pipeline in /app/compiler/:

validator.py — validation pipeline in strict order:
  validate\_schema()     → syntax\_error on malformed input
  validate\_operators()  → reference\_error if op not in registry
  validate\_types()      → type\_error via TypeChecker
  validate\_strategies() → reference\_error, type\_error, semantic\_error
  validate\_actions()    → reference\_error, semantic\_error
  validate\_graph()      → graph\_error on cycles or disconnected nodes

dag\_builder.py:
  - build\_dag(definition) → ExecutionGraph
  - Topological sort — execution order MUST be deterministic
  - Circular dependency detection is MANDATORY → graph\_error
  - Independent nodes identified as parallelizable\_groups
  - parallelizable\_groups determined at compile time, not runtime

ir\_generator.py:
  - hash\_graph(graph) → ir\_hash string
  - INVARIANT: same definition version → same ir\_hash on any machine
  - Use canonical serialisation: sorted keys, stable field order
  - Hash algorithm: SHA-256

Also implement compile\_explain\_plan():
  - Returns ExecutionPlan with steps, parallelizable\_groups,
    execution\_order, optimizations\_applied
  - Does NOT execute anything — inspect-only

Optimisation passes (minimum viable):
  - node\_deduplication: deduplicate identical subgraph nodes
  - dead\_node\_elimination: remove nodes not on any output path

Write tests in /tests/unit/test\_compiler.py:
- Same definition always produces same ir\_hash
- Circular dependency raises graph\_error
- Dead node elimination removes unreachable nodes
- Topological order is correct for chained dependencies
```

\---

## Sessions 6a–6f — Layer 3: Condition Strategies

One session per strategy. Never batch strategies — each has specific semantics that require individual verification.

**Template prompt (fill in STRATEGY\_NAME for each):**

```
Read py-instructions.md section "Condition Strategies —
STRATEGY\_NAME" and memintel\_type\_system.md.

Implement STRATEGY\_NAME strategy in /app/strategies/STRATEGY\_NAME.py.

Inherit from ConditionStrategy base class in /app/strategies/base.py.

Specification: \[paste exact spec from py-instructions.md]

Contract:
- evaluate(result, history, params) → DecisionValue
- Returns DecisionValue with provenance (condition\_id, version, entity)
- Does NOT return raw bool or str — DecisionValue only
- Raises type\_error if input type is invalid for this strategy
- Raises semantic\_error if params are missing or wrong type

Write tests in /tests/unit/test\_strategy\_STRATEGY\_NAME.py covering:
- Happy path: fires correctly given threshold
- Does NOT fire: value on wrong side of threshold
- type\_error: invalid input type
- semantic\_error: missing required param
- Output is DecisionValue, not bool
- decision\_type is correct (boolean or categorical)
```

**Strategies to implement in order:** threshold → percentile → z\_score → change → equals → composite

**After all six:** write a cross-strategy integration test:

```
Read py-instructions.md section "Condition Strategies" and
memintel\_type\_system.md.

Write /tests/integration/test\_strategies\_type\_compatibility.py:
- Run each strategy through TypeChecker to verify input type enforcement
- Verify equals rejects float/int/boolean input → type\_error
- Verify composite rejects decision<categorical> operand → type\_error
- Verify composite rejects nested composite → semantic\_error
- Verify all 6 strategies return DecisionValue (not raw bool)
- Verify decision\_type matches strategy output contract
```

\---

## Session 7 — Layer 3.5: LLM Output Fixtures ← NEW, CRITICAL

This step was missing from the original process. Without fixtures, Claude Code will invent inconsistent LLM output structures that break the compiler, strategies, and task creation flow. These fixtures are the ground truth for the task authoring pipeline during development.

**Prompt:**

```
Read developer\_api.yaml schemas for Task, ConditionDefinition,
and StrategyDefinition. Read py-instructions.md section
"POST /tasks" and "Condition Strategies".

Create four JSON fixtures in /app/llm/fixtures/ representing
valid LLM output for task creation. These are used as stubs
during development instead of calling a real LLM.

Each fixture must produce output that:
- Passes the full compiler validation pipeline
- Has a fully specified strategy.type and strategy.params
- Has a resolvable action binding
- Represents a realistic use case

fixtures/threshold\_task.json:
{
  "concept": { ... },      // concept definition for churn risk score
  "condition": {
    "id": "org.high\_churn\_risk",
    "strategy": {
      "type": "threshold",
      "params": { "direction": "above", "value": 0.80 }
    },
    ...
  },
  "action": { "id": "org.notify\_team", "version": "1.0" }
}

fixtures/z\_score\_task.json:
  // concept for payment failure rate
  // z\_score strategy: threshold 2.5, direction above, window 30d

fixtures/composite\_task.json:
  // concept for risk\_score
  // composite: AND of two boolean conditions
  // IMPORTANT: operands must be decision<boolean> conditions
  // NOT equals conditions — composite cannot wrap equals

fixtures/equals\_task.json:
  // concept with categorical output (risk\_category)
  // equals strategy: value "high\_risk"

After creating fixtures, implement LLMFixtureClient in
/app/llm/fixtures.py:
  class LLMFixtureClient:
      def generate\_task(self, intent: str, context: dict) -> dict:
          # Routes to fixture based on keywords in intent
          # Returns the appropriate fixture as parsed dict
          # Used by TaskAuthoringService when USE\_LLM\_FIXTURES=True
```

**Why this matters:** every session after this that touches task creation will use these fixtures. Inconsistent fixtures create cascading failures. Verify each fixture passes compiler validation before moving on.

\---

## Session 8 — Layer 4: Runtime and Execution Engine

**Prompt:**

```
Read py-instructions.md sections:
- "Deterministic Execution — Implementation Rules"
- "Caching Layer — Implementation Rules"
- "Action Execution — Best-Effort Contract"
- core-spec.md "DETERMINISM CONTRACT" and "DAG Execution
  Order Guarantee"

Implement the execution engine in /app/runtime/:

executor.py:
- execute(id, version, entity, timestamp, explain, cache,
  missing\_data\_policy) → Result
- DETERMINISM CONTRACT:
    timestamp present → deterministic=True, cache indefinitely
    timestamp absent → deterministic=False, DO NOT cache at all
    same inputs → always same output (hard invariant)
- Execution follows topological order from ExecutionGraph
- Independent nodes MAY execute concurrently (parallelizable\_groups)
- MUST NOT call LLM — any LLM call here is a critical bug

condition\_evaluator.py:
- evaluate(condition\_id, condition\_version, entity, timestamp,
  dry\_run) → DecisionValue
- Implicitly executes concept if not in cache (transparent)

action\_trigger.py:
- trigger\_bound\_actions(decision, condition, dry\_run)
  → list\[ActionTriggered]
- MUST NOT block pipeline response — fire and forget
- dry\_run=True: status='would\_trigger', no HTTP call made
- Failures: status='failed', error captured, never raised
- Skipped: status='skipped' (fire\_on mismatch)

cache.py:
- Cache key MUST be exactly: (concept\_id, version, entity, timestamp)
- None timestamp is a DIFFERENT key from any specific timestamp
- TTL: deterministic results → indefinite; snapshot → never cache

data\_resolver.py:
- fetch(primitive\_name, entity\_id, timestamp) → PrimitiveValue
- Apply missing\_data\_policy: null|zero|forward\_fill|backward\_fill
- Retry with exponential backoff on transient connector failures
- For now: implement with a MockConnector that returns test data
- Connector interface: stub with NotImplementedError for real connectors

Write tests in /tests/unit/test\_executor.py:
- Same inputs produce same output (run 3 times, assert identical)
- Timestamp present → deterministic=True
- Timestamp absent → deterministic=False
- Snapshot result NOT stored in cache across requests
- dry\_run → actions have status='would\_trigger'
- Action failure does NOT fail the pipeline (HTTP 200)
- Actions do NOT retry on failure (at-most-once per evaluation call)
- Same evaluation called twice fires actions twice (no cross-call dedup — caller responsibility)

Write tests in /tests/unit/test\_data\_resolver.py:
- timestamp fetch returns point-in-time data (not current state)
- forward\_fill: returns last known value when current is null
- backward\_fill: returns next known value when current is null
- null policy: returns T? (nullable) when data missing
- zero policy: returns 0 (non-nullable) when data missing
- retry logic: transient connector failure triggers backoff + retry
- auth failure: does NOT retry, fails immediately
- rate limit: retries with delay
- batch: multiple primitives for same entity fetched in one request
- request-scoped cache: same primitive+entity fetched only once per request
```

\---

## Session 9 — Layer 5: Registry

**Prompt:**

```
Read Memintel\_InternalPlatform\_API\_Reference.docx (Section 2 — Internal Platform,
Registry endpoints) and py-instructions.md "Registry Governance".

Implement /app/registry/definitions.py using DefinitionStore
from /app/persistence/stores.py.

register(definition, namespace) → DefinitionResponse
  - Returns HTTP 409 if id+version already exists
  - Definitions are IMMUTABLE — no in-place mutation, ever
  - Before registering: verify definition passes compiler validation
    (call validator.validate\_schema() + validate\_types())
  - This "definition freezing" check prevents garbage entering registry

get(id, version) → definition | HTTP 404
  - No implicit "latest" resolution — version is always required

list(type, namespace, tags, limit, cursor) → paginated
versions(id) → VersionListResult newest-first

deprecate(id, version, replacement\_version, reason)
  - Marks version as deprecated, does NOT delete it
  - Existing references continue to work
  - New registrations referencing this version get a warning

promote(id, version, from\_namespace, to\_namespace)
  - Namespace path: personal → team → org → global
  - Promoting to global requires elevated\_key=True on request context
  - Run semantic\_diff before promotion, block on 'breaking' status

semantic\_diff(id, version\_from, version\_to) → SemanticDiffResult
  - equivalence\_status: equivalent | compatible | breaking | unknown

Also implement /app/registry/features.py:
  register\_feature(feature, on\_duplicate) → FeatureRegistrationResult
  - on\_duplicate: 'warn' | 'reject' | 'merge'
  - Compute meaning\_hash on registration
  - Check for hash collision before registering

Write tests in /tests/unit/test\_registry.py:
- 409 on duplicate id+version
- Immutability: cannot update registered definition
- versions() returns newest first
- promote() to global fails without elevated key
- semantic\_diff returns 'equivalent' for identical definitions
- register() rejects unvalidated definitions (frozen check)
```

\---

## Session 10a — Layer 6: Task Authoring Service

**Prompt:**

```
Read py-instructions.md section "POST /tasks" and
"Task-Centric Execution Architecture".
Read llm-integration.md for prompt structure.
Read /app/llm/fixtures/ for the fixture contracts.

Implement TaskAuthoringService in /app/services/task\_authoring.py.

USE\_LLM\_FIXTURES environment variable:
  True (default during development) → use LLMFixtureClient
  False (production) → use real LLM from llm-integration.md

LLM context injection order — STRICT, DO NOT REORDER:
  \[1] Type system summary
  \[2] Guardrails (strategy registry, bounds, priors, bias rules)
  \[3] Application context
  \[4] Parameter bias rules
  \[5] Primitive registry

Strategy validation after LLM output:
  - Every condition MUST include strategy.type and strategy.params
  - Missing strategy → raise semantic\_error immediately
  - Do NOT proceed to compilation if strategy is absent

Action binding resolution order:
  user-specified → guardrails default → app\_context preferences
  → system default → HTTP 422 action\_binding\_failed if unresolved

LLM failure handling:
  - MAX\_RETRIES = 3 (configurable via env var)
  - On each failure: pass only errors back, ask LLM to fix only
    invalid parts, not regenerate from scratch
  - If MAX\_RETRIES exceeded → HTTP 422 semantic\_error,
    include last validation error in response
  - DO NOT persist any partial definition on failure
  - System must remain clean as if request was never made

dry\_run=True:
  - Return DryRunResult with concept + condition + action
  - DO NOT register, compile, or persist anything
  - DO NOT assign a task\_id

Write tests in /tests/unit/test\_task\_authoring.py using fixtures:
- threshold\_task.json creates valid Task
- dry\_run returns DryRunResult, nothing persisted
- Missing strategy.type in fixture → semantic\_error
- LLM failure after MAX\_RETRIES → 422 semantic\_error
- action\_binding\_failed when no action resolves
- Task is version-pinned (concept\_id, condition\_id immutable after create)
```

\---

## Session 10b — Layer 6: Calibration Service

**Prompt:**

```
Read py-instructions.md sections "POST /conditions/calibrate"
and "POST /conditions/apply-calibration".

Implement CalibrationService in /app/services/calibration.py.

calibrate():
  equals strategy → always return:
    status='no\_recommendation', reason='not\_applicable\_strategy'
  composite strategy → always return:
    status='no\_recommendation', reason='not\_applicable\_strategy'

  For all other strategies:
  - Aggregate stored feedback via FeedbackStore
  - derive\_direction(): majority false\_positive → 'tighten';
    majority false\_negative → 'relax'; tie → None
  - If feedback\_direction provided explicitly: use that, skip aggregation
  - adjust\_params() — strategy-aware:
      threshold:  relax → decrease 'value'; tighten → increase 'value'
      percentile: relax → increase 'value'; tighten → decrease 'value'
      change:     relax → decrease 'value'; tighten → increase 'value'
      z\_score:    relax → decrease 'threshold'; tighten → increase 'threshold'
  - Enforce bounds from guardrails:
      on\_bounds\_exceeded: 'clamp' → clip to bound;
                          'reject' → return no\_recommendation, bounds\_exceeded
  - Generate calibration\_token (24h expiry, single-use)

apply\_calibration():
  - calibration\_token is the ONLY input path
    DO NOT accept condition\_id + threshold as fallback
  - Validate token: not expired, not already used
  - Create NEW condition version — NEVER mutate existing
  - Invalidate token after use (single-use)
  - Return tasks\_pending\_rebind: \[{task\_id, intent}] — informational only
  - NEVER auto-rebind tasks — rebinding is always explicit by caller

Write tests in /tests/unit/test\_calibration.py:
- equals → always no\_recommendation
- composite → always no\_recommendation
- false\_positive majority → tighten direction
- bounds\_exceeded → no\_recommendation when reject policy
- apply-calibration creates new version, old version unchanged
- apply-calibration invalidates token (second use returns error)
- tasks\_pending\_rebind is informational — tasks NOT rebound
```

\---

## Session 10c — Layer 6: Feedback and Explanation Services

**Prompt:**

```
Read py-instructions.md sections "POST /feedback/decision"
and "POST /decisions/explain".

Implement in /app/services/:

feedback.py (FeedbackService):
  submit(condition\_id, condition\_version, entity, timestamp,
         feedback\_value, note) → FeedbackResponse
  - Valid values: false\_positive | false\_negative | correct
  - 'useful' and 'not\_useful' are NOT valid → raise parameter\_error
    BEFORE making any DB call
  - Must verify decision exists for condition\_id+entity+timestamp
    → HTTP 404 if no decision found at that timestamp
  - Store feedback record via FeedbackStore
  - Returns: { status: 'recorded', feedback\_id: str }

explanation.py (DecisionExplainService):
  explain(condition\_id, condition\_version, entity, timestamp)
    → DecisionExplanation
  - Requires deterministic execution (timestamp is required)
  - Re-execute concept with explain=True using exact same inputs
  - Strategy-aware field resolution:
      threshold/percentile/change: threshold\_applied = params\['value']
      z\_score: threshold\_applied = params\['threshold']
      equals: threshold\_applied = None, label\_matched = decision.value
      composite: threshold\_applied = None, label\_matched = None
  - INVARIANT: drivers\[].contribution must sum to 1.0
    If they do not, normalise before returning

Write tests in /tests/unit/test\_feedback.py:
- 'useful' value → parameter\_error (before any DB call)
- Decision not found → 404
- Duplicate submission (same condition\_id+version+entity+timestamp) → HTTP 409 conflict
- First submission succeeds, second identical submission returns 409
- Stores correctly for valid input

Write tests in /tests/unit/test\_explanation.py:
- threshold\_applied = params\['value'] for threshold strategy
- threshold\_applied = params\['threshold'] for z\_score
- threshold\_applied = None for equals
- drivers.contribution sums to 1.0
```

\---

## Session 10.5 — Observability Layer ← NEW

Add structured logging to all execution paths. Without this, debugging production issues requires log archaeology. This session is lightweight — 30 minutes — but pays dividends immediately.

**Prompt:**

```
Read py-instructions.md section "Observability and Logging
Requirements".

Add structured logging to /app/runtime/ and /app/services/:

Requirements:
- Use structlog for structured JSON logging
- Add log events to: executor.py, condition\_evaluator.py,
  action\_trigger.py, services/calibration.py,
  services/feedback.py, services/task\_authoring.py

Required log events and fields (from py-instructions.md):
  concept\_executed: concept\_id, version, entity, timestamp,
    deterministic, cache\_hit, compute\_time\_ms
  condition\_evaluated: condition\_id, condition\_version, entity,
    timestamp, decision\_value, strategy\_type, params\_applied,
    actions\_triggered\_count
  calibration\_recommended: condition\_id, old\_params,
    recommended\_params, delta\_alerts
  calibration\_applied: condition\_id, previous\_version, new\_version,
    params\_applied, tasks\_pending\_rebind count
  memintel\_error: error\_type, location, entity, concept\_id

NEVER LOG:
  - Resolved credential values
  - Raw primitive data or entity attribute values
  - Feedback note fields

Log format: structured JSON (structlog default)
Log level: INFO for all execution events, ERROR for errors

Write tests in /tests/unit/test\_observability.py:
  - concept\_executed event emitted on every execute() call
  - condition\_evaluated event emitted on every evaluate() call
  - memintel\_error event emitted on every raised MemintelError
  - No credential values appear in any log output
  - Log output is valid JSON
```

\---

## Session 11 — Layer 7: FastAPI Routes

Only wire routes after all services exist. Routes contain zero logic.

**Prompt:**

```
Read developer\_api.yaml and Memintel\_InternalPlatform\_API\_Reference.docx.
Read py-instructions.md "Framework and Structural Constraints".

Implement FastAPI routes in /app/api/routes/.

Rules — enforce all of these:
- All routes are async
- All request bodies are Pydantic models — no raw dict
- All error responses use ErrorResponse model
- HTTP status codes must match the YAML specs exactly
- Routes contain NO logic — delegate everything to service classes
- Internal platform endpoints require elevated\_key check:
    /compile, /execute/graph, /registry write operations,
    /agents/semantic-refine, /definitions/batch
    → HTTP 403 if not elevated\_key

Route responsibility:
  tasks.py    → POST/GET /tasks, GET/PATCH/DELETE /tasks/{id}
  conditions.py → GET /conditions/{id}, POST /conditions/explain,
                  POST /conditions/calibrate,
                  POST /conditions/apply-calibration
  decisions.py  → POST /decisions/explain
  feedback.py   → POST /feedback/decision
  execute.py    → POST /evaluate/full, POST /evaluate/condition,
                  POST /evaluate/condition/batch,
                  POST /execute, POST /execute/batch,
                  POST /execute/range, POST /execute/async
  jobs.py       → GET /jobs/{id}, DELETE /jobs/{id}

Wire all routes in /app/main.py with:
- Correct tags matching OpenAPI specs
- CORS middleware
- Exception handler that maps all exceptions to ErrorResponse
- Startup event: load guardrails + application\_context into stores
```

\---

## Session 12 — TypeScript SDK

Start a fresh Claude Code session. The Python backend should be running (or at least its routes should be defined) before this session.

**Prompt:**

```
Read ts-instructions.md, core-spec.md, and developer\_api.yaml.

Implement the @memintel/sdk TypeScript package.

Implement in this order:
1. All TypeScript types (from ts-instructions.md Type Definitions)
2. MemintelError class — .type, .message, .location,
   .suggestion, .retryAfterSeconds
3. Base HttpClient with post/get/patch/delete + error handling
4. Idempotency key support:
     client.tasks.create(params, { idempotencyKey })
     Sends as Idempotency-Key header
5. Sub-clients:
     TasksClient, ConditionsClient, DecisionsClient, FeedbackClient
     RegistryClient, FeaturesClient, ActionsClient, AgentsClient
6. All camelCase ↔ snake\_case mappers per ts-instructions.md
   naming table
7. Top-level methods on Memintel class:
     evaluateFull, execute, evaluateCondition,
     evaluateConditionBatch, explain, conditionImpact
8. Main Memintel class wiring all sub-clients

Rules:
- No 'any' in public API surface
- All methods return Promise<T>
- Branch on error.type — never on error.message
- actionsTriggered lives ONLY under decision — never at top level
- Do NOT implement Internal Platform endpoints

After implementation, write TypeScript tests covering:
- evaluateFull returns result.decision.actionsTriggered (not top-level)
- MemintelError.type is one of the ErrorType enum values
- tasks.create() accepts second options arg with idempotencyKey
- FeedbackValue type rejects anything other than the three valid values
- All camelCase fields map correctly from snake\_case responses
```

\---

## Session 13 — Full Pipeline Integration Test ← NEW, CRITICAL

This step was missing from the original process. Unit tests verify components in isolation. Integration tests reveal the bugs that only appear when the full pipeline runs end to end — strategy/runtime mismatches, type propagation errors, action triggering issues, and versioning inconsistencies.

**Prompt:**

```
Read core-spec.md section 1G (Alignment Test) and
py-instructions.md "Task-Centric Execution Architecture".

Write /tests/integration/test\_full\_pipeline.py covering the
complete task lifecycle using the LLM fixtures:

def test\_threshold\_task\_full\_pipeline():
  # 1. Create task using threshold fixture (USE\_LLM\_FIXTURES=True)
  task = task\_service.create\_task(CreateTaskRequest(
    intent='Alert when churn score exceeds 0.8',
    entity\_scope='user',
    delivery=DeliveryConfig(type='webhook', endpoint='...')
  ))
  assert task.task\_id is not None
  assert task.condition\_id is not None
  assert task.status == 'active'

  # 2. Execute full pipeline
  result = executor.evaluate\_full(
    concept\_id=task.concept\_id,
    concept\_version=task.concept\_version,
    condition\_id=task.condition\_id,
    condition\_version=task.condition\_version,
    entity='user\_test\_001',
    timestamp='2024-01-15T09:00:00Z'
  )
  assert result.result.deterministic == True
  assert result.decision.value in \[True, False]
  # actions\_triggered must be nested in decision, not top-level
  assert hasattr(result.decision, 'actions\_triggered')
  assert not hasattr(result, 'actions\_triggered')

  # 3. Submit feedback
  feedback = feedback\_service.submit(
    condition\_id=task.condition\_id,
    condition\_version=task.condition\_version,
    entity='user\_test\_001',
    timestamp='2024-01-15T09:00:00Z',
    feedback\_value='false\_positive'
  )
  assert feedback.status == 'recorded'

  # 4. Calibrate
  calibration = calibration\_service.calibrate(CalibrateRequest(
    condition\_id=task.condition\_id,
    condition\_version=task.condition\_version
  ))
  assert calibration.status == 'recommendation\_available'
  assert calibration.calibration\_token is not None

  # 5. Apply calibration
  applied = calibration\_service.apply\_calibration(
    ApplyCalibrationRequest(calibration\_token=calibration.calibration\_token)
  )
  assert applied.new\_version != task.condition\_version
  # Token is now invalid — second use must fail
  with pytest.raises(MemintelError) as exc:
    calibration\_service.apply\_calibration(
      ApplyCalibrationRequest(calibration\_token=calibration.calibration\_token)
    )
  assert exc.value.type == 'parameter\_error'

  # 6. Rebind task
  updated = task\_service.update\_task(task.task\_id, UpdateTaskRequest(
    condition\_version=applied.new\_version
  ))
  assert updated.condition\_version == applied.new\_version

  # 7. Re-run execution — determinism check
  result2 = executor.evaluate\_full(
    concept\_id=task.concept\_id,
    concept\_version=task.concept\_version,
    condition\_id=updated.condition\_id,
    condition\_version=updated.condition\_version,
    entity='user\_test\_001',
    timestamp='2024-01-15T09:00:00Z'
  )
  # Run a third time — all three must be identical
  result3 = executor.evaluate\_full(...)
  assert result2.result.value == result3.result.value
  assert result2.decision.value == result3.decision.value

Repeat this test structure for z\_score, composite, and equals fixtures.

Also write these standalone integration tests:

def test\_determinism\_harness():
  # Run same execution 3 times → assert ALL outputs identical
  # Catches hidden randomness and caching bugs

def test\_error\_injection():
  # Force type\_error: pass categorical to threshold strategy
  # Force semantic\_error: missing strategy.params
  # Force execution\_error: primitive unavailable (mock connector failure)
  # Force graph\_error: circular dependency in concept definition
  # Assert each raises the correct error.type

def test\_dry\_run\_propagation():
  # dry\_run=True → all actionsTriggered.status == 'would\_trigger'
  # No HTTP calls made to action endpoints

def test\_definition\_immutability():
  # Register a definition
  # Attempt to register same id+version again → 409
  # Attempt to update registered definition → error
  # Verify original definition unchanged
```

\---

## Session 14 — Explain Plan Validation

**Prompt:**

```
Read py-instructions.md "Compiler Layer" and core-spec.md
"DAG Execution Order Guarantee".

Write /tests/integration/test\_compiler\_pipeline.py:

def test\_explain\_plan\_before\_execution():
  # For each fixture definition:
  # 1. Compile definition → ExecutionGraph
  # 2. Call compile\_explain\_plan → ExecutionPlan
  # 3. Assert:
  #    - execution\_order has no duplicate nodes
  #    - all parallelizable\_groups contain independent nodes only
  #    - topological order is valid (no node before its dependencies)
  #    - plan is identical on repeated calls (deterministic)

def test\_ir\_hash\_stability():
  # Compile same definition 3 times
  # Assert ir\_hash is identical on all three
  # Assert ir\_hash is a valid SHA-256 hex string

def test\_circular\_dependency\_rejected():
  # Create a concept where A depends on B and B depends on A
  # Assert compile() raises graph\_error
  # Assert no graph is stored (clean failure)
```

\---

## The Prompting Discipline

These rules apply to every session, without exception:

**Cite the source document precisely.** "Read py-instructions.md section 'POST /conditions/calibrate'" produces better output than "implement the calibration service." Claude Code reads what you point at — point precisely.

**One layer at a time, one component at a time.** A focused prompt on one class produces better code than a broad prompt on a module. The temptation to batch will cost you more time in review than it saves in prompting.

**State invariants explicitly in every prompt.** The most critical ones: "same inputs → same output," "never mutate existing definitions," "never auto-rebind tasks," "LLM is not invoked during execution," "token is single-use." If you don't say them, they will be violated.

**Review before moving to the next session.** Fixing a wrong model is 5 minutes. Fixing a wrong model imported by 12 files is an hour. Every session should end with a review pass against the spec before the next session begins.

**Stub external dependencies immediately.** Real LLM calls, real database, real connectors — stub these as interfaces from day one. This lets you build and test core logic without infrastructure. The fixture system (Session 7) makes this explicit.

**Start a fresh session when context drifts.** A session running for 2 hours with dozens of files in context will drift. When you notice inconsistencies appearing, start fresh with explicit context. Do not try to course-correct a long session.

**Use `USE\_LLM\_FIXTURES=True`** throughout development until the full pipeline integration test (Session 13) passes. Only then wire up the real LLM.

**Backfill and batch execution pattern.** When using `executeRange` or `executeBatch` for historical data:

* Always provide `timestamp` — never mix snapshot and deterministic modes in the same batch
* `executeRange` for a single entity over time (historical replay): all results will have `deterministic=True`
* `executeBatch` for many entities at one point in time: provide a shared `timestamp`
* Never use `executeBatch` without a timestamp in a backfill context — results will be non-reproducible
* Use `executeAsync` for ranges longer than a few days — synchronous range calls time out

\---

## The Complete Session Sequence

```
Session 1:   Repository structure
Session 2:   Pydantic models + schemas
Session 2.5: Config \& environment bootstrapping           ← NEW
Session 3:   Persistence layer (DB schema, all stores)    ← NEW
Session 4:   Type checker + tests
Session 5:   Compiler pipeline + tests
Session 6a:  Strategy: threshold
Session 6b:  Strategy: percentile
Session 6c:  Strategy: z\_score
Session 6d:  Strategy: change
Session 6e:  Strategy: equals
Session 6f:  Strategy: composite
             + cross-strategy type compatibility test
Session 7:   LLM output fixtures                          ← NEW
Session 8:   Runtime executor + cache
Session 9:   Registry
Session 10a: Task authoring service
Session 10b: Calibration service
Session 10c: Feedback + explanation services
Session 10.5: Observability layer                         ← NEW
Session 11:  FastAPI routes
Session 12:  TypeScript SDK
Session 13:  Full pipeline integration test               ← NEW
Session 14:  Explain plan + determinism + error injection ← NEW
```

**20 focused sessions.** Each is 30 minutes to 2 hours. The whole implementation is achievable in 12–15 days of focused work. Every session has a single, well-scoped job, a source document to read, and explicit invariants to enforce.

The three new sessions (3, 7, 13) are not optional additions. Session 3 prevents data model chaos. Session 7 stabilises the entire task creation path. Session 13 is where all the hidden bugs surface — type propagation errors, action nesting bugs, calibration token issues — bugs that unit tests never catch because they only appear when the full pipeline runs end to end.

