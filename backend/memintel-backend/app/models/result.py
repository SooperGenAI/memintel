"""
app/models/result.py
──────────────────────────────────────────────────────────────────────────────
Execution result models for the ψ (psi), φ (phi), and α (alpha) layers,
plus async job models and dry-run / validation types.

Layer ownership
───────────────
  ψ  ConceptResult (Rₜ)     — concept execution output
  φ  DecisionResult (Aₜ)    — condition evaluation output; nests ActionTriggered[]
  α  ActionTriggered         — per-action delivery record (nested inside Aₜ)

Full pipeline: ψ → φ → α, assembled in FullPipelineResult.

Design notes
────────────
actions_triggered[] is nested inside DecisionResult, NOT at the top level of
  FullPipelineResult. This is a spec requirement (developer_api.yaml x-ts-note).
  Accidental flattening of this field is a common SDK mistake — the layout here
  enforces the correct nesting at the type level.

ConceptResult.deterministic must be set by the executor based on whether a
  timestamp was supplied. It is not derived from the result value — it is a
  statement about the execution mode used.

ConceptResult.explanation is None when explain=False (the default). The
  executor only populates it when the caller passes explain=True.

DryRunResult.concept is typed as dict[str, Any] because the concept definition
  model lives in concept.py (not yet a dependency of this file). The route handler
  for POST /tasks validates the concept dict against ConceptDefinition before
  constructing the DryRunResult.

Job.poll_interval_seconds uses the Python name; the DB column is poll_interval_s.
  The store maps between them. The Python name matches py-instructions.md's Job
  constructor example and is clearer at the API boundary.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.condition import DecisionType
from app.models.errors import ErrorResponse, ValidationErrorItem


# ── Enums ─────────────────────────────────────────────────────────────────────

class ConceptOutputType(str, Enum):
    """
    Declared output type of a concept. Matches the ConceptResult.type wire values.

    Determines which Python type result.value carries:
      float       → float
      boolean     → bool
      categorical → str  (a label from the declared label set)
    """
    FLOAT       = "float"
    BOOLEAN     = "boolean"
    CATEGORICAL = "categorical"


class ExplainMode(str, Enum):
    """
    Verbosity level for concept execution explanations.

    summary — final output + top contributions only
    full    — contributions + per-node trace (default)
    debug   — full trace + raw intermediate values
    """
    SUMMARY = "summary"
    FULL    = "full"
    DEBUG   = "debug"


class MissingDataPolicy(str, Enum):
    """
    Policy applied when a primitive fetch returns no data for an entity.

    null          — return null (T?); downstream operators must handle nullable
    zero          — substitute 0; forces non-nullable output
    forward_fill  — use last known value (non-nullable)
    backward_fill — use next known value (non-nullable)

    Specified per-request to override per-primitive defaults.
    """
    NULL          = "null"
    ZERO          = "zero"
    FORWARD_FILL  = "forward_fill"
    BACKWARD_FILL = "backward_fill"


class ActionTriggeredStatus(str, Enum):
    """
    Delivery status of a single action after a decision fires.

    triggered    — action was invoked and delivery succeeded
    skipped      — fire_on rule not met (e.g. fire_on='true' but decision=False)
    failed       — action was attempted but delivery raised an error
    would_trigger — dry_run mode; action would have fired but was not invoked
    """
    TRIGGERED    = "triggered"
    SKIPPED      = "skipped"
    FAILED       = "failed"
    WOULD_TRIGGER = "would_trigger"


class JobStatus(str, Enum):
    """
    Async job lifecycle status.

    Transitions (enforced in JobStore.update_status()):
      queued → running
      queued → cancelled
      running → completed
      running → failed
      running → cancelled
      completed / failed / cancelled → (terminal — no further transitions)
    """
    QUEUED    = "queued"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


#: Terminal states after which no further status transitions are allowed.
TERMINAL_JOB_STATUSES: frozenset[JobStatus] = frozenset({
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
})

#: Valid status transitions for JobStore.update_status().
#: Key = current status; value = set of statuses it may transition to.
VALID_JOB_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED:    frozenset({JobStatus.RUNNING, JobStatus.CANCELLED}),
    JobStatus.RUNNING:   frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED:    frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


# ── Execution request models ───────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    """
    Request body for POST /execute.

    timestamp presence controls execution mode:
      present → deterministic mode; result.deterministic=True; result is cached.
      absent  → snapshot mode; result.deterministic=False; result must NOT be cached
                beyond the current request (SNAPSHOT cache key rule).

    explain=True populates ConceptResult.explanation. The explain_mode controls
    verbosity. 'full' is the default and returns contributions + node trace.
    """
    id: str
    version: str
    entity: str
    timestamp: str | None = None
    explain: bool = False
    explain_mode: ExplainMode = ExplainMode.FULL
    cache: bool = True
    missing_data_policy: MissingDataPolicy | None = None


class ExecuteGraphRequest(BaseModel):
    """
    Request body for POST /execute/graph.

    Executes a pre-compiled graph directly, bypassing the compilation step.
    Use the hot path: compile once at startup → cache graph_id → call this
    endpoint on every evaluation.

    ir_hash, if provided, is verified against the stored graph before execution.
    Mismatch → HTTP 409 (audit trail mechanism; signals stale graph_id in caller).
    """
    graph_id: str
    entity: str
    ir_hash: str | None = None
    timestamp: str | None = None
    explain: bool = False
    explain_mode: ExplainMode = ExplainMode.FULL
    cache: bool = True
    missing_data_policy: MissingDataPolicy | None = None


# ── Concept execution result (ψ layer) ────────────────────────────────────────

class NodeTrace(BaseModel):
    """
    Computation record for a single graph node, in topological execution order.

    Populated only when explain=True. inputs and params are raw dicts because
    their structure varies by operator — the explain route serialises them
    faithfully rather than imposing a schema.
    """
    node_id: str
    op: str
    inputs: dict[str, Any]
    params: dict[str, Any]
    output_value: float | int | bool | str
    output_type: str


class ConceptExplanation(BaseModel):
    """
    Node-level explanation of a concept execution result.

    Populated when ExecuteRequest.explain=True. Null on responses where
    explain was False or omitted.

    output       — final computed value; matches ConceptResult.value.
    contributions — per-signal attribution: signal_name → contribution_amount.
                   Values are dimensionless; their sum equals 1.0 for normalised
                   concepts, or is unbounded for raw aggregations.
    nodes        — per-node computation records in topological execution order.
    trace        — step-by-step trace with intermediate values (debug mode only).
    """
    output: float | int | bool | str
    contributions: dict[str, float] = Field(default_factory=dict)
    nodes: list[NodeTrace] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class ConceptResult(BaseModel):
    """
    Output of concept execution (Rₜ). The ψ layer result.

    deterministic reflects the execution mode, not the value stability:
      True  → timestamp was provided; same inputs always produce the same result.
      False → snapshot mode; reflects current state at call time.

    explanation is None when explain=False (the default). The executor sets it
    only when the caller passes explain=True on the request.
    """
    value: float | int | bool | str | None
    type: ConceptOutputType
    entity: str
    version: str
    deterministic: bool
    timestamp: str | None = None
    explanation: ConceptExplanation | None = None


# ── Decision result (φ → α layers) ────────────────────────────────────────────

class ActionTriggered(BaseModel):
    """
    Delivery record for a single action invocation after a condition fires.

    Actions are best-effort — status='failed' does NOT roll back the decision.
    The full pipeline response returns HTTP 200 regardless of action outcomes.

    payload_sent is populated for triggered actions (webhook / workflow payloads).
    error is populated only when status='failed'.
    """
    action_id: str
    action_version: str
    status: ActionTriggeredStatus
    payload_sent: dict[str, Any] | None = None
    error: ErrorResponse | None = None


class DecisionResult(BaseModel):
    """
    Output of condition evaluation (Aₜ). The φ layer result.

    actions_triggered[] is nested HERE, not at the top level of FullPipelineResult.
    This is a hard API contract requirement. Do not flatten it.

    value is bool for boolean strategies (threshold/percentile/z_score/change/composite).
    value is str (the matched label) for the equals strategy on categorical input.

    reason is populated when z_score, percentile, or change strategies could not
    evaluate due to insufficient or unavailable history. Auditable and queryable.
      "insufficient_history" — fewer than the required minimum results available.
      "history_unavailable"  — history query failed; fallback to False.
    history_count reflects how many historical results were found (0 if the
    query failed). Set only when reason is also set.
    """
    value: bool | str
    type: DecisionType
    entity: str
    condition_id: str
    condition_version: str
    timestamp: str | None = None
    actions_triggered: list[ActionTriggered] = Field(default_factory=list)
    reason: str | None = None
    history_count: int | None = None


class FullPipelineResult(BaseModel):
    """
    Combined output of a full ψ → φ → α pipeline execution.

    Returned by POST /evaluate/full.

    result   — concept execution output (Rₜ); the ψ layer.
    decision — condition evaluation output (Aₜ); the φ layer.
               actions_triggered[] is nested inside decision, not here.
    dry_run  — True when the request was a simulation (actions were not invoked).
    """
    result: ConceptResult
    decision: DecisionResult
    dry_run: bool = False
    entity: str
    timestamp: str | None = None


# ── Validation models ──────────────────────────────────────────────────────────
# Used in DryRunResult and compiler responses.

class ValidationWarning(BaseModel):
    """
    A non-fatal compiler or validation warning.

    Warnings do not halt compilation. They surface advisory information —
    e.g. a deprecated definition reference, or a strategy choice that is
    valid but unusual for the primitive type.
    """
    type: str
    message: str


class ValidationResult(BaseModel):
    """
    Compiler / validator output for a definition or dry-run.

    valid=False means compilation was halted. errors[] contains the full list
    of problems found (the compiler collects all errors before raising, so the
    caller sees everything at once rather than fix-and-retry per error).

    valid=True with non-empty warnings[] means compilation succeeded but the
    caller should review the advisory notes.
    """
    valid: bool
    errors: list[ValidationErrorItem] = Field(default_factory=list)
    warnings: list[ValidationWarning] = Field(default_factory=list)


# ── Dry-run result ─────────────────────────────────────────────────────────────

class DryRunResult(BaseModel):
    """
    Returned by POST /tasks when dry_run=True.

    The task is NOT persisted. Use to verify intent resolution before committing
    a real task. When the request also provides entity and timestamp, the runtime
    simulates execution and populates would_trigger.

    concept is typed as dict[str, Any] because ConceptDefinition lives in
    concept.py. The route handler validates it against ConceptDefinition before
    constructing this response; at the wire layer it remains a plain object.
    """
    concept: dict[str, Any]
    condition: Any           # ConceptDefinition — forward ref; validated in route
    action_id: str
    action_version: str
    validation: ValidationResult
    would_trigger: bool | None = None


# ── Async job models ───────────────────────────────────────────────────────────

class Job(BaseModel):
    """
    Async job record. Returned by POST /execute/async and GET /jobs/{id}.

    poll_interval_seconds — caller should wait this many seconds between
      GET /jobs/{id} polls. The DB column is poll_interval_s; the store
      maps between them.

    Internal DB fields (excluded from API serialisation):
      enqueued_at, started_at, completed_at, updated_at — managed by JobStore.
      request_body — the original ExecuteRequest stored for audit / retry.
    """
    job_id: str | None = None        # None until DB assigns it
    job_type: str = "execute"
    status: JobStatus = JobStatus.QUEUED
    poll_interval_seconds: int = 2

    # Internal DB fields — excluded from API responses
    request_body: dict[str, Any] | None = Field(default=None, exclude=True)
    result_body: dict[str, Any] | None = Field(default=None, exclude=True)
    error_body: dict[str, Any] | None = Field(default=None, exclude=True)
    enqueued_at: datetime | None = Field(default=None, exclude=True)
    started_at: datetime | None = Field(default=None, exclude=True)
    completed_at: datetime | None = Field(default=None, exclude=True)
    updated_at: datetime | None = Field(default=None, exclude=True)

    model_config = {"populate_by_name": True}


class JobResult(BaseModel):
    """
    Response shape for GET /jobs/{id} and DELETE /jobs/{id}.

    result is populated (from result_body) only when status=completed.
    error  is populated (from error_body)  only when status=failed.
    Both are None for all other statuses.
    """
    job_id: str
    status: JobStatus
    result: ConceptResult | FullPipelineResult | None = None
    error: ErrorResponse | None = None
    poll_interval_seconds: int = 2


# ── Batch execution result ─────────────────────────────────────────────────────

class BatchExecuteItem(BaseModel):
    """
    Result for a single entity in a batch execution request.

    result is populated on success; error is populated on per-entity failure.
    A batch call returns HTTP 200 even if some individual entities fail —
    inspect each item's error field for per-entity status.
    """
    entity: str
    result: ConceptResult | None = None
    error: ErrorResponse | None = None


class BatchExecuteResult(BaseModel):
    """
    Response body for POST /execute/batch.

    results — one entry per input entity, in input order.
    total   — total entities in the request.
    failed  — count of entities with a non-null error.
    """
    results: list[BatchExecuteItem]
    total: int
    failed: int
