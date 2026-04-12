"""
app/models/concept_compile.py
──────────────────────────────────────────────────────────────────────────────
V7 models for the two-phase concept compilation flow.

Two-phase model
───────────────
Phase 1 — POST /concepts/compile
  SI (System Integrator) provides a concept description and signal names.
  Memintel runs a 4-step Chain of Reasoning (CoR) and returns:
    - compile_token  (single-use, 30-min TTL)
    - compiled_concept (formula summary + signal bindings)
    - reasoning_trace  (optional, when return_reasoning=True)
    - expires_at       (UTC datetime — for client countdown UI)

Phase 2 — POST /concepts/register
  Caller presents compile_token + identifier. Memintel atomically marks
  the token used and registers the concept, returning a stable concept_id.
  Idempotent: same (identifier, ir_hash) returns the existing concept_id.

Internal model
──────────────
CompileToken is the internal DB record — it is never returned to callers.
The opaque token_string is what callers see. The token_id is the PK in
the compile_tokens table (added by Alembic migration 0011 in Session M-2).

SSE event models
────────────────
CorStepEvent, CorCompleteEvent, CorErrorEvent are the payload shapes for
SSE streaming (stream=True). They are serialised into the `data:` field of
each SSE event line. See Session M-6 for the streaming implementation.

Dependency direction
────────────────────
  concept_compile.py → task.py  (ReasoningTrace)
  concept_compile.py → (no other app.models imports)

No other models module imports from concept_compile.py.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator

from app.models.task import ReasoningTrace


# ── Phase 1 request / response ────────────────────────────────────────────────

class CompileConceptRequest(BaseModel):
    """
    Request body for POST /concepts/compile.

    identifier    — unique name for the concept (e.g. "loan.repayment_ratio").
                    Locked at compile time: POST /concepts/register must supply
                    the same identifier or the request is rejected (HTTP 422).
    description   — plain-English definition of what the concept measures.
    output_type   — Memintel type string (e.g. "float", "integer", "boolean").
                    Must be non-empty. Not validated against MemintelType here —
                    the compiler validates type compatibility in Step 4 of CoR.
    signal_names  — opaque list of primitive/signal names from the caller's
                    domain. Canvas passes these from its own module store.
                    Memintel MUST NOT validate signal_names against any
                    internal registry — they are semantic hints to the LLM only.
    return_reasoning — when True, reasoning_trace is included in the response.
    stream           — when True, response is text/event-stream (SSE). The
                       synchronous JSON path is used when False (default).
    """
    identifier:       str
    description:      str
    output_type:      str
    signal_names:     list[str]
    return_reasoning: bool = False
    stream:           bool = False

    @field_validator("output_type")
    @classmethod
    def _output_type_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("output_type must be a non-empty string")
        return v


class SignalBinding(BaseModel):
    """
    A single signal→role binding in a compiled concept.

    Records which signal name was used, what role it plays in the formula,
    its relative weight (0–1, summing to 1.0 across all signals), and a
    one-sentence rationale explaining its contribution.
    """
    signal_name: str
    role:        str
    weight:      float | None = None   # 0–1; all weights sum to 1.0
    rationale:   str   | None = None   # Why this signal contributes to the formula


class CompiledConcept(BaseModel):
    """
    The compiled concept summary returned in CompileConceptResponse.

    This is NOT the full internal IR — it is the caller-facing summary.
    The full IR is hashed into ir_hash and stored in the compile_tokens table.

    formula_summary — plain-English description of what the formula computes
                      (e.g. "payments_on_time / payments_due, 90-day rolling window").
    signal_bindings — which signal names were used and in what roles.
    """
    identifier:      str
    output_type:     str
    formula_summary: str
    signal_bindings: list[SignalBinding]


class CompileConceptResponse(BaseModel):
    """
    Response from POST /concepts/compile.

    compile_token    — opaque single-use token required by POST /concepts/register.
                       Minimum entropy: 128 bits (generated via secrets.token_urlsafe(32)).
    compiled_concept — caller-facing summary of the compiled concept.
    reasoning_trace  — CoR trace; present only when return_reasoning=True.
                       Absent (not null) when return_reasoning=False.
    expires_at       — UTC datetime when compile_token expires (default TTL: 30 min).
                       Returned so the Canvas frontend can display a countdown timer.
    """
    compile_token:    str
    compiled_concept: CompiledConcept
    reasoning_trace:  ReasoningTrace | None = None
    expires_at:       datetime                    # UTC


# ── Phase 2 request / response ────────────────────────────────────────────────

class RegisterConceptRequest(BaseModel):
    """
    Request body for POST /concepts/register.

    compile_token — single-use token from POST /concepts/compile.
    identifier    — must match the identifier used at compile time.
                    Mismatch → HTTP 422 (identifier is locked at compile time).
    """
    compile_token: str
    identifier:    str


class RegisterConceptResponse(BaseModel):
    """
    Response from POST /concepts/register (HTTP 201 Created).

    concept_id    — stable Memintel concept identifier. Canvas stores this
                    in its module definition for use in POST /tasks (M5).
    version       — initial version string (defaults to "1.0.0").
    output_type   — the output type declared at compile time.
    registered_at — UTC timestamp of registration.

    Idempotency: if the same (identifier, ir_hash) is registered twice
    (via two separate compile→register flows), both calls return HTTP 201
    with the SAME concept_id. No duplicate row is created.
    """
    concept_id:    str
    identifier:    str
    version:       str
    output_type:   str
    registered_at: datetime


# ── Internal DB model (not returned to callers) ───────────────────────────────

class CompileToken(BaseModel):
    """
    Internal DB record for a compile token.

    Stored in the compile_tokens table (Alembic migration 0011, Session M-2).
    The token_string is the opaque value returned to callers. token_id is the
    DB primary key (UUID).

    Atomicity contract: CompileTokenStore.consume() uses a conditional UPDATE
    (WHERE used = FALSE AND expires_at > NOW()) to guarantee exactly one
    caller succeeds when two concurrent calls present the same token.

    used defaults to False — set to True atomically on successful consumption.
    """
    token_id:         str
    token_string:     str            # opaque token returned to the caller
    identifier:       str            # locked at compile time
    ir_hash:          str            # SHA-256 of the compiled concept IR
    output_type:      str            # declared at compile time; carried forward to Phase 2
    expires_at:       datetime
    used:             bool = False
    created_at:       datetime
    formula_summary:  str | None = None   # plain-English formula with % weights (Step 3)
    signal_bindings:  list[dict] | None = None  # [{signal_name, role, weight, rationale}] (Step 3)


# ── SSE event payload models ──────────────────────────────────────────────────

class CorStepEvent(BaseModel):
    """
    Payload of an SSE `event: cor_step` event.

    Emitted once per CoR step as each step completes. Four events are emitted
    before cor_complete (one per step, in order: 1, 2, 3, 4).

    candidates — concept/condition IDs or labels considered at this step.
                 Present only for steps where alternatives exist (typically
                 Step 2, Concept Selection).
    outcome    — free-form string describing what the step produced.
                 Unlike ReasoningStep.outcome, this is not a Literal — the
                 SSE wire format is less strictly typed than the internal model.
    """
    step_index: int
    label:      str
    summary:    str
    candidates: list[str] | None = None
    outcome:    str


class CorCompleteEvent(BaseModel):
    """
    Payload of the terminal SSE `event: cor_complete` event.

    Exactly one cor_complete (or cor_error) is emitted per stream; it is
    always the last event before the stream closes.

    Fields are mutually exclusive by endpoint:
      POST /tasks:            task_id is set; compile_token and concept_id are None.
      POST /concepts/compile: compile_token is set; task_id and concept_id are None.
      POST /concepts/register (future): concept_id is set; others are None.

    All fields are optional with None defaults to support all three call sites
    without requiring separate event models.
    """
    task_id:       str | None = None
    compile_token: str | None = None
    concept_id:    str | None = None
    status:        str | None = None


class CorErrorEvent(BaseModel):
    """
    Payload of the terminal SSE `event: cor_error` event.

    Emitted when CoR fails at any step. The stream closes immediately after
    this event — no further events are emitted.

    Memintel MUST emit cor_error within 10 seconds of detecting a step
    failure (including LLM timeout). This is a binding contract, not best-effort.

    failure_reason — machine-readable reason string (e.g. "vocabulary_mismatch",
                     "step_2_timed_out").
    failed_at_step — 1-based step index where failure occurred (None if the
                     failure occurred before any step started).
    suggestion     — actionable guidance for the caller (optional).
    """
    failure_reason: str
    failed_at_step: int | None = None
    suggestion:     str | None = None
