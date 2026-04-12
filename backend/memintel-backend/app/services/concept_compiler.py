"""
app/services/concept_compiler.py
──────────────────────────────────────────────────────────────────────────────
ConceptCompilerService — LLM-driven 4-step CoR pipeline for concept compilation.

Pipeline (Chain of Reasoning):
  Step 1 — Intent Parsing:       Parse identifier + description → structured intent.
  Step 2 — Signal Identification: Understand what signal_names imply (NOT validated).
  Step 3 — DAG Construction:     Produce formula strategy, signal_bindings, formula_summary.
  Step 4 — Type Validation:      Confirm output_type compatibility → CompiledConcept.

Streaming contract
──────────────────
The pipeline is an async generator (_run_pipeline) so the M-6 streaming path
can yield steps directly without duplicating pipeline logic. The non-streaming
path (this session) collects all yielded steps inside compile().

HARD RULE: signal_names identity boundary
──────────────────────────────────────────
Memintel MUST NOT validate signal_names against any internal registry.
Signal names come from the caller's domain. An unrecognised signal_name is
NOT an error. Step 2 treats them as semantic context hints for the LLM only.

LLM call pattern
────────────────
Each step calls self._llm.generate_task(step_prompt, step_context) — the same
pattern as TaskAuthoringService. On timeout or parse failure, _run_step raises
CompilationError with failed_at_step set to the offending step index.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator

import asyncpg
import structlog

from app.config.compile_token_ttl import get_compile_token_ttl
from app.llm.client_factory import create_llm_client
from app.models.concept_compile import (
    CompileConceptRequest,
    CompileConceptResponse,
    CompiledConcept,
    CompileToken,
    SignalBinding,
)
from app.models.errors import CompilationError, TypeMismatchError
from app.models.task import ReasoningStep, ReasoningTrace
from app.stores.compile_token import CompileTokenStore

log = structlog.get_logger(__name__)


class ConceptCompilerService:
    """
    LLM-driven concept compilation pipeline.

    compile() is the public entry point. It runs the 4-step CoR via the
    internal async generator and returns a CompileConceptResponse on success.

    The pipeline is generator-based so the M-6 streaming path can yield
    steps directly without duplicating pipeline logic.

    Parameters
    ──────────
    llm_client  — client with generate_task(prompt, context) → dict.
                  When None, auto-selected from USE_LLM_FIXTURES env var.
    token_store — optional CompileTokenStore for testing. When None, a
                  CompileTokenStore(pool) is created inside compile().
    """

    def __init__(
        self,
        llm_client: Any = None,
        token_store: Any = None,
    ) -> None:
        self._llm = llm_client if llm_client is not None else self._select_llm_client()
        self._token_store = token_store
        # Per-instance step output cache — safe because service is per-request.
        self._step_outputs: dict[int, dict] = {}

    @staticmethod
    def _select_llm_client() -> Any:
        """Select LLM client from USE_LLM_FIXTURES and LLM_PROVIDER env vars."""
        use_fixtures = os.environ.get("USE_LLM_FIXTURES", "true").lower() != "false"
        from app.models.config import LLMConfig
        config = LLMConfig.model_validate(
            {
                "provider": os.environ.get("LLM_PROVIDER", "anthropic"),
                "model": os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514",
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                "base_url": os.environ.get("LLM_BASE_URL"),
                "ssl_verify": os.environ.get("LLM_SSL_VERIFY", "true").lower() == "true",
                "timeout_seconds": int(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
            },
            context={"resolved": True},
        )
        return create_llm_client(config, use_fixtures)

    # ── Public API ───────────────────────────────────────────────────────────────

    async def compile(
        self,
        request: CompileConceptRequest,
        pool: asyncpg.Pool | None,
    ) -> CompileConceptResponse:
        """
        Run the 4-step CoR pipeline and return a CompileConceptResponse.

        Steps are collected via the internal async generator. On any step
        failure the appropriate error is raised immediately.

        On success:
          - Generates a single-use compile_token (secrets.token_urlsafe(32)).
          - Computes ir_hash: SHA-256 of canonical formula JSON (deterministic).
          - Persists CompileToken via CompileTokenStore.create().
          - Returns CompileConceptResponse.
        """
        from app.models.errors import CompilationError as _CE
        self._step_outputs = {}
        steps: list[ReasoningStep] = []
        pipeline_result: dict = {}

        try:
            async for step in self._run_pipeline(request, pipeline_result):
                steps.append(step)
        except asyncio.TimeoutError as exc:
            failed_at = next(
                (s.step_index for s in reversed(steps) if s.outcome == "failed"),
                None,
            )
            raise _CE(
                f"CoR step {failed_at} timed out: LLM did not respond within 10 seconds.",
                failed_at_step=failed_at,
                suggestion="Check LLM client configuration and retry.",
            ) from exc

        compiled_concept: CompiledConcept = pipeline_result["compiled_concept"]
        formula_data: dict = pipeline_result.get("formula_data", {})

        ir_hash = self._compute_ir_hash(formula_data)
        token_string = secrets.token_urlsafe(32)
        expires_at = self._compute_expires_at()
        now = datetime.now(tz=timezone.utc)

        token = CompileToken(
            token_id=str(uuid.uuid4()),
            token_string=token_string,
            identifier=request.identifier,
            ir_hash=ir_hash,
            output_type=request.output_type,
            expires_at=expires_at,
            used=False,
            created_at=now,
            formula_summary=compiled_concept.formula_summary,
            signal_bindings=[
                {
                    "signal_name": b.signal_name,
                    "role": b.role,
                    "weight": b.weight,
                    "rationale": b.rationale,
                }
                for b in compiled_concept.signal_bindings
            ],
        )

        store = self._token_store if self._token_store is not None else CompileTokenStore(pool)
        await store.create(token)

        log.info(
            "concept_compiled",
            identifier=request.identifier,
            ir_hash=ir_hash,
            expires_at=expires_at.isoformat(),
        )

        reasoning_trace: ReasoningTrace | None = None
        if request.return_reasoning:
            reasoning_trace = ReasoningTrace(steps=steps)

        return CompileConceptResponse(
            compile_token=token_string,
            compiled_concept=compiled_concept,
            reasoning_trace=reasoning_trace,
            expires_at=expires_at,
        )

    # ── Streaming public API ─────────────────────────────────────────────────────

    async def compile_stream(
        self,
        request: CompileConceptRequest,
        pool: asyncpg.Pool | None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream the 4-step CoR pipeline as SSE event dicts.

        Yields dicts with keys ``event_type`` and ``data``:
          cor_step     — one per pipeline step (4 total on success).
          cor_complete — final event; payload includes compile_token.
          cor_error    — terminal error event if any step fails or times out.

        HARD RULE: non-streaming path is untouched.
        This method is the ONLY streaming entry point — compile() is unchanged.
        """
        self._step_outputs = {}
        pipeline_result: dict = {}
        last_failed_step: int | None = None

        try:
            async for step in self._run_pipeline(request, pipeline_result):
                if step.outcome == "failed":
                    last_failed_step = step.step_index
                yield {"event_type": "cor_step", "data": step.model_dump()}

            # Pipeline succeeded — create compile token and yield cor_complete.
            compiled_concept: "CompiledConcept" = pipeline_result["compiled_concept"]
            formula_data: dict = pipeline_result.get("formula_data", {})
            ir_hash = self._compute_ir_hash(formula_data)
            token_string = secrets.token_urlsafe(32)
            expires_at = self._compute_expires_at()
            now = datetime.now(tz=timezone.utc)

            token = CompileToken(
                token_id=str(uuid.uuid4()),
                token_string=token_string,
                identifier=request.identifier,
                ir_hash=ir_hash,
                output_type=request.output_type,
                expires_at=expires_at,
                used=False,
                created_at=now,
                formula_summary=compiled_concept.formula_summary,
                signal_bindings=[
                    {"signal_name": b.signal_name, "role": b.role}
                    for b in compiled_concept.signal_bindings
                ],
            )
            store = (
                self._token_store
                if self._token_store is not None
                else CompileTokenStore(pool)
            )
            await store.create(token)

            log.info(
                "concept_compiled_stream",
                identifier=request.identifier,
                ir_hash=ir_hash,
            )

            yield {
                "event_type": "cor_complete",
                "data": {
                    "compile_token": token_string,
                    "status": "success",
                    "compiled_concept": compiled_concept.model_dump(),
                    "expires_at": expires_at.isoformat(),
                },
            }

        except asyncio.TimeoutError as exc:
            yield {
                "event_type": "cor_error",
                "data": {
                    "failure_reason": str(exc) or "LLM step timed out",
                    "failed_at_step": last_failed_step,
                },
            }
        except Exception as exc:
            yield {
                "event_type": "cor_error",
                "data": {
                    "failure_reason": str(exc),
                    "failed_at_step": getattr(exc, "failed_at_step", last_failed_step),
                },
            }

    # ── Pipeline (async generator) ───────────────────────────────────────────────

    async def _run_pipeline(
        self,
        request: CompileConceptRequest,
        result: dict,
    ) -> AsyncGenerator[ReasoningStep, None]:
        """
        4-step Chain of Reasoning — yields each ReasoningStep as it completes.

        result is mutated in-place; the caller reads result["compiled_concept"]
        and result["formula_data"] after the generator is exhausted.
        """
        # ── Step 1: Intent Parsing ─────────────────────────────────────────────
        try:
            step1 = await asyncio.wait_for(
                self._run_step(
                    step_index=1,
                    label="Intent Parsing",
                    llm_prompt=(
                        f"Parse concept intent: identifier='{request.identifier}', "
                        f"description='{request.description}'"
                    ),
                    context={
                        "step": 1,
                        "identifier": request.identifier,
                        "description": request.description,
                    },
                ),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            yield ReasoningStep(
                step_index=1,
                label="Intent Parsing",
                summary="LLM timed out at step 1",
                outcome="failed",
            )
            raise
        if step1.outcome == "failed":
            raise CompilationError(
                "Intent Parsing failed: could not parse concept description.",
                failed_at_step=1,
                suggestion=(
                    "Check that description clearly describes what the concept measures."
                ),
            )
        yield step1

        intent_data = self._step_outputs.get(1, {})

        # ── Step 2: Signal Identification ─────────────────────────────────────
        # signal_names are semantic context hints only — NOT validated against
        # any internal registry. An unrecognised signal_name is NOT an error.
        try:
            step2 = await asyncio.wait_for(
                self._run_step(
                    step_index=2,
                    label="Signal Identification",
                    llm_prompt=(
                        f"Understand signal context for concept '{request.identifier}': "
                        f"signal_names={request.signal_names}"
                    ),
                    context={
                        "step": 2,
                        "signal_names": request.signal_names,
                        "output_type": request.output_type,
                        "intent": intent_data,
                    },
                ),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            yield ReasoningStep(
                step_index=2,
                label="Signal Identification",
                summary="LLM timed out at step 2",
                outcome="failed",
            )
            raise
        if step2.outcome == "failed":
            raise CompilationError(
                "Signal Identification failed.",
                failed_at_step=2,
            )
        yield step2

        step2_output = self._step_outputs.get(2, {})
        if "signal_rationale" not in step2_output:
            log.warning(
                "signal_rationale_missing",
                step=2,
                keys=list(step2_output.keys()),
            )
        signal_context = step2_output

        # ── Step 3: DAG Construction ───────────────────────────────────────────
        try:
            step3 = await asyncio.wait_for(
                self._run_step(
                    step_index=3,
                    label="DAG Construction",
                    llm_prompt=(
                        f"Select formula strategy for concept: "
                        f"identifier='{request.identifier}', output_type='{request.output_type}'"
                    ),
                    context={
                        "step": 3,
                        "identifier": request.identifier,
                        "description": request.description,
                        "output_type": request.output_type,
                        "signal_names": request.signal_names,
                        "intent": intent_data,
                        "signal_context": signal_context,
                    },
                ),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            yield ReasoningStep(
                step_index=3,
                label="DAG Construction",
                summary="LLM timed out at step 3",
                outcome="failed",
            )
            raise
        if step3.outcome == "failed":
            raise CompilationError(
                "DAG Construction failed: could not select a formula strategy.",
                failed_at_step=3,
            )

        formula_output = self._step_outputs.get(3, {})
        formula_summary: str = formula_output.get(
            "formula_summary",
            (
                f"Computed {request.identifier} using "
                + (", ".join(request.signal_names) if request.signal_names else "available signals")
            ),
        )
        raw_bindings: list = formula_output.get("signal_bindings") or []
        signal_bindings: list[SignalBinding] = [
            SignalBinding(
                signal_name=b["signal_name"],
                role=b.get("role", "input"),
                weight=b.get("weight"),
                rationale=b.get("rationale"),
            )
            if isinstance(b, dict)
            else SignalBinding(signal_name=str(b), role="input")
            for b in raw_bindings
        ]
        # Fallback: derive bindings from signal_names when LLM provides none.
        if not signal_bindings and request.signal_names:
            signal_bindings = [
                SignalBinding(signal_name=sig, role="input")
                for sig in request.signal_names
            ]

        # Warn if any binding is missing weight — the Step 3 prompt requires it.
        missing_weights = [b.signal_name for b in signal_bindings if b.weight is None]
        if missing_weights:
            log.warning(
                "signal_binding_weights_missing",
                signals=missing_weights,
                step=3,
            )

        # Warn if any binding has an invalid role — prompt allows exactly three values.
        _VALID_ROLES = {"numerator", "denominator", "severity_multiplier"}
        for b in signal_bindings:
            if b.role not in _VALID_ROLES:
                log.warning(
                    "signal_binding_invalid_role",
                    signal=b.signal_name,
                    role=b.role,
                )

        formula_data: dict = {
            "identifier": request.identifier,
            "output_type": request.output_type,
            "formula_summary": formula_summary,
            "signal_bindings": [
                {
                    "signal_name": b.signal_name,
                    "role": b.role,
                    "weight": b.weight,
                    "rationale": b.rationale,
                }
                for b in signal_bindings
            ],
        }
        result["formula_data"] = formula_data

        log.info(
            "concept_compile_result",
            signals_resolved=len(signal_bindings),
            formula_structure=formula_summary,
            output_range=formula_output.get("output_range", ""),
            weights=[b.weight for b in signal_bindings],
        )

        yield step3

        # ── Step 4: Type Validation ────────────────────────────────────────────
        try:
            step4 = await asyncio.wait_for(
                self._run_step(
                    step_index=4,
                    label="Type Validation",
                    llm_prompt=(
                        f"Validate output type '{request.output_type}' "
                        f"for formula: {formula_summary}"
                    ),
                    context={
                        "step": 4,
                        "output_type": request.output_type,
                        "formula_summary": formula_summary,
                    },
                ),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            yield ReasoningStep(
                step_index=4,
                label="Type Validation",
                summary="LLM timed out at step 4",
                outcome="failed",
            )
            raise

        if step4.outcome == "failed":
            raise TypeMismatchError(
                f"output_type '{request.output_type}' is incompatible with the compiled formula.",
                suggestion=(
                    "Check that output_type matches the natural output type of the formula. "
                    "For ratio-based formulas use 'float'; for flags use 'boolean'."
                ),
            )

        compiled_concept = CompiledConcept(
            identifier=request.identifier,
            output_type=request.output_type,
            formula_summary=formula_summary,
            signal_bindings=signal_bindings,
        )
        result["compiled_concept"] = compiled_concept
        yield step4

    # ── Step execution helper ────────────────────────────────────────────────────

    async def _run_step(
        self,
        step_index: int,
        label: str,
        llm_prompt: str,
        context: dict | None = None,
    ) -> ReasoningStep:
        """
        Call the LLM for a single CoR step and return a ReasoningStep.

        Follows the exact LLM call pattern from task_authoring.py:
            output = self._llm.generate_task(intent, context)

        The raw LLM output is stored in self._step_outputs[step_index] so
        _run_pipeline can extract step-specific data (formula_summary, etc.).

        On LLM error or non-dict response: raises CompilationError with
        failed_at_step=step_index.
        """
        try:
            raw_output = self._llm.generate_compile_step(llm_prompt, context or {})
        except asyncio.TimeoutError:
            raise  # Propagate without wrapping — _run_pipeline catches it
        except Exception as exc:
            raise CompilationError(
                f"CoR step {step_index} ({label}) failed: {exc}",
                failed_at_step=step_index,
                suggestion="Check LLM client configuration and retry.",
            ) from exc

        if not isinstance(raw_output, dict):
            raise CompilationError(
                f"CoR step {step_index} ({label}): LLM returned non-dict output.",
                failed_at_step=step_index,
            )

        self._step_outputs[step_index] = raw_output

        log.info(
            "concept_compile_step_raw",
            step_index=step_index,
            label=label,
            summary=raw_output.get("summary", ""),
            outcome=raw_output.get("outcome", ""),
        )

        summary = raw_output.get("summary") or f"{label} complete"
        raw_candidates = raw_output.get("candidates")
        candidates = raw_candidates if isinstance(raw_candidates, list) else None
        raw_outcome = raw_output.get("outcome", "accepted")
        outcome: str = raw_outcome if raw_outcome in ("accepted", "skipped", "failed") else "accepted"

        # Extract signal_rationale from step 2 output — must be a dict keyed by
        # signal name. Passed through to ReasoningStep so Canvas can read it from
        # the streaming event / response trace.
        raw_sr = raw_output.get("signal_rationale")
        signal_rationale: dict[str, str] | None = raw_sr if isinstance(raw_sr, dict) else None

        return ReasoningStep(
            step_index=step_index,
            label=label,
            summary=summary,
            candidates=candidates,
            outcome=outcome,
            signal_rationale=signal_rationale,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _compute_expires_at(self) -> datetime:
        """Return UTC now + compile token TTL seconds."""
        return datetime.now(tz=timezone.utc) + timedelta(seconds=get_compile_token_ttl())

    def _compute_ir_hash(self, formula_data: dict) -> str:
        """
        SHA-256 of canonical formula JSON.

        json.dumps with sort_keys=True ensures determinism:
        same formula_data → same ir_hash across calls and processes.
        """
        canonical = json.dumps(formula_data, sort_keys=True)
        return "sha256_" + hashlib.sha256(canonical.encode()).hexdigest()
