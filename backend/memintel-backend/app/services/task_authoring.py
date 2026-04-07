"""
app/services/task_authoring.py
──────────────────────────────────────────────────────────────────────────────
TaskAuthoringService — LLM-driven task creation pipeline.

The ONLY service that calls the LLM.  Implements the full pipeline:
  1. Build prompt context in strict injection order
  2. Call LLM to generate concept + condition + action definitions
  3. Validate strategy presence (hard fail — no retry)
  4. Resolve action binding
  5. Enter refinement loop on validation failure (up to MAX_RETRIES)
  6. Register definitions + persist Task  OR  return DryRunResult

LLM context injection order — STRICT, DO NOT REORDER:
  [1] Type system summary
  [2] Guardrails (strategy registry, bounds, priors, bias rules)
  [3] Application context
  [4] Parameter bias rules
  [5] Primitive registry

Strategy validation:
  Every condition MUST include strategy.type and strategy.params.
  Missing strategy raises MemintelError(SEMANTIC_ERROR) immediately —
  no retry, no fallback.

Action binding resolution order:
  user-specified (LLM output) → guardrails default → app_context preferences
  → system default → ACTION_BINDING_FAILED (HTTP 422)

LLM failure handling:
  MAX_RETRIES = 3 (overridable via MAX_RETRIES env var or constructor param).
  On each failure: pass errors back to refine_task() if the client supports it;
  otherwise falls back to generate_task() for all retries.
  If MAX_RETRIES exceeded → MemintelError(SEMANTIC_ERROR) with last error.
  DO NOT persist any partial definition on failure — system remains clean.

dry_run=True:
  Returns DryRunResult immediately.  Nothing registered, compiled, or persisted.
  No task_id assigned.

USE_LLM_FIXTURES environment variable:
  True (default) → LLMFixtureClient  (development / test)
  False          → real LLM client   (production; falls back to fixtures until
                                     the real client is implemented)
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator

import structlog

from app.llm.client_factory import create_llm_client
from app.llm.prompts import build_context_prefix
from app.models.action import ActionDefinition
from app.models.concept import ConceptDefinition
from app.models.condition import (
    ConditionDefinition,
    StrategyType,
    ThresholdParams,
    ThresholdStrategy,
)
from app.models.context import ApplicationContext
from app.models.concept import MAX_VOCABULARY_IDS, VocabularyContext
from app.models.errors import (
    ConceptNotFoundError,
    ConflictError,
    ErrorType,
    MemintelError,
    NotFoundError,
    VocabularyContextTooLargeError,
    VocabularyMismatchError,
)
from app.models.guardrails import Guardrails
from app.models.result import DryRunResult, ValidationResult
from app.models.task import (
    CreateTaskRequest,
    ReasoningStep,
    ReasoningTrace,
    Task,
    TaskStatus,
    TaskUpdateRequest,
)

log = structlog.get_logger(__name__)

_MAX_RETRIES_DEFAULT = 3

# [1] Type system summary — always the first block injected into the LLM context.
_TYPE_SYSTEM_SUMMARY: dict[str, Any] = {
    "scalar_types":    ["float", "int", "boolean", "string"],
    "container_types": ["time_series<float>", "time_series<int>", "categorical"],
    "nullable_suffix": "?",
    "strategies":      ["threshold", "percentile", "z_score", "change", "equals", "composite"],
    "strategy_params": {
        "threshold":  {"direction": "above|below",             "value": "float"},
        "percentile": {"direction": "top|bottom",              "value": "float [0,100]"},
        "z_score":    {"threshold": "float >0",                "direction": "above|below|any", "window": "str"},
        "change":     {"direction": "increase|decrease|any",   "value": "float >=0",           "window": "str"},
        "equals":     {"value": "str",                         "labels": "list[str] | null"},
        "composite":  {"operator": "AND|OR",                   "operands": "list[condition_id] min=2"},
    },
}


class TaskAuthoringService:
    """
    LLM-driven task authoring service.

    create_task() is the only public entry point.
    All LLM calls are encapsulated here — no other service calls the LLM.

    Parameters
    ──────────
    task_store          — must implement ``async create(task: Task) → Task``.
    definition_registry — must implement
                          ``async register(body, namespace, ...) → DefinitionResponse``.
    guardrails          — Guardrails model for LLM context injection.
                          When None only the type system summary is injected.
    llm_client          — client with ``generate_task(intent, context) → dict``.
                          When None, auto-selected from USE_LLM_FIXTURES env var.
    max_retries         — total LLM call attempts (including the first).
                          Defaults to MAX_RETRIES env var, then 3.
    """

    def __init__(
        self,
        task_store: Any,
        definition_registry: Any,
        guardrails: Guardrails | None = None,
        llm_client: Any = None,
        max_retries: int | None = None,
        context_store: Any = None,
        guardrails_store: Any = None,
    ) -> None:
        self._task_store          = task_store
        self._definition_registry = definition_registry
        self._guardrails          = guardrails
        self._llm                 = llm_client if llm_client is not None else self._select_llm_client()
        self._max_retries         = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("MAX_RETRIES", _MAX_RETRIES_DEFAULT))
        )
        self._context_store       = context_store
        self._guardrails_store    = guardrails_store  # app.config.guardrails_store.GuardrailsStore

    @staticmethod
    def _select_llm_client() -> Any:
        """Select LLM client from USE_LLM_FIXTURES and LLM_PROVIDER env vars."""
        use_fixtures = os.environ.get("USE_LLM_FIXTURES", "true").lower() != "false"
        from app.models.config import LLMConfig
        config = LLMConfig.model_validate({
            "provider": os.environ.get("LLM_PROVIDER", "anthropic"),
            "model": os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514",
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "base_url": os.environ.get("LLM_BASE_URL"),
            "ssl_verify": os.environ.get("LLM_SSL_VERIFY", "true").lower() == "true",
            "timeout_seconds": int(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
        }, context={"resolved": True})
        return create_llm_client(config, use_fixtures)

    # ── Public API ──────────────────────────────────────────────────────────────

    async def create_task(self, request: CreateTaskRequest) -> Task | DryRunResult:
        """
        Main entry point for POST /tasks (non-streaming path — behaviour unchanged).

        Collects all items from ``_run_cor_pipeline()`` and returns the final
        Task or DryRunResult.  reasoning_trace is attached when return_reasoning=True.

        Raises:
          VocabularyMismatchError          — vocabulary_context both-empty or no match.
          VocabularyContextTooLargeError   — either list exceeds MAX_VOCABULARY_IDS.
          ConceptNotFoundError             — concept_id not in registry.
          MemintelError(SEMANTIC_ERROR)    — strategy absent, or LLM output
                                            invalid after all retries.
          MemintelError(ACTION_BINDING_FAILED) — action could not be resolved.
        """
        steps: list[ReasoningStep] = []
        result: Task | DryRunResult | None = None

        async for item in self._run_cor_pipeline(request):
            if isinstance(item, ReasoningStep):
                steps.append(item)
            else:
                result = item

        # Attach reasoning trace when caller opts in (Hard Rule 3: absent when False).
        if request.return_reasoning and isinstance(result, Task):
            result.reasoning_trace = ReasoningTrace(steps=steps)

        return result  # type: ignore[return-value]

    async def create_task_stream(
        self,
        request: CreateTaskRequest,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream the CoR pipeline as SSE event dicts (streaming path only).

        Yields dicts with keys ``event_type`` and ``data``:
          cor_step     — one per pipeline step (4 total on success).
          cor_complete — final event; payload includes task_id (None for dry_run).
          cor_error    — terminal error event on any failure or LLM timeout.

        HARD RULE: this method is ONLY called when stream=True.
        The non-streaming ``create_task()`` is completely unchanged.
        """
        last_failed_step: int | None = None

        try:
            async for item in self._run_cor_pipeline(request):
                if isinstance(item, ReasoningStep):
                    if item.outcome == "failed":
                        last_failed_step = item.step_index
                    yield {"event_type": "cor_step", "data": item.model_dump()}
                elif isinstance(item, Task):
                    yield {
                        "event_type": "cor_complete",
                        "data": {
                            "task_id": item.task_id,
                            "status": item.status.value if item.status else None,
                        },
                    }
                else:
                    # DryRunResult
                    yield {
                        "event_type": "cor_complete",
                        "data": {"task_id": None, "dry_run": True},
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
                    "failed_at_step": last_failed_step,
                },
            }

    # ── Internal CoR pipeline (shared by streaming and non-streaming) ───────────

    async def _run_cor_pipeline(
        self,
        request: CreateTaskRequest,
    ) -> AsyncGenerator[ReasoningStep | Task | DryRunResult, None]:
        """
        4-step Chain of Reasoning — yields ReasoningStep items and a final
        Task or DryRunResult.

        Both ``create_task()`` (non-streaming) and ``create_task_stream()``
        (streaming) iterate this generator. Do NOT call this method directly.

        Step layout:
          Step 1 — Intent Parsing:    pre-LLM validation + context build.
          Step 2 — Concept Selection: single LLM call (wrapped with wait_for).
          Step 3 — Condition Strategy: strategy validation + model parsing.
          Step 4 — Action Binding:    guardrails + persist or dry-run.
          Final  — Task or DryRunResult.

        On asyncio.TimeoutError at Step 2:
          yield ReasoningStep(step_index=2, outcome="failed")
          then raise — let the caller handle cor_error emission.
        """
        # ── Pre-LLM: vocabulary_context validation ────────────────────────────
        if request.vocabulary_context is not None:
            vc = request.vocabulary_context
            concept_ids   = vc.available_concept_ids or []
            condition_ids = vc.available_condition_ids or []

            if not concept_ids and not condition_ids:
                raise VocabularyMismatchError(
                    "vocabulary_context provided but both available_concept_ids "
                    "and available_condition_ids are empty.",
                )
            if len(concept_ids) > MAX_VOCABULARY_IDS:
                raise VocabularyContextTooLargeError(
                    f"available_concept_ids exceeds the maximum of "
                    f"{MAX_VOCABULARY_IDS} IDs per list.",
                )
            if len(condition_ids) > MAX_VOCABULARY_IDS:
                raise VocabularyContextTooLargeError(
                    f"available_condition_ids exceeds the maximum of "
                    f"{MAX_VOCABULARY_IDS} IDs per list.",
                )

        # ── Pre-LLM: concept_id existence check ───────────────────────────────
        pre_compiled_concept_id: str | None = None
        pre_compiled_concept_version: str | None = None
        if request.concept_id is not None:
            try:
                version_list = await self._definition_registry.versions(
                    request.concept_id
                )
                pre_compiled_concept_version = version_list.versions[0].version
                pre_compiled_concept_id = request.concept_id
            except NotFoundError:
                raise ConceptNotFoundError(
                    f"concept '{request.concept_id}' not found in registry.",
                )

        # Fetch active context — never raises; proceed without context on any error.
        app_context: ApplicationContext | None = None
        if self._context_store is not None:
            try:
                app_context = await self._context_store.get_active()
            except Exception as exc:
                log.error("context_fetch_failed", error=str(exc))

        context = self._build_context(request, app_context)

        # ── Step 1: Intent Parsing ─────────────────────────────────────────────
        skipped = pre_compiled_concept_id is not None
        step1 = ReasoningStep(
            step_index=1,
            label="Intent Parsing",
            summary="skipped (concept pre-compiled)" if skipped
                    else "Intent parsed; concept and condition context established.",
            outcome="skipped" if skipped else "accepted",
        )
        yield step1

        # ── Step 2: Concept Selection (LLM call — wrapped with wait_for) ──────
        try:
            llm_output = await asyncio.wait_for(
                self._generate_with_retries(request.intent, context),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            yield ReasoningStep(
                step_index=2,
                label="Concept Selection",
                summary="LLM timed out at step 2",
                outcome="failed",
            )
            raise  # let create_task_stream() emit cor_error

        concept_dict   = llm_output.get("concept")   or {}
        condition_dict = llm_output.get("condition") or {}
        action_dict    = llm_output.get("action")    or {}

        # Post-LLM: vocabulary check
        if (
            request.vocabulary_context is not None
            and request.vocabulary_context.available_concept_ids
        ):
            generated_cid = concept_dict.get("concept_id", "")
            if generated_cid not in request.vocabulary_context.available_concept_ids:
                log.warning(
                    "vocab_check_failed",
                    llm_concept_id=generated_cid,
                    available=request.vocabulary_context.available_concept_ids,
                    full_concept_dict=concept_dict,
                    full_llm_output=llm_output,
                )
                raise VocabularyMismatchError(
                    f"LLM selected concept '{generated_cid}' which is not in "
                    "vocabulary_context.available_concept_ids.",
                )
            else:
                log.info(
                    "vocab_check_passed",
                    llm_concept_id=generated_cid,
                    full_concept_dict=concept_dict,
                )

        if (
            request.vocabulary_context is not None
            and request.vocabulary_context.available_condition_ids
        ):
            generated_condition_id = condition_dict.get("condition_id", "")
            if generated_condition_id not in request.vocabulary_context.available_condition_ids:
                raise VocabularyMismatchError(
                    f"LLM selected condition '{generated_condition_id}' which is not in "
                    "vocabulary_context.available_condition_ids.",
                )

        _step2_concept_id   = pre_compiled_concept_id or concept_dict.get("concept_id")
        _step2_output_type  = concept_dict.get("output_type") if not skipped else None
        step2 = ReasoningStep(
            step_index=2,
            label="Concept Selection",
            summary="skipped (concept pre-compiled)" if skipped
                    else "Concept selected from intent and registry.",
            outcome="skipped" if skipped else "accepted",
            concept_id=_step2_concept_id,
            output_type=_step2_output_type,
        )
        yield step2

        # ── Step 3: Condition Strategy ─────────────────────────────────────────
        # Strategy validation — hard fail; do NOT proceed if absent.
        self._validate_strategy_presence(condition_dict)

        # Action binding resolution (priority order: LLM → guardrails → ctx → sys).
        resolved_action_dict = self._resolve_action(action_dict, request)

        # Pin composite operand versions before parsing.
        await self._pin_composite_operand_versions(condition_dict)

        # Parse into domain models.
        concept: ConceptDefinition | None = None
        if pre_compiled_concept_id is None:
            try:
                concept = ConceptDefinition(**concept_dict)
            except Exception as exc:
                raise MemintelError(
                    ErrorType.SEMANTIC_ERROR,
                    f"LLM output could not be parsed into domain models: {exc}",
                    suggestion=(
                        "Check that the LLM output conforms to ConceptDefinition, "
                        "ConditionDefinition, and ActionDefinition schemas."
                    ),
                ) from exc

        try:
            condition = ConditionDefinition(**condition_dict)
            action    = ActionDefinition(**resolved_action_dict)
        except Exception as exc:
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                f"LLM output could not be parsed into domain models: {exc}",
                suggestion=(
                    "Check that the LLM output conforms to ConceptDefinition, "
                    "ConditionDefinition, and ActionDefinition schemas."
                ),
            ) from exc

        _strat       = condition.strategy
        _strat_type  = _strat.type.value if hasattr(_strat.type, "value") else str(_strat.type)
        _params      = _strat.params
        _direction   = str(getattr(_params, "direction", None) or "")
        _direction   = _direction.value if hasattr(_direction, "value") else _direction or None
        _threshold   = getattr(_params, "value", None) or getattr(_params, "threshold", None)
        yield ReasoningStep(
            step_index=3,
            label="Condition Strategy",
            summary="Condition strategy selected and parameters resolved.",
            outcome="accepted",
            strategy_type=_strat_type,
            direction=_direction,
            threshold=float(_threshold) if _threshold is not None else None,
        )

        # ── Step 4: Action Binding (guardrails + persist) ──────────────────────
        # FIX 1: Validate compiled strategy against guardrails.
        if self._guardrails is not None:
            self._validate_strategy_allowed(condition)

        # FIX 2: Apply deterministic bias rules.
        if concept is not None:
            condition = self._apply_bias_rules(condition, concept, request.intent)

        # FIX 3: Validate concept primitives.
        if self._guardrails is not None and concept is not None:
            self._validate_primitives_registered(concept)

        _action_type = action.config.type.value if hasattr(action.config.type, "value") else str(action.config.type)
        _channel     = getattr(action.config, "channel", None)
        yield ReasoningStep(
            step_index=4,
            label="Action Binding",
            summary="Action binding resolved.",
            outcome="accepted",
            action_type=_action_type,
            channel=_channel,
        )

        # ── Final: dry_run or register + persist ──────────────────────────────
        if request.dry_run:
            if concept is not None:
                yield self._build_dry_run_result(concept, condition, action)
            else:
                yield DryRunResult(
                    concept={
                        "concept_id": pre_compiled_concept_id,
                        "version": pre_compiled_concept_version,
                    },
                    condition=condition,
                    action_id=action.action_id,
                    action_version=action.version,
                    validation=ValidationResult(valid=True),
                )
            return

        context_version = app_context.version if app_context is not None else None
        context_warning: str | None = None
        if app_context is None:
            context_warning = (
                "No application context defined. Task created without domain context. "
                "Define context via POST /context for more accurate results."
            )

        guardrails_version: str | None = None
        if self._guardrails_store is not None:
            active_gr_version = self._guardrails_store.get_active_version()
            if active_gr_version is not None:
                guardrails_version = active_gr_version.version

        task = await self._register_and_persist(
            concept, condition, action, request,
            context_version, context_warning, guardrails_version,
            pre_compiled_concept_id=pre_compiled_concept_id,
            pre_compiled_concept_version=pre_compiled_concept_version,
        )

        yield task

    async def update_task(self, task_id: str, body: TaskUpdateRequest) -> Task:
        """
        Update a task's operational settings.

        Validates condition_version rebinding before persisting:
          - Fetches the current task to obtain condition_id.
          - Verifies the new condition_version exists in the registry.
            Raises NotFoundError → HTTP 404 if missing.
          - Logs a warning if the target version is deprecated (does not block).

        ConflictError (HTTP 409) is raised by TaskStore.update() if the task
        has already been deleted.

        Raises:
          NotFoundError — task not found, or condition_version does not exist.
          ConflictError — task is in a terminal (deleted) state.
        """
        task = await self._task_store.get(task_id)
        if task is None:
            raise NotFoundError(f"Task '{task_id}' not found.", location="task_id")

        if body.condition_version is not None:
            try:
                condition_body = await self._definition_registry.get(
                    task.condition_id, body.condition_version
                )
            except NotFoundError:
                raise NotFoundError(
                    f"Condition version '{body.condition_version}' does not exist "
                    f"for condition '{task.condition_id}'.",
                    location="condition_version",
                    suggestion="Check available versions via GET /conditions/{id}.",
                )
            if condition_body.get("deprecated"):
                log.warning(
                    "rebinding_to_deprecated_condition",
                    task_id=task_id,
                    condition_id=task.condition_id,
                    condition_version=body.condition_version,
                )

        return await self._task_store.update(task_id, body.to_patch_dict())

    async def delete_task(self, task_id: str) -> Task:
        """
        Soft-delete a task (status → 'deleted').

        Already-deleted tasks are treated as not found — raises NotFoundError
        → HTTP 404. The row is retained for audit; deletion is irreversible
        via the API.

        Raises:
          NotFoundError — task not found or already deleted.
        """
        task = await self._task_store.get(task_id)
        if task is None or task.status == TaskStatus.DELETED:
            raise NotFoundError(f"Task '{task_id}' not found.", location="task_id")
        return await self._task_store.update(task_id, {"status": TaskStatus.DELETED.value})

    # ── LLM generation with retry loop ─────────────────────────────────────────

    async def _generate_with_retries(self, intent: str, context: dict) -> dict:
        """
        Call the LLM up to self._max_retries times.

        First call always uses generate_task().  Subsequent calls use
        refine_task() when available (targeted correction, not full regeneration);
        otherwise falls back to generate_task() for all retries.

        Raises MemintelError(SEMANTIC_ERROR) after all retries are exhausted,
        including the last LLM error in the message.
        """
        previous_output: dict = {}
        last_error: str = ""

        for attempt in range(self._max_retries):
            try:
                if attempt == 0 or not hasattr(self._llm, "refine_task"):
                    output = self._llm.generate_task(intent, context)
                else:
                    output = self._llm.refine_task(
                        intent,
                        context,
                        previous_output=previous_output,
                        errors=[last_error],
                    )
            except Exception as exc:
                last_error = str(exc)
                previous_output = {}
                log.warning(
                    "llm_call_failed",
                    attempt=attempt + 1,
                    max=self._max_retries,
                    error=last_error,
                )
                if attempt == self._max_retries - 1:
                    raise MemintelError(
                        ErrorType.SEMANTIC_ERROR,
                        f"LLM failed after {self._max_retries} attempt(s). "
                        f"Last error: {last_error}",
                        suggestion="Check LLM client configuration and retry.",
                    ) from exc
                continue

            error = self._validate_llm_output_structure(output)
            if error is None:
                return output

            last_error = error
            previous_output = output
            log.warning(
                "llm_output_invalid",
                attempt=attempt + 1,
                max=self._max_retries,
                error=last_error,
            )
            if attempt == self._max_retries - 1:
                raise MemintelError(
                    ErrorType.SEMANTIC_ERROR,
                    f"LLM output failed validation after {self._max_retries} attempt(s). "
                    f"Last error: {last_error}",
                    suggestion=(
                        "Ensure the LLM output is a JSON object with 'concept', "
                        "'condition', and 'action' keys."
                    ),
                )

        # Unreachable — the loop always returns or raises.
        raise MemintelError(  # pragma: no cover
            ErrorType.SEMANTIC_ERROR,
            f"LLM output failed validation after {self._max_retries} attempt(s). "
            f"Last error: {last_error}",
        )

    @staticmethod
    def _validate_llm_output_structure(output: Any) -> str | None:
        """
        Check that the LLM output has the required top-level structure.

        Returns None on success, or an error string describing the problem.
        Does NOT check strategy presence — that is a separate hard-fail step.
        """
        if not isinstance(output, dict):
            return f"LLM output must be a JSON object; got {type(output).__name__}"
        for key in ("concept", "condition", "action"):
            if key not in output or not isinstance(output.get(key), dict):
                return f"LLM output missing required key '{key}' (must be a JSON object)"
        return None

    # ── Composite operand version pinning ──────────────────────────────────────

    async def _pin_composite_operand_versions(self, condition_dict: dict) -> None:
        """
        Convert plain string operands to version-pinned OperandRef dicts in-place.

        The LLM emits composite operands as bare condition_ids, e.g.:
          {"operator": "AND", "operands": ["org.high_churn_risk", "org.high_ltv_customer"]}

        This method replaces each string with the OperandRef form, resolved to
        the latest registered version at authoring time:
          {"operator": "AND", "operands": [
              {"condition_id": "org.high_churn_risk", "condition_version": "1.0"},
              {"condition_id": "org.high_ltv_customer", "condition_version": "2.0"},
          ]}

        OperandRef dicts that are already pinned (dicts with condition_id /
        condition_version) are passed through unchanged so idempotent callers
        and direct-API registrations are not broken.

        Raises:
          NotFoundError — if an operand condition_id has no registered versions.
        """
        strategy = condition_dict.get("strategy") or {}
        if strategy.get("type") != "composite":
            return
        params = strategy.get("params") or {}
        raw_operands = params.get("operands") or []
        pinned: list[dict] = []
        for operand in raw_operands:
            if isinstance(operand, dict):
                # Already a pinned OperandRef — pass through unchanged.
                pinned.append(operand)
            else:
                # Plain string condition_id — resolve to latest version.
                operand_id: str = operand
                version_list = await self._definition_registry.versions(operand_id)
                latest_version = version_list.versions[0].version
                pinned.append({
                    "condition_id": operand_id,
                    "condition_version": latest_version,
                })
        params["operands"] = pinned

    # ── Strategy validation ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_strategy_presence(condition_dict: dict) -> None:
        """
        Enforce that every condition includes strategy.type and strategy.params.

        Raises MemintelError(SEMANTIC_ERROR) immediately.  This check is
        intentionally outside the retry loop — a missing strategy is a
        hard authoring error, not a retryable validation failure.
        """
        strategy = condition_dict.get("strategy")
        if not strategy:
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                "Condition is missing 'strategy'.  Every condition must include "
                "strategy.type and strategy.params.",
                location="condition.strategy",
                suggestion=(
                    "Specify a strategy: threshold, percentile, z_score, change, "
                    "equals, or composite."
                ),
            )
        if not isinstance(strategy, dict) or not strategy.get("type"):
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                "Condition strategy is missing 'type'.  Every condition must include "
                "strategy.type and strategy.params.",
                location="condition.strategy.type",
            )
        if "params" not in strategy:
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                "Condition strategy is missing 'params'.  Every condition must include "
                "strategy.type and strategy.params.",
                location="condition.strategy.params",
            )

    # ── Reasoning trace builder ─────────────────────────────────────────────────

    @staticmethod
    def _build_reasoning_steps(skipped_steps_1_and_2: bool = False) -> list[ReasoningStep]:
        """
        Build the 4-step CoR reasoning trace as synthetic annotations.

        The task authoring pipeline uses a single LLM call (not a step-by-step
        generator), so steps are annotated post-hoc to represent the logical
        CoR phases.

        When skipped_steps_1_and_2=True (concept_id shortcut), Steps 1 and 2
        are marked as outcome='skipped' with an explanatory summary.
        """
        if skipped_steps_1_and_2:
            step1 = ReasoningStep(
                step_index=1, label="Intent Parsing",
                summary="skipped (concept pre-compiled)", outcome="skipped",
            )
            step2 = ReasoningStep(
                step_index=2, label="Concept Selection",
                summary="skipped (concept pre-compiled)", outcome="skipped",
            )
        else:
            step1 = ReasoningStep(
                step_index=1, label="Intent Parsing",
                summary="Intent parsed; concept and condition context established.",
                outcome="accepted",
            )
            step2 = ReasoningStep(
                step_index=2, label="Concept Selection",
                summary="Concept selected from intent and registry.",
                outcome="accepted",
            )

        step3 = ReasoningStep(
            step_index=3, label="Condition Strategy",
            summary="Condition strategy selected and parameters resolved.",
            outcome="accepted",
        )
        step4 = ReasoningStep(
            step_index=4, label="Action Binding",
            summary="Action binding resolved.",
            outcome="accepted",
        )
        return [step1, step2, step3, step4]

    # ── Post-compilation guardrails enforcement ────────────────────────────────

    def _validate_strategy_allowed(self, condition: ConditionDefinition) -> None:
        """
        Raise MemintelError(SEMANTIC_ERROR) if the compiled strategy type is in
        guardrails.constraints.disallowed_strategies.

        self._guardrails is non-None at the call site.
        """
        disallowed = self._guardrails.constraints.disallowed_strategies
        if not disallowed:
            return
        strategy_type: str = condition.strategy.type.value
        if strategy_type in disallowed:
            allowed = [s for s in self._guardrails.strategy_registry if s not in disallowed]
            raise MemintelError(
                ErrorType.SEMANTIC_ERROR,
                f"Strategy '{strategy_type}' is not permitted by guardrails constraints "
                f"(disallowed_strategies includes '{strategy_type}').",
                location="condition.strategy.type",
                suggestion=(
                    f"Use one of the allowed strategies: {allowed}."
                    if allowed
                    else "All strategies are currently disallowed — check guardrails configuration."
                ),
            )

    def _apply_bias_rules(
        self,
        condition: ConditionDefinition,
        concept: ConceptDefinition,
        intent: str,
    ) -> ConditionDefinition:
        """
        Apply guardrails parameter_bias_rules to deterministically override the
        compiled threshold value with a primitive-level prior.

        Only applies when all of the following hold:
          - guardrails is loaded with a non-empty parameter_bias_rules list
          - the compiled condition uses the threshold strategy
          - a bias rule keyword matches the intent (case-insensitive, first-match)
          - the concept's primitive has a threshold_priors entry for the severity

        Severity resolution:
          Base tier = medium (index 1 of ["low", "medium", "high"]).
          The matched rule's severity_shift shifts the tier:
            +1 → high, 0 → medium, -1 → low.
          Tier index is clamped to [0, 2].

        Returns the (possibly updated) ConditionDefinition.
        """
        if self._guardrails is None:
            return condition
        bias_rules = self._guardrails.parameter_bias_rules
        if not bias_rules:
            return condition

        # Only threshold strategy exposes an overrideable numeric value.
        if condition.strategy.type != StrategyType.THRESHOLD:
            return condition

        # Find the first matching bias rule in the intent.
        intent_lower = intent.lower()
        matching_rule = None
        for rule in bias_rules:
            if rule.if_instruction_contains.lower() in intent_lower:
                matching_rule = rule
                break

        if matching_rule is None:
            return condition

        # Resolve severity tier: base=medium(1), apply shift, clamp to [0,2].
        _TIERS = ("low", "medium", "high")
        tier_idx = max(0, min(2, 1 + matching_rule.effect.severity_shift))
        severity = _TIERS[tier_idx]

        # Look up threshold prior for the concept's primitive at this severity.
        primitives = self._guardrails.primitives
        if not primitives:
            return condition

        prior_value: float | None = None
        for prim_id, prim_hint in primitives.items():
            if prim_id in concept.primitives and prim_hint.threshold_priors:
                strategy_priors = prim_hint.threshold_priors.get("threshold")
                if strategy_priors is not None:
                    prior_value = strategy_priors.get(severity)
                    if prior_value is not None:
                        break

        if prior_value is None:
            return condition

        # Build a new condition with the overridden threshold value.
        new_params = ThresholdParams(
            direction=condition.strategy.params.direction,  # type: ignore[union-attr]
            value=prior_value,
        )
        new_strategy = ThresholdStrategy(type=StrategyType.THRESHOLD, params=new_params)
        log.debug(
            "bias_rule_applied",
            intent_keyword=matching_rule.if_instruction_contains,
            severity=severity,
            prior_value=prior_value,
        )
        return condition.model_copy(update={"strategy": new_strategy})

    def _validate_primitives_registered(self, concept: ConceptDefinition) -> None:
        """
        Raise MemintelError(REFERENCE_ERROR) if a primitive referenced in the
        compiled concept is not declared in guardrails.primitives.

        Validation is skipped when guardrails.primitives is empty — an empty
        dict means no primitive registry is configured for this guardrails version,
        so no constraint is enforced.

        self._guardrails is non-None at the call site.
        """
        guardrails_primitives = self._guardrails.primitives
        if not guardrails_primitives:
            return

        for prim_id in concept.primitives:
            if prim_id not in guardrails_primitives:
                raise MemintelError(
                    ErrorType.REFERENCE_ERROR,
                    f"Primitive '{prim_id}' referenced in the compiled concept is not "
                    f"registered in the guardrails primitive registry.",
                    location=f"concept.primitives.{prim_id}",
                    suggestion=(
                        "Register the primitive in the guardrails configuration, or "
                        "check that the LLM used a declared primitive name."
                    ),
                )

    # ── Action binding resolution ───────────────────────────────────────────────

    def _resolve_action(
        self,
        action_dict: dict,
        request: CreateTaskRequest,
    ) -> dict:
        """
        Resolve the action binding in priority order:
          1. LLM-provided (user-specified via LLM output)
          2. Guardrails default
          3. Application context preferences
          4. System default (built from request delivery config)
          5. MemintelError(ACTION_BINDING_FAILED)
        """
        # 1. LLM-provided — complete if action_id, config, and trigger are present.
        if self._action_dict_is_complete(action_dict):
            return action_dict

        # 2. Guardrails default (reserved for future guardrails expansion).
        guardrails_default = self._guardrails_default_action()
        if guardrails_default:
            return guardrails_default

        # 3. Application context preferences (reserved for future expansion).
        if self._guardrails and self._guardrails.application_context:
            ctx_action = self._app_context_default_action(
                self._guardrails.application_context, request
            )
            if ctx_action:
                return ctx_action

        # 4. System default: synthesise action config from request delivery config.
        system_action = self._system_default_action(action_dict, request)
        if system_action:
            return system_action

        # 5. Unresolved.
        raise MemintelError(
            ErrorType.ACTION_BINDING_FAILED,
            "Action binding failed: no action could be resolved from LLM output, "
            "guardrails defaults, application context, or delivery config.",
            suggestion=(
                "Ensure the LLM returns a complete action dict with action_id, "
                "config, and trigger.  Or configure a default action in the guardrails."
            ),
        )

    @staticmethod
    def _action_dict_is_complete(action_dict: dict) -> bool:
        """Return True if the action dict contains action_id, config, and trigger."""
        return bool(
            action_dict
            and action_dict.get("action_id")
            and action_dict.get("config")
            and action_dict.get("trigger")
        )

    def _guardrails_default_action(self) -> dict | None:
        """Reserved: extract a default action from guardrails. Not yet implemented."""
        return None

    @staticmethod
    def _app_context_default_action(
        app_context: Any,
        request: CreateTaskRequest,
    ) -> dict | None:
        """Reserved: build a default action from application context. Not yet implemented."""
        return None

    @staticmethod
    def _system_default_action(
        action_dict: dict,
        request: CreateTaskRequest,
    ) -> dict | None:
        """
        Last-resort action: synthesise a config from the request delivery config.

        Requires action_dict to have at least action_id and trigger (condition
        binding).  Only the config block is synthesised from delivery.
        """
        if not action_dict.get("action_id") or not action_dict.get("trigger"):
            return None

        from app.models.task import DeliveryType

        delivery = request.delivery
        if delivery.type == DeliveryType.WEBHOOK and delivery.endpoint:
            config: dict = {"type": "webhook", "endpoint": delivery.endpoint}
        elif delivery.type in (DeliveryType.NOTIFICATION, DeliveryType.EMAIL) and delivery.channel:
            config = {"type": "notification", "channel": delivery.channel}
        elif delivery.type == DeliveryType.WORKFLOW and delivery.workflow_id:
            config = {"type": "workflow", "workflow_id": delivery.workflow_id}
        else:
            return None

        return {**action_dict, "config": config}

    # ── LLM prompt context ──────────────────────────────────────────────────────

    def _build_context(
        self,
        request: CreateTaskRequest,
        app_context: ApplicationContext | None = None,
    ) -> dict:
        """
        Build the LLM prompt context.

        Injection order is STRICT — do not reorder:
          [0] Application context prefix (dynamic domain knowledge; injected
              BEFORE all other instructions when an active context exists)
          [1] Type system summary
          [2] Guardrails (strategy registry, bounds, priors, bias rules)
          [3] Application context (guardrails-level, reserved)
          [4] Parameter bias rules
          [5] Primitive registry
        """
        context: dict = {}

        # [0] Application context prefix — injected first so the LLM sees domain
        # knowledge before any technical instructions. Empty string when no context.
        prefix = build_context_prefix(app_context)
        if prefix:
            context["context_prefix"] = prefix

        # [1] Type system summary — always present.
        context["type_system"] = _TYPE_SYSTEM_SUMMARY

        if self._guardrails:
            gr = self._guardrails

            # [2] Guardrails core.
            context["guardrails"] = {
                "strategy_registry": {
                    name: entry.model_dump()
                    for name, entry in gr.strategy_registry.items()
                },
                "constraints": gr.constraints.model_dump(),
                "thresholds": {
                    name: priors.model_dump()
                    for name, priors in gr.thresholds.items()
                },
            }

            # [3] Application context.
            if gr.application_context:
                context["application_context"] = gr.application_context.model_dump()

            # [4] Parameter bias rules.
            if gr.parameter_bias_rules:
                context["parameter_bias_rules"] = [
                    rule.model_dump() for rule in gr.parameter_bias_rules
                ]

            # [5] Primitive registry.
            if gr.primitives:
                context["primitives"] = {
                    name: hint.model_dump() for name, hint in gr.primitives.items()
                }

        # Request constraints forwarded as authoring hints.
        if request.constraints:
            context["request_constraints"] = request.constraints.model_dump(
                exclude_none=True
            )

        # vocabulary_context — forwarded to LLM so it can restrict concept selection.
        if request.vocabulary_context is not None:
            context["vocabulary_context"] = request.vocabulary_context.model_dump()

        return context

    # ── dry_run ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_dry_run_result(
        concept: ConceptDefinition,
        condition: ConditionDefinition,
        action: ActionDefinition,
    ) -> DryRunResult:
        """
        Build a DryRunResult preview.

        Nothing is registered, compiled, or persisted.  No task_id is assigned.
        """
        return DryRunResult(
            concept=concept.model_dump(),
            condition=condition,
            action_id=action.action_id,
            action_version=action.version,
            validation=ValidationResult(valid=True),
        )

    # ── Register + persist ───────────────────────────────────────────────────────

    async def _register_and_persist(
        self,
        concept: ConceptDefinition | None,
        condition: ConditionDefinition,
        action: ActionDefinition,
        request: CreateTaskRequest,
        context_version: str | None = None,
        context_warning: str | None = None,
        guardrails_version: str | None = None,
        pre_compiled_concept_id: str | None = None,
        pre_compiled_concept_version: str | None = None,
    ) -> Task:
        """
        Register concept, condition, and action definitions; then create the Task.

        The system remains clean on any failure — no partial state is written.
        The returned Task is version-pinned: concept_id, concept_version,
        condition_id, action_id, and action_version are set at creation and
        never change (enforced by IMMUTABLE_TASK_FIELDS / TaskStore.update()).

        When pre_compiled_concept_id is set, the concept was already registered
        via POST /concepts/register — concept registration is skipped and the
        pre_compiled ids are used for the Task's concept_id/concept_version.
        """
        # Namespace: request constraints override the concept's own namespace.
        if request.constraints and request.constraints.namespace:
            ns = request.constraints.namespace.value
        elif concept is not None:
            ns = concept.namespace.value if concept.namespace else "personal"
        else:
            ns = "personal"

        # Register concept only when it was LLM-generated (not pre-compiled).
        # ConflictError means a concurrent request already registered the same
        # (id, version) — safe to ignore because definitions are immutable.
        if concept is not None:
            # Runs _freeze_check: validate_schema + validate_types.
            try:
                await self._definition_registry.register(concept, namespace=ns)
            except ConflictError:
                pass  # already registered by a concurrent request

        # Register condition (freeze check is skipped for non-concept types).
        try:
            await self._definition_registry.register(
                condition.model_dump(),
                namespace=ns,
                definition_type="condition",
            )
        except ConflictError:
            pass  # already registered by a concurrent request

        # Register action.
        try:
            await self._definition_registry.register(
                action.model_dump(),
                namespace=ns,
                definition_type="action",
            )
        except ConflictError:
            pass  # already registered by a concurrent request

        # Determine concept_id/version: pre-compiled or LLM-generated.
        task_concept_id      = pre_compiled_concept_id      or concept.concept_id
        task_concept_version = pre_compiled_concept_version or concept.version

        # Build the version-pinned Task and persist it.
        task = Task(
            intent=request.intent,
            concept_id=task_concept_id,
            concept_version=task_concept_version,
            condition_id=condition.condition_id,
            condition_version=condition.version,
            action_id=action.action_id,
            action_version=action.version,
            entity_scope=request.entity_scope,
            delivery=request.delivery,
            status=TaskStatus.ACTIVE,
            context_version=context_version,
            guardrails_version=guardrails_version,
        )
        created_task = await self._task_store.create(task)

        # context_warning is not a DB field — set it on the returned object
        # so callers receive the informational message when no context existed.
        created_task.context_warning = context_warning

        log.info(
            "task_created",
            task_id=created_task.task_id,
            concept_id=task_concept_id,
            concept_version=task_concept_version,
            condition_id=condition.condition_id,
            condition_version=condition.version,
            action_id=action.action_id,
            context_version=context_version,
            guardrails_version=guardrails_version,
        )
        return created_task
