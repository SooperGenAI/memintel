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

import os
from typing import Any

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
from app.models.errors import ErrorType, MemintelError, NotFoundError
from app.models.guardrails import Guardrails
from app.models.result import DryRunResult, ValidationResult
from app.models.task import CreateTaskRequest, Task, TaskStatus, TaskUpdateRequest

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
        Main entry point for POST /tasks.

        Pipeline:
          1. Build LLM prompt context (strict injection order [1]–[5]).
          2. Call LLM with retry / refinement loop.
          3. Validate strategy presence — immediate raise, no retry.
          4. Resolve action binding.
          5. Parse domain models (concept, condition, action).
          6a. dry_run=True  → return DryRunResult; nothing persisted.
          6b. dry_run=False → register definitions + create Task.

        Raises:
          MemintelError(SEMANTIC_ERROR)        — strategy absent, or LLM output
                                                invalid after all retries.
          MemintelError(ACTION_BINDING_FAILED) — action could not be resolved.
        """
        # Fetch active context — never raises; proceed without context on any error.
        # This is the only point where the context store is consulted; context is
        # never consulted on the evaluation path (invariant: context injection is
        # task-creation-only).
        app_context: ApplicationContext | None = None
        if self._context_store is not None:
            try:
                app_context = await self._context_store.get_active()
            except Exception as exc:
                log.error("context_fetch_failed", error=str(exc))

        context = self._build_context(request, app_context)
        llm_output = await self._generate_with_retries(request.intent, context)

        concept_dict   = llm_output.get("concept")   or {}
        condition_dict = llm_output.get("condition") or {}
        action_dict    = llm_output.get("action")    or {}

        # Strategy validation — hard fail; do NOT proceed if absent.
        self._validate_strategy_presence(condition_dict)

        # Action binding resolution (priority order: LLM → guardrails → ctx → sys).
        resolved_action_dict = self._resolve_action(action_dict, request)

        # Parse into domain models.
        try:
            concept   = ConceptDefinition(**concept_dict)
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

        # FIX 1: Validate compiled strategy against guardrails.constraints.disallowed_strategies.
        if self._guardrails is not None:
            self._validate_strategy_allowed(condition)

        # FIX 2: Apply deterministic bias rules to override threshold value from priors.
        condition = self._apply_bias_rules(condition, concept, request.intent)

        # FIX 3: Validate concept primitives exist in guardrails primitive registry.
        if self._guardrails is not None:
            self._validate_primitives_registered(concept)

        # dry_run — preview only; nothing registered or persisted.
        if request.dry_run:
            return self._build_dry_run_result(concept, condition, action)

        # Determine context_version and context_warning for the persisted task.
        context_version = app_context.version if app_context is not None else None
        context_warning: str | None = None
        if app_context is None:
            context_warning = (
                "No application context defined. Task created without domain context. "
                "Define context via POST /context for more accurate results."
            )

        # Determine guardrails_version for the persisted task.
        guardrails_version: str | None = None
        if self._guardrails_store is not None:
            active_gr_version = self._guardrails_store.get_active_version()
            if active_gr_version is not None:
                guardrails_version = active_gr_version.version

        # Register and persist.
        return await self._register_and_persist(
            concept, condition, action, request,
            context_version, context_warning, guardrails_version,
        )

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
        concept: ConceptDefinition,
        condition: ConditionDefinition,
        action: ActionDefinition,
        request: CreateTaskRequest,
        context_version: str | None = None,
        context_warning: str | None = None,
        guardrails_version: str | None = None,
    ) -> Task:
        """
        Register concept, condition, and action definitions; then create the Task.

        The system remains clean on any failure — no partial state is written.
        The returned Task is version-pinned: concept_id, concept_version,
        condition_id, action_id, and action_version are set at creation and
        never change (enforced by IMMUTABLE_TASK_FIELDS / TaskStore.update()).
        """
        # Namespace: request constraints override the concept's own namespace.
        if request.constraints and request.constraints.namespace:
            ns = request.constraints.namespace.value
        else:
            ns = concept.namespace.value if concept.namespace else "personal"

        # Register concept (runs _freeze_check: validate_schema + validate_types).
        await self._definition_registry.register(concept, namespace=ns)

        # Register condition (freeze check is skipped for non-concept types).
        await self._definition_registry.register(
            condition.model_dump(),
            namespace=ns,
            definition_type="condition",
        )

        # Register action.
        await self._definition_registry.register(
            action.model_dump(),
            namespace=ns,
            definition_type="action",
        )

        # Build the version-pinned Task and persist it.
        task = Task(
            intent=request.intent,
            concept_id=concept.concept_id,
            concept_version=concept.version,
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
            concept_id=concept.concept_id,
            concept_version=concept.version,
            condition_id=condition.condition_id,
            condition_version=condition.version,
            action_id=action.action_id,
            context_version=context_version,
            guardrails_version=guardrails_version,
        )
        return created_task
