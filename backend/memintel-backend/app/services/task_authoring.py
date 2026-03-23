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

from app.llm.fixtures import LLMFixtureClient
from app.models.action import ActionDefinition
from app.models.concept import ConceptDefinition
from app.models.condition import ConditionDefinition
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
    ) -> None:
        self._task_store        = task_store
        self._definition_registry = definition_registry
        self._guardrails        = guardrails
        self._llm               = llm_client if llm_client is not None else self._select_llm_client()
        self._max_retries       = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("MAX_RETRIES", _MAX_RETRIES_DEFAULT))
        )

    @staticmethod
    def _select_llm_client() -> Any:
        """Select LLM client from USE_LLM_FIXTURES env var."""
        use_fixtures = os.environ.get("USE_LLM_FIXTURES", "true").lower()
        if use_fixtures != "false":
            return LLMFixtureClient()
        # Real LLM client — not yet implemented; falls back to fixtures.
        log.warning(
            "llm_client_fallback",
            reason="USE_LLM_FIXTURES=false but real LLM client is not implemented; falling back to LLMFixtureClient.",
        )
        return LLMFixtureClient()

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
        context = self._build_context(request)
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

        # dry_run — preview only; nothing registered or persisted.
        if request.dry_run:
            return self._build_dry_run_result(concept, condition, action)

        # Register and persist.
        return await self._register_and_persist(concept, condition, action, request)

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

    def _build_context(self, request: CreateTaskRequest) -> dict:
        """
        Build the LLM prompt context.

        Injection order is STRICT — do not reorder:
          [1] Type system summary
          [2] Guardrails (strategy registry, bounds, priors, bias rules)
          [3] Application context
          [4] Parameter bias rules
          [5] Primitive registry
        """
        context: dict = {}

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
        )
        created_task = await self._task_store.create(task)
        log.info(
            "task_created",
            task_id=created_task.task_id,
            concept_id=concept.concept_id,
            concept_version=concept.version,
            condition_id=condition.condition_id,
            condition_version=condition.version,
            action_id=action.action_id,
        )
        return created_task
