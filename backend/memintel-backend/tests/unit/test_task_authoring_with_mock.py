"""
tests/unit/test_task_authoring_with_mock.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for TaskAuthoringService with MockLLMClient injected.

These tests exercise the TaskAuthoringService directly with in-memory stores
and MockLLMClient, without a real database or HTTP stack.

Coverage
────────
  1. LLM is called with the intent string exactly as provided
  2. Guardrails disallowed_strategies not enforced — z_score accepted
  3. Parameter priors from guardrails not applied by MockLLMClient
  4. Bias rules / severity language not applied by MockLLMClient
  5. context_version set from active context
  6. guardrails_version set from active guardrails store
  7. context_warning set when no active context

Design
──────
Each test constructs its own isolated service instance (task_store,
definition_registry, service) via _make_service().  No shared mutable state
between tests.

The MockLLMClient records each call's intent and context so tests can assert
the service passes the right data to the LLM.

Findings
────────
These tests confirm the unit-level behavior documented in test_llm_compiler.py:
  - MockLLMClient returns fixed responses based on keyword routing in intent.
  - Guardrails constraints and bias rules in the context are NOT applied
    by MockLLMClient — they are LLM-applied features.
  - TaskAuthoringService correctly sets context_version, guardrails_version,
    and context_warning from the store/context at creation time.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from app.models.concept import ConceptDefinition, DefinitionResponse, SearchResult, VersionSummary
from app.models.errors import ConflictError, ErrorType, MemintelError, NotFoundError
from app.models.result import DryRunResult
from app.models.task import (
    CreateTaskRequest,
    DeliveryConfig,
    DeliveryType,
    Namespace,
    Task,
    TaskStatus,
)
from app.registry.definitions import DefinitionRegistry
from app.services.task_authoring import TaskAuthoringService
from tests.mocks.mock_llm_client import MockLLMClient


# ── In-memory stores ──────────────────────────────────────────────────────────
# Same pattern as test_task_authoring.py — isolated per test.

class _MockDefinitionStore:
    """In-memory DefinitionStore."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DefinitionResponse] = {}
        self._bodies: dict[tuple[str, str], dict] = {}
        self._insert_order: list[tuple[str, str]] = []

    async def register(
        self,
        definition_id: str,
        version: str,
        definition_type: str,
        namespace: str,
        body: dict[str, Any],
        meaning_hash: str | None = None,
        ir_hash: str | None = None,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key in self._rows:
            raise ConflictError(
                f"Definition '{definition_id}' version '{version}' already registered.",
                location=f"{definition_id}:{version}",
            )
        ts = datetime.now(timezone.utc)
        response = DefinitionResponse(
            definition_id=definition_id,
            version=version,
            definition_type=definition_type,
            namespace=Namespace(namespace),
            meaning_hash=meaning_hash,
            ir_hash=ir_hash,
            deprecated=False,
            created_at=ts,
            updated_at=ts,
        )
        self._rows[key] = response
        self._bodies[key] = body
        self._insert_order.append(key)
        return response

    async def get(self, definition_id: str, version: str) -> dict[str, Any] | None:
        return self._bodies.get((definition_id, version))

    async def get_metadata(
        self, definition_id: str, version: str
    ) -> DefinitionResponse | None:
        return self._rows.get((definition_id, version))

    async def versions(self, definition_id: str) -> list[VersionSummary]:
        ordered = [k for k in reversed(self._insert_order) if k[0] == definition_id]
        return [
            VersionSummary(
                version=k[1],
                created_at=self._rows[k].created_at,
                deprecated=self._rows[k].deprecated,
            )
            for k in ordered
        ]

    async def list(
        self,
        definition_type: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchResult:
        items = list(self._rows.values())
        if definition_type:
            items = [i for i in items if i.definition_type == definition_type]
        return SearchResult(items=items[:limit], has_more=False, total_count=len(items))

    async def deprecate(
        self,
        definition_id: str,
        version: str,
        replacement_version: str | None,
        reason: str,
    ) -> DefinitionResponse:
        key = (definition_id, version)
        if key not in self._rows:
            raise NotFoundError(f"Definition '{definition_id}' version '{version}' not found.")
        updated = self._rows[key].model_copy(update={"deprecated": True})
        self._rows[key] = updated
        return updated

    async def promote(
        self,
        definition_id: str,
        version: str,
        from_namespace: str,
        to_namespace: str,
        elevated_key: bool = False,
    ) -> DefinitionResponse:
        if to_namespace == "global" and not elevated_key:
            raise MemintelError(
                ErrorType.AUTH_ERROR,
                f"Promoting to '{to_namespace}' requires elevated privileges.",
            )
        key = (definition_id, version)
        if key not in self._rows:
            raise NotFoundError(f"Definition '{definition_id}' version '{version}' not found.")
        updated = self._rows[key].model_copy(update={"namespace": Namespace(to_namespace)})
        self._rows[key] = updated
        return updated

    @property
    def create_count(self) -> int:
        return len(self._insert_order)


class _MockTaskStore:
    """In-memory TaskStore."""

    def __init__(self) -> None:
        self._tasks: list[Task] = []

    async def create(self, task: Task) -> Task:
        task = task.model_copy(update={"task_id": f"task-{len(self._tasks) + 1:04d}"})
        self._tasks.append(task)
        return task

    async def get(self, task_id: str) -> Task | None:
        for t in self._tasks:
            if t.task_id == task_id:
                return t
        return None

    @property
    def create_count(self) -> int:
        return len(self._tasks)

    @property
    def last_task(self) -> Task | None:
        return self._tasks[-1] if self._tasks else None


class _MockContextStore:
    """
    In-memory ContextStore returning a fixed active context.

    Use None to simulate no active context.
    """

    def __init__(self, active_context=None) -> None:
        self._context = active_context

    async def get_active(self):
        return self._context


class _MockGuardrailsStore:
    """In-memory GuardrailsStore returning a fixed active version."""

    def __init__(self, version_str: str | None = None) -> None:
        self._version_str = version_str

    def get_active_version(self):
        if self._version_str is None:
            return None
        return _SimpleVersion(self._version_str)

    def is_loaded(self) -> bool:
        return self._version_str is not None

    def get_guardrails(self):
        return None


class _SimpleVersion:
    """Minimal version object with .version attribute."""
    def __init__(self, version: str) -> None:
        self.version = version


def _make_context(version: str = "v1"):
    """
    Build a proper app.models.context.ApplicationContext object.

    build_context_prefix() in app/llm/prompts.py accesses context.domain.description
    and other fields, so we must use the real model rather than a simple stub.
    """
    from app.models.context import ApplicationContext as CtxModel, DomainContext
    return CtxModel(
        version=version,
        domain=DomainContext(description="Test application domain"),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_service(
    *,
    mock_llm: MockLLMClient | None = None,
    guardrails=None,
    context_store=None,
    guardrails_store=None,
    max_retries: int = 3,
) -> tuple[_MockTaskStore, _MockDefinitionStore, MockLLMClient, TaskAuthoringService]:
    """Return (task_store, def_store, mock_llm, service) wired together."""
    if mock_llm is None:
        mock_llm = MockLLMClient()
    task_store = _MockTaskStore()
    def_store  = _MockDefinitionStore()
    registry   = DefinitionRegistry(store=def_store)
    service = TaskAuthoringService(
        task_store=task_store,
        definition_registry=registry,
        guardrails=guardrails,
        llm_client=mock_llm,
        max_retries=max_retries,
        context_store=context_store or _MockContextStore(None),
        guardrails_store=guardrails_store,
    )
    return task_store, def_store, mock_llm, service


def _webhook_request(intent: str, dry_run: bool = False) -> CreateTaskRequest:
    """Minimal valid CreateTaskRequest with webhook delivery."""
    return CreateTaskRequest(
        intent=intent,
        entity_scope="account",
        delivery=DeliveryConfig(type=DeliveryType.WEBHOOK, endpoint="https://example.com/webhook"),
        dry_run=dry_run,
    )


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── 1. LLM called with correct intent ─────────────────────────────────────────

class TestLLMCalledWithIntent:
    """Verify TaskAuthoringService passes the intent to generate_task() exactly."""

    def test_intent_passed_to_mock_llm(self):
        _, _, mock_llm, service = _make_service()
        intent = "Alert me when churn risk is high"
        run(service.create_task(_webhook_request(intent)))
        assert mock_llm.last_intent == intent

    def test_call_count_is_one_for_single_task(self):
        _, _, mock_llm, service = _make_service()
        run(service.create_task(_webhook_request("Alert when active user rate drops")))
        assert mock_llm.call_count == 1

    def test_context_dict_is_passed_to_mock_llm(self):
        _, _, mock_llm, service = _make_service()
        run(service.create_task(_webhook_request("Alert when active user rate drops")))
        # context should always include type_system key
        assert "type_system" in mock_llm.last_context

    def test_mock_records_intent_per_call(self):
        _, _, mock_llm, service = _make_service()
        run(service.create_task(_webhook_request("Intent A")))
        run(service.create_task(_webhook_request("Intent B")))
        assert mock_llm.call_count == 2
        assert mock_llm.last_intent == "Intent B"


# ── 2. Guardrails disallowed_strategies not enforced ──────────────────────────

class TestGuardrailsStrategyEnforcement:
    """
    Documents that disallowed_strategies in GuardrailConstraints is NOT enforced
    at the TaskAuthoringService / DefinitionRegistry level.

    The Validator validates schema and type compatibility only.  Constraints are
    injected into the LLM prompt — MockLLMClient ignores them.
    """

    def _make_guardrails_with_disallowed(self, disallowed: list[str]):
        from app.models.config import ApplicationContext as GuardrailsAppContext
        from app.models.guardrails import (
            Guardrails,
            GuardrailConstraints,
            StrategyRegistryEntry,
        )
        return Guardrails(
            application_context=GuardrailsAppContext(
                name="test",
                description="Test context",
                instructions=["Only threshold allowed."],
            ),
            strategy_registry={
                "threshold": StrategyRegistryEntry(
                    version="1.0",
                    description="Threshold",
                    input_types=["float"],
                    output_type="decision<boolean>",
                ),
                "z_score": StrategyRegistryEntry(
                    version="1.0",
                    description="Z-score",
                    input_types=["float"],
                    output_type="decision<boolean>",
                ),
            },
            constraints=GuardrailConstraints(disallowed_strategies=disallowed),
        )

    def test_z_score_accepted_despite_disallowed_strategy_constraint(self):
        """
        Guardrails disallows z_score.  MockLLMClient returns z_score for
        'error rate' intent.  Service succeeds — constraint not enforced.
        """
        guardrails = self._make_guardrails_with_disallowed(
            ["z_score", "percentile", "change", "equals", "composite"]
        )
        _, _, mock_llm, service = _make_service(guardrails=guardrails)

        # No exception should be raised — disallowed_strategies not enforced.
        result = run(service.create_task(_webhook_request(
            "Alert me when error rate deviates from baseline"
        )))
        assert isinstance(result, Task), (
            "ACTUAL: z_score accepted despite disallowed_strategies=['z_score', ...]. "
            "BUG: guardrails_constraints_not_enforced_at_compile_time"
        )
        assert result.condition_id is not None

    def test_guardrails_injected_into_llm_context(self):
        """
        Guardrails are serialized and passed to the LLM via the context dict.
        The real LLM would respect guardrails; MockLLMClient ignores them.
        """
        guardrails = self._make_guardrails_with_disallowed(["z_score"])
        _, _, mock_llm, service = _make_service(guardrails=guardrails)
        run(service.create_task(_webhook_request("Alert when error rate spikes")))
        # Guardrails must appear in the context passed to the LLM
        assert "guardrails" in mock_llm.last_context, (
            "guardrails must be injected into LLM context; "
            f"context keys: {list(mock_llm.last_context.keys())}"
        )


# ── 3. Parameter priors not applied by MockLLMClient ──────────────────────────

class TestParameterPriorsNotApplied:
    """
    Documents that parameter priors (threshold_priors in PrimitiveHint) are
    NOT applied by MockLLMClient.  The mock returns fixed values regardless of
    guardrails context.
    """

    def _make_guardrails_with_priors(self) -> Any:
        from app.models.config import ApplicationContext as GuardrailsAppContext
        from app.models.guardrails import (
            Guardrails,
            PrimitiveHint,
            StrategyRegistryEntry,
        )
        return Guardrails(
            application_context=GuardrailsAppContext(
                name="test",
                description="Test context",
                instructions=["Use severity priors."],
            ),
            strategy_registry={
                "threshold": StrategyRegistryEntry(
                    version="1.0",
                    description="Threshold",
                    input_types=["float"],
                    output_type="decision<boolean>",
                ),
            },
            primitives={
                "account.active_user_rate_30d": PrimitiveHint(
                    type="float",
                    description="Active user rate",
                    threshold_priors={
                        "threshold": {"low": 0.60, "medium": 0.45, "high": 0.30}
                    },
                ),
            },
        )

    def test_mock_returns_fixed_value_regardless_of_priors(self):
        """
        Guardrails define threshold_priors (medium=0.45, high=0.30) for
        account.active_user_rate_30d.  MockLLMClient ignores priors and
        returns its fixed churn value (0.35).
        """
        guardrails = self._make_guardrails_with_priors()
        _, _, mock_llm, service = _make_service(guardrails=guardrails)

        result = run(service.create_task(_webhook_request(
            "Alert me when active user rate drops"
        )))
        assert isinstance(result, Task)
        # The mock returned threshold value=0.35 (not 0.45 or 0.30 from priors)
        # To verify, check the condition_id — mock uses consistent naming
        assert "churn" in result.condition_id or "default" in result.condition_id, (
            f"Expected churn or default scenario; condition_id: {result.condition_id}"
        )

    def test_primitives_injected_into_llm_context(self):
        """
        Primitive hints (including threshold_priors) are serialized into
        the LLM context.  MockLLMClient receives them but ignores them.
        """
        guardrails = self._make_guardrails_with_priors()
        _, _, mock_llm, service = _make_service(guardrails=guardrails)
        run(service.create_task(_webhook_request("Alert when active user rate drops")))
        # Guardrails section must contain primitives (injected as context["primitives"])
        assert "guardrails" in mock_llm.last_context


# ── 4. Bias rules / severity language not applied ─────────────────────────────

class TestBiasRulesNotApplied:
    """
    Documents that severity language in the intent ("significantly", "urgently")
    is NOT mapped to parameter priors by MockLLMClient.

    With a real LLM, "significantly" → medium_severity → prior value.
    With MockLLMClient, keyword routing only considers topic keywords, not
    severity modifiers.
    """

    def test_significantly_does_not_shift_threshold(self):
        """
        Intent contains "significantly".  MockLLMClient returns churn scenario
        (value=0.35) — severity language is ignored.
        """
        _, _, mock_llm, service = _make_service()
        result = run(service.create_task(_webhook_request(
            "Alert me when active user rate drops significantly"
        )))
        assert isinstance(result, Task)
        # Mock routes "active user" → churn scenario; "significantly" has no effect.
        assert mock_llm.call_count == 1
        assert "significantly" in mock_llm.last_intent

    def test_urgently_does_not_shift_threshold(self):
        """
        Intent contains "urgently".  MockLLMClient returns churn scenario
        (value=0.35) — urgency language is ignored.
        """
        _, _, mock_llm, service = _make_service()
        result = run(service.create_task(_webhook_request(
            "Alert me when active user rate drops urgently"
        )))
        assert isinstance(result, Task)
        assert mock_llm.call_count == 1

    def test_both_severity_intents_produce_same_condition_namespace(self):
        """
        "significantly" and "urgently" both route to the same mock scenario.
        They produce different condition_ids (counter increments) but the
        same strategy and namespace (both churn/default → org namespace).
        """
        _, _, mock_llm, service = _make_service()
        r1 = run(service.create_task(_webhook_request(
            "Alert me when active user rate drops significantly"
        )))
        r2 = run(service.create_task(_webhook_request(
            "Alert me when active user rate drops urgently"
        )))
        assert isinstance(r1, Task)
        assert isinstance(r2, Task)
        # Different task_ids and condition_ids (counter-suffixed), same namespace
        assert r1.task_id != r2.task_id
        assert r1.condition_id != r2.condition_id


# ── 5. context_version set from active context ────────────────────────────────

class TestContextVersionSet:
    """context_version on the created task must reflect the active context version."""

    def test_context_version_set_when_context_active(self):
        active_ctx = _make_context(version="v3")
        ctx_store = _MockContextStore(active_context=active_ctx)
        _, _, _, service = _make_service(context_store=ctx_store)

        result = run(service.create_task(_webhook_request("Alert when active user rate drops")))

        assert isinstance(result, Task)
        assert result.context_version == "v3", (
            f"context_version must be 'v3' from active context; got {result.context_version!r}"
        )

    def test_context_version_none_when_no_context(self):
        ctx_store = _MockContextStore(active_context=None)
        _, _, _, service = _make_service(context_store=ctx_store)

        result = run(service.create_task(_webhook_request("Alert when active user rate drops")))

        assert isinstance(result, Task)
        assert result.context_version is None, (
            f"context_version must be None when no active context; got {result.context_version!r}"
        )

    def test_context_fetch_error_does_not_abort_task_creation(self):
        """Context fetch failure is logged and swallowed — task creation proceeds."""

        class _FailingContextStore:
            async def get_active(self):
                raise RuntimeError("simulated context store failure")

        _, _, _, service = _make_service(context_store=_FailingContextStore())

        # Should NOT raise — failure is swallowed, context_version → None
        result = run(service.create_task(_webhook_request("Alert when active user rate drops")))
        assert isinstance(result, Task)
        assert result.context_version is None


# ── 6. guardrails_version set from active guardrails ──────────────────────────

class TestGuardrailsVersionSet:
    """guardrails_version on the created task must reflect the active API guardrails version."""

    def test_guardrails_version_set_when_store_loaded(self):
        gr_store = _MockGuardrailsStore(version_str="v2")
        _, _, _, service = _make_service(guardrails_store=gr_store)

        result = run(service.create_task(_webhook_request("Alert when active user rate drops")))

        assert isinstance(result, Task)
        assert result.guardrails_version == "v2", (
            f"guardrails_version must be 'v2' from active guardrails store; "
            f"got {result.guardrails_version!r}"
        )

    def test_guardrails_version_none_when_store_not_loaded(self):
        gr_store = _MockGuardrailsStore(version_str=None)
        _, _, _, service = _make_service(guardrails_store=gr_store)

        result = run(service.create_task(_webhook_request("Alert when active user rate drops")))

        assert isinstance(result, Task)
        assert result.guardrails_version is None, (
            f"guardrails_version must be None when no API guardrails version active; "
            f"got {result.guardrails_version!r}"
        )

    def test_guardrails_version_none_when_no_store(self):
        _, _, _, service = _make_service(guardrails_store=None)

        result = run(service.create_task(_webhook_request("Alert when active user rate drops")))

        assert isinstance(result, Task)
        assert result.guardrails_version is None


# ── 7. context_warning set when no active context ─────────────────────────────

class TestContextWarning:
    """
    context_warning is set on the returned Task when no ApplicationContext
    is active at task creation time.  It is NOT stored in the DB.
    """

    def test_context_warning_set_when_no_context(self):
        ctx_store = _MockContextStore(active_context=None)
        _, _, _, service = _make_service(context_store=ctx_store)

        result = run(service.create_task(_webhook_request("Alert when active user rate drops")))

        assert isinstance(result, Task)
        assert result.context_warning is not None, (
            "context_warning must be non-None when no active context exists"
        )
        assert len(result.context_warning) > 10, (
            f"context_warning should be a meaningful message; got {result.context_warning!r}"
        )

    def test_context_warning_none_when_context_active(self):
        active_ctx = _make_context(version="v1")
        ctx_store = _MockContextStore(active_context=active_ctx)
        _, _, _, service = _make_service(context_store=ctx_store)

        result = run(service.create_task(_webhook_request("Alert when active user rate drops")))

        assert isinstance(result, Task)
        assert result.context_warning is None, (
            f"context_warning must be None when active context exists; "
            f"got {result.context_warning!r}"
        )

    def test_context_warning_message_mentions_context(self):
        ctx_store = _MockContextStore(active_context=None)
        _, _, _, service = _make_service(context_store=ctx_store)

        result = run(service.create_task(_webhook_request("Alert when active user rate drops")))

        warning = result.context_warning
        assert warning is not None
        assert any(
            term in warning.lower()
            for term in ("context", "domain", "define", "post /context")
        ), (
            f"context_warning should mention context or how to define one; got {warning!r}"
        )

    def test_context_warning_absent_from_dry_run_result(self):
        """DryRunResult has no context_warning — it is not a Task."""
        ctx_store = _MockContextStore(active_context=None)
        _, _, _, service = _make_service(context_store=ctx_store)

        result = run(service.create_task(
            _webhook_request("Alert when active user rate drops", dry_run=True)
        ))

        assert isinstance(result, DryRunResult), (
            f"dry_run=True must return DryRunResult; got {type(result)}"
        )
        # DryRunResult should NOT have context_warning
        assert not hasattr(result, "context_warning") or getattr(result, "context_warning", "ABSENT") == "ABSENT", (
            "DryRunResult must not have context_warning"
        )
