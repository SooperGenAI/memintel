"""
tests/unit/test_context_integration.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for Session B: LLM prompt injection and calibration bias adjustment.

Coverage
────────
  build_context_prefix():
    1. Returns empty string when context is None
    2. Includes domain description
    3. Includes all semantic hints
    4. Omits calibration section when calibration_bias is None

  TaskAuthoringService context integration:
    5. Task created with active context → context_version set on returned task
    6. Task created without active context → context_version is None
                                              AND context_warning is set

  CalibrationService bias adjustment:
    7. recall bias → context_adjusted < statistically_optimal
    8. precision bias → context_adjusted > statistically_optimal
    9. no context → recommended equals statistically_optimal, context_adjusted is None
   10. Calibration adjustment never produces a value outside [0.0, 1.0]
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

# Stub aioredis before any app import (module removed in Python 3.12+)
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

import pytest

from app.llm.prompts import build_context_prefix
from app.models.calibration import CalibrateRequest, CalibrationStatus
from app.models.context import (
    ApplicationContext,
    BehaviouralContext,
    CalibrationBias,
    DomainContext,
    EntityDeclaration,
    SemanticHint,
)
from app.models.task import CreateTaskRequest, DeliveryConfig, DeliveryType, Task
from app.services.calibration import CalibrationService
from app.services.task_authoring import TaskAuthoringService


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_domain(**kwargs) -> DomainContext:
    return DomainContext(description=kwargs.get("description", "Test domain"))


def _make_context(
    *,
    version: str = "v1",
    calibration_bias: CalibrationBias | None = None,
    semantic_hints: list[SemanticHint] | None = None,
    entities: list[EntityDeclaration] | None = None,
    decisions: list[str] | None = None,
    behavioural: BehaviouralContext | None = None,
) -> ApplicationContext:
    domain = DomainContext(
        description="Fraud detection platform",
        entities=entities or [],
        decisions=decisions or [],
    )
    return ApplicationContext(
        domain=domain,
        version=version,
        calibration_bias=calibration_bias,
        semantic_hints=semantic_hints or [],
        behavioural=behavioural or BehaviouralContext(),
    )


# ── CalibrationService mock dependencies ──────────────────────────────────────

_THRESHOLD_COND_BODY = {
    "condition_id": "test.cond",
    "version": "1.0",
    "concept_id": "test.concept",
    "concept_version": "1.0",
    "namespace": "personal",
    "strategy": {"type": "threshold", "params": {"direction": "above", "value": 0.80}},
}


class _MockRegistry:
    """Returns the threshold condition body for any (id, version)."""
    async def get(self, cid: str, version: str) -> dict:
        return dict(_THRESHOLD_COND_BODY)

    async def register(self, body, namespace=None, definition_type=None):
        return {}


class _MockFeedbackStore:
    async def get_by_condition(self, cid, version):
        return []


class _MockTokenStore:
    async def create(self, token_obj) -> str:
        return "mock-token-string"


class _MockTaskStoreCalibration:
    async def find_by_condition_version(self, cid, version):
        return []


class _MockGuardrailsStore:
    def get_threshold_bounds(self, strategy: str) -> dict:
        return {"min": 0.0, "max": 1.0}

    def get_guardrails(self):
        g = MagicMock()
        g.constraints.on_bounds_exceeded = "clamp"
        return g


def _make_calibration_service(context_store=None) -> CalibrationService:
    return CalibrationService(
        feedback_store=_MockFeedbackStore(),
        token_store=_MockTokenStore(),
        task_store=_MockTaskStoreCalibration(),
        definition_registry=_MockRegistry(),
        guardrails_store=_MockGuardrailsStore(),
        context_store=context_store,
    )


def _calibrate_sync(service: CalibrationService, req: CalibrateRequest):
    return asyncio.run(service.calibrate(req))


# ── TaskAuthoringService mock dependencies ─────────────────────────────────────

class _MockTaskStore:
    async def create(self, task: Task) -> Task:
        task.task_id = "mock-task-id"
        return task


class _MockDefinitionRegistry:
    async def register(self, body, namespace=None, definition_type=None):
        return {}


def _make_task_request() -> CreateTaskRequest:
    # "churn" routes to threshold fixture in LLMFixtureClient
    return CreateTaskRequest(
        intent="alert me when churn risk is above threshold",
        entity_scope="user:all",
        delivery=DeliveryConfig(type=DeliveryType.NOTIFICATION, channel="slack"),
    )


# ── Tests 1–4: build_context_prefix ───────────────────────────────────────────

class TestBuildContextPrefix:

    def test_none_returns_empty_string(self) -> None:
        """build_context_prefix(None) must return empty string."""
        result = build_context_prefix(None)
        assert result == ""

    def test_includes_domain_description(self) -> None:
        """Prefix includes the domain description."""
        ctx = _make_context()
        prefix = build_context_prefix(ctx)
        assert "Fraud detection platform" in prefix
        assert "=== APPLICATION CONTEXT ===" in prefix
        assert "=== END APPLICATION CONTEXT ===" in prefix

    def test_includes_all_semantic_hints(self) -> None:
        """All semantic hints appear in the prefix, one per line."""
        hints = [
            SemanticHint(term="churn", definition="Customer stops using the product"),
            SemanticHint(term="LTV", definition="Lifetime value of a customer"),
        ]
        ctx = _make_context(semantic_hints=hints)
        prefix = build_context_prefix(ctx)
        assert "- churn: Customer stops using the product" in prefix
        assert "- LTV: Lifetime value of a customer" in prefix

    def test_omits_calibration_section_when_no_bias(self) -> None:
        """Calibration sensitivity section is omitted when calibration_bias is None."""
        ctx = _make_context(calibration_bias=None)
        prefix = build_context_prefix(ctx)
        assert "Calibration sensitivity" not in prefix
        assert "False negative cost" not in prefix
        assert "Bias direction" not in prefix


# ── Tests 5–6: TaskAuthoringService context integration ───────────────────────

class TestTaskAuthoringContextIntegration:

    def test_context_version_set_when_context_exists(self) -> None:
        """Task created with active context → context_version matches context.version."""
        class _ContextStoreWithCtx:
            async def get_active(self):
                return _make_context(version="v3")

        service = TaskAuthoringService(
            task_store=_MockTaskStore(),
            definition_registry=_MockDefinitionRegistry(),
            context_store=_ContextStoreWithCtx(),
        )
        result = asyncio.run(service.create_task(_make_task_request()))

        assert isinstance(result, Task)
        assert result.context_version == "v3"
        assert result.context_warning is None

    def test_context_version_none_and_warning_set_when_no_context(self) -> None:
        """Task created without active context → context_version is None
        AND context_warning carries the informational message."""
        class _ContextStoreEmpty:
            async def get_active(self):
                return None

        service = TaskAuthoringService(
            task_store=_MockTaskStore(),
            definition_registry=_MockDefinitionRegistry(),
            context_store=_ContextStoreEmpty(),
        )
        result = asyncio.run(service.create_task(_make_task_request()))

        assert isinstance(result, Task)
        assert result.context_version is None
        assert result.context_warning is not None
        assert "POST /context" in result.context_warning


# ── Tests 7–10: CalibrationService bias adjustment ────────────────────────────

class TestCalibrationBiasAdjustment:
    """
    All tests use explicit feedback_direction to skip FeedbackStore aggregation
    and focus only on the bias adjustment step.

    Starting condition: threshold strategy, value=0.80.
    With direction="relax": adjust_params decreases value:
      step = max(0.08, 0.1) = 0.1
      new_val = 0.80 - 0.1 = 0.70  → statistically_optimal = 0.70
    """

    def test_recall_bias_lowers_adjusted_threshold(self) -> None:
        """
        recall bias (false_negative_cost > false_positive_cost) must produce
        context_adjusted < statistically_optimal.
        """
        class _RecallContextStore:
            async def get_active(self):
                return _make_context(
                    calibration_bias=CalibrationBias(
                        false_negative_cost="high",
                        false_positive_cost="low",
                    )
                )

        service = _make_calibration_service(context_store=_RecallContextStore())
        req = CalibrateRequest(
            condition_id="test.cond",
            condition_version="1.0",
            feedback_direction="relax",
        )
        result = _calibrate_sync(service, req)

        assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
        assert result.statistically_optimal is not None
        assert result.context_adjusted is not None
        assert result.context_adjusted < result.statistically_optimal, (
            f"Expected context_adjusted ({result.context_adjusted}) < "
            f"statistically_optimal ({result.statistically_optimal})"
        )

    def test_precision_bias_raises_adjusted_threshold(self) -> None:
        """
        precision bias (false_positive_cost > false_negative_cost) must produce
        context_adjusted > statistically_optimal.
        """
        class _PrecisionContextStore:
            async def get_active(self):
                return _make_context(
                    calibration_bias=CalibrationBias(
                        false_negative_cost="low",
                        false_positive_cost="high",
                    )
                )

        service = _make_calibration_service(context_store=_PrecisionContextStore())
        req = CalibrateRequest(
            condition_id="test.cond",
            condition_version="1.0",
            feedback_direction="relax",
        )
        result = _calibrate_sync(service, req)

        assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
        assert result.statistically_optimal is not None
        assert result.context_adjusted is not None
        assert result.context_adjusted > result.statistically_optimal, (
            f"Expected context_adjusted ({result.context_adjusted}) > "
            f"statistically_optimal ({result.statistically_optimal})"
        )

    def test_no_context_recommended_equals_statistically_optimal(self) -> None:
        """
        When no context_store is provided, recommended_params carries the raw
        statistically_optimal value and context_adjusted is None.
        """
        service = _make_calibration_service(context_store=None)
        req = CalibrateRequest(
            condition_id="test.cond",
            condition_version="1.0",
            feedback_direction="relax",
        )
        result = _calibrate_sync(service, req)

        assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
        assert result.context_adjusted is None
        assert result.adjustment_explanation is None
        assert result.statistically_optimal is not None
        # recommended_params["value"] must equal statistically_optimal
        assert result.recommended_params is not None
        assert abs(result.recommended_params["value"] - result.statistically_optimal) < 1e-9

    def test_bias_adjustment_never_exceeds_unit_interval(self) -> None:
        """
        precision bias applied to a high threshold must be clamped to ≤ 1.0.

        Condition value = 0.84.
        direction = tighten → step = max(0.084, 0.1) = 0.1 → new_val = 0.94
        statistically_optimal = 0.94
        precision (high, factor=0.10) → 0.94 * 1.10 = 1.034 → clamped to 1.0
        """
        _TIGHTEN_COND_BODY = {
            "condition_id": "test.high",
            "version": "1.0",
            "concept_id": "test.concept",
            "concept_version": "1.0",
            "namespace": "personal",
            "strategy": {
                "type": "threshold",
                "params": {"direction": "above", "value": 0.84},
            },
        }

        class _HighRegistry:
            async def get(self, cid, version):
                return dict(_TIGHTEN_COND_BODY)

        class _PrecisionContextStore:
            async def get_active(self):
                return _make_context(
                    calibration_bias=CalibrationBias(
                        false_negative_cost="low",
                        false_positive_cost="high",
                    )
                )

        service = CalibrationService(
            feedback_store=_MockFeedbackStore(),
            token_store=_MockTokenStore(),
            task_store=_MockTaskStoreCalibration(),
            definition_registry=_HighRegistry(),
            guardrails_store=_MockGuardrailsStore(),
            context_store=_PrecisionContextStore(),
        )
        req = CalibrateRequest(
            condition_id="test.high",
            condition_version="1.0",
            feedback_direction="tighten",
        )
        result = asyncio.run(service.calibrate(req))

        assert result.status == CalibrationStatus.RECOMMENDATION_AVAILABLE
        assert result.context_adjusted is not None
        assert result.context_adjusted <= 1.0, (
            f"context_adjusted ({result.context_adjusted}) exceeds 1.0"
        )
        assert result.recommended_params is not None
        assert result.recommended_params["value"] <= 1.0, (
            f"recommended_params['value'] ({result.recommended_params['value']}) exceeds 1.0"
        )
