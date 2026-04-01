"""
tests/unit/test_regression_suite.py
──────────────────────────────────────────────────────────────────────────────
Regression tests for all bugs fixed across rounds 1–5.

For each bug, this module either:
  (a) documents the existing test that provides adequate coverage, OR
  (b) provides a new test_regression_<description> that FAILS on the
      original broken code and PASSES on the fixed code.

Coverage map
────────────
Bug 01 — Composite condition always returned False
    COVERED → test_strategy_composite.py::TestCompositeAND::test_and_both_true_fires
              Asserts r.value is True; fails if composite always returns False.

Bug 02 — z_score/percentile/change returned False (history was empty list)
    COVERED → test_execute_service.py::test_z_score_with_30_history_rows_fires
              test_execute_service.py::test_percentile_with_30_history_rows_fires
              test_execute_service.py::test_change_with_3_history_rows_fires
              Each provides non-empty history and asserts value is True.

Bug 03 — Action delivery was no-op
    COVERED → test_action_service.py::test_trigger_fire_on_any_triggers
              test_action_trigger.py (webhook/notification/workflow delivery tests)

Bug 04 — ExplanationService(pool=pool) wrong constructor
    COVERED → test_explanation.py::test_explain_condition_service_wiring_from_conditions_route
              Calls the factory; would raise TypeError with wrong constructor.

Bug 05 — explain_condition() method did not exist
    COVERED → test_explanation.py::test_explain_condition_threshold_returns_condition_explanation
              Calls svc.explain_condition(); AttributeError if absent.

Bug 06 — _fetch_bound_actions did full table scan
    COVERED → test_execute_service.py::test_fetch_bound_actions_uses_parameterized_query
              Inspects SQL; fails if condition_id/version filter is missing.

Bug 07 — list_features returned body: {}
    COVERED → test_registry.py::TestListFeaturesReturnsBody::test_list_features_returns_body_from_db
              Asserts item["body"] != {}.

Bug 08 — executor.execute() missing await on get_by_concept()
    COVERED → test_fixes_round2.py::test_executor_execute_awaits_graph_store
              AsyncMock confirms await was used; AttributeError on coroutine if not.

Bug 09 — ConditionEvaluator had no async aevaluate()
    COVERED → test_fixes_round2.py::test_aevaluate_returns_decision_value_from_cache
              Calls aevaluate(); AttributeError if absent.

Bug 10 — explanation.py snapshot-mode crash (sync evaluate with timestamp=None)
    COVERED → test_fixes_round2.py::test_explain_decision_snapshot_mode_does_not_crash

Bug 11 — execute_static blocked event loop (sync execute_graph)
    COVERED → test_fixes_round2.py::test_execute_static_uses_aexecute_graph

Bug 12 — decisions.py ExplanationService wired with MockConnector
    COVERED → test_fixes_round2.py::test_get_explanation_service_wires_connector_registry

Bug 13 — calibration apply_calibration field name
    NOT A BUG — verified correct during audit; no test needed.

Bug 14 — DecisionRecord.concept_value: float only (dropped bool/str)
    COVERED → test_fixes_round2.py::test_decision_record_accepts_bool_concept_value
              test_fixes_round2.py::test_decision_record_accepts_str_concept_value

Bug 15 — float(row["value"]) crashed on non-numeric history
    COVERED → test_fixes_round2.py::test_fetch_history_filters_null_values
              test_fixes_round2.py::test_fetch_history_query_excludes_null_rows

Bug 16 — data_resolver time.sleep() in async context
    COVERED → test_fixes_round4.py::TestAsyncWithRetry::test_async_with_retry_uses_asyncio_sleep_not_time_sleep

Bug 17 — VALID_DEFINITION_TYPES missing "feature"
    PARTIALLY COVERED → test_fixes_round4.py::TestRegisterDefinitionRequestValidation::test_valid_feature_accepted
                        (covers the Pydantic Literal on the request model)
    NEW → test_regression_valid_definition_types_contains_feature
          (covers the store-level frozenset constant directly)

Bug 18 — guardrails_version never written/read in task store
    COVERED → test_fix3_guardrails_version.py (comprehensive INSERT/SELECT verification)

Bug 19 — feature type violated DB CHECK constraint
    NEW → test_regression_migration_0005_adds_feature_to_constraint
          (verifies the alembic migration upgrade SQL includes 'feature')

Bug 20 — jobs.py result/error never populated from result_body
    COVERED → test_fix5_job_result.py (comprehensive)

Bug 21 — actions.py total=len(actions) not DB count
    COVERED → test_actions_list.py::test_list_actions_total_reflects_db_count_not_page_count

Bug 22 — explain_mode param passed to executor (doesn't exist)
    COVERED → test_explanation.py::test_explain_decision_does_not_pass_explain_mode_to_executor

Bug 23 — result.py ConceptResult.value bool union ordering (True → 1.0)
    COVERED → test_fixes_round3.py::TestConceptResultValueType::test_bool_not_coerced_to_float

Bug 24 — calibration.py timezone replace() vs astimezone()
    COVERED → test_fixes_round3.py::TestCalibrationTokenIsExpired

Bug 25 — errors.py all errors logged at WARNING including 404/409
    COVERED → test_observability.py (asserts NOT_FOUND logs at debug, not warning)

Bug 26 — actions.py asyncio.ensure_future() → asyncio.create_task()
    NEW → test_regression_actions_uses_create_task_not_ensure_future

Bug 27 — NodeTrace.output_value bool union ordering (True → 1.0)
    COVERED → test_fixes_round5.py::TestNodeTraceOutputValueType::test_bool_not_coerced_to_float

Bug 28 — ConceptExplanation.output bool union ordering (True → 1.0)
    COVERED → test_fixes_round5.py::TestConceptExplanationOutputType::test_bool_not_coerced_to_float

Bug 29 — request models missing max_length on string fields
    COVERED → test_fixes_round5.py::TestExecuteRequestMaxLength

Bug 30 — stdlib logging in stores/services/registry (not structlog)
    NEW → test_regression_stores_use_structlog_not_stdlib_logging
          test_regression_services_use_structlog_not_stdlib_logging
          test_regression_registry_modules_use_structlog_not_stdlib_logging

Bug 31 — DecisionRecord.concept_value bool union ordering (True → 1.0)
    COVERED → test_fixes_round4.py::TestDecisionRecordConceptValueType::test_bool_true_preserved_as_bool
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os

import pytest


# ── Bug 17: VALID_DEFINITION_TYPES contains 'feature' ─────────────────────────

def test_regression_valid_definition_types_contains_feature():
    """
    Bug 17: VALID_DEFINITION_TYPES in app/stores/definition.py did not include
    'feature', so DefinitionStore.register() raised SEMANTIC_ERROR when
    RegistryService.register_feature() tried to store a feature record.

    The Pydantic Literal on RegisterDefinitionRequest was also fixed (covered
    separately in test_fixes_round4.py), but this test guards the store-level
    constant so both layers stay in sync.

    Fails if 'feature' is removed from VALID_DEFINITION_TYPES.
    Passes with the current frozenset that includes 'feature'.
    """
    from app.stores.definition import VALID_DEFINITION_TYPES

    assert "feature" in VALID_DEFINITION_TYPES, (
        "VALID_DEFINITION_TYPES must include 'feature' — "
        "required for RegistryService.register_feature() to persist feature records "
        "without a SEMANTIC_ERROR"
    )
    # Verify the complete allowed set so no future type is accidentally dropped.
    expected = {"concept", "condition", "action", "primitive", "feature"}
    assert expected <= VALID_DEFINITION_TYPES, (
        f"VALID_DEFINITION_TYPES is missing expected types. "
        f"Expected at least {expected}, got {VALID_DEFINITION_TYPES}"
    )


# ── Bug 19: Migration 0005 adds 'feature' to DB CHECK constraint ───────────────

def test_regression_migration_0005_adds_feature_to_constraint():
    """
    Bug 19: The definitions table CHECK constraint in the initial schema only
    allowed definition_type IN ('concept','condition','action','primitive').
    When RegistryService.register_feature() issued INSERT ... definition_type='feature'
    the DB rejected it with asyncpg.CheckViolationError.

    Migration 0005 re-created the CHECK constraint to include 'feature'.

    This test reads the migration source and verifies the upgrade() SQL contains
    'feature' in the allowed values list.

    Fails if migration 0005 is reverted or 'feature' is removed from the SQL.
    Passes with the current migration that includes 'feature'.
    """
    migration_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../../alembic/versions/0005_add_feature_to_definition_type_check.py",
        )
    )
    assert os.path.isfile(migration_path), (
        f"Migration 0005 not found at expected path: {migration_path}"
    )

    with open(migration_path, encoding="utf-8") as f:
        content = f.read()

    assert "'feature'" in content, (
        "Migration 0005 must include 'feature' in the CHECK constraint upgrade SQL"
    )
    # Also confirm the migration has the correct revision chain.
    assert "revision = \"0005\"" in content or "revision = '0005'" in content, (
        "Migration 0005 must declare revision = '0005'"
    )
    assert "down_revision = \"0004\"" in content or "down_revision = '0004'" in content, (
        "Migration 0005 must chain from revision 0004"
    )


# ── Bug 26: actions.py ensure_future() → create_task() ────────────────────────

def test_regression_actions_uses_create_task_not_ensure_future():
    """
    Bug 26: _list_actions_with_total() in app/api/routes/actions.py used
    asyncio.ensure_future() to schedule concurrent page and count queries.
    asyncio.ensure_future() is deprecated in Python 3.10+ and emits
    DeprecationWarning; asyncio.create_task() is the correct Python 3.7+ API
    for scheduling coroutines within a running event loop.

    Fails if ensure_future() is reintroduced in _list_actions_with_total().
    Passes with the current implementation that uses create_task().
    """
    from app.api.routes import actions as actions_module

    source = inspect.getsource(actions_module._list_actions_with_total)

    assert "asyncio.create_task(" in source, (
        "_list_actions_with_total() must use asyncio.create_task() "
        "to schedule concurrent store queries"
    )
    assert "asyncio.ensure_future(" not in source, (
        "_list_actions_with_total() must not use asyncio.ensure_future() "
        "(deprecated in Python 3.10+; use create_task() for coroutines)"
    )


# ── Bug 30: stdlib logging → structlog in stores ──────────────────────────────

_STORE_MODULES = [
    "app.stores.definition",
    "app.stores.task",
    "app.stores.graph",
    "app.stores.job",
    "app.stores.decision",
    "app.stores.feedback",
    "app.stores.calibration_token",
    "app.stores.concept_result",
]

_SERVICE_MODULES = [
    "app.services.explanation",
    "app.services.registry",
]

_REGISTRY_MODULES = [
    "app.registry.definitions",
    "app.registry.features",
]


@pytest.mark.parametrize("module_name", _STORE_MODULES)
def test_regression_stores_use_structlog_not_stdlib_logging(module_name: str):
    """
    Bug 30: All eight store modules used stdlib logging.getLogger() which
    does not propagate structlog's bound context variables (request IDs,
    entity IDs, trace IDs). Losing this context makes distributed tracing
    impossible across the store layer.

    Fixed by replacing:
        import logging
        log = logging.getLogger(__name__)
    with:
        import structlog
        log = structlog.get_logger(__name__)

    Parametrised across all 8 store modules so a regression in any one of them
    is reported individually.

    Fails if a store is reverted to logging.getLogger().
    Passes with the current structlog-based implementation.
    """
    mod = importlib.import_module(module_name)
    source = inspect.getsource(mod)

    assert "structlog.get_logger" in source, (
        f"{module_name} must use structlog.get_logger() — "
        "stdlib logging does not propagate request-bound context"
    )
    assert "logging.getLogger" not in source, (
        f"{module_name} must not use stdlib logging.getLogger() — "
        "switch to structlog.get_logger(__name__)"
    )


@pytest.mark.parametrize("module_name", _SERVICE_MODULES)
def test_regression_services_use_structlog_not_stdlib_logging(module_name: str):
    """
    Bug 30: services/explanation.py and services/registry.py used stdlib
    logging.getLogger(), breaking structlog context propagation in service
    layer calls.

    Parametrised across both affected service modules.

    Fails if either service is reverted to logging.getLogger().
    Passes with the current structlog-based implementation.
    """
    mod = importlib.import_module(module_name)
    source = inspect.getsource(mod)

    assert "structlog.get_logger" in source, (
        f"{module_name} must use structlog.get_logger()"
    )
    assert "logging.getLogger" not in source, (
        f"{module_name} must not use stdlib logging.getLogger()"
    )


@pytest.mark.parametrize("module_name", _REGISTRY_MODULES)
def test_regression_registry_modules_use_structlog_not_stdlib_logging(module_name: str):
    """
    Bug 30: registry/definitions.py and registry/features.py used stdlib
    logging.getLogger(), breaking structlog context propagation in the
    registry layer.

    Parametrised across both affected registry modules.

    Fails if either module is reverted to logging.getLogger().
    Passes with the current structlog-based implementation.
    """
    mod = importlib.import_module(module_name)
    source = inspect.getsource(mod)

    assert "structlog.get_logger" in source, (
        f"{module_name} must use structlog.get_logger()"
    )
    assert "logging.getLogger" not in source, (
        f"{module_name} must not use stdlib logging.getLogger()"
    )
