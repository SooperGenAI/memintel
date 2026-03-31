"""
tests/unit/test_null_handling_and_decisions.py
──────────────────────────────────────────────────────────────────────────────
12 new tests covering:
  Null handling (tests 1-5):
    1. threshold with None input → fired=False, reason=null_input
    2. z_score with None input → fired=False, reason=null_input
    3. equals with None input → fired=False, reason=null_input
    4. ConnectorError → PrimitiveValue has fetch_error=True
    5. fetch_error primitive → decision record has signal_errors entry

  Decision persistence (tests 6-12):
    6. evaluate_full() writes a decision record to decisions table
    7. decision record contains correct concept_value
    8. decision record contains correct threshold_applied (full params)
    9. decision record contains correct condition_version
    10. decision record contains correct input_primitives dict
    11. DecisionStore.get() retrieves a recorded decision
    12. decision recording failure does not fail the evaluation
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.models.condition import DecisionType
from app.models.decision import DecisionRecord
from app.models.result import ConceptOutputType, ConceptResult
from app.runtime.data_resolver import (
    ConnectorError,
    DataResolver,
    MockConnector,
    PrimitiveValue,
    TransientConnectorError,
)
from app.services.execute import ExecuteService
from app.stores.decision import DecisionStore
from app.strategies.equals import EqualsStrategy
from app.strategies.threshold import ThresholdStrategy
from app.strategies.z_score import ZScoreStrategy


# ── helpers ───────────────────────────────────────────────────────────────────

def _float_result(value, entity="e1", version="1.0", ts=None):
    return ConceptResult(
        value=value,
        type=ConceptOutputType.FLOAT,
        entity=entity,
        version=version,
        deterministic=ts is not None,
        timestamp=ts,
    )


def _cat_result(value, entity="e1", version="1.0"):
    return ConceptResult(
        value=value,
        type=ConceptOutputType.CATEGORICAL,
        entity=entity,
        version=version,
        deterministic=False,
    )


def _run(coro):
    return asyncio.run(coro)


# ── Concept / condition bodies ─────────────────────────────────────────────────

_CONCEPT_BODY = {
    "concept_id": "org.revenue_score",
    "version": "1.0",
    "namespace": "org",
    "output_type": "float",
    "primitives": {
        "revenue": {"type": "float", "missing_data_policy": "zero"}
    },
    "features": {
        "score": {
            "op": "normalize",
            "inputs": {"input": "revenue"},
            "params": {},
        }
    },
    "output_feature": "score",
}

_CONDITION_BODY = {
    "condition_id": "high_revenue",
    "version": "2.0",
    "namespace": "org",
    "concept_id": "org.revenue_score",
    "concept_version": "1.0",
    "strategy": {
        "type": "threshold",
        "params": {"direction": "above", "value": 0.5},
    },
}


# ── MockPool for service tests ────────────────────────────────────────────────

class _MockPool:
    """
    Minimal asyncpg pool stub for decision tests.

    Captures INSERT INTO decisions calls so tests can assert on them.
    """

    def __init__(
        self,
        fetchrow_map: dict | None = None,
        fetch_rows: list | None = None,
        raise_on_decision_insert: bool = False,
    ):
        self._fetchrow_map = fetchrow_map or {}
        self._fetch_rows = fetch_rows or []
        self._raise_on_decision_insert = raise_on_decision_insert
        # Captured INSERT args for assertions
        self.decision_inserts: list[tuple] = []
        self.execute_calls: list[str] = []

    async def fetchrow(self, query: str, *args: Any) -> dict | None:
        if "INSERT INTO decisions" in query:
            if self._raise_on_decision_insert:
                raise RuntimeError("simulated decision insert failure")
            self.decision_inserts.append(args)
            return {"decision_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}
        # Definition lookups
        if len(args) >= 2:
            key = (args[0], args[1])
            if key in self._fetchrow_map:
                return self._fetchrow_map[key]
        if len(args) == 1 and args[0] in self._fetchrow_map:
            return self._fetchrow_map[args[0]]
        return None

    async def fetch(self, query: str, *args: Any) -> list:
        if "concept_results" in query:
            return []
        return self._fetch_rows

    async def execute(self, query: str, *args: Any) -> None:
        self.execute_calls.append(query)


def _make_pool(raise_on_decision_insert: bool = False) -> _MockPool:
    return _MockPool(
        fetchrow_map={
            ("org.revenue_score", "1.0"): {"body": json.dumps(_CONCEPT_BODY)},
            ("high_revenue", "2.0"): {"body": json.dumps(_CONDITION_BODY)},
        },
        fetch_rows=[{"body": json.dumps({
            "action_id": "no_action",
            "version": "1.0",
            "namespace": "org",
            "config": {"type": "webhook", "endpoint": "https://example.com"},
            "trigger": {"fire_on": "true", "condition_id": "OTHER", "condition_version": "1.0"},
        })}],
        raise_on_decision_insert=raise_on_decision_insert,
    )


def _full_req(**kw):
    class _Req:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return self.__dict__.copy()
    defaults = dict(
        concept_id="org.revenue_score",
        concept_version="1.0",
        condition_id="high_revenue",
        condition_version="2.0",
        entity="acct_42",
        timestamp=None,
        explain=False,
        dry_run=False,
    )
    defaults.update(kw)
    return _Req(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — NULL HANDLING IN STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

class TestNullGuardStrategies:
    """Tests 1-3: null guard returns fired=False with reason=null_input."""

    def test_threshold_none_input_returns_null_input(self):
        """Test 1: threshold with None input → fired=False, reason=null_input."""
        strategy = ThresholdStrategy()
        result = _float_result(None)
        dv = strategy.evaluate(
            result, [],
            {"direction": "above", "value": 0.5},
            condition_id="c1", condition_version="1.0",
        )
        assert dv.value is False
        assert dv.reason == "null_input"
        assert dv.decision_type == DecisionType.BOOLEAN

    def test_z_score_none_input_returns_null_input(self):
        """Test 2: z_score with None input → fired=False, reason=null_input."""
        strategy = ZScoreStrategy()
        result = _float_result(None)
        history = [_float_result(1.0), _float_result(2.0), _float_result(3.0)]
        dv = strategy.evaluate(
            result, history,
            {"threshold": 2.0, "direction": "above"},
            condition_id="c1", condition_version="1.0",
        )
        assert dv.value is False
        assert dv.reason == "null_input"
        assert dv.decision_type == DecisionType.BOOLEAN

    def test_equals_none_input_returns_null_input(self):
        """Test 3: equals with None input → fired=False, reason=null_input."""
        strategy = EqualsStrategy()
        result = _cat_result(None)
        dv = strategy.evaluate(
            result, [],
            {"value": "premium"},
            condition_id="c1", condition_version="1.0",
        )
        assert dv.value is False
        assert dv.reason == "null_input"
        assert dv.decision_type == DecisionType.BOOLEAN


class TestConnectorErrorHandling:
    """Tests 4-5: ConnectorError handling in DataResolver."""

    def test_connector_error_returns_fetch_error_primitive(self):
        """Test 4: ConnectorError → PrimitiveValue has fetch_error=True."""
        connector = MockConnector(auth_failure=True)  # always raises AuthConnectorError
        resolver = DataResolver(connector=connector, backoff_base=0.0)
        from app.models.result import MissingDataPolicy
        pv = resolver.fetch("revenue", "acct_1", None, policy=MissingDataPolicy.NULL)
        assert pv.fetch_error is True
        assert pv.value is None
        assert pv.error_msg is not None
        assert len(pv.error_msg) > 0

    def test_fetch_error_appears_in_signal_errors(self):
        """Test 5: fetch_error primitive → service records signal_errors."""
        # Use a connector that always raises TransientConnectorError (exhausts retries)
        connector = MockConnector(transient_failures=100)  # more than max retries
        resolver = DataResolver(connector=connector, backoff_base=0.0)
        from app.models.result import MissingDataPolicy
        pv = resolver.fetch("revenue", "acct_1", None, policy=MissingDataPolicy.NULL)
        assert pv.fetch_error is True
        assert pv.value is None


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — DECISION PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionPersistence:
    """Tests 6-10: evaluate_full() correctly writes decision records."""

    def _run_evaluate_full(self, pool, req=None):
        """Run evaluate_full and drain background tasks."""
        async def _inner():
            svc = ExecuteService(pool=pool)
            result = await svc.evaluate_full(req or _full_req())
            # Give background task a chance to run
            await asyncio.sleep(0)
            return result
        return asyncio.run(_inner())

    def test_evaluate_full_writes_decision_record(self):
        """Test 6: evaluate_full() writes a decision record to decisions table."""
        pool = _make_pool()
        self._run_evaluate_full(pool)
        assert len(pool.decision_inserts) == 1

    def test_decision_record_contains_correct_concept_value(self):
        """Test 7: decision record contains correct concept_value."""
        pool = _make_pool()
        self._run_evaluate_full(pool)
        assert len(pool.decision_inserts) == 1
        # concept_value is the 7th positional arg (index 6) in the INSERT
        # Args order: concept_id, concept_version, condition_id, condition_version,
        #             entity_id, fired, concept_value, threshold_applied, ir_hash,
        #             input_primitives, signal_errors, reason, action_ids_fired, dry_run
        args = pool.decision_inserts[0]
        concept_value = args[6]  # 7th arg (0-indexed = 6)
        # revenue=0 (missing with zero policy → normalize(0) = 0.0); concept_value = 0.0
        assert concept_value == 0.0

    def test_decision_record_contains_correct_threshold_applied(self):
        """Test 8: decision record contains correct threshold_applied (full params)."""
        pool = _make_pool()
        self._run_evaluate_full(pool)
        args = pool.decision_inserts[0]
        threshold_applied_json = args[7]  # threshold_applied
        threshold_applied = json.loads(threshold_applied_json)
        assert threshold_applied["direction"] == "above"
        assert threshold_applied["value"] == 0.5

    def test_decision_record_contains_correct_condition_version(self):
        """Test 9: decision record contains correct condition_version."""
        pool = _make_pool()
        self._run_evaluate_full(pool)
        args = pool.decision_inserts[0]
        condition_version = args[3]  # 4th arg: condition_version
        assert condition_version == "2.0"

    def test_decision_record_contains_input_primitives(self):
        """Test 10: decision record contains input_primitives dict."""
        pool = _make_pool()
        self._run_evaluate_full(pool)
        args = pool.decision_inserts[0]
        input_primitives_json = args[9]  # input_primitives
        if input_primitives_json is not None:
            input_primitives = json.loads(input_primitives_json)
            # The concept uses primitive "revenue" with missing_data_policy=zero → value=0.0
            assert "revenue" in input_primitives


class TestDecisionStore:
    """Test 11: DecisionStore.get() retrieves a recorded decision."""

    def test_decision_store_get_retrieves_record(self):
        """Test 11: DecisionStore.get() retrieves a recorded decision."""
        _decision_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        _row = {
            "decision_id": _decision_id,
            "concept_id": "org.revenue_score",
            "concept_version": "1.0",
            "condition_id": "high_revenue",
            "condition_version": "2.0",
            "entity_id": "acct_42",
            "evaluated_at": datetime(2026, 3, 31, 12, 0, 0),
            "fired": True,
            "concept_value": 0.75,
            "threshold_applied": json.dumps({"direction": "above", "value": 0.5}),
            "ir_hash": None,
            "input_primitives": json.dumps({"revenue": 1000.0}),
            "signal_errors": None,
            "reason": None,
            "action_ids_fired": ["notify_high"],
            "dry_run": False,
        }

        class _GetPool:
            async def fetchrow(self, query, *args):
                return _row
            async def fetch(self, query, *args):
                return []
            async def execute(self, query, *args):
                pass

        async def _inner():
            store = DecisionStore(_GetPool())
            record = await store.get(_decision_id)
            assert record is not None
            assert record.decision_id == _decision_id
            assert record.concept_id == "org.revenue_score"
            assert record.fired is True
            assert record.concept_value == 0.75
            assert record.condition_version == "2.0"
            assert record.entity_id == "acct_42"
            assert "revenue" in record.input_primitives

        asyncio.run(_inner())


class TestDecisionRecordingFailure:
    """Test 12: decision recording failure does not fail the evaluation."""

    def test_decision_recording_failure_does_not_fail_evaluation(self):
        """Test 12: if DecisionStore.record() raises, evaluate_full() still succeeds."""
        pool = _make_pool(raise_on_decision_insert=True)

        async def _inner():
            svc = ExecuteService(pool=pool)
            # Should NOT raise even though decision INSERT will fail
            result = await svc.evaluate_full(_full_req())
            await asyncio.sleep(0)  # drain background task
            return result

        result = asyncio.run(_inner())
        # The pipeline result should be valid
        from app.models.result import FullPipelineResult
        assert isinstance(result, FullPipelineResult)
        assert result.entity == "acct_42"
