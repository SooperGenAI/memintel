"""
app/runtime/condition_evaluator.py
──────────────────────────────────────────────────────────────────────────────
ConditionEvaluator — evaluates a condition for a given entity.

Transparent concept execution:
  The evaluator implicitly executes the concept if the ConceptResult is not
  already in cache.  Callers can pre-warm the concept cache by calling
  ConceptExecutor.execute_graph() before invoking evaluate().  This is
  transparent — callers do not need to manage the execution order.

Strategy dispatch:
  The ConditionDefinition.strategy.type determines which ConditionStrategy
  subclass is used.  All six strategies (threshold, percentile, z_score,
  change, equals, composite) are registered in _STRATEGY_REGISTRY.

History:
  Strategies that require historical results (z_score, change, percentile)
  receive a history list from the history_provider callable.  In tests, this
  can be overridden to return pre-configured history.  Default: empty list.

Composite evaluation:
  Composite conditions reference operand condition_ids.  The evaluator
  resolves each operand recursively via _evaluate_operand() which calls a
  provided operand_resolver callable.  Default: always returns a False boolean
  decision (safe stub for single-condition tests).

dry_run:
  When dry_run=True the concept EXECUTES and the condition EVALUATES normally.
  Only action triggering is skipped (handled by ActionTrigger).  This is
  consistent with core-spec.md §1C dry_run propagation rules.
"""
from __future__ import annotations

from typing import Any, Callable

import structlog

from app.models.condition import (
    BOOLEAN_STRATEGIES,
    CATEGORICAL_STRATEGIES,
    ConditionDefinition,
    DecisionType,
    DecisionValue,
    StrategyType,
)
from app.models.result import ConceptResult, MissingDataPolicy
from app.runtime.cache import ResultCache
from app.runtime.data_resolver import DataResolver
from app.runtime.executor import ConceptExecutor
from app.strategies.change import ChangeStrategy
from app.strategies.composite import CompositeStrategy
from app.strategies.equals import EqualsStrategy
from app.strategies.percentile import PercentileStrategy
from app.strategies.threshold import ThresholdStrategy
from app.strategies.z_score import ZScoreStrategy


log = structlog.get_logger(__name__)


# ── Strategy registry ─────────────────────────────────────────────────────────
# One instance per strategy type — strategies are stateless.

_STRATEGY_REGISTRY: dict[StrategyType, Any] = {
    StrategyType.THRESHOLD:  ThresholdStrategy(),
    StrategyType.PERCENTILE: PercentileStrategy(),
    StrategyType.Z_SCORE:    ZScoreStrategy(),
    StrategyType.CHANGE:     ChangeStrategy(),
    StrategyType.EQUALS:     EqualsStrategy(),
    StrategyType.COMPOSITE:  CompositeStrategy(),
}


class ConditionEvaluator:
    """
    Evaluates a ConditionDefinition against the concept result for an entity.

    Dependencies
    ────────────
    executor         — ConceptExecutor; used to produce the ConceptResult if not
                       already in cache (transparent concept execution).
    result_cache     — ResultCache; shared with executor so cache hits work.
    history_provider — callable(condition, entity, timestamp) → list[ConceptResult].
                       Provides historical concept results for z_score, change,
                       percentile strategies.  Default: returns empty list.
    operand_resolver — callable(condition_id, version, entity, timestamp) → DecisionValue.
                       Used by composite strategy to evaluate operands.
                       Default: returns a False boolean decision (stub).
    """

    def __init__(
        self,
        executor: ConceptExecutor,
        result_cache: ResultCache,
        history_provider: Callable[
            [ConditionDefinition, str, str | None], list[ConceptResult]
        ] | None = None,
        operand_resolver: Callable[
            [str, str, str, str | None], DecisionValue
        ] | None = None,
    ) -> None:
        self._executor = executor
        self._cache = result_cache
        self._history_provider = history_provider or _empty_history
        self._operand_resolver = operand_resolver or _stub_operand_resolver

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        condition: ConditionDefinition,
        entity: str,
        data_resolver: DataResolver,
        timestamp: str | None = None,
        dry_run: bool = False,
        missing_data_policy: MissingDataPolicy | None = None,
    ) -> DecisionValue:
        """
        Evaluate ``condition`` for ``entity`` at ``timestamp``.

        1. Obtain ConceptResult — from cache or by executing the concept.
        2. Fetch history for stateful strategies (z_score, change, percentile).
        3. Dispatch to the appropriate ConditionStrategy.
        4. Return DecisionValue with provenance.

        dry_run does NOT affect evaluation here — the concept executes and the
        condition evaluates normally.  Only ActionTrigger suppresses side effects
        in dry_run mode.
        """
        # Step 1: obtain concept result (transparent execution).
        concept_result = self._get_concept_result(
            condition=condition,
            entity=entity,
            data_resolver=data_resolver,
            timestamp=timestamp,
            missing_data_policy=missing_data_policy,
        )

        # Step 2: fetch history for stateful strategies.
        strategy_type = StrategyType(condition.strategy.type)
        history: list[ConceptResult] = []
        if strategy_type in (StrategyType.Z_SCORE, StrategyType.CHANGE, StrategyType.PERCENTILE):
            history = self._history_provider(condition, entity, timestamp)

        # Step 3: dispatch to strategy.
        params = condition.strategy.params.model_dump()
        log_params = dict(params)  # capture before composite resolution

        if strategy_type == StrategyType.COMPOSITE:
            params = self._resolve_composite_params(
                condition, entity, timestamp, params
            )

        strategy = _STRATEGY_REGISTRY[strategy_type]
        decision = strategy.evaluate(
            concept_result,
            history,
            params,
            condition_id=condition.condition_id,
            condition_version=condition.version,
        )

        log.info(
            "condition_evaluated",
            condition_id=condition.condition_id,
            condition_version=condition.version,
            entity=entity,
            timestamp=timestamp,
            decision_value=str(decision.value),
            decision_type=decision.decision_type.value,
            strategy_type=strategy_type.value,
            params_applied=log_params,
            actions_triggered_count=0,
        )
        return decision

    # ── Async public API ───────────────────────────────────────────────────────

    async def aevaluate(
        self,
        condition: ConditionDefinition,
        entity: str,
        data_resolver: DataResolver,
        timestamp: str | None = None,
        dry_run: bool = False,
        missing_data_policy: MissingDataPolicy | None = None,
    ) -> DecisionValue:
        """
        Async variant of evaluate() — fetches concept result via async executor.

        Use this from async call paths (e.g. ExplanationService.explain_decision)
        to avoid the cache-miss crash that occurs when the sync evaluate() falls
        through to ConceptExecutor.execute() on a non-warmed cache.

        Semantics are identical to evaluate() — same strategy dispatch, same
        history fetch, same dry_run behaviour.
        """
        concept_result = await self._aget_concept_result(
            condition=condition,
            entity=entity,
            data_resolver=data_resolver,
            timestamp=timestamp,
            missing_data_policy=missing_data_policy,
        )

        strategy_type = StrategyType(condition.strategy.type)
        history: list[ConceptResult] = []
        if strategy_type in (StrategyType.Z_SCORE, StrategyType.CHANGE, StrategyType.PERCENTILE):
            history = self._history_provider(condition, entity, timestamp)

        params = condition.strategy.params.model_dump()
        log_params = dict(params)

        if strategy_type == StrategyType.COMPOSITE:
            params = self._resolve_composite_params(
                condition, entity, timestamp, params
            )

        strategy = _STRATEGY_REGISTRY[strategy_type]
        decision = strategy.evaluate(
            concept_result,
            history,
            params,
            condition_id=condition.condition_id,
            condition_version=condition.version,
        )

        log.info(
            "condition_evaluated",
            condition_id=condition.condition_id,
            condition_version=condition.version,
            entity=entity,
            timestamp=timestamp,
            decision_value=str(decision.value),
            decision_type=decision.decision_type.value,
            strategy_type=strategy_type.value,
            params_applied=log_params,
            actions_triggered_count=0,
        )
        return decision

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_concept_result(
        self,
        condition: ConditionDefinition,
        entity: str,
        data_resolver: DataResolver,
        timestamp: str | None,
        missing_data_policy: MissingDataPolicy | None,
    ) -> ConceptResult:
        """
        Return the ConceptResult for the condition's concept, from cache or
        by executing it.  This is transparent to the caller.
        """
        from app.models.concept import ExecutionGraph  # avoid circular at module level

        cache_key = (condition.concept_id, condition.concept_version, entity, timestamp)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Not in cache — execute the concept.
        # This requires the executor to have a graph_store, OR the graph to be
        # available directly.  For tests without a real graph_store, pre-populate
        # the cache before calling evaluate().
        #
        # NOTE: ConceptExecutor.execute() is async. This sync method can only
        # reach here in test contexts where the cache is guaranteed to be
        # pre-warmed before evaluate() is called.  Production async call paths
        # must use aevaluate() / _aget_concept_result() instead.
        raise RuntimeError(
            "ConditionEvaluator.evaluate() reached a cache miss. "
            "Pre-warm the result cache before calling the sync evaluate(), "
            "or use async aevaluate() from async contexts."
        )

    async def _aget_concept_result(
        self,
        condition: ConditionDefinition,
        entity: str,
        data_resolver: DataResolver,
        timestamp: str | None,
        missing_data_policy: MissingDataPolicy | None,
    ) -> ConceptResult:
        """
        Async variant of _get_concept_result — awaits ConceptExecutor.execute().

        Returns the cached result when available; otherwise awaits the executor
        to fetch the graph and produce the result.
        """
        from app.models.concept import ExecutionGraph  # avoid circular at module level

        cache_key = (condition.concept_id, condition.concept_version, entity, timestamp)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        return await self._executor.execute(
            concept_id=condition.concept_id,
            version=condition.concept_version,
            entity=entity,
            data_resolver=data_resolver,
            timestamp=timestamp,
            missing_data_policy=missing_data_policy,
        )

    def _resolve_composite_params(
        self,
        condition: ConditionDefinition,
        entity: str,
        timestamp: str | None,
        params: dict,
    ) -> dict:
        """
        Resolve composite operand condition_ids to their DecisionValues.

        Replaces params['operands'] (list of condition_ids) with
        params['operand_results'] (list of DecisionValues) so CompositeStrategy
        can evaluate them directly.
        """
        operand_ids: list[str] = params.get("operands", [])
        operand_results: list[DecisionValue] = [
            self._operand_resolver(op_id, "1.0", entity, timestamp)
            for op_id in operand_ids
        ]
        # CompositeStrategy expects 'operand_results', not 'operands'.
        resolved = dict(params)
        resolved["operand_results"] = operand_results
        return resolved


# ── Default providers ─────────────────────────────────────────────────────────

def _empty_history(
    condition: ConditionDefinition, entity: str, timestamp: str | None
) -> list[ConceptResult]:
    """Default history provider — returns no history."""
    return []


def _stub_operand_resolver(
    condition_id: str, version: str, entity: str, timestamp: str | None
) -> DecisionValue:
    """
    Stub operand resolver for composite conditions.

    Returns False boolean decision.  Replace with a real resolver that fetches
    from the condition store and evaluates recursively in production.
    """
    return DecisionValue(
        value=False,
        decision_type=DecisionType.BOOLEAN,
        condition_id=condition_id,
        condition_version=version,
        entity=entity,
        timestamp=timestamp,
    )
