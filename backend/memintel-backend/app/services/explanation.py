"""
app/services/explanation.py
──────────────────────────────────────────────────────────────────────────────
ExplanationService — deterministic decision explanation generation.

Explains WHY a condition fired or did not fire for a given entity at a given
timestamp by re-executing the concept with explain=True and re-evaluating
the condition. No LLM involvement at any stage.

Pipeline for explain_decision():
  1. Load condition from definition registry → parse ConditionDefinition.
     Raises NotFoundError → HTTP 404 if not registered.
  2. Execute concept with explain=True → ConceptResult with ConceptExplanation.
     Provides per-signal attribution (contributions dict).
  3. Evaluate condition → DecisionValue.
     Determines boolean/categorical decision outcome.
  4. rank_drivers() — convert contributions dict to sorted DriverContribution list.
     Normalises contributions to sum to 1.0 (INVARIANT).
  5. Resolve strategy-aware fields:
       threshold/percentile/change → threshold_applied = params.value
       z_score                    → threshold_applied = params.threshold
       equals                     → label_matched = decision.value (str)
       composite                  → both threshold_applied and label_matched are None
  6. Return DecisionExplanation.

Invariants
──────────
  - No LLM calls. Ever.
  - drivers[].contribution values sum to 1.0 (normalisation in _rank_drivers).
  - threshold_applied is None for equals and composite strategies.
  - label_matched is None for all non-equals strategies.
  - Results are deterministic for any (condition_id, condition_version,
    entity, timestamp) tuple — same inputs always produce the same explanation.
"""
from __future__ import annotations

import logging
from typing import Any

from app.models.condition import (
    ConditionDefinition,
    ConditionExplanation,
    DecisionExplanation,
    DriverContribution,
    StrategyDefinition,
    StrategyType,
)
from app.models.result import ConceptResult

log = logging.getLogger(__name__)


class ExplanationService:
    """
    Generates DecisionExplanation objects by re-executing the concept and
    condition evaluation paths deterministically.

    explain_decision() uses the same execution path as POST /evaluate/full,
    with explain=True to capture per-signal driver contributions.

    Parameters
    ──────────
    definition_registry — must implement async get(id, version) → dict.
                          Raises NotFoundError if not registered.
    concept_executor    — must implement async aexecute(concept_id, version,
                          entity, data_resolver, timestamp, explain)
                          → ConceptResult.
    condition_evaluator — must implement evaluate(condition, entity,
                          data_resolver, timestamp) → DecisionValue.
    data_resolver       — passed through to executor and evaluator.
                          May be None; callers are responsible for providing
                          a real resolver for production use.
    """

    def __init__(
        self,
        definition_registry: Any,
        concept_executor: Any,
        condition_evaluator: Any,
        data_resolver: Any = None,
    ) -> None:
        self._registry = definition_registry
        self._executor = concept_executor
        self._evaluator = condition_evaluator
        self._data_resolver = data_resolver

    # ── Public API ──────────────────────────────────────────────────────────────

    async def explain_condition(
        self,
        condition_id: str,
        condition_version: str,
        timestamp: str | None = None,
    ) -> ConditionExplanation:
        """
        Return a human-readable explanation of a condition definition.

        Derives natural_language_summary and parameter_rationale from the
        condition's strategy type and parameters. Fully deterministic — no
        concept execution, no LLM, no entity required.

        timestamp is accepted for API symmetry but is not used in the
        derivation — condition explanations are definition-level, not
        instance-level.

        Raises NotFoundError if (condition_id, condition_version) is not
        registered in the definition registry.
        """
        body = await self._registry.get(condition_id, condition_version)
        condition = ConditionDefinition.model_validate(body)

        summary, rationale = _explain_strategy(condition.strategy)

        log.info(
            "condition_explained",
            extra={
                "condition_id": condition_id,
                "condition_version": condition_version,
                "strategy_type": condition.strategy.type.value,
            },
        )

        return ConditionExplanation(
            condition_id=condition_id,
            condition_version=condition_version,
            strategy=condition.strategy,
            concept_id=condition.concept_id,
            concept_version=condition.concept_version,
            natural_language_summary=summary,
            parameter_rationale=rationale,
        )

    async def explain_decision(
        self,
        condition_id: str,
        condition_version: str,
        entity: str,
        timestamp: str | None,
    ) -> DecisionExplanation:
        """
        Return a full explanation of a specific decision result.

        Re-executes concept execution and condition evaluation deterministically.
        Results are identical for any given (condition_id, condition_version,
        entity, timestamp) tuple.

        Raises NotFoundError if (condition_id, condition_version) is not
        registered in the definition registry.
        """
        # 1. Load and parse condition definition.
        body = await self._registry.get(condition_id, condition_version)
        condition = ConditionDefinition.model_validate(body)

        # 2. Execute concept with explain=True for driver attribution.
        concept_result: ConceptResult = await self._executor.aexecute(
            concept_id=condition.concept_id,
            version=condition.concept_version,
            entity=entity,
            data_resolver=self._data_resolver,
            timestamp=timestamp,
            explain=True,
        )

        # 3. Evaluate condition to get the decision value.
        decision = self._evaluator.evaluate(
            condition=condition,
            entity=entity,
            data_resolver=self._data_resolver,
            timestamp=timestamp,
        )

        # 4. Rank and normalise driver contributions.
        #    INVARIANT: sum(d.contribution for d in drivers) == 1.0.
        drivers = _rank_drivers(concept_result)

        # 5. Resolve strategy-aware fields.
        strategy_type = condition.strategy.type
        threshold_applied: float | None = None
        label_matched: str | None = None

        if strategy_type in (
            StrategyType.THRESHOLD,
            StrategyType.PERCENTILE,
            StrategyType.CHANGE,
        ):
            threshold_applied = condition.strategy.params.value
        elif strategy_type == StrategyType.Z_SCORE:
            threshold_applied = condition.strategy.params.threshold
        elif strategy_type == StrategyType.EQUALS:
            # Categorical match — no numeric threshold.
            label_matched = (
                decision.value if isinstance(decision.value, str) else None
            )
        # composite: both remain None (operator, not a single threshold)

        log.info(
            "decision_explained",
            extra={
                "condition_id": condition_id,
                "condition_version": condition_version,
                "entity": entity,
                "strategy_type": strategy_type.value,
                "decision": decision.value,
                "drivers_count": len(drivers),
            },
        )

        return DecisionExplanation(
            condition_id=condition_id,
            condition_version=condition_version,
            entity=entity,
            timestamp=timestamp,
            decision=decision.value,
            decision_type=decision.decision_type,
            concept_value=concept_result.value,
            strategy_type=strategy_type,
            threshold_applied=threshold_applied,
            label_matched=label_matched,
            drivers=drivers,
        )


# ── Private helpers ───────────────────────────────────────────────────────────

def _explain_strategy(strategy: StrategyDefinition) -> tuple[str, str]:
    """
    Derive (natural_language_summary, parameter_rationale) for a strategy.

    Fully deterministic — no LLM, no entity, no execution.
    Returns a 2-tuple: (summary, rationale).
    """
    p = strategy.params
    st = strategy.type

    if st == StrategyType.THRESHOLD:
        summary = (
            f"Fires when the concept value is {p.direction} the fixed threshold {p.value}."
        )
        rationale = (
            f"direction={p.direction!r}, value={p.value} — fixed cutoff; "
            "adjust value to raise or lower sensitivity."
        )
    elif st == StrategyType.PERCENTILE:
        summary = (
            f"Fires when the concept value falls in the {p.direction} {p.value}% "
            "of historical values."
        )
        rationale = (
            f"direction={p.direction!r}, value={p.value} — relative cutoff; "
            "adjust value to change the fraction of entities that trigger."
        )
    elif st == StrategyType.Z_SCORE:
        summary = (
            f"Fires when the concept value is more than {p.threshold} standard "
            f"deviations {p.direction} the historical mean (window: {p.window})."
        )
        rationale = (
            f"threshold={p.threshold}, direction={p.direction!r}, window={p.window!r} — "
            "higher threshold reduces sensitivity; wider window smooths the baseline."
        )
    elif st == StrategyType.CHANGE:
        summary = (
            f"Fires when the concept value changes by more than {p.value}% "
            f"({p.direction}) within the last {p.window}."
        )
        rationale = (
            f"direction={p.direction!r}, value={p.value}, window={p.window!r} — "
            "adjust value to set the minimum % change; adjust window to control lookback."
        )
    elif st == StrategyType.EQUALS:
        summary = f"Fires when the concept value equals {p.value!r}."
        rationale = (
            f"value={p.value!r} — categorical match; no numeric parameter to calibrate."
        )
    else:  # COMPOSITE
        summary = (
            f"Combines {len(p.operands)} conditions with a logical {p.operator} operator."
        )
        rationale = (
            f"operator={p.operator!r}, operands={p.operands} — "
            "fires when the logical combination of operands evaluates to True."
        )

    return summary, rationale


def _rank_drivers(concept_result: ConceptResult) -> list[DriverContribution]:
    """
    Convert ConceptExplanation.contributions to a sorted, normalised list of
    DriverContribution objects.

    Normalisation: contributions are scaled so they sum to 1.0.
    Ordering: highest contribution driver listed first.
    Signal values: resolved from ConceptExplanation.nodes (matched by node_id).
                   Defaults to 0.0 when not found in the node trace.

    Returns an empty list when explanation is None or contributions is empty.

    INVARIANT: sum(d.contribution for d in result) == 1.0 when result is
    non-empty.
    """
    explanation = concept_result.explanation
    if explanation is None or not explanation.contributions:
        return []

    # Build node_id → output_value lookup from the trace.
    node_values: dict[str, Any] = {}
    if explanation.nodes:
        for node in explanation.nodes:
            node_values[node.node_id] = node.output_value

    contributions = dict(explanation.contributions)

    # Normalise to sum to 1.0 if needed.
    total = sum(contributions.values())
    if total > 0.0 and abs(total - 1.0) > 1e-9:
        contributions = {k: v / total for k, v in contributions.items()}

    # Sort descending by contribution (highest driver first).
    return [
        DriverContribution(
            signal=signal,
            contribution=round(contrib, 6),
            value=node_values.get(signal, 0.0),
        )
        for signal, contrib in sorted(
            contributions.items(), key=lambda kv: kv[1], reverse=True
        )
    ]
