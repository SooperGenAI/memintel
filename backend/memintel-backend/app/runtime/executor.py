"""
app/runtime/executor.py
──────────────────────────────────────────────────────────────────────────────
ConceptExecutor — executes a compiled ExecutionGraph for a given entity.

DETERMINISM CONTRACT (from py-instructions.md and core-spec.md §1C):
  Rule 1: timestamp present  → deterministic=True, cache indefinitely.
           All data lookups MUST use timestamp as point-in-time reference.
  Rule 2: same inputs        → always same output (hard invariant, not best-effort).
  Rule 3: timestamp absent   → snapshot mode, deterministic=False.
           DO NOT cache snapshot results beyond the current request.
  Rule 4: cache key          → exactly (concept_id, version, entity, timestamp).
           None timestamp is a DIFFERENT key — never alias or combine.
  Rule 5: ir_hash check      → verify stored graph hash before executing.

NO LLM IN RUNTIME — ABSOLUTE RULE:
  The executor MUST NEVER call the LLM under any circumstance.
  Any LLM call in this file or any code it calls is a critical bug.

Execution order:
  Nodes execute in the deterministic topological order stored in
  ExecutionGraph.topological_order (fixed at compile time).  Independent
  nodes at the same level MAY execute concurrently (parallelizable_groups),
  but MUST NOT be reordered outside their group.  The current implementation
  runs groups sequentially — the MAY in the spec permits this.

Operator dispatch:
  Each graph node's op is dispatched via _OPERATORS, a flat dict of
  op-name → callable.  Operators receive resolved input values and a params
  dict.  Primitive fetch is handled specially: primitive_fetch nodes are
  resolved via DataResolver before operator dispatch begins.
"""
from __future__ import annotations

import math
import time
from typing import Any

import structlog

from app.models.concept import ExecutionGraph, GraphNode
from app.models.errors import ErrorType, MemintelError
from app.models.result import (
    ConceptExplanation,
    ConceptOutputType,
    ConceptResult,
    MissingDataPolicy,
    NodeTrace,
)
from app.runtime.cache import CacheKey, ResultCache
from app.runtime.data_resolver import DataResolver, PrimitiveValue


# ── Operator implementations ──────────────────────────────────────────────────
# Each callable receives (inputs: dict[str, Any], params: dict[str, Any]) → Any.
# Inputs are already-resolved values (not node_ids).
# Raises MemintelError(EXECUTION_ERROR) on runtime failures (e.g. div-by-zero).


def _guard_none(value: Any, params: dict) -> float | None:
    """
    Coerce a scalar primitive value to float, respecting missing_data_policy
    when the value is None.

    Called by scalar numeric operators before touching their input.  If a
    primitive_fetch node's missing_data_policy is 'null' (or not set), the
    DataResolver propagates None downstream.  Operators must not crash on None.

    Policy resolution (from params, falls back to null propagation):
      'zero'                         → substitute 0.0
      'null' / 'forward_fill' /
      'backward_fill' / unset / None → propagate None (caller must short-circuit)
    """
    if value is None:
        if params.get("missing_data_policy") == "zero":
            return 0.0
        return None
    return float(value)


def _guard_none_list(vals: Any) -> list | None:
    """Return None if vals itself is None (upstream produced no value)."""
    return None if vals is None else vals


def _extract_ts_value(v: Any) -> float:
    """
    Extract a numeric value from a time_series element.

    Accepts both plain float/int values and the canonical time_series dict
    format {'timestamp': <ISO 8601 str>, 'value': <number>} that connectors
    return for time_series<float> and time_series<int> primitives.

    The dict form is documented in app/models/config.py:
      list[dict[str, Any]] — for time_series types; each entry has
                             'timestamp' (ISO 8601 str) and 'value' keys.

    Falls back to float(v) for plain scalars to preserve backward compatibility
    with callers that already pass numeric values (e.g. tests using raw lists).
    """
    if isinstance(v, dict) and "value" in v:
        return float(v["value"])
    return float(v)


def _op_add(inputs: dict, params: dict) -> float | None:
    a = _guard_none(inputs["a"], params)
    b = _guard_none(inputs["b"], params)
    if a is None or b is None:
        return None
    return a + b


def _op_subtract(inputs: dict, params: dict) -> float | None:
    a = _guard_none(inputs["a"], params)
    b = _guard_none(inputs["b"], params)
    if a is None or b is None:
        return None
    return a - b


def _op_multiply(inputs: dict, params: dict) -> float | None:
    a = _guard_none(inputs["a"], params)
    b = _guard_none(inputs["b"], params)
    if a is None or b is None:
        return None
    return a * b


def _op_divide(inputs: dict, params: dict) -> float | None:
    a = _guard_none(inputs["a"], params)
    b = _guard_none(inputs["b"], params)
    if a is None or b is None:
        return None
    if b == 0.0:
        raise MemintelError(ErrorType.EXECUTION_ERROR, "Division by zero in 'divide' operator.")
    return a / b


def _op_mean(inputs: dict, params: dict) -> float | None:
    vals = _guard_none_list(inputs["input"])
    if vals is None:
        return None
    if not vals:
        return 0.0
    return sum(_extract_ts_value(v) for v in vals) / len(vals)


def _op_sum(inputs: dict, params: dict) -> float | None:
    vals = _guard_none_list(inputs["input"])
    if vals is None:
        return None
    return sum(_extract_ts_value(v) for v in vals)


def _op_min(inputs: dict, params: dict) -> float | None:
    vals = _guard_none_list(inputs["input"])
    if vals is None:
        return None
    return min(_extract_ts_value(v) for v in vals) if vals else 0.0


def _op_max(inputs: dict, params: dict) -> float | None:
    vals = _guard_none_list(inputs["input"])
    if vals is None:
        return None
    return max(_extract_ts_value(v) for v in vals) if vals else 0.0


def _op_count(inputs: dict, params: dict) -> int:
    return len(inputs["input"])


def _op_pct_change(inputs: dict, params: dict) -> float | None:
    vals = _guard_none_list(inputs["input"])
    if vals is None:
        return None
    if len(vals) < 2:
        return 0.0
    prev = _extract_ts_value(vals[-2])
    curr = _extract_ts_value(vals[-1])
    if prev == 0.0:
        return 0.0
    return (curr - prev) / abs(prev)


def _op_rate_of_change(inputs: dict, params: dict) -> float | None:
    vals = _guard_none_list(inputs["input"])
    if vals is None:
        return None
    if len(vals) < 2:
        return 0.0
    return _extract_ts_value(vals[-1]) - _extract_ts_value(vals[-2])


def _op_moving_average(inputs: dict, params: dict) -> float | None:
    vals = _guard_none_list(inputs["input"])
    if vals is None:
        return None
    if not vals:
        return 0.0
    window = int(params.get("window", len(vals)))
    window = max(1, window)
    recent = vals[-window:]
    return sum(_extract_ts_value(v) for v in recent) / len(recent)


def _op_z_score_op(inputs: dict, params: dict) -> float | None:
    # Concept-level z_score op: normalises a scalar relative to a population.
    # Without population statistics available at node-eval time, returns the
    # raw value.  For full z-score anomaly detection use the z_score condition
    # strategy, which operates on the ConceptResult + historical results.
    v = _guard_none(inputs["input"], params)
    if v is None:
        return None
    return v


def _op_percentile_op(inputs: dict, params: dict) -> float | None:
    # Same note as z_score_op above.
    v = _guard_none(inputs["input"], params)
    if v is None:
        return None
    return v


def _op_normalize(inputs: dict, params: dict) -> float | None:
    # Sigmoid normalisation: maps any float to the open interval (0, 1).
    # Deterministic and monotone-increasing — larger inputs yield larger outputs.
    x = _guard_none(inputs["input"], params)
    if x is None:
        return None
    return x / (1.0 + abs(x))


def _op_weighted_sum(inputs: dict, params: dict) -> float | None:
    vals = inputs["values"]
    if vals is None:
        return None
    weights = params.get("weights") or [1.0] * len(vals)
    return sum(float(v) * float(w) for v, w in zip(vals, weights))


def _op_to_int(inputs: dict, params: dict) -> int | None:
    v = _guard_none(inputs["input"], params)
    if v is None:
        return None
    if math.isnan(v) or math.isinf(v):
        raise MemintelError(ErrorType.EXECUTION_ERROR, "to_int: input is NaN or Inf.")
    return int(v)


def _op_coalesce(inputs: dict, params: dict) -> float:
    v = inputs["input"]
    return float(v) if v is not None else float(inputs["default"])


def _op_drop_null(inputs: dict, params: dict) -> list:
    return [v for v in inputs["input"] if v is not None]


def _op_fill_null(inputs: dict, params: dict) -> list:
    fill_val = float(inputs["value"])
    return [float(v) if v is not None else fill_val for v in inputs["input"]]


def _op_unwrap_decision(inputs: dict, params: dict) -> Any:
    dv = inputs["input"]
    return dv.value if hasattr(dv, "value") else dv


def _op_passthrough(inputs: dict, params: dict) -> Any:
    return inputs["input"]


#: Registry mapping op name → execution function.
_OPERATORS: dict[str, Any] = {
    "add":            _op_add,
    "subtract":       _op_subtract,
    "multiply":       _op_multiply,
    "divide":         _op_divide,
    "mean":           _op_mean,
    "sum":            _op_sum,
    "min":            _op_min,
    "max":            _op_max,
    "count":          _op_count,
    "pct_change":     _op_pct_change,
    "rate_of_change": _op_rate_of_change,
    "moving_average": _op_moving_average,
    "z_score_op":     _op_z_score_op,
    "percentile_op":  _op_percentile_op,
    "normalize":      _op_normalize,
    "weighted_sum":   _op_weighted_sum,
    "to_int":         _op_to_int,
    "coalesce":       _op_coalesce,
    "drop_null":      _op_drop_null,
    "fill_null":      _op_fill_null,
    "unwrap_decision": _op_unwrap_decision,
    "passthrough":    _op_passthrough,
}

_PRIMITIVE_FETCH_OP = "primitive_fetch"

log = structlog.get_logger(__name__)


# ── ConceptExecutor ────────────────────────────────────────────────────────────

class ConceptExecutor:
    """
    Executes a compiled ExecutionGraph to produce a ConceptResult.

    Dependencies are injected so the executor is testable without a real DB:
      result_cache   — ResultCache instance; shared across requests for the
                       deterministic cache to be useful.
      data_resolver  — DataResolver instance; create ONE per execute() call so
                       the request-scoped primitive cache resets correctly.
                       Use the data_resolver_factory parameter to provide a
                       factory function when calling execute_by_id().

    Two entry points:
      execute_graph()  — executes a pre-fetched ExecutionGraph directly.
                         Used in tests and by the hot path.
      execute()        — fetches the graph from graph_store first, then delegates.
    """

    def __init__(
        self,
        result_cache: ResultCache,
        graph_store: Any | None = None,
    ) -> None:
        self._cache = result_cache
        self._graph_store = graph_store  # optional; only needed for execute()

    # ── Primary entry point (direct graph) ────────────────────────────────────

    def execute_graph(
        self,
        graph: ExecutionGraph,
        entity: str,
        data_resolver: DataResolver,
        timestamp: str | None = None,
        explain: bool = False,
        cache: bool = True,
        missing_data_policy: MissingDataPolicy | None = None,
    ) -> ConceptResult:
        """
        Execute ``graph`` for ``entity`` at ``timestamp`` and return a ConceptResult.

        Determinism contract enforced here:
          - timestamp present  → deterministic=True, cache if cache=True.
          - timestamp absent   → deterministic=False, NEVER cache.
          - same inputs        → always same output.
        """
        # Rule 4: cache key = (concept_id, version, entity, timestamp)
        cache_key: CacheKey = (graph.concept_id, graph.version, entity, timestamp)

        # Check cache only for deterministic executions.
        if cache and timestamp is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                log.info(
                    "concept_executed",
                    concept_id=graph.concept_id,
                    version=graph.version,
                    entity=entity,
                    timestamp=timestamp,
                    deterministic=True,
                    cache_hit=True,
                    compute_time_ms=0,
                    result_type=cached.type.value,
                )
                return cached

        # Execute the graph.
        t0 = time.monotonic()
        result = self._run(
            graph=graph,
            entity=entity,
            data_resolver=data_resolver,
            timestamp=timestamp,
            explain=explain,
            missing_data_policy=missing_data_policy,
        )
        compute_time_ms = int((time.monotonic() - t0) * 1000)

        # Rule 1 / Rule 3: cache iff timestamp is present (deterministic).
        # ResultCache.set() is already a no-op for None-timestamp keys, but we
        # honour the intent explicitly to satisfy Rule 3.
        if cache and timestamp is not None:
            self._cache.set(cache_key, result)

        log.info(
            "concept_executed",
            concept_id=graph.concept_id,
            version=graph.version,
            entity=entity,
            timestamp=timestamp,
            deterministic=result.deterministic,
            cache_hit=False,
            compute_time_ms=compute_time_ms,
            result_type=result.type.value,
        )
        return result

    # ── Secondary entry point (fetch graph by concept_id + version) ───────────

    def execute(
        self,
        concept_id: str,
        version: str,
        entity: str,
        data_resolver: DataResolver,
        timestamp: str | None = None,
        explain: bool = False,
        cache: bool = True,
        missing_data_policy: MissingDataPolicy | None = None,
        ir_hash: str | None = None,
    ) -> ConceptResult:
        """
        Fetch the graph for (concept_id, version) then delegate to execute_graph().

        Raises MemintelError(NOT_FOUND)  if the graph does not exist.
        Raises MemintelError(CONFLICT)   if ir_hash is provided and mismatches.
        Raises RuntimeError              if graph_store was not injected.
        """
        if self._graph_store is None:
            raise RuntimeError(
                "ConceptExecutor.execute() requires a graph_store. "
                "Either inject one at construction, or call execute_graph() directly."
            )

        graph = self._graph_store.get_by_concept(concept_id, version)
        if graph is None:
            raise MemintelError(
                ErrorType.NOT_FOUND,
                f"No compiled graph found for concept '{concept_id}' version '{version}'.",
                location=f"{concept_id}:{version}",
            )

        if ir_hash is not None and graph.ir_hash != ir_hash:
            raise MemintelError(
                ErrorType.CONFLICT,
                f"ir_hash mismatch for '{concept_id}:{version}'. "
                f"Expected '{ir_hash}', found '{graph.ir_hash}'. Re-compile required.",
                location=f"{concept_id}:{version}",
            )

        return self.execute_graph(
            graph=graph,
            entity=entity,
            data_resolver=data_resolver,
            timestamp=timestamp,
            explain=explain,
            cache=cache,
            missing_data_policy=missing_data_policy,
        )

    # ── Internal graph execution ───────────────────────────────────────────────

    def _run(
        self,
        graph: ExecutionGraph,
        entity: str,
        data_resolver: DataResolver,
        timestamp: str | None,
        explain: bool,
        missing_data_policy: MissingDataPolicy | None,
    ) -> ConceptResult:
        """Execute graph nodes in topological order; return a ConceptResult."""
        node_by_id: dict[str, GraphNode] = {n.node_id: n for n in graph.nodes}
        node_values: dict[str, Any] = {}  # node_id → computed value
        node_traces: list[NodeTrace] = [] if explain else []

        for node_id in graph.topological_order:
            node = node_by_id[node_id]
            value = self._execute_node(
                node=node,
                node_values=node_values,
                entity=entity,
                timestamp=timestamp,
                data_resolver=data_resolver,
                missing_data_policy=missing_data_policy,
            )
            node_values[node_id] = value

            if explain:
                resolved_inputs = {
                    slot: node_values.get(src_id, src_id)
                    for slot, src_id in node.inputs.items()
                    if isinstance(src_id, str)
                }
                node_traces.append(NodeTrace(
                    node_id=node_id,
                    op=node.op,
                    inputs=resolved_inputs,
                    params=dict(node.params),
                    output_value=value,
                    output_type=node.output_type,
                ))

        output_value = node_values[graph.output_node_id]
        deterministic = timestamp is not None

        # Build explanation if requested.
        explanation: ConceptExplanation | None = None
        if explain:
            prim_nodes = [n for n in graph.nodes if n.op == _PRIMITIVE_FETCH_OP]
            contributions = _equal_contributions(
                [n.params.get("source_name", n.node_id) for n in prim_nodes],
                output_value,
            )
            explanation = ConceptExplanation(
                output=output_value,
                contributions=contributions,
                nodes=node_traces,
            )

        return ConceptResult(
            value=output_value,
            type=_output_type(graph.output_type),
            entity=entity,
            version=graph.version,
            deterministic=deterministic,
            timestamp=timestamp,
            explanation=explanation,
        )

    def _execute_node(
        self,
        node: GraphNode,
        node_values: dict[str, Any],
        entity: str,
        timestamp: str | None,
        data_resolver: DataResolver,
        missing_data_policy: MissingDataPolicy | None,
    ) -> Any:
        """Compute a single node's output value."""
        if node.op == _PRIMITIVE_FETCH_OP:
            # Primitive fetch: delegate to DataResolver.
            source_name = node.params.get("source_name", "")
            declared_policy = node.params.get("missing_data_policy")
            # Use the per-request override, the node-level policy, or NULL.
            policy = (
                missing_data_policy
                or (MissingDataPolicy(declared_policy) if declared_policy else MissingDataPolicy.NULL)
            )
            pv = data_resolver.fetch(source_name, entity, timestamp, policy=policy)
            return pv.value

        # Feature node: resolve inputs from node_values, then dispatch.
        resolved_inputs: dict[str, Any] = {}
        for slot_name, src in node.inputs.items():
            if isinstance(src, str):
                resolved_inputs[slot_name] = node_values.get(src)
            else:
                # Literal value in the input slot.
                resolved_inputs[slot_name] = src

        op_fn = _OPERATORS.get(node.op)
        if op_fn is None:
            raise MemintelError(
                ErrorType.EXECUTION_ERROR,
                f"Operator '{node.op}' is not registered in the executor's operator table.",
                location=node.node_id,
            )

        return op_fn(resolved_inputs, dict(node.params))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _output_type(type_str: str) -> ConceptOutputType:
    """Map a Memintel type string to ConceptOutputType enum."""
    # Strip nullable modifier so float? → float, string? → string, etc.
    base = type_str[:-1] if type_str.endswith("?") else type_str
    mapping = {
        "float": ConceptOutputType.FLOAT,
        "int":   ConceptOutputType.FLOAT,    # int widens to float at concept output
        "boolean": ConceptOutputType.BOOLEAN,
        "categorical": ConceptOutputType.CATEGORICAL,
        "string": ConceptOutputType.CATEGORICAL,  # string concepts map to CATEGORICAL slot
    }
    result = mapping.get(base)
    if result is not None:
        return result
    # labeled categorical: categorical{a,b,c} or categorical{a,b,c}? → CATEGORICAL
    if base.startswith("categorical{"):
        return ConceptOutputType.CATEGORICAL
    return ConceptOutputType.FLOAT


def _equal_contributions(signal_names: list[str], output_value: Any) -> dict[str, float]:
    """Return equal attribution weights across all primitive signals."""
    if not signal_names:
        return {}
    weight = round(1.0 / len(signal_names), 6)
    return {name: weight for name in signal_names}
