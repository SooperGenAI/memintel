"""
app/services/execute.py
──────────────────────────────────────────────────────────────────────────────
ExecuteService — concept execution and full pipeline dispatch.

Drives the ψ → φ → α pipeline:
  ψ  ConceptExecutor  — fetches primitives, evaluates the graph
  φ  condition strategy — evaluates the condition against the concept result
  α  ActionTrigger    — dispatches bound actions (best-effort)

Also handles:
  - Batch execution (execute_batch): runs ψ for N entities, short-circuits on
    per-entity errors without failing the whole batch.
  - Range execution (execute_range): repeats ψ for one entity over a time range.
  - Async execution (execute_async): enqueues a job and returns immediately.
  - Graph execution (execute_graph): executes a pre-compiled graph, bypassing
    compilation.

Connector note
──────────────
No production data connector (SQL, REST) is implemented yet.  All execution
paths use MockConnector with empty data — primitives return None and the null
missing_data_policy applies.  Wire a real connector here when the connector
layer is built.

Composite conditions
────────────────────
Composite conditions are evaluated recursively: each operand condition_id is
resolved to its latest registered version, its concept is executed, and the
resulting DecisionValue is collected. The CompositeStrategy then applies its
AND/OR operator across the collected boolean results.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

import asyncio

import asyncpg
import structlog

from app.compiler.dag_builder import DAGBuilder
from app.models.action import ActionDefinition
from app.models.condition import ConditionDefinition, DecisionType, DecisionValue, StrategyType
from app.models.concept import ConceptDefinition
from app.models.errors import (
    ErrorDetail,
    ErrorResponse,
    ErrorType,
    MemintelError,
    NotFoundError,
)
from app.models.decision import DecisionRecord
from app.models.result import (
    ActionTriggered,
    ActionTriggeredStatus,
    BatchExecuteItem,
    BatchExecuteResult,
    ConceptOutputType,
    ConceptResult,
    DecisionResult,
    ExecuteGraphRequest,
    ExecuteRequest,
    FullPipelineResult,
    Job,
)
from app.stores.decision import DecisionStore
from app.runtime.action_trigger import ActionTrigger
from app.runtime.cache import ResultCache
from app.runtime.data_resolver import DataResolver, MockConnector
from app.runtime.executor import ConceptExecutor
from app.stores.concept_result import ConceptResultStore
from app.stores.graph import GraphStore
from app.stores.job import JobStore
from app.strategies.change import ChangeStrategy
from app.strategies.composite import CompositeStrategy
from app.strategies.equals import EqualsStrategy
from app.strategies.percentile import PercentileStrategy
from app.strategies.threshold import ThresholdStrategy
from app.strategies.z_score import ZScoreStrategy


log = structlog.get_logger(__name__)

#: Strategies that require a historical reference frame (prior concept results
#: for the same entity). These are the only strategies where history matters —
#: threshold, equals, and composite do not use history at all.
_HISTORY_STRATEGIES: frozenset[StrategyType] = frozenset({
    StrategyType.Z_SCORE,
    StrategyType.PERCENTILE,
    StrategyType.CHANGE,
})

#: Minimum number of stored concept results required before a history-based
#: strategy will evaluate. Below this, returns reason="insufficient_history".
_HISTORY_MIN_RESULTS: int = 3

#: Maximum number of historical results fetched per evaluation.
_HISTORY_WINDOW: int = 30


# ── Strategy registry ──────────────────────────────────────────────────────────
# One stateless instance per strategy type — shared across all service calls.

_STRATEGY_REGISTRY: dict[StrategyType, Any] = {
    StrategyType.THRESHOLD:  ThresholdStrategy(),
    StrategyType.PERCENTILE: PercentileStrategy(),
    StrategyType.Z_SCORE:    ZScoreStrategy(),
    StrategyType.CHANGE:     ChangeStrategy(),
    StrategyType.EQUALS:     EqualsStrategy(),
    StrategyType.COMPOSITE:  CompositeStrategy(),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_body(raw: Any) -> dict:
    """Deserialise a JSONB value that may arrive as str or dict."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw or {}


def _parse_iso_duration(duration: str) -> timedelta:
    """
    Parse an ISO 8601 duration string into a timedelta.

    Supports P[n]Y[n]M[n]W[n]DT[n]H[n]M[n]S.
    Years and months are approximated (365 days/year, 30 days/month).
    Raises ValueError for unrecognised formats.
    """
    pattern = (
        r"^P"
        r"(?:(\d+(?:\.\d+)?)Y)?"
        r"(?:(\d+(?:\.\d+)?)M)?"
        r"(?:(\d+(?:\.\d+)?)W)?"
        r"(?:(\d+(?:\.\d+)?)D)?"
        r"(?:T"
        r"(?:(\d+(?:\.\d+)?)H)?"
        r"(?:(\d+(?:\.\d+)?)M)?"
        r"(?:(\d+(?:\.\d+)?)S)?"
        r")?$"
    )
    m = re.match(pattern, duration)
    if not m:
        raise ValueError(f"Cannot parse ISO 8601 duration: {duration!r}")
    years, months, weeks, days, hours, minutes, seconds = (
        float(g) if g else 0.0 for g in m.groups()
    )
    total_days = years * 365.0 + months * 30.0 + weeks * 7.0 + days
    total_seconds = hours * 3600.0 + minutes * 60.0 + seconds
    return timedelta(days=total_days, seconds=total_seconds)


def _make_connector() -> MockConnector:
    """
    Return the data connector for primitive fetching.

    No production connector is implemented yet.  MockConnector with empty data
    is used — all primitives return None; null missing_data_policy applies.
    """
    return MockConnector(data={})


class ExecuteService:
    """
    Drives concept and full pipeline execution.

    evaluate_full()            — ψ → φ → α; returns FullPipelineResult.
    evaluate_condition()       — φ layer for one entity; returns DecisionResult.
    evaluate_condition_batch() — φ for N entities; returns list[DecisionResult].
    execute()                  — ψ layer only; returns ConceptResult.
    execute_batch()            — ψ for N entities; returns BatchExecuteResult.
    execute_range()            — ψ over a time range; returns list[ConceptResult].
    execute_async()            — enqueues job; returns Job.
    execute_graph()            — executes a pre-compiled graph; returns ConceptResult.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        connector_registry: Any = None,
        primitive_sources: dict | None = None,
    ) -> None:
        self._pool = pool
        # Shared in-memory result cache — deterministic results survive across
        # execute() calls within the service instance lifetime.
        self._cache = ResultCache()
        # Store for persisting concept results used as history by stateful strategies.
        self._result_store = ConceptResultStore(pool)
        self._connector_registry = connector_registry  # ConnectorRegistry or None
        self._primitive_sources = primitive_sources or {}

    # ── DB helpers ─────────────────────────────────────────────────────────────

    async def _fetch_concept(self, concept_id: str, version: str) -> ConceptDefinition:
        """Load and parse a concept definition from the definitions table."""
        row = await self._pool.fetchrow(
            """
            SELECT body FROM definitions
            WHERE definition_id = $1 AND version = $2
              AND definition_type = 'concept'
            LIMIT 1
            """,
            concept_id,
            version,
        )
        if row is None:
            raise NotFoundError(
                f"Concept '{concept_id}' version '{version}' not found.",
                location=f"{concept_id}:{version}",
                suggestion="Register the concept first via POST /registry/definitions.",
            )
        return ConceptDefinition(**_parse_body(row["body"]))

    async def _fetch_condition(
        self, condition_id: str, condition_version: str
    ) -> ConditionDefinition:
        """Load and parse a condition definition from the definitions table."""
        row = await self._pool.fetchrow(
            """
            SELECT body FROM definitions
            WHERE definition_id = $1 AND version = $2
              AND definition_type = 'condition'
            LIMIT 1
            """,
            condition_id,
            condition_version,
        )
        if row is None:
            raise NotFoundError(
                f"Condition '{condition_id}' version '{condition_version}' not found.",
                location=f"{condition_id}:{condition_version}",
                suggestion="Register the condition first via POST /registry/definitions.",
            )
        return ConditionDefinition(**_parse_body(row["body"]))

    async def _fetch_condition_latest(self, condition_id: str) -> ConditionDefinition:
        """
        Load the most recently registered version of a condition.

        Used by composite operand resolution where operands store only a
        condition_id (no pinned version). Raises NotFoundError when the
        condition_id is not registered.
        """
        row = await self._pool.fetchrow(
            """
            SELECT body FROM definitions
            WHERE definition_id = $1
              AND definition_type = 'condition'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            condition_id,
        )
        if row is None:
            raise NotFoundError(
                f"Composite operand condition '{condition_id}' not found.",
                location="operands",
                suggestion="Register the operand condition first via POST /registry/definitions.",
            )
        return ConditionDefinition(**_parse_body(row["body"]))

    async def _fetch_bound_actions(
        self, condition_id: str, condition_version: str
    ) -> list[ActionDefinition]:
        """
        Return all registered actions whose trigger binds to (condition_id, condition_version).

        Silently skips malformed action bodies — best-effort contract.
        """
        rows = await self._pool.fetch(
            """
            SELECT body FROM definitions
            WHERE definition_type = 'action'
              AND body->'trigger'->>'condition_id' = $1
              AND body->'trigger'->>'condition_version' = $2
            """,
            condition_id,
            condition_version,
        )
        actions: list[ActionDefinition] = []
        for row in rows:
            try:
                body = _parse_body(row["body"])
                actions.append(ActionDefinition(**body))
            except Exception:
                pass  # skip malformed records
        return actions

    # ── Runtime helpers ────────────────────────────────────────────────────────

    def _make_executor(self) -> ConceptExecutor:
        """Create a ConceptExecutor backed by the shared result cache."""
        return ConceptExecutor(result_cache=self._cache)

    def _make_resolver(self) -> DataResolver:
        """
        Create a fresh DataResolver per execute call.

        A new instance resets the request-scoped primitive cache so separate
        execute() calls do not share fetched primitive values.
        """
        async_registry = {}
        if self._connector_registry is not None:
            async_registry = self._connector_registry._registry
        return DataResolver(
            connector=_make_connector(),
            backoff_base=0.0,
            primitive_sources=self._primitive_sources,
            async_connector_registry=async_registry,
        )

    async def _store_concept_result(
        self, result: ConceptResult, concept_id: str
    ) -> None:
        """
        Persist a concept result for future history retrieval. Best-effort.

        All result types are stored:
          FLOAT       — value stored as float (supports history for numeric strategies)
          BOOLEAN     — stored as 1.0 (True) or 0.0 (False)
          CATEGORICAL — value=None; text stored in output_text column

        Exceptions are caught and logged rather than propagated — history
        storage must never block or fail the primary evaluation path.
        """
        try:
            if result.type == ConceptOutputType.FLOAT:
                store_value = float(result.value) if result.value is not None else None
                store_text = None
            elif result.type == ConceptOutputType.BOOLEAN:
                store_value = 1.0 if result.value else 0.0
                store_text = None
            elif result.type == ConceptOutputType.CATEGORICAL:
                store_value = None
                store_text = str(result.value) if result.value is not None else None
            else:
                return
            await self._result_store.store(
                concept_id=concept_id,
                version=result.version,
                entity=result.entity,
                value=store_value,
                output_type=result.type.value,
                output_text=store_text,
            )
        except Exception as exc:
            log.warning(
                "concept_result_store_failed",
                concept_id=concept_id,
                entity=result.entity,
                error=str(exc),
            )

    async def _evaluate_strategy(
        self,
        condition: ConditionDefinition,
        concept_result: ConceptResult,
        entity: str,
        timestamp: str | None,
        executor: ConceptExecutor,
        resolver: DataResolver,
    ) -> DecisionValue:
        """
        Evaluate the condition strategy against concept_result.

        History-based strategies (z_score, percentile, change):
          Fetches up to _HISTORY_WINDOW prior concept results from the
          concept_results table for this (concept_id, entity) pair.
          If fewer than _HISTORY_MIN_RESULTS are available, returns
          DecisionValue(value=False, reason="insufficient_history") without
          calling the strategy — the evaluation cannot be meaningful yet.
          If the history query fails, falls back to reason="history_unavailable".

        Composite strategy:
          Recursively evaluates each operand condition (resolved to its latest
          registered version) and passes the resulting DecisionValue list to
          CompositeStrategy.

        All other strategies (threshold, equals):
          Evaluate directly from concept_result; history is not used.

        Returns a DecisionValue.
        """
        strategy_type = StrategyType(condition.strategy.type)
        params = condition.strategy.params.model_dump()

        # ── History fetch for stateful strategies ──────────────────────────────
        history: list[ConceptResult] = []

        if strategy_type in _HISTORY_STRATEGIES:
            history_reason: str | None = None
            try:
                rows = await self._result_store.fetch_history(
                    concept_id=condition.concept_id,
                    entity=entity,
                    limit=_HISTORY_WINDOW,
                )
                # Rows arrive oldest-first from the store (already reversed).
                history = [
                    ConceptResult(
                        value=float(row["value"]),
                        type=ConceptOutputType.FLOAT,
                        entity=row["entity"],
                        version=row["version"],
                        deterministic=True,
                        timestamp=(
                            row["evaluated_at"].isoformat()
                            if row.get("evaluated_at") is not None
                            else None
                        ),
                    )
                    for row in rows
                ]
            except Exception as exc:
                log.warning(
                    "concept_history_fetch_failed",
                    concept_id=condition.concept_id,
                    entity=entity,
                    error=str(exc),
                )
                history_reason = "history_unavailable"

            if history_reason is None and len(history) < _HISTORY_MIN_RESULTS:
                history_reason = "insufficient_history"

            if history_reason is not None:
                return DecisionValue(
                    value=False,
                    decision_type=DecisionType.BOOLEAN,
                    condition_id=condition.condition_id,
                    condition_version=condition.version,
                    entity=entity,
                    timestamp=timestamp,
                    reason=history_reason,
                    history_count=len(history),
                )

        # ── Composite recursive evaluation ─────────────────────────────────────
        if strategy_type == StrategyType.COMPOSITE:
            operand_results = []
            for operand_id in condition.strategy.params.operands:
                operand_condition = await self._fetch_condition_latest(operand_id)
                operand_concept = await self._fetch_concept(
                    operand_condition.concept_id, operand_condition.concept_version
                )
                operand_graph = DAGBuilder().build_dag(operand_concept)
                operand_concept_result = await executor.aexecute_graph(
                    graph=operand_graph,
                    entity=entity,
                    data_resolver=resolver,
                    timestamp=timestamp,
                    explain=False,
                    cache=True,
                )
                operand_decision = await self._evaluate_strategy(
                    operand_condition,
                    operand_concept_result,
                    entity,
                    timestamp,
                    executor,
                    resolver,
                )
                operand_results.append(operand_decision)
            params["operand_results"] = operand_results

        strategy = _STRATEGY_REGISTRY[strategy_type]
        return strategy.evaluate(
            concept_result,
            history,
            params,
            condition_id=condition.condition_id,
            condition_version=condition.version,
        )

    # ── ψ layer: concept execution ─────────────────────────────────────────────

    async def execute(self, req: ExecuteRequest) -> ConceptResult:
        """
        ψ layer — compile and execute a concept for a single entity.

        Loads the concept from DB, compiles via DAGBuilder, executes the graph.

        Determinism contract:
          timestamp present → deterministic=True, result cached.
          timestamp absent  → snapshot mode, deterministic=False, not cached.

        Raises NotFoundError (→ HTTP 404) when the concept is not registered.
        """
        concept = await self._fetch_concept(req.id, req.version)
        graph = DAGBuilder().build_dag(concept)
        executor = self._make_executor()
        resolver = self._make_resolver()

        result = await executor.aexecute_graph(
            graph=graph,
            entity=req.entity,
            data_resolver=resolver,
            timestamp=req.timestamp,
            explain=req.explain,
            cache=req.cache,
        )
        log.info(
            "service_execute",
            concept_id=req.id,
            version=req.version,
            entity=req.entity,
            timestamp=req.timestamp,
        )
        return result

    async def execute_graph(self, req: ExecuteGraphRequest) -> ConceptResult:
        """
        ψ layer via pre-compiled graph — execute by graph_id.

        Retrieves the stored ExecutionGraph from GraphStore and executes it.
        Verifies ir_hash when provided — mismatch raises ConflictError (→ 409).

        Raises NotFoundError  (→ HTTP 404) when graph_id is not in GraphStore.
        Raises ConflictError  (→ HTTP 409) on ir_hash mismatch.
        """
        graph_store = GraphStore(self._pool)
        graph = await graph_store.get(req.graph_id)
        if graph is None:
            raise NotFoundError(
                f"Graph '{req.graph_id}' not found.",
                location="graph_id",
                suggestion="Compile the concept first via POST /compile.",
            )

        if req.ir_hash is not None and graph.ir_hash != req.ir_hash:
            raise MemintelError(
                ErrorType.CONFLICT,
                f"ir_hash mismatch for graph '{req.graph_id}'. "
                f"Expected '{req.ir_hash}', stored '{graph.ir_hash}'. "
                "Recompile the concept and update your cached graph_id.",
                location="ir_hash",
            )

        executor = self._make_executor()
        resolver = self._make_resolver()

        result = await executor.aexecute_graph(
            graph=graph,
            entity=req.entity,
            data_resolver=resolver,
            timestamp=req.timestamp,
            explain=req.explain,
            cache=req.cache,
        )
        log.info(
            "service_execute_graph",
            graph_id=req.graph_id,
            entity=req.entity,
            timestamp=req.timestamp,
        )
        return result

    async def execute_batch(self, req: Any) -> BatchExecuteResult:
        """
        ψ layer for N entities — evaluate the same concept for multiple entities.

        Concept is compiled once; per-entity failures are captured in
        BatchExecuteItem.error and do NOT abort the batch.  Always returns
        HTTP 200 with results in the same order as req.entities.

        Raises NotFoundError (→ HTTP 404) when the concept is not registered.
        """
        concept = await self._fetch_concept(req.id, req.version)
        graph = DAGBuilder().build_dag(concept)
        executor = self._make_executor()

        items: list[BatchExecuteItem] = []
        for entity in req.entities:
            try:
                resolver = self._make_resolver()
                result = await executor.aexecute_graph(
                    graph=graph,
                    entity=entity,
                    data_resolver=resolver,
                    timestamp=req.timestamp,
                    explain=getattr(req, "explain", False),
                    cache=True,
                )
                items.append(BatchExecuteItem(entity=entity, result=result))
            except Exception as exc:
                err_type = (
                    exc.error_type
                    if isinstance(exc, MemintelError)
                    else ErrorType.EXECUTION_ERROR
                )
                items.append(BatchExecuteItem(
                    entity=entity,
                    error=ErrorResponse(
                        error=ErrorDetail(type=err_type, message=str(exc))
                    ),
                ))

        failed = sum(1 for item in items if item.error is not None)
        log.info(
            "service_execute_batch",
            concept_id=req.id,
            version=req.version,
            total=len(items),
            failed=failed,
        )
        return BatchExecuteResult(results=items, total=len(items), failed=failed)

    async def execute_range(self, req: Any) -> list[ConceptResult]:
        """
        ψ layer over a time range — execute a concept for one entity at each
        step within [from_timestamp, to_timestamp] using the given interval.

        Returns one ConceptResult per step in chronological order.  All results
        are deterministic (timestamps are explicit).

        Raises NotFoundError (→ HTTP 404) when the concept is not registered.
        Raises MemintelError(EXECUTION_ERROR) for invalid interval / timestamps.
        """
        try:
            step = _parse_iso_duration(req.interval)
        except ValueError as exc:
            raise MemintelError(
                ErrorType.EXECUTION_ERROR,
                f"Cannot parse interval '{req.interval}': {exc}",
                location="interval",
            ) from exc

        if step.total_seconds() <= 0:
            raise MemintelError(
                ErrorType.EXECUTION_ERROR,
                "Interval must be a positive duration.",
                location="interval",
            )

        try:
            current = datetime.fromisoformat(
                req.from_timestamp.replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                req.to_timestamp.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise MemintelError(
                ErrorType.EXECUTION_ERROR,
                f"Cannot parse timestamp: {exc}",
                location="from_timestamp",
            ) from exc

        if current > end:
            raise MemintelError(
                ErrorType.EXECUTION_ERROR,
                "from_timestamp must not be after to_timestamp.",
                location="from_timestamp",
            )

        concept = await self._fetch_concept(req.id, req.version)
        graph = DAGBuilder().build_dag(concept)
        executor = self._make_executor()

        results: list[ConceptResult] = []
        while current <= end:
            ts = current.strftime("%Y-%m-%dT%H:%M:%SZ")
            resolver = self._make_resolver()
            result = await executor.aexecute_graph(
                graph=graph,
                entity=req.entity,
                data_resolver=resolver,
                timestamp=ts,
                explain=getattr(req, "explain", False),
                cache=True,
            )
            results.append(result)
            current += step

        log.info(
            "service_execute_range",
            concept_id=req.id,
            version=req.version,
            entity=req.entity,
            steps=len(results),
        )
        return results

    async def execute_async(self, req: ExecuteRequest) -> Job:
        """
        Enqueue a concept execution job and return immediately (HTTP 202).

        Stores the job in the jobs table with status='queued'.  The caller
        polls GET /jobs/{job_id} for status and result.

        Note: background job execution is not yet implemented.  The job is
        enqueued but not automatically picked up by a worker.
        """
        job_store = JobStore(self._pool)
        job = await job_store.enqueue(req.model_dump())
        log.info(
            "service_execute_async",
            job_id=job.job_id,
            concept_id=req.id,
            version=req.version,
            entity=req.entity,
        )
        return job

    # ── φ layer: condition evaluation ─────────────────────────────────────────

    async def evaluate_condition(self, req: Any) -> DecisionResult:
        """
        φ layer — evaluate a condition for a single entity.

        1. Fetches condition + linked concept from DB.
        2. Compiles and executes the concept (ψ) → ConceptResult.
        3. Evaluates the condition strategy → DecisionValue.
        4. Returns DecisionResult with empty actions_triggered.

        Raises NotFoundError (→ HTTP 404) when condition or concept not found.
        """
        condition = await self._fetch_condition(req.condition_id, req.condition_version)
        concept = await self._fetch_concept(condition.concept_id, condition.concept_version)

        graph = DAGBuilder().build_dag(concept)
        executor = self._make_executor()
        resolver = self._make_resolver()

        concept_result = await executor.aexecute_graph(
            graph=graph,
            entity=req.entity,
            data_resolver=resolver,
            timestamp=req.timestamp,
            explain=getattr(req, "explain", False),
            cache=True,
        )

        decision = await self._evaluate_strategy(
            condition, concept_result, req.entity, req.timestamp, executor, resolver
        )
        await self._store_concept_result(concept_result, condition.concept_id)

        log.info(
            "service_evaluate_condition",
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            entity=req.entity,
            timestamp=req.timestamp,
            decision=str(decision.value),
        )
        return DecisionResult(
            value=decision.value,
            type=decision.decision_type,
            entity=req.entity,
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            timestamp=req.timestamp,
            actions_triggered=[],
            reason=decision.reason,
            history_count=decision.history_count,
        )

    async def evaluate_condition_batch(self, req: Any) -> list[DecisionResult]:
        """
        φ layer for N entities — evaluate the same condition for multiple entities.

        Condition + concept loaded once; evaluated independently per entity.
        Per-entity failures propagate to the caller (unlike execute_batch which
        absorbs them).

        Raises NotFoundError (→ HTTP 404) when condition or concept not found.
        """
        condition = await self._fetch_condition(req.condition_id, req.condition_version)
        concept = await self._fetch_concept(condition.concept_id, condition.concept_version)

        graph = DAGBuilder().build_dag(concept)
        executor = self._make_executor()

        results: list[DecisionResult] = []
        for entity in req.entities:
            resolver = self._make_resolver()
            concept_result = await executor.aexecute_graph(
                graph=graph,
                entity=entity,
                data_resolver=resolver,
                timestamp=req.timestamp,
                cache=True,
            )
            decision = await self._evaluate_strategy(
                condition, concept_result, entity, req.timestamp, executor, resolver
            )
            await self._store_concept_result(concept_result, condition.concept_id)
            results.append(DecisionResult(
                value=decision.value,
                type=decision.decision_type,
                entity=entity,
                condition_id=req.condition_id,
                condition_version=req.condition_version,
                timestamp=req.timestamp,
                actions_triggered=[],
                reason=decision.reason,
                history_count=decision.history_count,
            ))

        log.info(
            "service_evaluate_condition_batch",
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            entity_count=len(results),
        )
        return results

    # ── ψ → φ → α pipeline ────────────────────────────────────────────────────

    async def evaluate_full(self, req: Any) -> FullPipelineResult:
        """
        Full ψ → φ → α pipeline — concept + condition + actions.

        1. ψ: compile and execute concept → ConceptResult.
        2. φ: evaluate condition strategy → DecisionValue.
        3. α: load bound actions from DB, trigger each (best-effort, never blocks).
        4. δ: record decision to decisions table (fire-and-forget, never fails pipeline).

        dry_run=True: ψ and φ execute fully; α returns status='would_trigger'
        without making real deliveries.

        actions_triggered[] is nested inside decision (DecisionResult), NOT at
        the top level of FullPipelineResult — this is a hard API contract.

        Raises NotFoundError (→ HTTP 404) when concept or condition not found.
        """
        condition = await self._fetch_condition(req.condition_id, req.condition_version)
        concept = await self._fetch_concept(req.concept_id, req.concept_version)

        # ψ: compile and execute concept; collect primitive values for audit.
        graph = DAGBuilder().build_dag(concept)
        executor = self._make_executor()
        resolver = self._make_resolver()
        primitive_collector: dict = {}

        concept_result = await executor.aexecute_graph(
            graph=graph,
            entity=req.entity,
            data_resolver=resolver,
            timestamp=req.timestamp,
            explain=getattr(req, "explain", False),
            cache=True,
            primitive_collector=primitive_collector,
        )

        # Task status (paused, deleted) is not checked here. evaluate_full takes
        # concept_id/condition_id directly and evaluates regardless of task status.
        # The scheduler layer (Canvas) is responsible for not dispatching evaluations
        # for paused or deleted tasks.

        # φ: evaluate condition strategy.
        decision = await self._evaluate_strategy(
            condition, concept_result, req.entity, req.timestamp, executor, resolver
        )
        if not getattr(req, "dry_run", False):
            await self._store_concept_result(concept_result, condition.concept_id)

        # α: load and trigger bound actions (best-effort).
        actions = await self._fetch_bound_actions(req.condition_id, req.condition_version)
        trigger = ActionTrigger()
        triggered: list[ActionTriggered] = await trigger.trigger_bound_actions(
            decision=decision,
            actions=actions,
            dry_run=getattr(req, "dry_run", False),
        )

        decision_result = DecisionResult(
            value=decision.value,
            type=decision.decision_type,
            entity=req.entity,
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            timestamp=req.timestamp,
            actions_triggered=triggered,
            reason=decision.reason,
            history_count=decision.history_count,
        )

        # δ: build and persist decision record (fire-and-forget — never fails pipeline).
        from app.runtime.data_resolver import PrimitiveValue as _PV  # local to avoid circular

        input_primitives: dict[str, Any] = {}
        signal_errors: dict[str, str] = {}
        for prim_name, pv in primitive_collector.items():
            input_primitives[prim_name] = pv.value if hasattr(pv, "value") else pv
            if hasattr(pv, "fetch_error") and pv.fetch_error:
                signal_errors[prim_name] = pv.error_msg or ""

        # Determine concept_value for the decision record.
        if concept_result.value is None:
            _concept_value: float | None = None
        elif concept_result.type == ConceptOutputType.FLOAT:
            _concept_value = float(concept_result.value)
        elif concept_result.type == ConceptOutputType.BOOLEAN:
            _concept_value = 1.0 if concept_result.value else 0.0
        else:
            _concept_value = None

        action_ids_fired = [
            t.action_id for t in triggered
            if t.status == ActionTriggeredStatus.TRIGGERED
        ]

        ir_hash_val = getattr(condition, "ir_hash", None)

        decision_record = DecisionRecord(
            concept_id=req.concept_id,
            concept_version=req.concept_version,
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            entity_id=req.entity,
            fired=bool(decision.value),
            concept_value=_concept_value,
            threshold_applied=condition.strategy.params.model_dump(),
            ir_hash=ir_hash_val,
            input_primitives=input_primitives if input_primitives else None,
            signal_errors=signal_errors if signal_errors else None,
            reason=decision.reason,
            action_ids_fired=action_ids_fired,
            dry_run=getattr(req, "dry_run", False),
        )

        _decision_store = DecisionStore(self._pool)

        async def _record_decision() -> None:
            try:
                await _decision_store.record(decision_record)
            except Exception as exc:
                log.warning(
                    "decision_record_failed",
                    condition_id=req.condition_id,
                    entity=req.entity,
                    error=str(exc),
                )

        if not getattr(req, "dry_run", False):
            asyncio.create_task(_record_decision())

        log.info(
            "service_evaluate_full",
            concept_id=req.concept_id,
            concept_version=req.concept_version,
            condition_id=req.condition_id,
            condition_version=req.condition_version,
            entity=req.entity,
            timestamp=req.timestamp,
            dry_run=getattr(req, "dry_run", False),
            actions_count=len(triggered),
        )
        return FullPipelineResult(
            result=concept_result,
            decision=decision_result,
            dry_run=getattr(req, "dry_run", False),
            entity=req.entity,
            timestamp=req.timestamp,
        )
