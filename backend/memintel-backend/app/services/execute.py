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
Recursive operand evaluation for composite conditions is not yet implemented.
Composite strategies receive empty operand_results, which evaluates to the
strategy's identity element (False for AND, False for OR).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

import asyncpg
import structlog

from app.compiler.dag_builder import DAGBuilder
from app.models.action import ActionDefinition
from app.models.condition import ConditionDefinition, DecisionType, StrategyType
from app.models.concept import ConceptDefinition
from app.models.errors import (
    ErrorDetail,
    ErrorResponse,
    ErrorType,
    MemintelError,
    NotFoundError,
)
from app.models.result import (
    ActionTriggered,
    BatchExecuteItem,
    BatchExecuteResult,
    ConceptResult,
    DecisionResult,
    ExecuteGraphRequest,
    ExecuteRequest,
    FullPipelineResult,
    Job,
)
from app.runtime.action_trigger import ActionTrigger
from app.runtime.cache import ResultCache
from app.runtime.data_resolver import DataResolver, MockConnector
from app.runtime.executor import ConceptExecutor
from app.stores.graph import GraphStore
from app.stores.job import JobStore
from app.strategies.change import ChangeStrategy
from app.strategies.composite import CompositeStrategy
from app.strategies.equals import EqualsStrategy
from app.strategies.percentile import PercentileStrategy
from app.strategies.threshold import ThresholdStrategy
from app.strategies.z_score import ZScoreStrategy


log = structlog.get_logger(__name__)


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

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        # Shared in-memory result cache — deterministic results survive across
        # execute() calls within the service instance lifetime.
        self._cache = ResultCache()

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
            """,
        )
        actions: list[ActionDefinition] = []
        for row in rows:
            try:
                body = _parse_body(row["body"])
                action = ActionDefinition(**body)
                if (
                    action.trigger.condition_id == condition_id
                    and action.trigger.condition_version == condition_version
                ):
                    actions.append(action)
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
        return DataResolver(connector=_make_connector(), backoff_base=0.0)

    def _evaluate_strategy(
        self,
        condition: ConditionDefinition,
        concept_result: ConceptResult,
        entity: str,
        timestamp: str | None,
    ):
        """
        Evaluate the condition strategy against concept_result.

        Returns a DecisionValue.  Composite conditions receive empty
        operand_results (all operands treated as False) until recursive
        evaluation is implemented.
        """
        strategy_type = StrategyType(condition.strategy.type)
        params = condition.strategy.params.model_dump()

        if strategy_type == StrategyType.COMPOSITE:
            # Recursive operand evaluation not yet implemented.
            # Pass empty operand_results → strategy's identity element.
            params["operand_results"] = []

        strategy = _STRATEGY_REGISTRY[strategy_type]
        return strategy.evaluate(
            concept_result,
            [],  # history — stateful strategies (z_score, change, percentile) use []
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

        result = executor.execute_graph(
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

        result = executor.execute_graph(
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
                result = executor.execute_graph(
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
            result = executor.execute_graph(
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

        concept_result = executor.execute_graph(
            graph=graph,
            entity=req.entity,
            data_resolver=resolver,
            timestamp=req.timestamp,
            explain=getattr(req, "explain", False),
            cache=True,
        )

        decision = self._evaluate_strategy(condition, concept_result, req.entity, req.timestamp)

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
            concept_result = executor.execute_graph(
                graph=graph,
                entity=entity,
                data_resolver=resolver,
                timestamp=req.timestamp,
                cache=True,
            )
            decision = self._evaluate_strategy(condition, concept_result, entity, req.timestamp)
            results.append(DecisionResult(
                value=decision.value,
                type=decision.decision_type,
                entity=entity,
                condition_id=req.condition_id,
                condition_version=req.condition_version,
                timestamp=req.timestamp,
                actions_triggered=[],
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

        dry_run=True: ψ and φ execute fully; α returns status='would_trigger'
        without making real deliveries.

        actions_triggered[] is nested inside decision (DecisionResult), NOT at
        the top level of FullPipelineResult — this is a hard API contract.

        Raises NotFoundError (→ HTTP 404) when concept or condition not found.
        """
        condition = await self._fetch_condition(req.condition_id, req.condition_version)
        concept = await self._fetch_concept(req.concept_id, req.concept_version)

        # ψ: compile and execute concept.
        graph = DAGBuilder().build_dag(concept)
        executor = self._make_executor()
        resolver = self._make_resolver()

        concept_result = executor.execute_graph(
            graph=graph,
            entity=req.entity,
            data_resolver=resolver,
            timestamp=req.timestamp,
            explain=getattr(req, "explain", False),
            cache=True,
        )

        # φ: evaluate condition strategy.
        decision = self._evaluate_strategy(
            condition, concept_result, req.entity, req.timestamp
        )

        # α: load and trigger bound actions (best-effort).
        actions = await self._fetch_bound_actions(req.condition_id, req.condition_version)
        trigger = ActionTrigger()
        triggered: list[ActionTriggered] = trigger.trigger_bound_actions(
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
        )

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
