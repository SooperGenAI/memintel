"""
app/services/execute.py
──────────────────────────────────────────────────────────────────────────────
ExecuteService — concept execution and full pipeline dispatch.

Drives the ψ → φ → α pipeline:
  ψ  ConceptExecutor  — fetches primitives, evaluates the graph
  φ  ConditionExecutor — evaluates the condition strategy
  α  ActionExecutor    — dispatches bound actions (best-effort)

Also handles:
  - Batch execution (execute_batch): runs ψ for N entities, short-circuits on
    per-entity errors without failing the whole batch.
  - Range execution (execute_range): repeats ψ for one entity over a time range.
  - Async execution (execute_async): enqueues a job and returns immediately.
  - Graph execution (execute_graph): executes a pre-compiled graph, bypassing
    compilation.

TODO: full implementation in a future session.
"""
from __future__ import annotations

import asyncpg


class ExecuteService:
    """
    Drives concept and full pipeline execution.

    evaluate_full()  — ψ → φ → α; returns FullPipelineResult.
    execute()        — ψ layer only; returns ConceptResult.
    execute_batch()  — ψ for N entities; returns BatchExecuteResult.
    execute_range()  — ψ over a time range; returns list[ConceptResult].
    execute_async()  — enqueues job; returns Job.
    execute_graph()  — executes a pre-compiled graph; returns ConceptResult.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
