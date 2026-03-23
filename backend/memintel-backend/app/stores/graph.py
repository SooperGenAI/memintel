"""
app/stores/graph.py
──────────────────────────────────────────────────────────────────────────────
GraphStore — asyncpg-backed persistence for the `execution_graphs` table.

Graph replacement invariant
────────────────────────────
Before any upsert, store() checks whether a graph already exists for
(concept_id, version). If it does:

  new_ir_hash == existing_ir_hash → safe; the upsert is a no-op in practice
                                    (same content, same hash — idempotent)
  new_ir_hash != existing_ir_hash → CRITICAL: raises CompilerInvariantError,
                                    logs at ERROR, and aborts without touching
                                    the existing graph row

A hash mismatch means either the definition body was mutated after
registration (should be impossible given the immutability contract) or the
compiler is non-deterministic. Neither case should ever occur in production —
the error is a signal to investigate immediately.

graph_body layout
─────────────────
The full ExecutionGraph model is serialised to JSONB via model_dump(). On
read, json.loads() + ExecutionGraph.model_validate() reconstructs the object.
The top-level columns (graph_id, concept_id, version, ir_hash) are denormed
from graph_body for indexing — they must always agree with graph_body content.

Column ↔ field mapping
──────────────────────
DB column   Python field (ExecutionGraph)
──────────  ──────────────────────────────
graph_id    graph_id
concept_id  concept_id
version     version
ir_hash     ir_hash
graph_body  (the full serialised model — not a separate field on the model)
created_at  created_at
"""
from __future__ import annotations

import json
import logging

import asyncpg

from app.models.concept import ExecutionGraph
from app.models.errors import CompilerInvariantError

log = logging.getLogger(__name__)


class GraphStore:
    """Async store for the `execution_graphs` table."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── store ─────────────────────────────────────────────────────────────────

    async def store(self, graph: ExecutionGraph) -> str:
        """
        Persist a compiled execution graph and return its graph_id.

        If a graph already exists for (concept_id, version):
          - Same ir_hash  → upsert proceeds (idempotent recompilation).
          - Different hash → CompilerInvariantError raised; existing graph
                             is NOT overwritten.

        Raises CompilerInvariantError on ir_hash mismatch. This is a critical
        compiler bug and must be investigated before any retry.
        """
        existing = await self.get_by_concept(graph.concept_id, graph.version)
        if existing is not None and existing.ir_hash != graph.ir_hash:
            log.error(
                "compiler_invariant_violation",
                extra={
                    "concept_id": graph.concept_id,
                    "version": graph.version,
                    "existing_ir_hash": existing.ir_hash,
                    "new_ir_hash": graph.ir_hash,
                },
            )
            raise CompilerInvariantError(
                concept_id=graph.concept_id,
                version=graph.version,
                existing_hash=existing.ir_hash,
                new_hash=graph.ir_hash,
            )

        graph_body = json.dumps(graph.model_dump(mode="json"))

        await self._pool.execute(
            """
            INSERT INTO execution_graphs (
                graph_id, concept_id, version, ir_hash, graph_body
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (concept_id, version)
            DO UPDATE SET
                graph_id   = EXCLUDED.graph_id,
                ir_hash    = EXCLUDED.ir_hash,
                graph_body = EXCLUDED.graph_body
            """,
            graph.graph_id,
            graph.concept_id,
            graph.version,
            graph.ir_hash,
            graph_body,
        )

        log.info(
            "graph_stored",
            extra={
                "graph_id": graph.graph_id,
                "concept_id": graph.concept_id,
                "version": graph.version,
                "ir_hash": graph.ir_hash,
            },
        )
        return graph.graph_id

    # ── get ───────────────────────────────────────────────────────────────────

    async def get(self, graph_id: str) -> ExecutionGraph | None:
        """
        Return an ExecutionGraph by graph_id, or None if not found.

        Used by POST /execute/graph. The route handler raises HTTP 404 on None.
        """
        row = await self._pool.fetchrow(
            """
            SELECT graph_body, created_at
            FROM execution_graphs
            WHERE graph_id = $1
            """,
            graph_id,
        )
        return _row_to_graph(row) if row else None

    # ── get_by_concept ────────────────────────────────────────────────────────

    async def get_by_concept(
        self, concept_id: str, version: str
    ) -> ExecutionGraph | None:
        """
        Return the compiled graph for a (concept_id, version) pair, or None.

        Used by the compiler (invariant check in store()) and by the executor
        hot path (compile-once → cache graph_id → call execute/graph).
        """
        row = await self._pool.fetchrow(
            """
            SELECT graph_body, created_at
            FROM execution_graphs
            WHERE concept_id = $1 AND version = $2
            """,
            concept_id,
            version,
        )
        return _row_to_graph(row) if row else None


# ── Row mapping helper ────────────────────────────────────────────────────────

def _row_to_graph(row: asyncpg.Record) -> ExecutionGraph:
    """
    Reconstruct an ExecutionGraph from an asyncpg Record.

    graph_body is the full serialised model. created_at from the DB row is
    injected into the model so callers can inspect when the graph was compiled.
    """
    body_raw = row["graph_body"]
    body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw

    # Inject created_at from the DB row (the body may predate this field)
    body["created_at"] = row["created_at"].isoformat() if row["created_at"] else None

    return ExecutionGraph.model_validate(body)
