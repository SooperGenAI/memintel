"""
app/services/compile.py
──────────────────────────────────────────────────────────────────────────────
CompileService — concept compilation to execution graphs.

Compiles a ConceptDefinition into an ExecutionGraph (IR), validates the
graph against the type system, and stores the result in the execution_graphs
table via GraphStore.

Also produces:
  - SemanticGraph (POST /compile/semantic) — canonical semantic view,
    stable meaning_hash for deduplication and equivalence detection.
  - ExecutionPlan (POST /compile/explain-plan) — SQL EXPLAIN equivalent;
    returns execution order and parallelizable groups without executing.

Graph replacement invariant:
  Recompiling an unchanged definition must produce the same ir_hash. If
  the existing graph has a different ir_hash, CompilerInvariantError is raised.

TODO: full implementation in a future session.
"""
from __future__ import annotations

import asyncpg


class CompileService:
    """
    Compiles concept definitions to execution graphs.

    compile()         — validates, compiles, stores graph; returns ExecutionGraph.
    compile_semantic() — produces SemanticGraph with stable semantic_hash.
    explain_plan()    — returns ExecutionPlan without compiling or storing.
    get_graph()       — retrieves a stored graph by graph_id.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
