"""
app/services/compile.py
──────────────────────────────────────────────────────────────────────────────
CompileService — concept compilation to execution graphs.

Compiles a ConceptDefinition into an ExecutionGraph (IR), validates the
graph against the type system, and stores the result in the execution_graphs
table via GraphStore.

Pipeline for compile():
  1. Validator.validate()     — six-phase validation; raises on first error.
  2. DAGBuilder.build_dag()   — produces ExecutionGraph with ir_hash=''.
  3. IRGenerator.hash_graph() — computes and stamps the ir_hash in-place.
  4. GraphStore.store()       — persists; raises CompilerInvariantError when
                                an existing graph's ir_hash does not match.

compile_semantic():
  Runs phases 1–3, then computes a semantic_hash that is stable across
  different concept_id/version pairs that compute the same function.
  Does NOT persist anything.

explain_plan():
  Runs phases 1–2, then delegates to IRGenerator.compile_explain_plan().
  Does NOT persist anything.

Graph replacement invariant:
  Recompiling an unchanged definition must produce the same ir_hash. If
  the existing graph has a different ir_hash, CompilerInvariantError is raised.
  This maps to HTTP 500 (a critical compiler bug, not a client error).
"""
from __future__ import annotations

import hashlib
import json

import asyncpg
import structlog

from app.compiler.dag_builder import DAGBuilder
from app.compiler.ir_generator import IRGenerator
from app.compiler.validator import Validator
from app.models.concept import (
    ConceptDefinition,
    ExecutionGraph,
    ExecutionPlan,
    SemanticGraph,
)
from app.models.errors import ErrorType, MemintelError
from app.stores.graph import GraphStore

log = structlog.get_logger(__name__)


class CompileService:
    """
    Compiles concept definitions to execution graphs.

    compile()          — validates, compiles, stores graph; returns ExecutionGraph.
    compile_semantic() — produces SemanticGraph with stable semantic_hash.
    explain_plan()     — returns ExecutionPlan without compiling or storing.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── compile ────────────────────────────────────────────────────────────────

    async def compile(self, concept: ConceptDefinition) -> ExecutionGraph:
        """
        Validate, compile, and store a concept as an ExecutionGraph.

        Pipeline: validate → build_dag → hash_graph → GraphStore.store().

        If a graph already exists for (concept_id, version):
          - Same ir_hash  → idempotent; returns the stored graph.
          - Different hash → CompilerInvariantError (HTTP 500 — compiler bug).

        Raises MemintelError (HTTP 400) on validation failure.
        Raises CompilerInvariantError (HTTP 500) on ir_hash invariant violation.
        Returns the compiled and stored ExecutionGraph.
        """
        self._validate_or_raise(concept)

        graph = DAGBuilder().build_dag(concept)
        IRGenerator().hash_graph(graph)

        store = GraphStore(self._pool)
        await store.store(graph)

        log.info(
            "compile_complete",
            concept_id=concept.concept_id,
            version=concept.version,
            graph_id=graph.graph_id,
            ir_hash=graph.ir_hash,
        )
        return graph

    # ── compile_semantic ───────────────────────────────────────────────────────

    async def compile_semantic(self, concept: ConceptDefinition) -> SemanticGraph:
        """
        Produce a SemanticGraph with a stable semantic_hash.

        The semantic_hash captures the mathematical meaning of the concept —
        two concepts that compute the same function over the same input types
        produce the same hash, even if their concept_id or version differ.
        The hash is computed from the compiled graph structure with
        concept_id/version/node_id labels stripped.

        Does NOT store anything.

        Raises MemintelError (HTTP 400) on validation failure.
        """
        self._validate_or_raise(concept)

        graph = DAGBuilder().build_dag(concept)

        semantic_hash = _compute_semantic_hash(graph)

        primitive_names = sorted(
            n.params.get("source_name", n.node_id)
            for n in graph.nodes
            if n.op == "primitive_fetch"
        )
        feature_ids = sorted(
            n.node_id for n in graph.nodes if n.op != "primitive_fetch"
        )

        log.info(
            "compile_semantic_complete",
            concept_id=concept.concept_id,
            version=concept.version,
            semantic_hash=semantic_hash,
        )
        return SemanticGraph(
            concept_id=concept.concept_id,
            version=concept.version,
            semantic_hash=semantic_hash,
            features=feature_ids,
            input_primitives=primitive_names,
            equivalences=[],  # cross-registry equivalence lookup not yet implemented
        )

    # ── explain_plan ──────────────────────────────────────────────────────────

    async def explain_plan(self, concept: ConceptDefinition) -> ExecutionPlan:
        """
        Return an ExecutionPlan for the concept without storing anything.

        Equivalent to SQL EXPLAIN — shows execution_order, parallelizable_groups,
        primitive_fetches, and critical_path_length. No data is touched.

        Raises MemintelError (HTTP 400) on validation failure.
        """
        self._validate_or_raise(concept)

        graph = DAGBuilder().build_dag(concept)
        plan = IRGenerator().compile_explain_plan(graph, concept)

        log.info(
            "explain_plan_complete",
            concept_id=concept.concept_id,
            version=concept.version,
            node_count=plan.node_count,
            critical_path_length=plan.critical_path_length,
        )
        return plan

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _validate_or_raise(self, concept: ConceptDefinition) -> None:
        """
        Run the full six-phase validator.  Raise MemintelError on the first error.

        Collecting all errors and raising only the first preserves the single-error
        HTTP response contract used by every other service in this codebase.
        """
        errors = Validator().validate(concept)
        if errors:
            first = errors[0]
            raise MemintelError(
                ErrorType(first.type),   # ValidationErrorItem stores as str (use_enum_values=True)
                first.message,
                location=first.location,
            )


# ── Semantic hash helper ───────────────────────────────────────────────────────

def _compute_semantic_hash(graph: ExecutionGraph) -> str:
    """
    Compute a stable semantic_hash for the compiled graph.

    The hash captures the mathematical meaning of the computation:
      - Topological structure (expressed as positional indices, not node_ids)
      - Operator names and params for each position in the DAG
      - Output type of each node
      - Edge structure (from-position → to-position + input_slot)

    Deliberately EXCLUDES:
      - concept_id and version (naming, not meaning)
      - node_id label strings (positional index used instead)
      - created_at and other metadata

    This means two concepts named differently that compute the same function
    over the same input types produce the same semantic_hash.
    """
    # Build a stable position map: sort nodes by topological order
    topo = graph.topological_order
    topo_idx: dict[str, int] = {}
    for node_id in topo:
        if node_id not in topo_idx:
            topo_idx[node_id] = len(topo_idx)
    # Nodes not in topological_order get a stable fallback index
    for node in sorted(graph.nodes, key=lambda n: n.node_id):
        if node.node_id not in topo_idx:
            topo_idx[node.node_id] = len(topo_idx)

    canonical_nodes = [
        {
            "op": n.op,
            "params": {
                k: v for k, v in sorted(n.params.items())
                # Omit internal meta-params that aren't semantic
                if k != "missing_data_policy" or n.op == "primitive_fetch"
            },
            "output_type": n.output_type,
        }
        for n in sorted(graph.nodes, key=lambda n: topo_idx.get(n.node_id, 9999))
    ]

    canonical_edges = sorted(
        [
            {
                "from_idx": topo_idx.get(e.from_node_id, -1),
                "to_idx": topo_idx.get(e.to_node_id, -1),
                "input_slot": str(e.input_slot),
            }
            for e in graph.edges
        ],
        key=lambda e: (e["from_idx"], e["to_idx"], e["input_slot"]),
    )

    canonical = {
        "output_type": graph.output_type,
        "nodes": canonical_nodes,
        "edges": canonical_edges,
    }
    serialised = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()
