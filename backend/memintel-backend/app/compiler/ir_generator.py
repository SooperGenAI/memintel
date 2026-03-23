"""
app/compiler/ir_generator.py
────────────────────────────────────────────────────────────────────────────────
IR hash computation and execution plan generation.

INVARIANT (core-spec.md §1B, py-instructions.md Compiler Layer):
  Same (concept_id, version) definition → same ir_hash on any machine, always.

hash_graph() achieves this via canonical JSON serialisation:
  - All dict keys sorted alphabetically.
  - Node list sorted by node_id.
  - Edge list sorted by (from_node_id, to_node_id, input_slot).
  - Timestamps, graph_id, and the ir_hash field itself are excluded
    (they are derived or mutable metadata, not semantic content).
  - Hash algorithm: SHA-256 (hexdigest, 64 hex chars).

compile_explain_plan() is the SQL-EXPLAIN equivalent — it returns an
ExecutionPlan from a compiled graph without executing anything.
"""
from __future__ import annotations

import hashlib
import json

from app.models.concept import ConceptDefinition, ExecutionGraph, ExecutionPlan


class IRGenerator:
    """
    Computes ir_hash for a compiled ExecutionGraph and produces ExecutionPlan.

    Public API
    ----------
    hash_graph(graph) → str
        SHA-256 of the canonical graph JSON.
        Sets graph.ir_hash in-place and also returns the value.

    compile_explain_plan(graph, definition) → ExecutionPlan
        Inspection-only plan — does NOT execute the graph.
        Equivalent to SQL EXPLAIN.
    """

    # ── ir_hash ────────────────────────────────────────────────────────────────

    def hash_graph(self, graph: ExecutionGraph) -> str:
        """
        Compute the ir_hash for ``graph``, set it in-place, and return it.

        Canonical serialisation rules:
          - Node list sorted by node_id.
          - Each node's inputs and params dicts have keys sorted.
          - Edge list sorted by (from_node_id, to_node_id, input_slot).
          - Excluded fields: graph_id, ir_hash, created_at (non-semantic metadata).
          - JSON separators: no whitespace (compact, deterministic byte sequence).
          - Encoding: UTF-8.
        """
        canonical = {
            "concept_id":    graph.concept_id,
            "version":       graph.version,
            "output_node_id": graph.output_node_id,
            "output_type":   graph.output_type,
            "nodes": sorted(
                [
                    {
                        "node_id":     n.node_id,
                        "op":          n.op,
                        "inputs":      dict(sorted(n.inputs.items())),
                        "params":      dict(sorted(n.params.items())),
                        "output_type": n.output_type,
                    }
                    for n in graph.nodes
                ],
                key=lambda x: x["node_id"],
            ),
            "edges": sorted(
                [
                    {
                        "from_node_id": e.from_node_id,
                        "to_node_id":   e.to_node_id,
                        "input_slot":   e.input_slot,
                    }
                    for e in graph.edges
                ],
                key=lambda x: (x["from_node_id"], x["to_node_id"], x["input_slot"]),
            ),
        }

        serialised = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
        ir_hash = hashlib.sha256(serialised.encode("utf-8")).hexdigest()

        graph.ir_hash = ir_hash
        return ir_hash

    # ── compile_explain_plan ───────────────────────────────────────────────────

    def compile_explain_plan(
        self,
        graph: ExecutionGraph,
        definition: ConceptDefinition,
    ) -> ExecutionPlan:
        """
        Return the ExecutionPlan for ``graph`` without executing it.

        The plan exposes:
          - execution_order         — node_ids in deterministic topological order.
          - parallelizable_groups   — groups of node_ids safe to run concurrently.
          - primitive_fetches       — primitive names resolved at runtime.
          - critical_path_length    — count of nodes on the longest dependency chain.

        Does NOT touch any data source or execute any operator.
        """
        primitive_fetches = sorted(
            n.params.get("source_name", n.node_id)
            for n in graph.nodes
            if n.op == "primitive_fetch"
        )

        return ExecutionPlan(
            concept_id=graph.concept_id,
            version=graph.version,
            node_count=len(graph.nodes),
            execution_order=list(graph.topological_order),
            parallelizable_groups=[list(g) for g in graph.parallelizable_groups],
            primitive_fetches=primitive_fetches,
            critical_path_length=self._critical_path_length(graph),
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _critical_path_length(self, graph: ExecutionGraph) -> int:
        """
        Return the number of nodes on the longest dependency chain (critical path).

        Uses dynamic programming in topological order: the longest path to each
        node is 1 + max(longest path to each of its direct predecessors).
        """
        if not graph.nodes:
            return 0

        node_ids = {n.node_id for n in graph.nodes}

        # upstream[x] = set of immediate predecessors of x
        upstream: dict[str, set[str]] = {n.node_id: set() for n in graph.nodes}
        for edge in graph.edges:
            if edge.to_node_id in upstream:
                upstream[edge.to_node_id].add(edge.from_node_id)

        longest: dict[str, int] = {}
        for nid in graph.topological_order:
            if nid not in node_ids:
                continue
            preds = upstream.get(nid, set())
            if not preds:
                longest[nid] = 1
            else:
                longest[nid] = 1 + max(longest.get(p, 0) for p in preds)

        return max(longest.values()) if longest else 0
